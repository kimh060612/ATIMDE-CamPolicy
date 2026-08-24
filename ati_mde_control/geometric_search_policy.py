from __future__ import annotations

import math
from collections.abc import Sequence

from hardware.utils import ContextKey, QScore, SensorCell

from .config import PolicyConfig, SafetyPolicy
from .context import LIGHT_LABEL_TO_STATE
from .geometric_search import (
    GeometricProposal,
    GeometricProposalKind,
    GeometricSearchState,
    GeometricVertex,
    cell_to_point,
    contraction_target,
    expansion_target,
    project_safe_untested,
    ranked_vertices,
    reflection_geometry,
    symmetric_axis_candidates,
)
from .pairwise_policy import PairwisePolicy
from .simplex_memory import SimplexMemory, SimplexSnapshot
from .state import LocalEdgeState
from .types import PairMode, PairStatus, PairwiseDecision, SwitchEvent


class GeometricSearchPolicy(PairwisePolicy):
    """Uncertainty-aware pairwise policy backed by a remembered 2D simplex."""

    def __init__(
        self,
        config: PolicyConfig,
        safety: SafetyPolicy,
        simplex_memory_ttl_rounds: int = 0,
    ) -> None:
        if simplex_memory_ttl_rounds < 0:
            raise ValueError("Simplex memory TTL must be non-negative.")
        super().__init__(config, safety)
        self.geometric_states: dict[str, GeometricSearchState] = {}
        self.simplex_memory = SimplexMemory()
        self.simplex_memory_ttl_rounds = simplex_memory_ttl_rounds
        self.current_round = 0
        self.active_context: ContextKey | None = None
        self._brightness_cold_starts: set[ContextKey] = set()

    def state(self, context: ContextKey) -> GeometricSearchState:
        context.validate()
        state = self.geometric_states.get(context.table_key)
        if state is None:
            state = GeometricSearchState(search_axis=self.config.initial_search_axis)
            self.geometric_states[context.table_key] = state
        return state

    def on_context(self, context: ContextKey) -> None:
        """Suspend the active simplex whenever the latched context changes."""

        context.validate()
        if self.active_context == context:
            return
        if self.active_context is not None:
            previous = self.active_context
            self._end_episode(
                previous,
                self.state(previous),
                "context_changed",
                wait_for_periodic=False,
            )
        state = self.state(context)
        self._clear_episode(state, "context_changed", wait_for_periodic=False)
        self.active_context = context

    def reset_for_context_change(self, context: ContextKey) -> None:
        self._end_episode(
            context,
            self.state(context),
            "context_changed",
            wait_for_periodic=False,
        )

    def reset_for_brightness_change(
        self,
        context: ContextKey,
        reason: str = "brightness_state_changed",
    ) -> None:
        """Discard scores observed under the previous brightness envelope."""

        state = self.state(context)
        self.simplex_memory.discard(context)
        self._clear_episode(state, reason, wait_for_periodic=False)
        self._brightness_cold_starts.add(context)

    def begin_round(self, context: ContextKey, round_index: int) -> SwitchEvent:
        if round_index < 0:
            raise ValueError("Round index must be non-negative.")
        self.current_round = round_index
        state = self.state(context)
        for edge in state.local_edges.values():
            edge.invalid_cooldown = max(0, edge.invalid_cooldown - 1)
            edge.ambiguous_cooldown = max(0, edge.ambiguous_cooldown - 1)
        return SwitchEvent.NONE

    def proposal_after_current(
        self,
        context: ContextKey,
        current: SensorCell,
        current_score: QScore,
        *,
        allow_probe: bool = True,
        admissible_cells: Sequence[SensorCell] | None = None,
        prefer_brighter: bool | None = None,
        forced_challenger: SensorCell | None = None,
    ) -> SensorCell | None:
        """Observe the reused current score and select at most one challenger."""

        state = self.state(context)
        hard_safe = self.safety.safe_cells(context)
        if current not in hard_safe:
            raise RuntimeError(
                f"Current cell {current.cell_id} is unsafe for {context.table_key}."
            )
        if admissible_cells is None:
            safe = hard_safe
        else:
            admissible = set(admissible_cells)
            safe = [cell for cell in hard_safe if cell in admissible]
            if current not in safe:
                safe.append(current)
        if forced_challenger is not None:
            if forced_challenger == current or forced_challenger not in safe:
                raise ValueError("Forced brightness recovery must be an admissible hard-safe challenger.")
            self.simplex_memory.discard(context)
            self._clear_episode(
                state,
                GeometricProposalKind.BRIGHTNESS_RECOVERY.value,
                wait_for_periodic=False,
            )
            state.pending_proposal = GeometricProposal(
                forced_challenger,
                GeometricProposalKind.BRIGHTNESS_RECOVERY,
                tuple(map(float, cell_to_point(forced_challenger))),
            )
            state.probe_pending = True
            state.last_geometric_reason = GeometricProposalKind.BRIGHTNESS_RECOVERY.value
            return forced_challenger
        current_mu, current_std = self._values(current_score)
        state.rounds_since_valid_probe += 1
        self._refresh_vertex(
            state, current, current_mu, current_std, self.current_round
        )
        if state.episode_active and self.simplex_memory.vertices_expired(
            state.vertices,
            self.current_round,
            self.simplex_memory_ttl_rounds,
        ):
            self.simplex_memory.discard(context)
            self._clear_episode(
                state,
                "simplex_memory_expired",
                wait_for_periodic=False,
            )
        if state.episode_active:
            self.simplex_memory.update(context, state)

        if state.episode_active and current_mu <= self.config.probe_trigger_threshold:
            self._end_episode(
                context,
                state,
                "good_enough_risk",
                wait_for_periodic=False,
            )
            return None

        if not state.episode_active:
            periodic = (
                state.rounds_since_valid_probe
                >= self.config.periodic_reprobe_interval
            )
            risk_triggered = current_mu > self.config.probe_trigger_threshold
            if state.wait_for_periodic_reprobe:
                risk_triggered = False
            if not (risk_triggered or periodic):
                state.probe_pending = False
                state.last_geometric_reason = (
                    "good_enough_risk"
                    if current_mu <= self.config.probe_trigger_threshold
                    else "current_only"
                )
                return None
            if not self._start_episode(
                context,
                state,
                current,
                current_mu,
                current_std,
                safe,
                prefer_brighter,
            ):
                self._end_episode(
                    context,
                    state,
                    "geometric_search_complete",
                    wait_for_periodic=True,
                )
                return None

        if not allow_probe:
            state.probe_pending = True
            state.last_geometric_reason = "invalid_pair_recovery"
            return None

        proposal = self._next_proposal(context, state, safe)
        if proposal is None:
            return None
        # Mark on selection so an apply/capture failure cannot cause the same
        # expensive camera cell to be proposed again in this episode.
        state.tested_cell_ids.add(proposal.cell.cell_id)
        state.pending_proposal = proposal
        state.probe_pending = True
        state.last_geometric_reason = proposal.kind.value
        return proposal.cell

    def resolve(
        self,
        context: ContextKey,
        current: SensorCell,
        current_score: QScore,
        challenger: SensorCell,
        challenger_score: QScore,
        round_index: int,
        pair_mode: PairMode | None = None,
    ) -> PairwiseDecision:
        state = self.state(context)
        proposal = state.pending_proposal
        if proposal is None or proposal.cell != challenger:
            raise RuntimeError("Geometric pair resolved without its selected proposal.")
        if challenger not in self.safety.safe_cells(context):
            raise RuntimeError(f"Refusing unsafe challenger {challenger.cell_id}.")

        current_mu, current_std = self._values(current_score)
        challenger_mu, challenger_std = self._values(challenger_score)
        delta_mu = current_mu - challenger_mu
        pair_std = math.hypot(current_std, challenger_std)
        margin = (
            self.config.switch_margin
            + self.config.pair_uncertainty_weight * pair_std
        )
        forced_recovery = proposal.kind is GeometricProposalKind.BRIGHTNESS_RECOVERY
        if forced_recovery or delta_mu > margin:
            status = PairStatus.CHALLENGER_WON
        elif delta_mu < -margin:
            status = PairStatus.CURRENT_WON
        else:
            status = PairStatus.AMBIGUOUS
        edge = state.local_edges.setdefault(
            (current.cell_id, challenger.cell_id), LocalEdgeState()
        )
        edge.valid_count += 1
        edge.last_delta_mu = delta_mu
        edge.last_observed_round = round_index
        if status is PairStatus.AMBIGUOUS:
            edge.ambiguous_count += 1
        state.consecutive_invalid_pairs = 0
        state.rounds_since_valid_probe = 0
        state.pending_proposal = None

        selected = current
        event = SwitchEvent.NONE
        if status is PairStatus.CHALLENGER_WON:
            selected = challenger
            state.active_cell_id = challenger.cell_id
            state.cycle_had_switch = True
            event = SwitchEvent.IMMEDIATE_COMMIT

        if forced_recovery:
            pair_z = delta_mu / max(pair_std, self.config.pending_std_floor)
            self.reset_for_brightness_change(context, "brightness_recovery_complete")
            return PairwiseDecision(
                status=status,
                selected_cell=challenger,
                delta_mu=delta_mu,
                pair_std=pair_std,
                effective_margin=margin,
                pair_mode=pair_mode or PairMode.NORMAL_SEARCH,
                switch_event=SwitchEvent.IMMEDIATE_COMMIT,
                pair_z=pair_z,
            )

        observed = GeometricVertex(
            challenger, challenger_mu, challenger_std, round_index
        )
        kind = proposal.kind
        if kind in (
            GeometricProposalKind.INITIAL_EXPOSURE,
            GeometricProposalKind.INITIAL_GAIN,
        ):
            self._append_vertex(state, observed)
            if status is PairStatus.AMBIGUOUS and len(state.vertices) == 3:
                self._schedule_contraction_from_simplex(state)
        elif kind is GeometricProposalKind.REFLECTION:
            if status is PairStatus.CHALLENGER_WON:
                self._replace_vertex(state, proposal.replace_cell_id, observed)
                if proposal.centroid is None:
                    raise RuntimeError("Reflection proposal lost its centroid.")
                self._defer(
                    state,
                    GeometricProposalKind.EXPANSION,
                    expansion_target(proposal.centroid, challenger),
                    challenger.cell_id,
                )
            else:
                if proposal.centroid is None or proposal.replace_cell_id is None:
                    raise RuntimeError("Reflection proposal lost simplex geometry.")
                worst = self._vertex(state, proposal.replace_cell_id).cell
                self._defer(
                    state,
                    GeometricProposalKind.CONTRACTION,
                    contraction_target(proposal.centroid, worst),
                    worst.cell_id,
                )
        elif kind is GeometricProposalKind.EXPANSION:
            if status is PairStatus.CHALLENGER_WON:
                self._replace_vertex(state, proposal.replace_cell_id, observed)
            elif status is PairStatus.AMBIGUOUS:
                self._schedule_contraction_from_simplex(state)
        elif kind is GeometricProposalKind.CONTRACTION:
            if status is PairStatus.CHALLENGER_WON:
                self._replace_vertex(state, proposal.replace_cell_id, observed)
            else:
                self._end_episode(
                    context,
                    state,
                    "contraction_failure",
                    wait_for_periodic=True,
                )

        if state.episode_active:
            state.probe_pending = True
            state.search_cycle_complete = False
            self.simplex_memory.update(context, state)
        pair_z = delta_mu / max(pair_std, self.config.pending_std_floor)
        return PairwiseDecision(
            status=status,
            selected_cell=selected,
            delta_mu=delta_mu,
            pair_std=pair_std,
            effective_margin=margin,
            pair_mode=pair_mode or PairMode.NORMAL_SEARCH,
            switch_event=event,
            pair_z=pair_z,
        )

    def record_invalid(
        self,
        context: ContextKey,
        current: SensorCell,
        challenger: SensorCell,
        round_index: int,
    ) -> LocalEdgeState:
        state = self.state(context)
        edge = state.local_edges.setdefault(
            (current.cell_id, challenger.cell_id), LocalEdgeState()
        )
        edge.invalid_count += 1
        edge.invalid_cooldown = self.config.invalid_edge_cooldown_rounds
        edge.last_observed_round = round_index
        state.consecutive_invalid_pairs += 1
        state.rounds_since_valid_probe += 1
        proposal = state.pending_proposal
        reason = (
            "contraction_failure"
            if proposal is not None
            and proposal.kind is GeometricProposalKind.CONTRACTION
            else "invalid_pair"
        )
        self._end_episode(context, state, reason, wait_for_periodic=True)
        if state.consecutive_invalid_pairs >= self.config.max_consecutive_invalid_pairs:
            state.force_current_only_rounds = self.config.recovery_current_only_rounds
            state.consecutive_invalid_pairs = 0
        return edge

    def current_reason(self, context: ContextKey) -> str:
        return self.state(context).last_geometric_reason

    def _start_episode(
        self,
        context: ContextKey,
        state: GeometricSearchState,
        current: SensorCell,
        mu: float,
        uncertainty: float,
        safe: list[SensorCell],
        prefer_brighter: bool | None,
    ) -> bool:
        cold_start = context in self._brightness_cold_starts
        self._brightness_cold_starts.discard(context)
        if not cold_start and self._restore_episode(
            context, state, current, mu, uncertainty, safe
        ):
            return True

        initial_tested = {current.cell_id}
        positive_first = (
            context.light_state != LIGHT_LABEL_TO_STATE["normal"]
            if prefer_brighter is None
            else prefer_brighter
        )
        exposure_candidates = symmetric_axis_candidates(
            current, "exposure", safe, initial_tested, positive_first
        )
        gain_candidates = symmetric_axis_candidates(
            current, "gain", safe, initial_tested, positive_first
        )
        if not exposure_candidates or not gain_candidates:
            return False
        exposure = exposure_candidates[0]
        gain = gain_candidates[0]
        state.episode_id += 1
        state.episode_active = True
        state.search_cycle_complete = False
        state.probe_pending = True
        state.cycle_had_switch = False
        # Local-edge rows are only diagnostics.  Clearing them at episode start
        # makes it impossible for any score-derived value from an older episode
        # to influence or accompany the new simplex.
        state.local_edges.clear()
        state.vertices = [
            GeometricVertex(current, mu, uncertainty, self.current_round)
        ]
        state.tested_cell_ids = {current.cell_id}
        state.initialization_queue = [
            (GeometricProposalKind.INITIAL_EXPOSURE, exposure),
            (GeometricProposalKind.INITIAL_GAIN, gain),
        ]
        state.pending_proposal = None
        state.deferred_kind = None
        state.deferred_target = None
        state.deferred_replace_cell_id = None
        state.wait_for_periodic_reprobe = False
        self.simplex_memory.update(context, state)
        return True

    def _restore_episode(
        self,
        context: ContextKey,
        state: GeometricSearchState,
        current: SensorCell,
        mu: float,
        uncertainty: float,
        safe: list[SensorCell],
    ) -> bool:
        safe_cells = set(safe)
        for snapshot in self.simplex_memory.recall_candidates(
            context,
            self.current_round,
            self.simplex_memory_ttl_rounds,
        ):
            restored = self._compatible_snapshot(
                snapshot,
                current,
                mu,
                uncertainty,
                self.current_round,
                safe_cells,
            )
            if restored is None:
                continue
            vertices, tested_cell_ids, initialization_queue = restored
            state.episode_id = max(state.episode_id, snapshot.episode_id)
            state.episode_active = True
            state.search_cycle_complete = False
            state.probe_pending = True
            state.cycle_had_switch = snapshot.cycle_had_switch
            state.local_edges.clear()
            state.vertices = vertices
            state.tested_cell_ids = tested_cell_ids
            state.initialization_queue = initialization_queue
            state.pending_proposal = None
            state.deferred_kind = snapshot.deferred_kind
            state.deferred_target = snapshot.deferred_target
            state.deferred_replace_cell_id = snapshot.deferred_replace_cell_id
            state.wait_for_periodic_reprobe = False
            self.simplex_memory.update(context, state)
            return True
        return False

    @staticmethod
    def _compatible_snapshot(
        snapshot: SimplexSnapshot,
        current: SensorCell,
        mu: float,
        uncertainty: float,
        observed_round: int,
        safe_cells: set[SensorCell],
    ) -> tuple[
        list[GeometricVertex],
        set[str],
        list[tuple[GeometricProposalKind, SensorCell]],
    ] | None:
        if not snapshot.vertices or any(
            vertex.cell not in safe_cells for vertex in snapshot.vertices
        ):
            return None

        vertices = list(snapshot.vertices)
        tested_cell_ids = set(snapshot.tested_cell_ids)
        initialization_queue = [
            (kind, cell)
            for kind, cell in snapshot.initialization_queue
            if cell in safe_cells and cell.cell_id not in tested_cell_ids
        ]

        for index, vertex in enumerate(vertices):
            if vertex.cell == current:
                vertices[index] = GeometricVertex(
                    current, mu, uncertainty, observed_round
                )
                break
        else:
            if len(vertices) < 3:
                vertices.append(
                    GeometricVertex(current, mu, uncertainty, observed_round)
                )
                initialization_queue = [
                    item for item in initialization_queue if item[1] != current
                ]
        tested_cell_ids.add(current.cell_id)

        if len(vertices) < 3:
            available = {vertex.cell.cell_id for vertex in vertices}
            available.update(cell.cell_id for _, cell in initialization_queue)
            if len(available) < 3:
                return None
        else:
            initialization_queue = []

        if snapshot.deferred_kind is not None:
            vertex_ids = {vertex.cell.cell_id for vertex in vertices}
            if (
                len(vertices) != 3
                or snapshot.deferred_target is None
                or snapshot.deferred_replace_cell_id not in vertex_ids
            ):
                return None

        return vertices, tested_cell_ids, initialization_queue

    def _next_proposal(
        self,
        context: ContextKey,
        state: GeometricSearchState,
        safe: list[SensorCell],
    ) -> GeometricProposal | None:
        if state.deferred_kind is not None:
            kind = state.deferred_kind
            target = state.deferred_target
            replace = state.deferred_replace_cell_id
            if target is None:
                raise RuntimeError("Deferred geometric proposal lost its target.")
            state.deferred_kind = None
            state.deferred_target = None
            state.deferred_replace_cell_id = None
            cell = project_safe_untested(target, safe, state.tested_cell_ids)
            if cell is None:
                reason = (
                    "contraction_failure"
                    if kind is GeometricProposalKind.CONTRACTION
                    else "geometric_search_complete"
                )
                self._end_episode(
                    context, state, reason, wait_for_periodic=True
                )
                return None
            return GeometricProposal(cell, kind, target, replace_cell_id=replace)

        while state.initialization_queue:
            kind, cell = state.initialization_queue.pop(0)
            if cell.cell_id not in state.tested_cell_ids and cell in safe:
                return GeometricProposal(
                    cell, kind, tuple(map(float, cell_to_point(cell)))
                )
        if len(state.vertices) != 3:
            self._end_episode(
                context,
                state,
                "geometric_search_complete",
                wait_for_periodic=True,
            )
            return None

        centroid, target, worst = reflection_geometry(state.vertices)
        cell = project_safe_untested(target, safe, state.tested_cell_ids)
        if cell is None:
            self._end_episode(
                context,
                state,
                "geometric_search_complete",
                wait_for_periodic=True,
            )
            return None
        return GeometricProposal(
            cell,
            GeometricProposalKind.REFLECTION,
            target,
            centroid=centroid,
            replace_cell_id=worst.cell.cell_id,
        )

    @staticmethod
    def _defer(
        state: GeometricSearchState,
        kind: GeometricProposalKind,
        target: tuple[float, float],
        replace_cell_id: str,
    ) -> None:
        state.deferred_kind = kind
        state.deferred_target = target
        state.deferred_replace_cell_id = replace_cell_id

    def _schedule_contraction_from_simplex(
        self, state: GeometricSearchState
    ) -> None:
        best, second, worst = ranked_vertices(state.vertices)
        best_point = cell_to_point(best.cell)
        second_point = cell_to_point(second.cell)
        centroid = (
            (best_point[0] + second_point[0]) / 2.0,
            (best_point[1] + second_point[1]) / 2.0,
        )
        self._defer(
            state,
            GeometricProposalKind.CONTRACTION,
            contraction_target(centroid, worst.cell),
            worst.cell.cell_id,
        )

    @staticmethod
    def _vertex(state: GeometricSearchState, cell_id: str) -> GeometricVertex:
        for vertex in state.vertices:
            if vertex.cell.cell_id == cell_id:
                return vertex
        raise RuntimeError(f"Simplex vertex {cell_id} is missing.")

    @staticmethod
    def _refresh_vertex(
        state: GeometricSearchState,
        cell: SensorCell,
        mu: float,
        uncertainty: float,
        observed_round: int,
    ) -> None:
        if not state.episode_active:
            return
        for index, vertex in enumerate(state.vertices):
            if vertex.cell == cell:
                state.vertices[index] = GeometricVertex(
                    cell, mu, uncertainty, observed_round
                )
                return

    @staticmethod
    def _append_vertex(
        state: GeometricSearchState, vertex: GeometricVertex
    ) -> None:
        state.vertices = [item for item in state.vertices if item.cell != vertex.cell]
        state.vertices.append(vertex)
        if len(state.vertices) > 3:
            state.vertices = state.vertices[-3:]

    @staticmethod
    def _replace_vertex(
        state: GeometricSearchState,
        replace_cell_id: str | None,
        vertex: GeometricVertex,
    ) -> None:
        if replace_cell_id is None:
            raise RuntimeError("Geometric proposal lost its replacement vertex.")
        replaced = False
        updated: list[GeometricVertex] = []
        for item in state.vertices:
            if item.cell.cell_id == replace_cell_id:
                if not replaced:
                    updated.append(vertex)
                    replaced = True
            else:
                updated.append(item)
        if not replaced:
            raise RuntimeError(f"Simplex vertex {replace_cell_id} is missing.")
        state.vertices = updated

    def _end_episode(
        self,
        context: ContextKey,
        state: GeometricSearchState,
        reason: str,
        *,
        wait_for_periodic: bool,
    ) -> None:
        # A selected but unresolved proposal has no risk observation yet. Keep
        # the last stable snapshot instead of caching an incomplete mutation.
        if state.pending_proposal is None:
            self.simplex_memory.update(context, state)
        self._clear_episode(state, reason, wait_for_periodic=wait_for_periodic)

    @staticmethod
    def _clear_episode(
        state: GeometricSearchState,
        reason: str,
        *,
        wait_for_periodic: bool,
    ) -> None:
        state.episode_active = False
        state.vertices = []
        state.tested_cell_ids = set()
        state.initialization_queue = []
        state.pending_proposal = None
        state.deferred_kind = None
        state.deferred_target = None
        state.deferred_replace_cell_id = None
        state.search_cycle_complete = True
        state.probe_pending = False
        state.cycle_had_switch = False
        state.pending_edge_from_id = None
        state.pending_edge_to_id = None
        state.pending_axis = None
        state.pending_direction = None
        state.last_geometric_reason = reason
        state.wait_for_periodic_reprobe = wait_for_periodic
        if wait_for_periodic:
            state.rounds_since_valid_probe = 0
