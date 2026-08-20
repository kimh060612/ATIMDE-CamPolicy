from __future__ import annotations

import math

from hardware.utils import ContextKey, QScore, SensorCell

from .config import PolicyConfig, SafetyPolicy
from .geometric_search import (
    GeometricProposal,
    GeometricProposalKind,
    GeometricSearchState,
    GeometricVertex,
    cell_to_point,
    contraction_target,
    expansion_target,
    nearest_axis_cell,
    project_safe_untested,
    ranked_vertices,
    reflection_geometry,
)
from .pairwise_policy import PairwisePolicy
from .state import LocalEdgeState
from .types import PairMode, PairStatus, PairwiseDecision, SwitchEvent


class GeometricSearchPolicy(PairwisePolicy):
    """Uncertainty-aware pairwise policy backed by a short-lived 2D simplex."""

    def __init__(self, config: PolicyConfig, safety: SafetyPolicy) -> None:
        super().__init__(config, safety)
        self.geometric_states: dict[str, GeometricSearchState] = {}
        self.active_context_key: str | None = None

    def state(self, context: ContextKey) -> GeometricSearchState:
        context.validate()
        state = self.geometric_states.get(context.table_key)
        if state is None:
            state = GeometricSearchState(search_axis=self.config.initial_search_axis)
            self.geometric_states[context.table_key] = state
        return state

    def on_context(self, context: ContextKey) -> None:
        """Discard simplex observations whenever the latched context changes."""

        context.validate()
        key = context.table_key
        if self.active_context_key == key:
            return
        if self.active_context_key in self.geometric_states:
            self._end_episode(
                self.geometric_states[self.active_context_key],
                "context_changed",
                wait_for_periodic=False,
            )
        state = self.state(context)
        self._end_episode(state, "context_changed", wait_for_periodic=False)
        self.active_context_key = key

    def reset_for_context_change(self, context: ContextKey) -> None:
        self._end_episode(
            self.state(context), "context_changed", wait_for_periodic=False
        )

    def begin_round(self, context: ContextKey, round_index: int) -> SwitchEvent:
        del round_index
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
    ) -> SensorCell | None:
        """Observe the reused current score and select at most one challenger."""

        state = self.state(context)
        safe = self.safety.safe_cells(context)
        if current not in safe:
            raise RuntimeError(
                f"Current cell {current.cell_id} is unsafe for {context.table_key}."
            )
        current_mu, current_std = self._values(current_score)
        state.rounds_since_valid_probe += 1
        self._refresh_vertex(state, current, current_mu, current_std)

        if state.episode_active and current_mu <= self.config.probe_trigger_threshold:
            self._end_episode(
                state, "good_enough_risk", wait_for_periodic=False
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
            if not self._start_episode(state, current, current_mu, current_std, safe):
                self._end_episode(
                    state, "geometric_search_complete", wait_for_periodic=True
                )
                return None

        if not allow_probe:
            state.probe_pending = True
            state.last_geometric_reason = "invalid_pair_recovery"
            return None

        proposal = self._next_proposal(state, safe)
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
        status = (
            PairStatus.CHALLENGER_WON
            if delta_mu > margin
            else PairStatus.CURRENT_WON
            if delta_mu < -margin
            else PairStatus.AMBIGUOUS
        )
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

        observed = GeometricVertex(challenger, challenger_mu, challenger_std)
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
                    state, "contraction_failure", wait_for_periodic=True
                )

        if state.episode_active:
            state.probe_pending = True
            state.search_cycle_complete = False
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
        self._end_episode(state, reason, wait_for_periodic=True)
        if state.consecutive_invalid_pairs >= self.config.max_consecutive_invalid_pairs:
            state.force_current_only_rounds = self.config.recovery_current_only_rounds
            state.consecutive_invalid_pairs = 0
        return edge

    def current_reason(self, context: ContextKey) -> str:
        return self.state(context).last_geometric_reason

    def _start_episode(
        self,
        state: GeometricSearchState,
        current: SensorCell,
        mu: float,
        uncertainty: float,
        safe: list[SensorCell],
    ) -> bool:
        exposure = nearest_axis_cell(current, "exposure", safe)
        gain = nearest_axis_cell(current, "gain", safe)
        if exposure is None or gain is None or exposure == gain:
            return False
        state.episode_id += 1
        state.episode_active = True
        state.search_cycle_complete = False
        state.probe_pending = True
        state.cycle_had_switch = False
        # Local-edge rows are only diagnostics.  Clearing them at episode start
        # makes it impossible for any score-derived value from an older episode
        # to influence or accompany the new simplex.
        state.local_edges.clear()
        state.vertices = [GeometricVertex(current, mu, uncertainty)]
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
        return True

    def _next_proposal(
        self, state: GeometricSearchState, safe: list[SensorCell]
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
                self._end_episode(state, reason, wait_for_periodic=True)
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
                state, "geometric_search_complete", wait_for_periodic=True
            )
            return None

        centroid, target, worst = reflection_geometry(state.vertices)
        cell = project_safe_untested(target, safe, state.tested_cell_ids)
        if cell is None:
            self._end_episode(
                state, "geometric_search_complete", wait_for_periodic=True
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
    ) -> None:
        if not state.episode_active:
            return
        for index, vertex in enumerate(state.vertices):
            if vertex.cell == cell:
                state.vertices[index] = GeometricVertex(cell, mu, uncertainty)
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

    @staticmethod
    def _end_episode(
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
