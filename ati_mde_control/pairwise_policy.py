from __future__ import annotations

import math
import os
import shlex
import subprocess

from hardware.utils import CELL_BY_ID, ContextKey, QScore, SensorCell

from .config import PolicyConfig, SafetyPolicy
from . import local_search
from .state import LocalEdgeState, StateStore, clear_pending
from .types import PairMode, PairStatus, PairwiseDecision, SearchDirection, SwitchEvent


class PairwisePolicy:
    def __init__(self, config: PolicyConfig, safety: SafetyPolicy) -> None:
        self.config = config
        self.safety = safety
        self.states = StateStore(config.initial_search_axis)

    def state(self, context: ContextKey):
        context.validate()
        return self.states.get(context.table_key)

    def committed_cell(self, context: ContextKey, fallback: SensorCell) -> SensorCell:
        safe = self.safety.safe_cells(context)
        state = self.state(context)
        if state.active_cell_id is None:
            chosen = fallback if fallback in safe else safe[0]
            state.active_cell_id = chosen.cell_id
        if state.active_cell_id not in {cell.cell_id for cell in safe}:
            raise RuntimeError(f"Committed cell {state.active_cell_id} is unsafe for {context.table_key}.")
        return CELL_BY_ID[state.active_cell_id]

    def select_challenger(self, context: ContextKey, current: SensorCell) -> SensorCell | None:
        state = self.state(context)
        safe = self.safety.safe_cells(context)
        if state.pending_edge_from_id:
            key = (state.pending_edge_from_id, state.pending_edge_to_id or "")
            edge = state.local_edges.get(key)
            stale = (
                edge is None
                or not edge.pending
                or current.cell_id != key[0]
                or state.active_cell_id != key[0]
                or state.pending_axis is None
                or state.pending_direction is None
                or state.search_axis is not state.pending_axis
                or state.search_direction is not state.pending_direction
                or key[1] not in {cell.cell_id for cell in safe}
            )
            if stale:
                clear_pending(state)
            elif edge.invalid_cooldown or edge.ambiguous_cooldown:
                return None
            else:
                return CELL_BY_ID[key[1]]
        return local_search.select_challenger(
            state, current, safe, self.config.initial_search_axis
        )

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
        current_mu, current_std = self._values(current_score)
        challenger_mu, challenger_std = self._values(challenger_score)
        delta_mu = current_mu - challenger_mu
        pair_std = math.hypot(current_std, challenger_std)
        margin = self.config.switch_margin + self.config.pair_uncertainty_weight * pair_std
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
        if pair_mode is None:
            pair_mode = (
                PairMode.PENDING_RECHECK
                if edge.pending
                and state.pending_edge_from_id == current.cell_id
                and state.pending_edge_to_id == challenger.cell_id
                else PairMode.NORMAL_SEARCH
            )
        edge.valid_count += 1
        edge.last_delta_mu = delta_mu
        edge.last_observed_round = round_index
        state.consecutive_invalid_pairs = 0
        state.rounds_since_valid_probe = 0

        direction = state.pending_direction if pair_mode is PairMode.PENDING_RECHECK else state.search_direction
        if direction is None:
            raise RuntimeError("Pair resolved without a search direction.")
        event = SwitchEvent.NONE
        evidence = (0, 0.0, 0.0, None, None, None, None)
        if status is PairStatus.CHALLENGER_WON:
            if pair_mode is PairMode.PENDING_RECHECK:
                evidence = self._evidence(edge, round_index)
                event = SwitchEvent.PENDING_COMMITTED
            else:
                event = SwitchEvent.IMMEDIATE_COMMIT
            self._commit(state, challenger, direction)
        elif status is PairStatus.CURRENT_WON:
            if pair_mode is PairMode.PENDING_RECHECK:
                evidence = self._evidence(edge, round_index)
                event = SwitchEvent.PENDING_REJECTED
            self._reject(state, direction)
        else:
            edge.ambiguous_count += 1
            if delta_mu > 0 or pair_mode is PairMode.PENDING_RECHECK:
                if not edge.pending:
                    edge.pending = True
                    edge.pending_started_round = round_index
                    state.pending_edge_from_id = current.cell_id
                    state.pending_edge_to_id = challenger.cell_id
                    state.pending_axis = state.search_axis
                    state.pending_direction = direction
                    event = SwitchEvent.PENDING_STARTED
                else:
                    event = SwitchEvent.PENDING_UPDATED
                variance = max(
                    pair_std * pair_std,
                    self.config.pending_std_floor * self.config.pending_std_floor,
                )
                precision = 1.0 / variance
                edge.pending_weighted_delta_sum += delta_mu * precision
                edge.pending_precision_sum += precision
                edge.pending_observation_count += 1
                evidence = self._evidence(edge, round_index)
                count, _, _, aggregate, _, aggregate_margin, _ = evidence
                if aggregate is not None and aggregate > aggregate_margin:
                    event = SwitchEvent.PENDING_COMMITTED
                    self._commit(state, challenger, direction)
                elif aggregate is not None and aggregate < -aggregate_margin:
                    event = SwitchEvent.PENDING_REJECTED
                    self._reject(state, direction)
                elif count >= self.config.max_pending_observations:
                    event = SwitchEvent.PENDING_EXHAUSTED
                    edge.ambiguous_cooldown = self.config.ambiguous_edge_cooldown_rounds
                    clear_pending(state)
                    local_search.mark_direction_tested(state, direction)
                    state.search_direction = None
                else:
                    state.search_direction = direction
            else:
                edge.ambiguous_cooldown = self.config.ambiguous_edge_cooldown_rounds
                if edge.ambiguous_count >= 2:
                    local_search.mark_direction_tested(state, direction)
                state.search_direction = None
        state.probe_pending = True
        count, weighted, precision, aggregate, aggregate_std, aggregate_margin, age = evidence
        return PairwiseDecision(
            status,
            CELL_BY_ID[state.active_cell_id],
            delta_mu,
            pair_std,
            margin,
            pair_mode,
            event,
            count,
            weighted,
            precision,
            aggregate,
            aggregate_std,
            aggregate_margin,
            age,
        )

    def record_invalid(
        self, context: ContextKey, current: SensorCell, challenger: SensorCell, round_index: int
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
        if not edge.pending:
            state.search_direction = None
        state.probe_pending = True
        if state.consecutive_invalid_pairs >= self.config.max_consecutive_invalid_pairs:
            state.force_current_only_rounds = self.config.recovery_current_only_rounds
            state.consecutive_invalid_pairs = 0
        return edge

    def schedule_after_current(self, context: ContextKey, current_mu: float) -> None:
        state = self.state(context)
        state.rounds_since_valid_probe += 1
        if state.pending_edge_from_id:
            state.probe_pending = True
            return
        periodic = state.rounds_since_valid_probe >= self.config.periodic_reprobe_interval
        triggered = current_mu > self.config.probe_trigger_threshold or periodic
        if triggered and state.search_cycle_complete:
            clear_pending(state)
            local_search.restart_cycle(state, self.config.initial_search_axis)
        state.probe_pending = (
            triggered
            or not state.search_cycle_complete
        )

    def begin_round(self, context: ContextKey, round_index: int) -> SwitchEvent:
        state = self.state(context)
        for edge in state.local_edges.values():
            edge.invalid_cooldown = max(0, edge.invalid_cooldown - 1)
            edge.ambiguous_cooldown = max(0, edge.ambiguous_cooldown - 1)
        key = (state.pending_edge_from_id or "", state.pending_edge_to_id or "")
        edge = state.local_edges.get(key)
        if not edge or not edge.pending or edge.pending_started_round is None:
            if state.pending_edge_from_id:
                clear_pending(state)
            return SwitchEvent.NONE
        if round_index - edge.pending_started_round < self.config.pending_timeout_rounds:
            return SwitchEvent.NONE
        axis, direction = state.pending_axis, state.pending_direction
        edge.ambiguous_cooldown = self.config.ambiguous_edge_cooldown_rounds
        clear_pending(state)
        if axis is not None and direction is not None:
            state.search_axis = axis
            local_search.mark_direction_tested(state, direction)
        state.search_direction = None
        state.probe_pending = True
        return SwitchEvent.PENDING_TIMEOUT

    def _evidence(
        self, edge: LocalEdgeState, round_index: int
    ) -> tuple[int, float, float, float | None, float | None, float | None, int | None]:
        precision = edge.pending_precision_sum
        if not precision:
            return (0, 0.0, 0.0, None, None, None, None)
        delta = edge.pending_weighted_delta_sum / precision
        std = math.sqrt(1.0 / precision)
        margin = self.config.switch_margin + self.config.pair_uncertainty_weight * std
        age = (
            round_index - edge.pending_started_round
            if edge.pending_started_round is not None else None
        )
        return (
            edge.pending_observation_count,
            edge.pending_weighted_delta_sum,
            precision,
            delta,
            std,
            margin,
            age,
        )

    @staticmethod
    def _commit(state, challenger: SensorCell, direction: SearchDirection) -> None:
        clear_pending(state)
        state.active_cell_id = challenger.cell_id
        state.cycle_had_switch = True
        local_search.reset_axis(state, state.search_axis)
        state.search_direction = direction
        local_search.mark_direction_tested(
            state,
            SearchDirection.POSITIVE
            if direction is SearchDirection.NEGATIVE else SearchDirection.NEGATIVE,
        )

    @staticmethod
    def _reject(state, direction: SearchDirection) -> None:
        clear_pending(state)
        local_search.mark_direction_tested(state, direction)
        state.search_direction = None

    def offload(self, context: ContextKey, cell: SensorCell, score: QScore, status: PairStatus) -> tuple[bool, float]:
        mu, std = self._values(score)
        risk = mu + self.config.offload_uncertainty_weight * std
        requested = risk > self.config.offload_threshold or (
            status is PairStatus.AMBIGUOUS and mu > self.config.ambiguous_offload_threshold
        )
        if requested and self.config.offload_command:
            environment = os.environ.copy()
            environment.update(
                ATI_MOTION_STATE=str(context.motion_state),
                ATI_LIGHT_STATE=str(context.light_state),
                ATI_BEST_CELL=cell.cell_id,
                ATI_CAMERA_BIAS=str(mu),
                ATI_CAMERA_STD=str(std),
            )
            subprocess.run(shlex.split(self.config.offload_command), env=environment, check=False)
        return requested, risk

    @staticmethod
    def _values(score: QScore) -> tuple[float, float]:
        mu, std = float(score.mu), float(score.uncertainty)
        if not math.isfinite(mu) or not math.isfinite(std) or std < 0:
            raise ValueError("Scores require finite mu and non-negative std.")
        return mu, std
