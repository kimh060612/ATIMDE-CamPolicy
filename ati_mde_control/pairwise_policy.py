from __future__ import annotations

import math
import os
import shlex
import subprocess

from hardware.utils import CELL_BY_ID, ContextKey, QScore, SensorCell

from .config import PolicyConfig, SafetyPolicy
from . import local_search
from .state import ContextState, LocalEdgeState, StateStore
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

    def pair_mode(self, context: ContextKey) -> PairMode:
        state = self.state(context)
        if state.rollback_verification_pending and not state.post_switch_dwell_remaining:
            return PairMode.ROLLBACK_VERIFICATION
        if state.pending_switch_to_id:
            return PairMode.SWITCH_CONFIRMATION
        return PairMode.NORMAL_SEARCH

    def select_challenger(self, context: ContextKey, current: SensorCell) -> SensorCell | None:
        state = self.state(context)
        mode = self.pair_mode(context)
        if mode is PairMode.ROLLBACK_VERIFICATION:
            challenger = CELL_BY_ID[state.rollback_cell_id]
            edge = state.local_edges.get((current.cell_id, challenger.cell_id))
            return None if edge and (edge.invalid_cooldown or edge.ambiguous_cooldown) else challenger
        if mode is PairMode.SWITCH_CONFIRMATION:
            challenger = CELL_BY_ID[state.pending_switch_to_id]
            edge = state.local_edges.get((current.cell_id, challenger.cell_id))
            return None if edge and (edge.invalid_cooldown or edge.ambiguous_cooldown) else challenger
        return local_search.select_challenger(
            state, current, self.safety.safe_cells(context), self.config.initial_search_axis
        )

    def resolve(
        self,
        context: ContextKey,
        current: SensorCell,
        current_score: QScore,
        challenger: SensorCell,
        challenger_score: QScore,
        round_index: int,
        pair_mode: PairMode = PairMode.NORMAL_SEARCH,
    ) -> PairwiseDecision:
        state = self.state(context)
        if pair_mode is PairMode.NORMAL_SEARCH:
            pair_mode = self.pair_mode(context)
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
        edge.valid_count += 1
        edge.last_delta_mu = delta_mu
        edge.last_observed_round = round_index
        state.consecutive_invalid_pairs = 0
        state.rounds_since_valid_probe = 0

        if pair_mode is PairMode.ROLLBACK_VERIFICATION:
            return self._resolve_rollback(
                state, current, challenger, status, delta_mu, pair_std, margin, edge
            )

        if pair_mode is PairMode.SWITCH_CONFIRMATION:
            state.search_axis = state.pending_switch_axis
            state.search_direction = state.pending_switch_direction
        direction = state.search_direction
        if direction is None:
            raise RuntimeError("Pair resolved without a search direction.")
        if status is PairStatus.CHALLENGER_WON:
            same_edge = (
                state.pending_switch_from_id == current.cell_id
                and state.pending_switch_to_id == challenger.cell_id
            )
            if not same_edge:
                state.clear_pending_switch()
                state.pending_switch_from_id = current.cell_id
                state.pending_switch_to_id = challenger.cell_id
                state.pending_switch_started_round = round_index
                state.pending_switch_axis = state.search_axis
                state.pending_switch_direction = direction
            state.pending_switch_wins += 1
            if state.pending_switch_wins >= self.config.switch_confirmations:
                state.active_cell_id = challenger.cell_id
                state.cycle_had_switch = True
                local_search.reset_axis(state, state.search_axis)
                state.search_direction = direction
                local_search.mark_direction_tested(
                    state,
                    SearchDirection.POSITIVE if direction is SearchDirection.NEGATIVE else SearchDirection.NEGATIVE,
                )
                state.clear_pending_switch()
                state.rollback_cell_id = current.cell_id
                state.rollback_verification_pending = True
                state.post_switch_dwell_remaining = self.config.post_switch_dwell_rounds
                event = SwitchEvent.COMMITTED
                selected = challenger
            else:
                event = SwitchEvent.CONFIRMATION_PENDING
                selected = current
        elif status is PairStatus.CURRENT_WON:
            state.clear_pending_switch()
            local_search.mark_direction_tested(state, direction)
            state.search_direction = None
            event, selected = SwitchEvent.NONE, current
        else:
            edge.ambiguous_count += 1
            edge.ambiguous_cooldown = self.config.ambiguous_edge_cooldown_rounds
            if pair_mode is PairMode.SWITCH_CONFIRMATION:
                state.search_direction = state.pending_switch_direction
            elif edge.ambiguous_count >= 2:
                local_search.mark_direction_tested(state, direction)
                state.search_direction = None
            else:
                state.search_direction = None
            event, selected = SwitchEvent.NONE, current
        state.probe_pending = True
        return PairwiseDecision(status, selected, delta_mu, pair_std, margin, pair_mode, event)

    def record_invalid(
        self,
        context: ContextKey,
        current: SensorCell,
        challenger: SensorCell,
        round_index: int,
        pair_mode: PairMode = PairMode.NORMAL_SEARCH,
    ) -> LocalEdgeState:
        state = self.state(context)
        if pair_mode is PairMode.NORMAL_SEARCH:
            pair_mode = self.pair_mode(context)
        edge = state.local_edges.setdefault(
            (current.cell_id, challenger.cell_id), LocalEdgeState()
        )
        edge.invalid_count += 1
        edge.invalid_cooldown = self.config.invalid_edge_cooldown_rounds
        edge.last_observed_round = round_index
        state.consecutive_invalid_pairs += 1
        state.rounds_since_valid_probe += 1
        if pair_mode is PairMode.SWITCH_CONFIRMATION:
            state.search_direction = state.pending_switch_direction
        elif pair_mode is not PairMode.ROLLBACK_VERIFICATION:
            state.search_direction = None
        state.probe_pending = True
        if state.consecutive_invalid_pairs >= self.config.max_consecutive_invalid_pairs:
            state.force_current_only_rounds = self.config.recovery_current_only_rounds
            state.consecutive_invalid_pairs = 0
        return edge

    def schedule_after_current(self, context: ContextKey, current_mu: float) -> None:
        state = self.state(context)
        state.rounds_since_valid_probe += 1
        periodic = state.rounds_since_valid_probe >= self.config.periodic_reprobe_interval
        triggered = current_mu > self.config.probe_trigger_threshold or periodic
        if triggered and state.search_cycle_complete:
            local_search.restart_cycle(state, self.config.initial_search_axis)
        state.probe_pending = (
            triggered
            or not state.search_cycle_complete
        )

    def begin_round(self, context: ContextKey, round_index: int = 0) -> SwitchEvent:
        state = self.state(context)
        for edge in state.local_edges.values():
            edge.invalid_cooldown = max(0, edge.invalid_cooldown - 1)
            edge.ambiguous_cooldown = max(0, edge.ambiguous_cooldown - 1)
        if (
            state.pending_switch_started_round is not None
            and round_index - state.pending_switch_started_round
            >= self.config.switch_confirmation_timeout_rounds
        ):
            edge = state.local_edges[
                (state.pending_switch_from_id, state.pending_switch_to_id)
            ]
            edge.ambiguous_cooldown = self.config.ambiguous_edge_cooldown_rounds
            state.clear_pending_switch()
            state.search_direction = None
            return SwitchEvent.CONFIRMATION_TIMEOUT
        if state.rollback_verification_pending and not state.post_switch_dwell_remaining:
            if state.rollback_verification_started_round is None:
                state.rollback_verification_started_round = round_index
            elif (
                round_index - state.rollback_verification_started_round
                >= self.config.rollback_verification_timeout_rounds
            ):
                state.clear_rollback()
                state.probe_pending = True
                return SwitchEvent.ROLLBACK_TIMEOUT
        return SwitchEvent.NONE

    def _resolve_rollback(
        self,
        state: ContextState,
        current: SensorCell,
        challenger: SensorCell,
        status: PairStatus,
        delta_mu: float,
        pair_std: float,
        margin: float,
        edge: LocalEdgeState,
    ) -> PairwiseDecision:
        selected = current
        if status is PairStatus.CHALLENGER_WON:
            selected = challenger
            direction = state.search_direction
            state.active_cell_id = challenger.cell_id
            original_edge = state.local_edges.setdefault(
                (challenger.cell_id, current.cell_id), LocalEdgeState()
            )
            original_edge.ambiguous_cooldown = self.config.ambiguous_edge_cooldown_rounds
            if direction is not None:
                local_search.reset_axis(state, state.search_axis)
                local_search.mark_direction_tested(state, direction)
            state.clear_rollback()
            event = SwitchEvent.ROLLED_BACK
        elif status is PairStatus.CURRENT_WON:
            state.clear_rollback()
            event = SwitchEvent.ROLLBACK_VERIFIED
        else:
            edge.ambiguous_count += 1
            edge.ambiguous_cooldown = self.config.ambiguous_edge_cooldown_rounds
            state.clear_rollback()
            event = SwitchEvent.ROLLBACK_INCONCLUSIVE
        state.probe_pending = True
        return PairwiseDecision(
            status,
            selected,
            delta_mu,
            pair_std,
            margin,
            PairMode.ROLLBACK_VERIFICATION,
            event,
        )

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
