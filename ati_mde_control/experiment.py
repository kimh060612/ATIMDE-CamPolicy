from __future__ import annotations

import time
from typing import Any, Protocol

from hardware.utils import CELL_BY_ID, ContextKey, QScore, SensorCell

from .capture_runner import CaptureRunner
from .config import ExperimentConfig
from .evaluation import DepthEvaluator
from .logging import CaptureLogger
from .pairwise_policy import PairwisePolicy
from .state import ContextState
from .types import PairMode, PairStatus, PairwiseDecision, RoundResult, SwitchEvent


class Predictor(Protocol):
    def predict(self, image, context: ContextKey, exposure_us: float, gain: float) -> QScore: ...
    def predict_batch(self, images, contexts, exposure_us_values, gains) -> list[QScore]: ...


class CameraControlExperiment:
    def __init__(
        self,
        config: ExperimentConfig,
        capture_runner: CaptureRunner,
        predictor: Predictor,
        policy: PairwisePolicy,
        logger: CaptureLogger,
        evaluator: DepthEvaluator,
    ) -> None:
        self.config = config
        self.capture_runner = capture_runner
        self.predictor = predictor
        self.policy = policy
        self.logger = logger
        self.evaluator = evaluator
        self.round_index = 0
        self.last_applied_cell = config.default_cell

    def run_round(self) -> RoundResult:
        context, stable = self.capture_runner.context_state()
        context.validate()
        if not stable:
            result = self._current_only(context, self.last_applied_cell, False, "transition_hold")
        else:
            current = self.policy.committed_cell(context, self.config.default_cell)
            state = self.policy.state(context)
            round_event = self.policy.begin_round(context, self.round_index)
            if round_event is not SwitchEvent.NONE:
                result = self._current_only(context, current, False, round_event.value, round_event)
            elif state.post_switch_dwell_remaining:
                result = self._current_only(context, current, False, "post_switch_dwell")
            elif state.force_current_only_rounds:
                result = self._current_only(context, current, True, "invalid_pair_recovery")
            elif state.rollback_verification_pending:
                result = self._pair_round(context, current, PairMode.ROLLBACK_VERIFICATION)
            elif state.pending_switch_to_id:
                result = self._pair_round(context, current, PairMode.SWITCH_CONFIRMATION)
            elif state.probe_pending:
                result = self._pair_round(context, current, PairMode.NORMAL_SEARCH)
            else:
                result = self._current_only(context, current, False, "current_only")
        self.round_index += 1
        return result

    def _current_only(
        self,
        context: ContextKey,
        current: SensorCell,
        forced: bool,
        reason: str,
        switch_event: SwitchEvent = SwitchEvent.NONE,
    ) -> RoundResult:
        state = self.policy.state(context)
        active_before = state.active_cell_id or current.cell_id
        pending_before = state.probe_pending
        frame = self.capture_runner.capture(
            current, context, "initial", self.round_index, apply_cell=reason != "transition_hold"
        )
        self.last_applied_cell = current
        score = self.predictor.predict(
            frame.image, context, frame.exposure_us, float(frame.actual_gain or frame.cell.gain)
        )
        decision_context, decision_stable = self.capture_runner.context_state()
        capture_valid = frame.context_stable and frame.capture_context == context
        safeguarding = (
            reason == "post_switch_dwell"
            or state.pending_switch_to_id is not None
            or state.rollback_verification_pending
        )
        if reason != "transition_hold" and capture_valid and safeguarding:
            state.rounds_since_valid_probe += 1
            state.probe_pending = True
        elif reason != "transition_hold" and capture_valid:
            self.policy.schedule_after_current(context, float(score.mu))
        elif reason != "transition_hold":
            state.probe_pending = True
        if forced:
            state.force_current_only_rounds = max(0, state.force_current_only_rounds - 1)
        if reason == "post_switch_dwell":
            state.post_switch_dwell_remaining -= 1
        if reason == "transition_hold":
            requested = False
            risk = float(score.mu) + self.config.policy.offload_uncertainty_weight * float(score.uncertainty)
        else:
            requested, risk = self.policy.offload(context, current, score, PairStatus.NOT_PROBED)
            self._apply_for_decision(context, decision_context, decision_stable, current)
        decision_ns = time.time_ns()
        values = self._state_values(state, active_before, pending_before, self.round_index)
        values.update(
            pair_status=PairStatus.NOT_PROBED.value,
            pair_mode="",
            switch_event=switch_event.value,
            committed_switch=0,
            rollback_applied=0,
            round_reason=reason,
            forced_current_only=int(forced),
            selected=1,
            offload_risk=risk,
            offload_requested=int(requested),
            control_decision_delay_ms=(decision_ns - frame.timestamp_ns) / 1_000_000.0,
        )
        self.logger.record(frame, score, values)
        self._print_round(
            context, state, current, None, current,
            PairStatus.NOT_PROBED, None, switch_event, reason,
        )
        self._warn(frame.camera_parameter_ms, float(score.extra.get("mde_inference_ms", 0)), values["control_decision_delay_ms"])
        return RoundResult(frame, score, offload_requested=requested, reason=reason)

    def _pair_round(
        self, context: ContextKey, current: SensorCell, pair_mode: PairMode
    ) -> RoundResult:
        state = self.policy.state(context)
        active_before = state.active_cell_id or current.cell_id
        pending_before = state.probe_pending
        challenger = self.policy.select_challenger(context, current)
        if challenger is None:
            reason = (
                "search_complete" if state.search_cycle_complete
                else f"{pair_mode.value}_cooldown"
            )
            return self._current_only(context, current, False, reason)

        pair = self.capture_runner.capture_pair(current, challenger, context, self.round_index)
        self.last_applied_cell = challenger
        scores = self.predictor.predict_batch(
            [pair.current.image, pair.challenger.image],
            [context, context],
            [pair.current.exposure_us, pair.challenger.exposure_us],
            [
                float(pair.current.actual_gain or current.gain),
                float(pair.challenger.actual_gain or challenger.gain),
            ],
        )
        if len(scores) != 2:
            raise RuntimeError("Pair inference must return exactly two scores.")
        current_score, challenger_score = scores
        decision_context, decision_stable = self.capture_runner.context_state()
        edge_key = (current.cell_id, challenger.cell_id)
        if pair.valid:
            decision = self.policy.resolve(
                context,
                current,
                current_score,
                challenger,
                challenger_score,
                self.round_index,
                pair_mode,
            )
            selected_score = challenger_score if decision.selected_cell == challenger else current_score
            requested, risk = self.policy.offload(context, decision.selected_cell, selected_score, decision.status)
        else:
            self.policy.record_invalid(
                context, current, challenger, self.round_index, pair_mode
            )
            decision = PairwiseDecision(
                PairStatus.INVALID_PAIR,
                current,
                0.0,
                0.0,
                0.0,
                pair_mode,
                SwitchEvent.NONE,
            )
            selected_score, requested = current_score, False
            risk = float(current_score.mu) + self.config.policy.offload_uncertainty_weight * float(current_score.uncertainty)

        self._apply_for_decision(context, decision_context, decision_stable, decision.selected_cell)
        decision_ns = time.time_ns()
        edge = state.local_edges[edge_key]
        common = self._state_values(state, active_before, pending_before, self.round_index)
        common.update(
            challenger_cell_id=challenger.cell_id,
            pair_status=decision.status.value,
            pair_mode=decision.pair_mode.value,
            switch_event=decision.switch_event.value,
            committed_switch=int(decision.switch_event is SwitchEvent.COMMITTED),
            rollback_applied=int(decision.switch_event is SwitchEvent.ROLLED_BACK),
            round_reason=decision.pair_mode.value,
            delta_mu=decision.delta_mu if pair.valid else "",
            pair_std=decision.pair_std if pair.valid else "",
            effective_margin=decision.effective_margin if pair.valid else "",
            pair_capture_gap_ms=pair.gap_ms,
            capture_valid_pair=int(pair.valid),
            pair_invalid_reason=pair.invalid_reason,
            edge_valid_count=edge.valid_count,
            edge_invalid_count=edge.invalid_count,
            edge_invalid_cooldown=edge.invalid_cooldown,
            edge_ambiguous_count=edge.ambiguous_count,
            edge_ambiguous_cooldown=edge.ambiguous_cooldown,
            offload_risk=risk,
            offload_requested=int(requested),
        )
        for frame, score in zip((pair.current, pair.challenger), scores):
            values = dict(common)
            values.update(
                selected=int(frame.cell == decision.selected_cell),
                control_decision_delay_ms=(decision_ns - frame.timestamp_ns) / 1_000_000.0,
            )
            self.logger.record(frame, score, values)
            self._warn(frame.camera_parameter_ms, float(score.extra.get("mde_inference_ms", 0)), values["control_decision_delay_ms"])
        self._print_round(
            context, state, current, challenger, decision.selected_cell,
            decision.status, decision.pair_mode, decision.switch_event,
            decision.pair_mode.value,
        )
        return RoundResult(
            pair.current,
            current_score,
            pair.challenger,
            challenger_score,
            decision,
            offload_requested=requested,
            reason=pair.invalid_reason,
        )

    def _apply_for_decision(
        self,
        latched: ContextKey,
        current_context: ContextKey,
        context_stable: bool,
        selected: SensorCell,
    ) -> None:
        if not context_stable:
            target = self.policy.committed_cell(latched, self.config.default_cell)
        elif current_context == latched:
            target = selected
        else:
            target = self.policy.committed_cell(current_context, self.config.default_cell)
        if target != self.last_applied_cell:
            self.capture_runner.camera.apply_cell(target)
            self.last_applied_cell = target

    def _state_values(
        self,
        state: ContextState,
        active_before: str,
        pending_before: bool,
        round_index: int,
    ) -> dict[str, Any]:
        return {
            "active_cell_before": active_before,
            "active_cell_after": state.active_cell_id or active_before,
            "search_axis": state.search_axis.value,
            "search_direction": state.search_direction.name.lower() if state.search_direction else "",
            "search_cycle_complete": int(state.search_cycle_complete),
            "cycle_had_switch": int(state.cycle_had_switch),
            "probe_pending_before": int(pending_before),
            "probe_pending_after": int(state.probe_pending),
            "rounds_since_valid_probe": state.rounds_since_valid_probe,
            "consecutive_invalid_pairs": state.consecutive_invalid_pairs,
            "force_current_only_rounds": state.force_current_only_rounds,
            "pending_switch_from_id": state.pending_switch_from_id or "",
            "pending_switch_to_id": state.pending_switch_to_id or "",
            "pending_switch_wins": state.pending_switch_wins,
            "switch_confirmations_required": self.config.policy.switch_confirmations,
            "switch_confirmation_age": (
                round_index - state.pending_switch_started_round
                if state.pending_switch_started_round is not None else ""
            ),
            "rollback_cell_id": state.rollback_cell_id or "",
            "rollback_verification_pending": int(state.rollback_verification_pending),
            "post_switch_dwell_remaining": state.post_switch_dwell_remaining,
            "rollback_verification_age": (
                round_index - state.rollback_verification_started_round
                if state.rollback_verification_started_round is not None else ""
            ),
        }

    def _warn(self, camera_ms: float, inference_ms: float, decision_ms: float) -> None:
        for name, value, limit in (
            ("camera_parameter", camera_ms, self.config.camera_parameter_warn_ms),
            ("mde_inference", inference_ms, self.config.mde_inference_warn_ms),
            ("control_decision", decision_ms, self.config.control_decision_warn_ms),
        ):
            if value > limit:
                print(f"[LATENCY WARNING] round={self.round_index} {name}_ms={value:.2f}")

    def _print_round(
        self,
        context: ContextKey,
        state: ContextState,
        current: SensorCell,
        challenger: SensorCell | None,
        selected: SensorCell,
        status: PairStatus,
        mode: PairMode | None,
        event: SwitchEvent,
        reason: str,
    ) -> None:
        active = CELL_BY_ID.get(state.active_cell_id or "", current)
        pair = f"{current.cell_id}->{challenger.cell_id}" if challenger else "none"
        direction = state.search_direction.name.lower() if state.search_direction else "none"
        print(
            f"[Control] round={self.round_index} ctx={context.table_key} "
            f"active={active.cell_id} E={active.exposure_ms}ms G={active.gain} "
            f"search={state.search_axis.value}/{direction} pair={pair} "
            f"selected={selected.cell_id} decision={status.value}/{event.value} "
            f"flow={mode.value if mode else reason} "
            f"confirm={state.pending_switch_wins}/{self.config.policy.switch_confirmations} "
            f"rollback={state.rollback_cell_id or 'none'} "
            f"dwell={state.post_switch_dwell_remaining}"
        )

    def finalize(self) -> Any:
        failure = None
        try:
            self.evaluator.evaluate_rows(self.logger.rows)
            initial = [row for row in self.logger.rows if row["output_delivered"] == 1]
            if initial and not any(row["abs_rel"] != "" for row in initial):
                raise RuntimeError("AbsRel/A1 evaluation failed for every initial frame.")
        except (OSError, RuntimeError, ValueError) as error:
            failure = error
        path = self.logger.write()
        if failure:
            raise failure
        return path
