from __future__ import annotations

import time

from hardware.utils import ContextKey, QScore, SensorCell

from .experiment import CameraControlExperiment
from .types import (
    CapturedFrame,
    PairMode,
    PairStatus,
    PairwiseDecision,
    RoundResult,
    SwitchEvent,
)


_CAPTURE_ERRORS = (OSError, RuntimeError, TimeoutError, ValueError)


class LocalSearchUpdateExperiment(CameraControlExperiment):
    """Pairwise control that reuses the current capture and inference.

    The legacy experiment deliberately keeps its back-to-back pair capture path.
    This class changes only the execution order: every round captures and scores
    the committed cell first, then captures and scores only a challenger when the
    existing policy requests a probe.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The camera's startup parameter is unknown until the first successful
        # apply/capture.  None makes that first write explicit and keeps this
        # tracker aligned with writes made by this control path.
        self.last_applied_cell: SensorCell | None = None

    def run_round(self) -> RoundResult:
        context, initially_stable = self.capture_runner.context_state()
        context.validate()
        state = self.policy.state(context)
        round_event = SwitchEvent.NONE

        if initially_stable:
            current = self.policy.committed_cell(context, self.config.default_cell)
            round_event = self.policy.begin_round(context, self.round_index)
        else:
            # During a transition, hold the physical setting when it is known.
            # If this is the first round, establish the context's safe default.
            current = self.last_applied_cell or self.policy.committed_cell(
                context, self.config.default_cell
            )

        active_before = state.active_cell_id or current.cell_id
        pending_before = state.probe_pending
        apply_current = current != self.last_applied_cell
        current_frame = self.capture_runner.capture(
            current,
            context,
            "initial",
            self.round_index,
            apply_cell=apply_current,
        )
        self.last_applied_cell = current
        current_score = self._predict(current_frame, context)

        decision_context, decision_stable = self.capture_runner.context_state()
        current_valid, current_invalid_reason = self._current_validity(
            context,
            initially_stable,
            current_frame,
            decision_context,
            decision_stable,
        )

        forced = initially_stable and bool(state.force_current_only_rounds)
        if not current_valid and initially_stable:
            state.probe_pending = True
        if forced:
            state.force_current_only_rounds = max(0, state.force_current_only_rounds - 1)

        if not current_valid:
            # A failed setting verification needs an explicit rewrite because
            # last_applied_cell records the attempted setting, not its efficacy.
            if current_invalid_reason == "current_setting_ineffective":
                self.last_applied_cell = None
            reason = "transition_hold" if not initially_stable else current_invalid_reason
            result = self._finish_current_only(
                context,
                current,
                current_frame,
                current_score,
                active_before,
                pending_before,
                forced,
                reason,
                round_event,
                decision_context,
                decision_stable,
                allow_decision=False,
            )
        elif forced:
            self.policy.schedule_after_current(context, float(current_score.mu))
            result = self._finish_current_only(
                context,
                current,
                current_frame,
                current_score,
                active_before,
                pending_before,
                True,
                "invalid_pair_recovery",
                round_event,
                decision_context,
                decision_stable,
            )
        elif round_event is SwitchEvent.PENDING_TIMEOUT:
            self.policy.schedule_after_current(context, float(current_score.mu))
            result = self._finish_current_only(
                context,
                current,
                current_frame,
                current_score,
                active_before,
                pending_before,
                False,
                round_event.value,
                round_event,
                decision_context,
                decision_stable,
            )
        else:
            probe_was_pending = bool(
                state.pending_edge_from_id or state.probe_pending
            )
            if not probe_was_pending:
                self.policy.schedule_after_current(context, float(current_score.mu))

        if current_valid and not forced and round_event is not SwitchEvent.PENDING_TIMEOUT and (
            state.pending_edge_from_id or state.probe_pending
        ):
            challenger = self.policy.select_challenger(context, current)
            if challenger is None:
                if probe_was_pending:
                    # This is the same current-only scheduling performed by the
                    # legacy path after a pending probe finds no usable edge.
                    self.policy.schedule_after_current(
                        context, float(current_score.mu)
                    )
                reason = (
                    "pending_edge_cooldown"
                    if state.pending_edge_from_id
                    else "search_complete"
                    if state.search_cycle_complete
                    else "local_edges_cooling_down"
                )
                result = self._finish_current_only(
                    context,
                    current,
                    current_frame,
                    current_score,
                    active_before,
                    pending_before,
                    False,
                    reason,
                    round_event,
                    decision_context,
                    decision_stable,
                )
            else:
                result = self._finish_probe(
                    context,
                    current,
                    current_frame,
                    current_score,
                    challenger,
                    active_before,
                    pending_before,
                )
        elif current_valid and not forced and round_event is not SwitchEvent.PENDING_TIMEOUT:
            result = self._finish_current_only(
                context,
                current,
                current_frame,
                current_score,
                active_before,
                pending_before,
                False,
                "current_only",
                round_event,
                decision_context,
                decision_stable,
            )

        self.round_index += 1
        return result

    def _predict(self, frame: CapturedFrame, context: ContextKey) -> QScore:
        return self.predictor.predict(
            frame.image,
            context,
            frame.exposure_us,
            float(frame.actual_gain or frame.cell.gain),
        )

    @staticmethod
    def _current_validity(
        context: ContextKey,
        initially_stable: bool,
        frame: CapturedFrame,
        decision_context: ContextKey,
        decision_stable: bool,
    ) -> tuple[bool, str]:
        if not initially_stable or not frame.context_stable or not decision_stable:
            return False, "context_unstable"
        if frame.capture_context != context or decision_context != context:
            return False, "context_changed"
        if not frame.setting_effective:
            return False, "current_setting_ineffective"
        return True, ""

    def _finish_current_only(
        self,
        context: ContextKey,
        current: SensorCell,
        frame: CapturedFrame,
        score: QScore,
        active_before: str,
        pending_before: bool,
        forced: bool,
        reason: str,
        switch_event: SwitchEvent,
        decision_context: ContextKey,
        decision_stable: bool,
        *,
        allow_decision: bool = True,
    ) -> RoundResult:
        state = self.policy.state(context)
        if allow_decision:
            requested, risk = self.policy.offload(
                context, current, score, PairStatus.NOT_PROBED
            )
        else:
            requested = False
            risk = float(score.mu) + self.config.policy.offload_uncertainty_weight * float(
                score.uncertainty
            )

        self._apply_for_decision(
            context, decision_context, decision_stable, current
        )
        decision_ns = time.time_ns()
        values = self._state_values(state, active_before, pending_before)
        values.update(
            pair_status=PairStatus.NOT_PROBED.value,
            pair_mode="",
            switch_event=switch_event.value,
            immediate_commit=0,
            pending_commit=0,
            pending_timeout=int(switch_event is SwitchEvent.PENDING_TIMEOUT),
            forced_current_only=int(forced),
            selected=1,
            output_delivered=1,
            offload_risk=risk,
            offload_requested=int(requested),
            control_decision_delay_ms=(decision_ns - frame.timestamp_ns) / 1_000_000.0,
        )
        self.logger.record(frame, score, values)
        self._print_round(
            context,
            state,
            current,
            None,
            current,
            PairStatus.NOT_PROBED,
            None,
            switch_event,
            reason,
        )
        self._warn(
            frame.camera_parameter_ms,
            float(score.extra.get("mde_inference_ms", 0)),
            values["control_decision_delay_ms"],
        )
        return RoundResult(frame, score, offload_requested=requested, reason=reason)

    def _finish_probe(
        self,
        context: ContextKey,
        current: SensorCell,
        current_frame: CapturedFrame,
        current_score: QScore,
        challenger: SensorCell,
        active_before: str,
        pending_before: bool,
    ) -> RoundResult:
        state = self.policy.state(context)
        pair_mode = (
            PairMode.PENDING_RECHECK
            if state.pending_edge_from_id == current.cell_id
            and state.pending_edge_to_id == challenger.cell_id
            else PairMode.NORMAL_SEARCH
        )

        try:
            challenger_frame = self.capture_runner.capture(
                challenger,
                context,
                "challenger",
                self.round_index,
                apply_cell=True,
            )
        except _CAPTURE_ERRORS as error:
            return self._finish_failed_probe(
                context,
                current,
                current_frame,
                current_score,
                challenger,
                pair_mode,
                active_before,
                pending_before,
                error,
            )

        self.last_applied_cell = challenger
        challenger_score = self._predict(challenger_frame, context)
        decision_context, decision_stable = self.capture_runner.context_state()
        gap_ms = self._capture_gap_ms(current_frame, challenger_frame)
        valid, invalid_reason = self._challenger_validity(
            context, challenger_frame, decision_context, decision_stable
        )

        edge_key = (current.cell_id, challenger.cell_id)
        if valid:
            decision = self.policy.resolve(
                context,
                current,
                current_score,
                challenger,
                challenger_score,
                self.round_index,
                pair_mode,
            )
            selected_score = (
                challenger_score
                if decision.selected_cell == challenger
                else current_score
            )
            requested, risk = self.policy.offload(
                context, decision.selected_cell, selected_score, decision.status
            )
        else:
            self.policy.record_invalid(
                context, current, challenger, self.round_index
            )
            decision = PairwiseDecision(
                status=PairStatus.INVALID_PAIR,
                selected_cell=current,
                delta_mu=0.0,
                pair_std=0.0,
                effective_margin=0.0,
                pair_mode=pair_mode,
            )
            requested = False
            risk = float(current_score.mu) + self.config.policy.offload_uncertainty_weight * float(
                current_score.uncertainty
            )

        self._apply_for_decision(
            context, decision_context, decision_stable, decision.selected_cell
        )
        decision_ns = time.time_ns()
        edge = state.local_edges[edge_key]
        common = self._state_values(state, active_before, pending_before)
        common.update(
            challenger_cell_id=challenger.cell_id,
            pair_status=decision.status.value,
            pair_mode=decision.pair_mode.value,
            switch_event=decision.switch_event.value,
            delta_mu=decision.delta_mu if valid else "",
            pair_std=decision.pair_std if valid else "",
            pair_z=decision.pair_z if valid else "",
            effective_margin=decision.effective_margin if valid else "",
            pending_min_z=self.config.policy.pending_min_z,
            pending_admitted=int(decision.pending_admitted),
            pending_margin=self.config.policy.pending_margin,
            pending_commit_z=self.config.policy.pending_commit_z,
            pair_capture_gap_ms=gap_ms if gap_ms is not None else "",
            capture_valid_pair=int(valid),
            pair_invalid_reason=invalid_reason,
            edge_valid_count=edge.valid_count,
            edge_invalid_count=edge.invalid_count,
            edge_invalid_cooldown=edge.invalid_cooldown,
            edge_ambiguous_count=edge.ambiguous_count,
            edge_ambiguous_cooldown=edge.ambiguous_cooldown,
            immediate_commit=int(
                decision.switch_event is SwitchEvent.IMMEDIATE_COMMIT
            ),
            pending_commit=int(
                decision.switch_event is SwitchEvent.PENDING_COMMITTED
            ),
            pending_timeout=0,
            offload_risk=risk,
            offload_requested=int(requested),
        )
        if valid and decision.aggregated_delta_mu is not None:
            common.update(
                pending_edge_from_id=current.cell_id,
                pending_edge_to_id=challenger.cell_id,
                pending_observation_count=decision.pending_observation_count,
                pending_age_rounds=decision.pending_age_rounds,
                pending_weighted_delta_sum=decision.pending_weighted_delta_sum,
                pending_precision_sum=decision.pending_precision_sum,
                aggregated_delta_mu=decision.aggregated_delta_mu,
                aggregated_pair_std=decision.aggregated_pair_std,
                aggregated_z=decision.aggregated_z,
                aggregated_pending_margin=decision.aggregated_pending_margin,
            )

        for frame, score in (
            (current_frame, current_score),
            (challenger_frame, challenger_score),
        ):
            selected = frame.cell == decision.selected_cell
            values = dict(common)
            values.update(
                selected=int(selected),
                output_delivered=int(selected),
                control_decision_delay_ms=(decision_ns - frame.timestamp_ns)
                / 1_000_000.0,
            )
            self.logger.record(frame, score, values)
            self._warn(
                frame.camera_parameter_ms,
                float(score.extra.get("mde_inference_ms", 0)),
                values["control_decision_delay_ms"],
            )

        self._print_round(
            context,
            state,
            current,
            challenger,
            decision.selected_cell,
            decision.status,
            decision.pair_mode,
            decision.switch_event,
            decision.pair_mode.value,
        )
        return RoundResult(
            current_frame,
            current_score,
            challenger_frame,
            challenger_score,
            decision,
            offload_requested=requested,
            reason=invalid_reason,
        )

    def _finish_failed_probe(
        self,
        context: ContextKey,
        current: SensorCell,
        current_frame: CapturedFrame,
        current_score: QScore,
        challenger: SensorCell,
        pair_mode: PairMode,
        active_before: str,
        pending_before: bool,
        error: Exception,
    ) -> RoundResult:
        state = self.policy.state(context)
        edge = self.policy.record_invalid(
            context, current, challenger, self.round_index
        )
        reason = f"capture_failure:{type(error).__name__}"
        decision = PairwiseDecision(
            status=PairStatus.INVALID_PAIR,
            selected_cell=current,
            delta_mu=0.0,
            pair_std=0.0,
            effective_margin=0.0,
            pair_mode=pair_mode,
        )
        # capture() may fail after the challenger write.  Force a known safe
        # write instead of guessing which setting the camera retained.
        self.last_applied_cell = None
        decision_context, decision_stable = self.capture_runner.context_state()
        self._apply_for_decision(
            context, decision_context, decision_stable, current
        )
        decision_ns = time.time_ns()
        risk = float(current_score.mu) + self.config.policy.offload_uncertainty_weight * float(
            current_score.uncertainty
        )
        values = self._state_values(state, active_before, pending_before)
        values.update(
            challenger_cell_id=challenger.cell_id,
            pair_status=PairStatus.INVALID_PAIR.value,
            pair_mode=pair_mode.value,
            switch_event=SwitchEvent.NONE.value,
            pair_capture_gap_ms="",
            capture_valid_pair=0,
            pair_invalid_reason=reason,
            edge_valid_count=edge.valid_count,
            edge_invalid_count=edge.invalid_count,
            edge_invalid_cooldown=edge.invalid_cooldown,
            edge_ambiguous_count=edge.ambiguous_count,
            edge_ambiguous_cooldown=edge.ambiguous_cooldown,
            immediate_commit=0,
            pending_commit=0,
            pending_timeout=0,
            offload_risk=risk,
            offload_requested=0,
            selected=1,
            output_delivered=1,
            control_decision_delay_ms=(decision_ns - current_frame.timestamp_ns)
            / 1_000_000.0,
        )
        self.logger.record(current_frame, current_score, values)
        self._warn(
            current_frame.camera_parameter_ms,
            float(current_score.extra.get("mde_inference_ms", 0)),
            values["control_decision_delay_ms"],
        )
        self._print_round(
            context,
            state,
            current,
            challenger,
            current,
            PairStatus.INVALID_PAIR,
            pair_mode,
            SwitchEvent.NONE,
            reason,
        )
        return RoundResult(
            current_frame,
            current_score,
            decision=decision,
            reason=reason,
        )

    @staticmethod
    def _capture_gap_ms(
        current_frame: CapturedFrame, challenger_frame: CapturedFrame
    ) -> float:
        if (
            current_frame.color_timestamp_us is not None
            and challenger_frame.color_timestamp_us is not None
        ):
            return (
                challenger_frame.color_timestamp_us
                - current_frame.color_timestamp_us
            ) / 1000.0
        return (challenger_frame.timestamp_ns - current_frame.timestamp_ns) / 1_000_000.0

    @staticmethod
    def _challenger_validity(
        context: ContextKey,
        challenger_frame: CapturedFrame,
        decision_context: ContextKey,
        decision_stable: bool,
    ) -> tuple[bool, str]:
        if not challenger_frame.setting_effective:
            return False, "setting_ineffective"
        if not challenger_frame.context_stable or not decision_stable:
            return False, "context_unstable"
        if (
            challenger_frame.capture_context != context
            or decision_context != context
        ):
            return False, "context_changed"
        return True, ""
