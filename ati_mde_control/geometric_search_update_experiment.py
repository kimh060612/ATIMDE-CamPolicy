from __future__ import annotations

from hardware.utils import ContextKey, QScore, SensorCell

from .brightness_safety import (
    BrightnessDecision,
    BrightnessGuard,
    BrightnessGuardConfig,
    BrightnessGuardMode,
    BrightnessState,
    brightness_log_values,
)
from .geometric_search_policy import GeometricSearchPolicy
from .local_search_update_experiment import LocalSearchUpdateExperiment
from .types import CapturedFrame, RoundResult, SwitchEvent


class GeometricSearchUpdateExperiment(LocalSearchUpdateExperiment):
    """Current-score-reuse execution path with one geometric challenger."""

    policy: GeometricSearchPolicy

    def __init__(
        self,
        *args,
        brightness_guard_config: BrightnessGuardConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.brightness_guard_config = brightness_guard_config or BrightnessGuardConfig()
        self._brightness_guards: dict[ContextKey, BrightnessGuard] = {}

    def _predict(self, frame: CapturedFrame, context: ContextKey) -> QScore:
        # Prefer the control-only score head used by the reference route while
        # remaining compatible with predictor implementations that expose only
        # the original single-frame API.
        predict = getattr(self.predictor, "predict_scores", None)
        if predict is None:
            predict = self.predictor.predict
        return predict(
            frame.image,
            context,
            frame.exposure_us,
            float(frame.actual_gain or frame.cell.gain),
        )

    def run_round(self) -> RoundResult:
        first_row = len(self.logger.rows)
        brightness_decision: BrightnessDecision | None = None
        context, initially_stable = self.capture_runner.context_state()
        context.validate()
        self.policy.on_context(context)
        state = self.policy.state(context)
        round_event = self.policy.begin_round(context, self.round_index)

        if initially_stable:
            current = self.policy.committed_cell(context, self.config.default_cell)
        else:
            safe = self.policy.safety.safe_cells(context)
            current = (
                self.last_applied_cell
                if self.last_applied_cell in safe
                else self.policy.committed_cell(context, self.config.default_cell)
            )

        active_before = state.active_cell_id or current.cell_id
        pending_before = state.probe_pending
        current_frame = self.capture_runner.capture(
            current,
            context,
            "initial",
            self.round_index,
            apply_cell=current != self.last_applied_cell,
        )
        self.last_applied_cell = current
        current_score = self._predict(current_frame, context)

        decision_context, decision_stable = self.capture_runner.context_state()
        current_valid, invalid_reason = self._current_validity(
            context,
            initially_stable,
            current_frame,
            decision_context,
            decision_stable,
        )
        forced = initially_stable and bool(state.force_current_only_rounds)
        if forced:
            state.force_current_only_rounds = max(0, state.force_current_only_rounds - 1)

        if not current_valid:
            self.policy.reset_for_context_change(context)
            if invalid_reason == "current_setting_ineffective":
                self.last_applied_cell = None
            reason = "transition_hold" if not initially_stable else invalid_reason
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
        else:
            mode = self.brightness_guard_config.mode
            if mode is not BrightnessGuardMode.OFF:
                first_brightness_observation = context not in self._brightness_guards
                guard = self._brightness_guards.setdefault(
                    context,
                    BrightnessGuard(self.brightness_guard_config),
                )
                brightness_decision = guard.observe(
                    current_frame.image,
                    current,
                    self.policy.safety.safe_cells(context),
                )
                if (
                    (first_brightness_observation or brightness_decision.state_changed)
                    and mode in (BrightnessGuardMode.FILTER, BrightnessGuardMode.ENFORCE)
                ):
                    self.policy.reset_for_brightness_change(context)

            recovery_unavailable = (
                mode is BrightnessGuardMode.ENFORCE
                and brightness_decision is not None
                and brightness_decision.state is BrightnessState.SEVERE_OVER
                and brightness_decision.recovery_cell is None
            )
            if recovery_unavailable:
                self.policy.reset_for_brightness_change(
                    context, "brightness_recovery_unavailable"
                )
                challenger = None
            elif (
                brightness_decision is not None
                and mode in (BrightnessGuardMode.FILTER, BrightnessGuardMode.ENFORCE)
            ):
                challenger = self.policy.proposal_after_current(
                    context,
                    current,
                    current_score,
                    allow_probe=not forced,
                    admissible_cells=brightness_decision.admissible_cells,
                    prefer_brighter=brightness_decision.prefer_brighter,
                    forced_challenger=(
                        brightness_decision.recovery_cell
                        if brightness_decision.force_recovery
                        else None
                    ),
                )
            else:
                challenger = self.policy.proposal_after_current(
                    context,
                    current,
                    current_score,
                    allow_probe=not forced,
                )
            if challenger is not None:
                result = self._finish_probe(
                    context,
                    current,
                    current_frame,
                    current_score,
                    challenger,
                    active_before,
                    pending_before,
                )
            else:
                reason = (
                    "invalid_pair_recovery"
                    if forced
                    else self.policy.current_reason(context)
                )
                result = self._finish_current_only(
                    context,
                    current,
                    current_frame,
                    current_score,
                    active_before,
                    pending_before,
                    forced,
                    reason,
                    SwitchEvent.NONE,
                    decision_context,
                    decision_stable,
                )

        brightness_values = brightness_log_values(
            self.brightness_guard_config, brightness_decision
        )
        for row in self.logger.rows[first_row:]:
            row.update(brightness_values)
        self.round_index += 1
        return result
