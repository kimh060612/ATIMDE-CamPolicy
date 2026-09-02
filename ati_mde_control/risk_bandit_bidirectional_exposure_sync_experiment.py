from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from hardware.utils import QScore

from .bidirectional_exposure_guard import BidirectionalExposureGuard
from .capture_runner import CaptureRunner
from .config import ExperimentConfig
from .logging import CaptureLogger
from .risk_bandit_bidirectional_exposure_experiment import (
    BidirectionalExposureGuardLogger,
)
from .risk_bandit_experiment import RiskBanditExperiment
from .risk_bandit_saturation_experiment import (
    RiskScorePredictor,
    SaturationGuardedRoundResult,
)
from .saturation_guard import SaturationGuardedRiskBanditPolicy


METHOD_NAME = (
    "Bidirectional Exposure-Constrained "
    "Risk-Sensitive Contextual GP Bandit"
)


class RiskBanditBidirectionalExposureSyncExperiment(RiskBanditExperiment):
    """Single-capture GP control with current-round synchronous feedback."""

    def __init__(
        self,
        config: ExperimentConfig,
        capture_runner: CaptureRunner,
        predictor: RiskScorePredictor,
        policy: SaturationGuardedRiskBanditPolicy,
        logger: CaptureLogger,
        evaluator: Any,
        guard: BidirectionalExposureGuard,
    ) -> None:
        super().__init__(config, capture_runner, predictor, policy, logger, evaluator)
        self.policy = policy
        self.guard = guard
        self.saturation_logger = BidirectionalExposureGuardLogger(config.output_dir)

    def run_round(self) -> SaturationGuardedRoundResult:
        capture_context, _ = self.capture_runner.context_state()
        capture_context.validate()
        self._ensure_safe_pre_applied(capture_context)
        if self.current_cell is None or self._pre_applied is None:
            raise RuntimeError("A safe camera cell was not applied before capture.")

        frame = self.capture_runner.capture(
            self.current_cell,
            capture_context,
            "initial",
            self.round_index,
            apply_cell=False,
        )
        frame = self._with_pre_applied_metadata(frame, self._pre_applied)

        score: QScore | None = None
        prediction_error: Exception | None = None
        try:
            gain = frame.actual_gain if frame.actual_gain is not None else frame.cell.gain
            score = self.predictor.predict_scores(
                frame.image,
                frame.capture_context,
                frame.exposure_us,
                float(gain),
            )
        except Exception as error:
            prediction_error = error

        if prediction_error is not None:
            update_status = f"predictor_error:{type(prediction_error).__name__}"
        elif not frame.setting_effective:
            update_status = "setting_ineffective"
        elif score is not None and self.policy.add_observation(
            frame.capture_context,
            frame.cell,
            frame.timestamp_ns,
            score,
        ):
            update_status = "updated"
        else:
            update_status = self.policy.last_update_status

        selection_context, _ = self.capture_runner.context_state()
        selection_context.validate()
        observation = self.guard.observe(
            frame.capture_context,
            frame.cell,
            self.round_index,
            frame.image,
            setting_effective=frame.setting_effective,
        )
        base_candidates = self.policy.base_safe_cells(selection_context)
        filtered = self.guard.filter_candidates(
            selection_context,
            frame.cell,
            self.round_index,
            observation,
            base_candidates,
        )
        decision = self.policy.select_from_candidates(
            selection_context,
            frame.cell,
            time.time_ns(),
            filtered.candidates,
        )

        apply_error: Exception | None = None
        next_metadata = None
        apply_completed_ns = time.time_ns()
        try:
            next_metadata = self._apply_cell(decision.selected_cell)
            apply_completed_ns = time.time_ns()
        except Exception as error:
            apply_error = error
            apply_completed_ns = time.time_ns()
        if next_metadata is not None:
            self.current_cell = decision.selected_cell
            self._pre_applied = next_metadata

        if update_status != "updated":
            print(
                f"[RiskBanditBidirectionalSync] round={self.round_index} "
                f"GP update skipped: {update_status}"
            )

        active_after = decision.selected_cell if next_metadata is not None else frame.cell
        decision_delay_ms = (apply_completed_ns - frame.timestamp_ns) / 1_000_000.0
        self.logger.record(
            frame,
            score,
            {
                "active_cell_before": frame.cell.cell_id,
                "active_cell_after": active_after.cell_id,
                "pair_status": "not_probed",
                "selected": 1,
                "control_decision_delay_ms": decision_delay_ms,
            },
        )
        self.saturation_logger.record(
            frame, observation, filtered, decision.selected_cell
        )
        self._warn(frame, score, decision_delay_ms)
        print(
            f"[RiskBanditBidirectionalSync] round={self.round_index} "
            f"capture_ctx={frame.capture_context.table_key} "
            f"selection_ctx={selection_context.table_key} "
            f"captured={frame.cell.cell_id} next={active_after.cell_id} "
            f"guard={observation.guard_state} decision={decision.status} "
            f"update={update_status}"
        )
        self.round_index += 1

        failures = [
            f"camera apply failed: {apply_error}" if apply_error is not None else "",
            f"risk prediction failed: {prediction_error}"
            if prediction_error is not None
            else "",
        ]
        if any(failures):
            raise RuntimeError("; ".join(item for item in failures if item)) from (
                apply_error or prediction_error
            )
        return SaturationGuardedRoundResult(
            frame,
            score,
            decision,
            observation,
            filtered,
            update_status,
        )

    def finalize(self) -> Path:
        try:
            return super().finalize()
        finally:
            self.saturation_logger.write()
