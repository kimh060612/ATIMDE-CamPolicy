from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Protocol

from hardware.utils import ContextKey, QScore, SensorCell

from .capture_runner import CaptureRunner
from .config import ExperimentConfig
from .full_depth_predictor import (
    FullDepthBatchPrediction,
    save_raw_depth_prediction,
)
from .logging import CaptureLogger
from .risk_bandit_policy import RiskBanditDecision, RiskBanditPolicy
from .types import CapturedFrame


class RiskPredictor(Protocol):
    def predict_batch(
        self,
        images,
        contexts,
        exposure_us_values,
        gains,
    ) -> FullDepthBatchPrediction: ...


@dataclass(frozen=True)
class PreAppliedMetadata:
    cell: SensorCell
    requested_exposure_raw: int
    actual_exposure_raw: int | None
    actual_gain: int | None
    camera_parameter_ms: float
    sensor_settle_ms: float


@dataclass(frozen=True)
class RiskBanditRoundResult:
    frame: CapturedFrame
    score: QScore | None
    decision: RiskBanditDecision
    gp_update_status: str


class RiskBanditExperiment:
    """One-capture delayed-feedback runtime for the risk-sensitive GP bandit."""

    def __init__(
        self,
        config: ExperimentConfig,
        capture_runner: CaptureRunner,
        predictor: RiskPredictor,
        policy: RiskBanditPolicy,
        logger: CaptureLogger,
        evaluator: Any,
    ) -> None:
        self.config = config
        self.capture_runner = capture_runner
        self.predictor = predictor
        self.policy = policy
        self.logger = logger
        self.evaluator = evaluator
        self.round_index = 0
        self.current_cell: SensorCell | None = None
        self._pre_applied: PreAppliedMetadata | None = None

    def run_round(self) -> RiskBanditRoundResult:
        latched_context, _ = self.capture_runner.context_state()
        latched_context.validate()
        self._ensure_safe_pre_applied(latched_context)
        if self.current_cell is None or self._pre_applied is None:
            raise RuntimeError("A safe camera cell was not applied before capture.")

        frame = self.capture_runner.capture(
            self.current_cell,
            latched_context,
            "initial",
            self.round_index,
            apply_cell=False,
        )
        frame = self._with_pre_applied_metadata(frame, self._pre_applied)

        # Delayed feedback is preserved: round t+1 is selected using history
        # only through t-1, before the current frame's synchronous inference.
        selection_context, _ = self.capture_runner.context_state()
        selection_context.validate()
        decision = self.policy.select_action(
            selection_context,
            self.current_cell,
            time.time_ns(),
        )

        score: QScore | None = None
        prediction_error: Exception | None = None
        try:
            gain = frame.actual_gain if frame.actual_gain is not None else frame.cell.gain
            prediction = self.predictor.predict_batch(
                [frame.image],
                [frame.capture_context],
                [frame.exposure_us],
                [float(gain)],
            )
            if len(prediction.scores) != 1 or len(prediction.depth_maps) != 1:
                raise RuntimeError("Risk predictor returned the wrong batch size.")
            score = prediction.scores[0]
            save_raw_depth_prediction(
                self.config.output_dir,
                frame,
                prediction.depth_maps[0],
            )
        except Exception as error:
            prediction_error = error

        apply_error: Exception | None = None
        next_metadata: PreAppliedMetadata | None = None
        apply_completed_ns = time.time_ns()
        if prediction_error is None:
            try:
                next_metadata = self._apply_cell(decision.selected_cell)
                apply_completed_ns = time.time_ns()
            except Exception as error:
                apply_error = error
                apply_completed_ns = time.time_ns()

        if next_metadata is not None:
            self.current_cell = decision.selected_cell
            self._pre_applied = next_metadata

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

        if update_status != "updated":
            print(
                f"[RiskBandit] round={self.round_index} GP update skipped: "
                f"{update_status}"
            )

        active_after = (
            decision.selected_cell if next_metadata is not None else frame.cell
        )
        decision_delay_ms = (apply_completed_ns - frame.timestamp_ns) / 1_000_000.0
        values = {
            "active_cell_before": frame.cell.cell_id,
            "active_cell_after": active_after.cell_id,
            "pair_status": "not_probed",
            "selected": 1,
            "control_decision_delay_ms": decision_delay_ms,
        }
        self.logger.record(frame, score, values)
        self._warn(frame, score, decision_delay_ms)
        print(
            f"[RiskBandit] round={self.round_index} "
            f"capture_ctx={frame.capture_context.table_key} "
            f"selection_ctx={selection_context.table_key} "
            f"captured={frame.cell.cell_id} next={active_after.cell_id} "
            f"decision={decision.status} update={update_status}"
        )
        self.round_index += 1

        if apply_error is not None or prediction_error is not None:
            failures = [
                f"camera apply failed: {apply_error}" if apply_error is not None else "",
                f"risk prediction failed: {prediction_error}" if prediction_error is not None else "",
            ]
            raise RuntimeError("; ".join(item for item in failures if item)) from (
                apply_error or prediction_error
            )
        return RiskBanditRoundResult(frame, score, decision, update_status)

    def _ensure_safe_pre_applied(self, context: ContextKey) -> None:
        safe = self.policy.safe_cells(context)
        if self.current_cell is None:
            target = (
                self.config.default_cell
                if self.config.default_cell in safe
                else self.policy.safe_fallback(context)
            )
        elif self.current_cell not in safe:
            target = self.policy.safe_fallback(context)
        else:
            return
        self._pre_applied = self._apply_cell(target)
        self.current_cell = target

    def _apply_cell(self, cell: SensorCell) -> PreAppliedMetadata:
        started = time.perf_counter()
        requested, actual_exposure, actual_gain = self.capture_runner.camera.apply_cell(cell)
        parameter_ms = (time.perf_counter() - started) * 1000.0
        return PreAppliedMetadata(
            cell=cell,
            requested_exposure_raw=int(requested),
            actual_exposure_raw=actual_exposure,
            actual_gain=actual_gain,
            camera_parameter_ms=parameter_ms,
            sensor_settle_ms=float(
                getattr(self.capture_runner.camera, "sensor_settle_ms", 0.0)
            ),
        )

    def _with_pre_applied_metadata(
        self,
        frame: CapturedFrame,
        metadata: PreAppliedMetadata,
    ) -> CapturedFrame:
        if frame.cell != metadata.cell:
            raise RuntimeError("Pre-applied camera metadata belongs to a different cell.")
        exposure_us = frame.cell.exposure_ms * 1000.0
        if metadata.actual_exposure_raw is not None:
            exposure_us = (
                metadata.actual_exposure_raw
                / self.capture_runner.camera.exposure_value_per_ms
                * 1000.0
            )
        return replace(
            frame,
            exposure_us=float(exposure_us),
            requested_exposure_raw=metadata.requested_exposure_raw,
            actual_exposure_raw=metadata.actual_exposure_raw,
            actual_gain=metadata.actual_gain,
            camera_parameter_ms=metadata.camera_parameter_ms,
            sensor_settle_ms=metadata.sensor_settle_ms,
        )

    def _warn(
        self,
        frame: CapturedFrame,
        score: QScore | None,
        decision_ms: float,
    ) -> None:
        inference_ms = (
            float(score.extra.get("mde_inference_ms", 0.0)) if score is not None else 0.0
        )
        for name, value, limit in (
            ("camera_parameter", frame.camera_parameter_ms, self.config.camera_parameter_warn_ms),
            ("mde_inference", inference_ms, self.config.mde_inference_warn_ms),
            ("control_decision", decision_ms, self.config.control_decision_warn_ms),
        ):
            if math.isfinite(value) and value > limit:
                print(
                    f"[LATENCY WARNING] round={self.round_index} "
                    f"{name}_ms={value:.2f}"
                )

    def finalize(self) -> Any:
        failure: Exception | None = None
        try:
            self.evaluator.evaluate_rows(self.logger.rows)
            initial = [
                row for row in self.logger.rows if row.get("output_delivered") == 1
            ]
            if initial and not any(row.get("abs_rel", "") != "" for row in initial):
                raise RuntimeError("AbsRel/A1 evaluation failed for every initial frame.")
        except (OSError, RuntimeError, ValueError) as error:
            failure = error
        path = self.logger.write()
        if failure is not None:
            raise failure
        return path
