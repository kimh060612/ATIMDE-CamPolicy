from __future__ import annotations

import csv
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hardware.utils import QScore, SensorCell

from .capture_runner import CaptureRunner
from .config import ExperimentConfig
from .logging import CaptureLogger
from .risk_bandit_experiment import RiskBanditExperiment
from .risk_bandit_policy import RiskBanditDecision
from .saturation_guard import (
    CandidateFilterResult,
    SaturationGuard,
    SaturationGuardedRiskBanditPolicy,
    SaturationObservation,
    ev_index,
)
from .types import CapturedFrame


METHOD_NAME = "Saturation-Guarded Delay-Aware Risk-Sensitive Contextual GP Bandit"

SATURATION_CSV_FIELDS = (
    "round_index",
    "capture_index",
    "timestamp_ns",
    "motion_state",
    "light_state",
    "cell_id",
    "cell_ev_index",
    "mean_luminance",
    "channel_clip_ratio",
    "luminance_clip_ratio",
    "guard_state",
    "hard_overexposed",
    "soft_overexposed",
    "quarantine_active",
    "quarantine_expiry_round",
    "candidate_count_before_guard",
    "candidate_count_after_guard",
    "selected_next_cell",
    "selected_next_ev_index",
    "fallback_used",
)


class RiskScorePredictor(Protocol):
    def predict_scores(
        self,
        image,
        context,
        exposure_us: float,
        gain: float,
    ) -> QScore: ...


@dataclass(frozen=True)
class SaturationGuardedRoundResult:
    frame: CapturedFrame
    score: QScore | None
    decision: RiskBanditDecision
    observation: SaturationObservation
    candidate_filter: CandidateFilterResult
    gp_update_status: str


class SaturationGuardLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "saturation_guard.csv"
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        frame: CapturedFrame,
        observation: SaturationObservation,
        filtered: CandidateFilterResult,
        selected_cell: SensorCell,
    ) -> None:
        metrics = observation.metrics
        self.rows.append(
            {
                "round_index": frame.round_index,
                "capture_index": frame.capture_index,
                "timestamp_ns": frame.timestamp_ns,
                "motion_state": frame.capture_context.motion_state,
                "light_state": frame.capture_context.light_state,
                "cell_id": frame.cell.cell_id,
                "cell_ev_index": ev_index(frame.cell),
                "mean_luminance": metrics.mean_luminance,
                "channel_clip_ratio": metrics.channel_clip_ratio,
                "luminance_clip_ratio": metrics.luminance_clip_ratio,
                "guard_state": observation.guard_state,
                "hard_overexposed": int(metrics.hard_overexposed),
                "soft_overexposed": int(metrics.soft_overexposed),
                "quarantine_active": int(filtered.quarantine_active),
                "quarantine_expiry_round": filtered.quarantine_expiry_round
                if filtered.quarantine_expiry_round is not None
                else "",
                "candidate_count_before_guard": filtered.candidate_count_before_guard,
                "candidate_count_after_guard": filtered.candidate_count_after_guard,
                "selected_next_cell": selected_cell.cell_id,
                "selected_next_ev_index": ev_index(selected_cell),
                "fallback_used": int(filtered.fallback_used),
            }
        )

    def write(self) -> Path:
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=SATURATION_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)
        return self.path


class RiskBanditSaturationExperiment(RiskBanditExperiment):
    """Single-capture v2 runtime with a context-local saturation guard."""

    def __init__(
        self,
        config: ExperimentConfig,
        capture_runner: CaptureRunner,
        predictor: RiskScorePredictor,
        policy: SaturationGuardedRiskBanditPolicy,
        logger: CaptureLogger,
        evaluator: Any,
        guard: SaturationGuard,
    ) -> None:
        super().__init__(config, capture_runner, predictor, policy, logger, evaluator)
        self.policy = policy
        self.guard = guard
        self.saturation_logger = SaturationGuardLogger(config.output_dir)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="risk-bandit-saturation"
        )
        self._pending: Future[QScore] | None = None
        self._worker_closed = False

    def run_round(self) -> SaturationGuardedRoundResult:
        if self._worker_closed:
            raise RuntimeError("The saturation experiment has already been finalized.")
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
        future = self._submit_prediction(frame)

        score: QScore | None = None
        prediction_error: Exception | None = None
        try:
            observation = self.guard.observe(
                frame.capture_context,
                frame.cell,
                self.round_index,
                frame.image,
                setting_effective=frame.setting_effective,
            )
            selection_context, _ = self.capture_runner.context_state()
            selection_context.validate()
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
        finally:
            try:
                score = future.result()
            except Exception as error:
                prediction_error = error
            finally:
                self._pending = None

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
                f"[RiskBanditSaturation] round={self.round_index} "
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
            f"[RiskBanditSaturation] round={self.round_index} "
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

    def _submit_prediction(self, frame: CapturedFrame) -> Future[QScore]:
        if self._pending is not None:
            raise RuntimeError("Only one risk inference may be pending.")
        started = threading.Event()
        self._pending = self._executor.submit(self._predict_score, frame, started)
        if not started.wait(timeout=5.0):
            raise RuntimeError("Risk inference worker did not start.")
        return self._pending

    def _predict_score(self, frame: CapturedFrame, started: threading.Event) -> QScore:
        self._on_inference_start()
        started.set()
        gain = frame.actual_gain if frame.actual_gain is not None else frame.cell.gain
        return self.predictor.predict_scores(
            frame.image,
            frame.capture_context,
            frame.exposure_us,
            float(gain),
        )

    def _on_inference_start(self) -> None:
        pass

    def _shutdown_worker(self) -> None:
        if not self._worker_closed:
            self._executor.shutdown(wait=True)
            self._worker_closed = True
            self._pending = None

    def finalize(self) -> Path:
        self._shutdown_worker()
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

        common_path: Path | None = None
        try:
            common_path = self.logger.write()
        except (OSError, RuntimeError, ValueError) as error:
            failure = failure or error
        try:
            self.saturation_logger.write()
        except (OSError, RuntimeError, ValueError) as error:
            failure = failure or error
        if failure is not None:
            raise failure
        if common_path is None:
            raise RuntimeError("Common capture logger did not return an output path.")
        return common_path
