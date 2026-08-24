from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol

from hardware.utils import QScore

from .capture_runner import CaptureRunner
from .config import ExperimentConfig
from .full_depth_predictor import (
    FullDepthBatchPrediction,
    save_raw_depth_prediction,
)
from .logging import CaptureLogger
from .nelder_mead_policy import ContextualRiskNelderMeadPolicy
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
class NelderMeadRoundResult:
    frame: CapturedFrame
    score: QScore
    operation: str
    update_status: str


class RiskNelderMeadExperiment:
    """Single-capture control loop minimizing CameraErrorPredictor QScore.q."""

    def __init__(
        self,
        config: ExperimentConfig,
        capture_runner: CaptureRunner,
        predictor: RiskPredictor,
        policy: ContextualRiskNelderMeadPolicy,
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

    def run_round(self) -> NelderMeadRoundResult:
        selection_context, _ = self.capture_runner.context_state()
        selection_context.validate()
        cell = self.policy.next_cell(selection_context)
        if cell not in self.policy.safe_cells(selection_context):
            raise RuntimeError("Nelder-Mead proposed a cell outside the safety envelope.")
        operation = self.policy.operation(selection_context)
        frame = self.capture_runner.capture(
            cell,
            selection_context,
            "initial",
            self.round_index,
        )
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

        if not frame.setting_effective:
            update_status = "setting_ineffective"
        elif frame.capture_context != selection_context:
            update_status = "context_changed"
        elif self.policy.observe(selection_context, frame.cell, score):
            update_status = "updated"
        else:
            update_status = self.policy.last_update_status

        decision_ns = time.time_ns()
        decision_ms = (decision_ns - frame.timestamp_ns) / 1_000_000.0
        self.logger.record(
            frame,
            score,
            {
                "active_cell_before": frame.cell.cell_id,
                "active_cell_after": frame.cell.cell_id,
                "pair_status": "not_probed",
                "selected": 1,
                "control_decision_delay_ms": decision_ms,
            },
        )
        best = self.policy.best_cell(selection_context)
        best_risk = self.policy.best_risk(selection_context)
        print(
            f"[RiskNelderMead] round={self.round_index} "
            f"ctx={selection_context.table_key} op={operation} "
            f"captured={cell.cell_id} q={score.q} update={update_status} "
            f"best={best.cell_id} best_q={best_risk}"
        )
        self._warn(frame, score, decision_ms)
        self.round_index += 1
        return NelderMeadRoundResult(frame, score, operation, update_status)

    def _warn(self, frame: CapturedFrame, score: QScore, decision_ms: float) -> None:
        inference_ms = float(score.extra.get("mde_inference_ms", 0.0))
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
