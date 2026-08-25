from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from hardware.utils import SensorCell

from .capture_runner import CaptureRunner
from .config import ExperimentConfig
from .logging import CaptureLogger
from .predictive_saturation_guard import (
    PredictiveCandidateFilterResult,
    PredictiveSaturationGuard,
    PredictiveSaturationObservation,
)
from .risk_bandit_saturation_experiment import (
    RiskBanditSaturationExperiment,
    RiskScorePredictor,
)
from .saturation_guard import SaturationGuardedRiskBanditPolicy, ev_index
from .types import CapturedFrame


METHOD_NAME = (
    "Predictive Saturation-Constrained "
    "Delay-Aware Risk-Sensitive Contextual GP Bandit"
)

PREDICTIVE_SATURATION_CSV_FIELDS = (
    "round_index",
    "capture_index",
    "timestamp_ns",
    "motion_state",
    "light_state",
    "current_cell_id",
    "current_ev_index",
    "mean_luminance",
    "observed_channel_clip_ratio",
    "observed_luminance_clip_ratio",
    "guard_state",
    "hard_overexposed",
    "soft_overexposed",
    "quarantine_active",
    "blocked_ev_min",
    "quarantine_expiry_round",
    "candidate_count_safety",
    "candidate_count_after_quarantine",
    "candidate_count_after_observed_guard",
    "candidate_count_after_ev_step",
    "candidate_count_after_projection",
    "selected_next_cell",
    "selected_next_ev_index",
    "selected_delta_ev",
    "selected_projected_clip_ratio",
    "max_projected_clip_ratio",
    "projected_rejected_cell_ids",
    "ev_step_rejected_cell_ids",
    "fallback_used",
    "fallback_reason",
)


class PredictiveSaturationGuardLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "predictive_saturation_guard.csv"
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        frame: CapturedFrame,
        observation: PredictiveSaturationObservation,
        filtered: PredictiveCandidateFilterResult,
        selected_cell: SensorCell,
    ) -> None:
        metrics = observation.metrics
        projected = dict(filtered.projected_clip_ratios)
        selected_ev = ev_index(selected_cell)
        self.rows.append(
            {
                "round_index": frame.round_index,
                "capture_index": frame.capture_index,
                "timestamp_ns": frame.timestamp_ns,
                "motion_state": frame.capture_context.motion_state,
                "light_state": frame.capture_context.light_state,
                "current_cell_id": frame.cell.cell_id,
                "current_ev_index": ev_index(frame.cell),
                "mean_luminance": metrics.mean_luminance,
                "observed_channel_clip_ratio": metrics.channel_clip_ratio,
                "observed_luminance_clip_ratio": metrics.luminance_clip_ratio,
                "guard_state": observation.guard_state,
                "hard_overexposed": int(metrics.hard_overexposed),
                "soft_overexposed": int(metrics.soft_overexposed),
                "quarantine_active": int(filtered.quarantine_active),
                "blocked_ev_min": filtered.blocked_ev_min
                if filtered.blocked_ev_min is not None
                else "",
                "quarantine_expiry_round": filtered.quarantine_expiry_round
                if filtered.quarantine_expiry_round is not None
                else "",
                "candidate_count_safety": filtered.candidate_count_safety,
                "candidate_count_after_quarantine": (
                    filtered.candidate_count_after_quarantine
                ),
                "candidate_count_after_observed_guard": (
                    filtered.candidate_count_after_observed_guard
                ),
                "candidate_count_after_ev_step": filtered.candidate_count_after_ev_step,
                "candidate_count_after_projection": (
                    filtered.candidate_count_after_projection
                ),
                "selected_next_cell": selected_cell.cell_id,
                "selected_next_ev_index": selected_ev,
                "selected_delta_ev": selected_ev - ev_index(frame.cell),
                "selected_projected_clip_ratio": projected.get(selected_cell, ""),
                "max_projected_clip_ratio": max(projected.values())
                if projected
                else "",
                "projected_rejected_cell_ids": self._cell_ids(
                    filtered.projected_rejected_cells
                ),
                "ev_step_rejected_cell_ids": self._cell_ids(
                    filtered.ev_step_rejected_cells
                ),
                "fallback_used": int(filtered.fallback_used),
                "fallback_reason": filtered.fallback_reason,
            }
        )

    @staticmethod
    def _cell_ids(cells: tuple[SensorCell, ...]) -> str:
        return json.dumps(
            sorted(cell.cell_id for cell in cells), separators=(",", ":")
        )

    def write(self) -> Path:
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=PREDICTIVE_SATURATION_CSV_FIELDS
            )
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)
        return self.path


class RiskBanditPredictiveSaturationExperiment(RiskBanditSaturationExperiment):
    """Reactive single-capture runtime with predictive candidate filtering."""

    def __init__(
        self,
        config: ExperimentConfig,
        capture_runner: CaptureRunner,
        predictor: RiskScorePredictor,
        policy: SaturationGuardedRiskBanditPolicy,
        logger: CaptureLogger,
        evaluator: Any,
        guard: PredictiveSaturationGuard,
    ) -> None:
        super().__init__(
            config,
            capture_runner,
            predictor,
            policy,
            logger,
            evaluator,
            guard,
        )
        self.guard = guard
        self.saturation_logger = PredictiveSaturationGuardLogger(config.output_dir)
