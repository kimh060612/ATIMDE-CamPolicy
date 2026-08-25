from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from hardware.utils import SensorCell

from .bidirectional_exposure_guard import (
    BidirectionalCandidateFilterResult,
    BidirectionalExposureGuard,
    BidirectionalExposureObservation,
)
from .capture_runner import CaptureRunner
from .config import ExperimentConfig
from .logging import CaptureLogger
from .risk_bandit_predictive_saturation_experiment import (
    RiskBanditPredictiveSaturationExperiment,
)
from .risk_bandit_saturation_experiment import RiskScorePredictor
from .saturation_guard import SaturationGuardedRiskBanditPolicy, ev_index
from .types import CapturedFrame


METHOD_NAME = (
    "Bidirectional Exposure-Constrained "
    "Delay-Aware Risk-Sensitive Contextual GP Bandit"
)

BIDIRECTIONAL_EXPOSURE_CSV_FIELDS = (
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
    "observed_shadow_clip_ratio",
    "overexposure_state",
    "underexposure_state",
    "combined_exposure_state",
    "guard_conflict",
    "candidate_count_safety",
    "candidate_count_after_quarantine",
    "candidate_count_after_symmetric_step",
    "candidate_count_after_direction_guard",
    "candidate_count_after_projected_saturation",
    "candidate_count_after_projected_shadow",
    "selected_next_cell",
    "selected_next_ev_index",
    "selected_delta_ev",
    "selected_projected_saturation_ratio",
    "selected_projected_shadow_ratio",
    "ev_step_rejected_cell_ids",
    "direction_rejected_cell_ids",
    "saturation_rejected_cell_ids",
    "shadow_rejected_cell_ids",
    "fallback_used",
    "fallback_reason",
    "under_recovery_count",
    "over_recovery_count",
)


class BidirectionalExposureGuardLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "bidirectional_exposure_guard.csv"
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        frame: CapturedFrame,
        observation: BidirectionalExposureObservation,
        filtered: BidirectionalCandidateFilterResult,
        selected_cell: SensorCell,
    ) -> None:
        metrics = observation.metrics
        saturation = dict(filtered.projected_saturation_ratios)
        shadow = dict(filtered.projected_shadow_ratios)
        current_ev = ev_index(frame.cell)
        selected_ev = ev_index(selected_cell)
        self.rows.append(
            {
                "round_index": frame.round_index,
                "capture_index": frame.capture_index,
                "timestamp_ns": frame.timestamp_ns,
                "motion_state": frame.capture_context.motion_state,
                "light_state": frame.capture_context.light_state,
                "current_cell_id": frame.cell.cell_id,
                "current_ev_index": current_ev,
                "mean_luminance": metrics.mean_luminance,
                "observed_channel_clip_ratio": metrics.channel_clip_ratio,
                "observed_luminance_clip_ratio": metrics.luminance_clip_ratio,
                "observed_shadow_clip_ratio": metrics.shadow_clip_ratio,
                "overexposure_state": observation.overexposure_state,
                "underexposure_state": observation.underexposure_state,
                "combined_exposure_state": observation.guard_state,
                "guard_conflict": int(observation.guard_conflict),
                "candidate_count_safety": filtered.candidate_count_safety,
                "candidate_count_after_quarantine": (
                    filtered.candidate_count_after_quarantine
                ),
                "candidate_count_after_symmetric_step": (
                    filtered.candidate_count_after_symmetric_step
                ),
                "candidate_count_after_direction_guard": (
                    filtered.candidate_count_after_direction_guard
                ),
                "candidate_count_after_projected_saturation": (
                    filtered.candidate_count_after_projected_saturation
                ),
                "candidate_count_after_projected_shadow": (
                    filtered.candidate_count_after_projected_shadow
                ),
                "selected_next_cell": selected_cell.cell_id,
                "selected_next_ev_index": selected_ev,
                "selected_delta_ev": selected_ev - current_ev,
                "selected_projected_saturation_ratio": saturation.get(
                    selected_cell, ""
                ),
                "selected_projected_shadow_ratio": shadow.get(selected_cell, ""),
                "ev_step_rejected_cell_ids": self._cell_ids(
                    filtered.ev_step_rejected_cells
                ),
                "direction_rejected_cell_ids": self._cell_ids(
                    filtered.direction_rejected_cells
                ),
                "saturation_rejected_cell_ids": self._cell_ids(
                    filtered.saturation_rejected_cells
                ),
                "shadow_rejected_cell_ids": self._cell_ids(
                    filtered.shadow_rejected_cells
                ),
                "fallback_used": int(filtered.fallback_used),
                "fallback_reason": filtered.fallback_reason,
                "under_recovery_count": observation.under_recovery_count,
                "over_recovery_count": observation.over_recovery_count,
            }
        )

    @staticmethod
    def _cell_ids(cells: tuple[SensorCell, ...]) -> str:
        return json.dumps(sorted(cell.cell_id for cell in cells), separators=(",", ":"))

    def write(self) -> Path:
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=BIDIRECTIONAL_EXPOSURE_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)
        return self.path


class RiskBanditBidirectionalExposureExperiment(
    RiskBanditPredictiveSaturationExperiment
):
    """Predictive single-capture runtime with a bidirectional guard."""

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
        self.saturation_logger = BidirectionalExposureGuardLogger(config.output_dir)
