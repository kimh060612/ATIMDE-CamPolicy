import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import cv2
import numpy as np

EXPOSURE_MS_VALUES: tuple[int, ...] = (4, 8, 16, 32)
GAIN_VALUES: tuple[int, ...] = (16, 32, 64, 128)
MOTION_STATE_COUNT = 5
LIGHT_STATE_COUNT = 3
DEFAULT_CELLS_PER_CYCLE = 4
MOTION_LABEL_TO_STATE = {
    "stop": 0,
    "slow": 1,
    "fast": 2,
    "rotate": 3,
    "spin": 4,
}
LIGHT_LABEL_TO_STATE = {"normal": 0, "dim": 1, "dark": 2}
MOTION_STATE_TO_LABEL = {value: key for key, value in MOTION_LABEL_TO_STATE.items()}
LIGHT_STATE_TO_LABEL = {value: key for key, value in LIGHT_LABEL_TO_STATE.items()}

@dataclass(frozen=True, order=True)
class ContextKey:
    motion_state: int
    light_state: int

    def validate(self) -> None:
        if not 0 <= self.motion_state < MOTION_STATE_COUNT:
            raise ValueError(
                f"motion_state must be in [0, {MOTION_STATE_COUNT - 1}], "
                f"got {self.motion_state}."
            )
        if not 0 <= self.light_state < LIGHT_STATE_COUNT:
            raise ValueError(
                f"light_state must be in [0, {LIGHT_STATE_COUNT - 1}], "
                f"got {self.light_state}."
            )

    @property
    def table_key(self) -> str:
        return f"{self.motion_state},{self.light_state}"


@dataclass(frozen=True, order=True)
class SensorCell:
    exposure_ms: int
    gain: int

    @property
    def cell_id(self) -> str:
        return f"E{self.exposure_ms:02d}_G{self.gain:03d}"


ALL_CELLS: tuple[SensorCell, ...] = tuple(
    SensorCell(exposure_ms=e, gain=g)
    for e in EXPOSURE_MS_VALUES
    for g in GAIN_VALUES
)
CELL_BY_ID: dict[str, SensorCell] = {cell.cell_id: cell for cell in ALL_CELLS}


@dataclass(frozen=True)
class QScore:
    """Output of `compute_q_score`.

    Parameters
    ----------
    q:
        Scalar score used by the probing controller. Lower is better.
        For example, this may be `mu + beta * sigma`.
    uncertainty:
        Optional predictive uncertainty. When present, it can trigger an
        offload decision through `--offload-uncertainty-threshold`.
    mu:
        Optional predicted mean, logged for later analysis.
    extra:
        Optional scalar metadata to append to the JSON metadata field.
    """

    q: float
    uncertainty: Optional[float] = None
    mu: Optional[float] = None
    extra: dict[str, float] = field(default_factory=dict)
    

@dataclass
class CellStats:
    count: int = 0
    ema_score: Optional[float] = None
    last_q: Optional[float] = None
    last_uncertainty: Optional[float] = None
    last_mu: Optional[float] = None
    last_seen_cycle: int = -1
    belief_mean: Optional[float] = None
    belief_variance: Optional[float] = None
    last_scene_epoch: int = -1

    def update(
        self,
        score: QScore,
        *,
        alpha: float,
        cycle_index: int,
        observation_variance: Optional[float] = None,
        variance_floor: Optional[float] = None,
        scene_epoch: Optional[int] = None,
        process_variance: Optional[float] = None,
        maximum_belief_variance: Optional[float] = None,
    ) -> None:
        if not math.isfinite(score.q):
            raise ValueError(f"Non-finite q score: {score.q}")

        if self.ema_score is None:
            self.ema_score = score.q
        else:
            self.ema_score = (1.0 - alpha) * self.ema_score + alpha * score.q

        self.count += 1
        self.last_q = score.q
        self.last_uncertainty = score.uncertainty
        self.last_mu = score.mu
        self.last_seen_cycle = cycle_index

        # Belief arguments are optional so existing EMA-only users retain the
        # previous update interface.
        belief_arguments = (observation_variance, variance_floor, scene_epoch)
        if all(value is None for value in belief_arguments):
            return
        if any(value is None for value in belief_arguments):
            raise ValueError("All Gaussian belief update arguments are required.")
        if score.mu is None or not math.isfinite(score.mu):
            raise ValueError("QScore.mu must contain a finite belief observation.")
        assert observation_variance is not None
        assert variance_floor is not None
        assert scene_epoch is not None
        process_variance = 0.0 if process_variance is None else process_variance
        maximum_belief_variance = (
            math.inf
            if maximum_belief_variance is None
            else maximum_belief_variance
        )
        if not math.isfinite(observation_variance) or observation_variance <= 0.0:
            raise ValueError("observation_variance must be finite and positive.")
        if not math.isfinite(variance_floor) or variance_floor <= 0.0:
            raise ValueError("variance_floor must be finite and positive.")
        if not math.isfinite(process_variance) or process_variance < 0.0:
            raise ValueError("process_variance must be finite and non-negative.")
        if (
            math.isnan(maximum_belief_variance)
            or maximum_belief_variance < variance_floor
        ):
            raise ValueError(
                "maximum_belief_variance must be at least variance_floor."
            )

        observation_variance = max(observation_variance, variance_floor)
        if self.belief_mean is None or self.belief_variance is None:
            self.belief_mean = float(score.mu)
            self.belief_variance = observation_variance
        else:
            prior_variance = min(
                max(self.belief_variance, variance_floor) + process_variance,
                maximum_belief_variance,
            )
            fused_variance = max(
                1.0
                / (
                    1.0 / prior_variance
                    + 1.0 / observation_variance
                ),
                variance_floor,
            )
            self.belief_mean = fused_variance * (
                self.belief_mean / prior_variance
                + float(score.mu) / observation_variance
            )
            self.belief_variance = fused_variance
        self.last_scene_epoch = scene_epoch


@dataclass
class ContextState:
    cells: dict[str, CellStats] = field(
        default_factory=lambda: {
            cell.cell_id: CellStats() for cell in ALL_CELLS
        }
    )
    best_cell_id: Optional[str] = None
    challenger_cell_id: Optional[str] = None
    challenger_streak: int = 0
    committed_cycles: int = 0
    scene_epoch: int = 0
    scene_change_streak: int = 0
    candidate_cooldown: dict[str, int] = field(default_factory=dict)
    tentative_cell_id: Optional[str] = None
    previous_cell_id: Optional[str] = None
    previous_reference_q: Optional[float] = None


@dataclass(frozen=True)
class RawCaptureObservation:
    cell: SensorCell
    image_bgr: np.ndarray
    ground_truth_depth_m: np.ndarray
    image_path: Optional[str]
    depth_path: Optional[str]
    requested_exposure_raw: int
    actual_exposure_raw: Optional[int]
    actual_gain: Optional[int]
    capture_time_ns: int
    camera_parameter_ms: float


@dataclass(frozen=True)
class CaptureObservation:
    cell: SensorCell
    score: QScore
    image_bgr: np.ndarray
    ground_truth_depth_m: np.ndarray
    image_path: Optional[str]
    depth_path: Optional[str]
    requested_exposure_raw: int
    actual_exposure_raw: Optional[int]
    actual_gain: Optional[int]
    capture_time_ns: int
    camera_parameter_ms: float
