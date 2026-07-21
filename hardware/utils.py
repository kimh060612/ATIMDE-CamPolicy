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
from typing import Any, Iterable, Optional, Sequence, Union

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

    def update(self, score: QScore, *, alpha: float, cycle_index: int) -> None:
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
    round_robin_pointer: int = 0
    committed_cycles: int = 0

    def safe_unobserved(self, safe_cells: Sequence[SensorCell]) -> list[SensorCell]:
        return [
            cell for cell in safe_cells if self.cells[cell.cell_id].count == 0
        ]

    def initialization_complete(self, safe_cells: Sequence[SensorCell]) -> bool:
        return all(self.cells[cell.cell_id].count > 0 for cell in safe_cells)

    def best_from_scores(self, safe_cells: Sequence[SensorCell]) -> SensorCell:
        candidates = [
            cell
            for cell in safe_cells
            if self.cells[cell.cell_id].ema_score is not None
        ]
        if not candidates:
            raise RuntimeError("No scored safe cell is available.")

        # Stable deterministic tie-breaking follows ALL_CELLS ordering.
        return min(
            candidates,
            key=lambda cell: (
                float(self.cells[cell.cell_id].ema_score),
                ALL_CELLS.index(cell),
            ),
        )


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