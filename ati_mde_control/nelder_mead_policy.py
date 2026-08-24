from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hardware.utils import (
    ALL_CELLS,
    EXPOSURE_MS_VALUES,
    GAIN_VALUES,
    ContextKey,
    QScore,
    SensorCell,
)
from iqa_control.noise_aware_iqa_control import ControlSetting, NoiseAwareNelderMead

from .config import SafetyPolicy


@dataclass(frozen=True)
class RiskNelderMeadConfig:
    restart_frames: int = 60
    simplex_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if self.restart_frames < 3:
            raise ValueError("Nelder-Mead restart frames must be at least 3.")
        if not math.isfinite(self.simplex_tolerance) or self.simplex_tolerance < 0:
            raise ValueError(
                "Nelder-Mead simplex tolerance must be finite and non-negative."
            )


@dataclass(frozen=True)
class _RiskFeedback:
    # NoiseAwareNelderMead minimizes ``-score``. Supplying ``score=-q`` makes
    # its unchanged state machine minimize the predictor's raw QScore.q.
    score: float
    mean_intensity: float = 127.5


class SafetyAwareRiskNelderMead(NoiseAwareNelderMead):
    """Existing incremental Nelder-Mead projected onto one context's safe grid."""

    def __init__(
        self,
        initial: SensorCell,
        safe_cells: tuple[SensorCell, ...],
        config: RiskNelderMeadConfig,
    ) -> None:
        safe = tuple(sorted(set(safe_cells), key=lambda cell: cell.cell_id))
        if len(safe) < 3:
            raise RuntimeError(
                "Two-dimensional Nelder-Mead requires at least three safe cells."
            )
        self._safe_cells = safe
        self._exposures = tuple(dict.fromkeys(cell.exposure_ms for cell in ALL_CELLS))
        self._gains = tuple(dict.fromkeys(cell.gain for cell in ALL_CELLS))
        self._exposure_index = {
            value: index for index, value in enumerate(self._exposures)
        }
        self._gain_index = {value: index for index, value in enumerate(self._gains)}
        initial = self._nearest_safe((initial.exposure_ms, initial.gain))
        super().__init__(
            ControlSetting(initial.exposure_ms, initial.gain),
            exposure_bounds=(self._exposures[0], self._exposures[-1]),
            gain_bounds=(self._gains[0], self._gains[-1]),
            exposure_step=1,
            gain_step=1,
            restart_interval=config.restart_frames,
            simplex_tolerance=config.simplex_tolerance,
        )

    @property
    def best_score(self) -> float | None:
        """Best observed risk in raw QScore.q units."""
        return (
            min(vertex.objective for vertex in self.simplex)
            if self.simplex
            else None
        )

    @property
    def best_cell(self) -> SensorCell:
        setting = self.best_setting
        return SensorCell(setting.exposure_ms, setting.gain)

    def next_cell(self) -> SensorCell:
        setting = self.next_setting()
        return SensorCell(setting.exposure_ms, setting.gain)

    def observe_q(self, q: float) -> None:
        if not math.isfinite(q):
            raise ValueError("Nelder-Mead risk observation must be finite.")
        super().observe(_RiskFeedback(score=-float(q)))

    def _quantize(self, point: np.ndarray) -> np.ndarray:
        cell = self._nearest_safe((float(point[0]), float(point[1])))
        return np.asarray((cell.exposure_ms, cell.gain), dtype=np.float64)

    def _make_initial_candidates(
        self,
        anchor: np.ndarray,
        mean_intensity: float,
    ) -> list[np.ndarray]:
        del mean_intensity
        anchor_cell = self._nearest_safe((float(anchor[0]), float(anchor[1])))
        used = {anchor_cell}
        candidates: list[SensorCell] = []
        for axis in range(2):
            same_other_axis = [
                cell
                for cell in self._safe_cells
                if cell not in used
                and (
                    cell.gain == anchor_cell.gain
                    if axis == 0
                    else cell.exposure_ms == anchor_cell.exposure_ms
                )
            ]
            pool = same_other_axis or [
                cell for cell in self._safe_cells if cell not in used
            ]
            candidate = min(
                pool,
                key=lambda cell: (
                    self._grid_distance(cell, anchor_cell),
                    cell.cell_id,
                ),
            )
            candidates.append(candidate)
            used.add(candidate)
        return [
            np.asarray((cell.exposure_ms, cell.gain), dtype=np.float64)
            for cell in candidates
        ]

    def _nearest_safe(self, point: tuple[float, float]) -> SensorCell:
        exposure = max(float(self._exposures[0]), point[0])
        gain = max(float(self._gains[0]), point[1])
        exposure_position = float(
            np.interp(
                math.log2(exposure),
                np.log2(np.asarray(self._exposures, dtype=np.float64)),
                np.arange(len(self._exposures), dtype=np.float64),
            )
        )
        gain_position = float(
            np.interp(
                math.log2(gain),
                np.log2(np.asarray(self._gains, dtype=np.float64)),
                np.arange(len(self._gains), dtype=np.float64),
            )
        )
        return min(
            self._safe_cells,
            key=lambda cell: (
                (self._exposure_index[cell.exposure_ms] - exposure_position) ** 2
                + (self._gain_index[cell.gain] - gain_position) ** 2,
                cell.cell_id,
            ),
        )

    def _grid_distance(self, left: SensorCell, right: SensorCell) -> int:
        return (
            (self._exposure_index[left.exposure_ms] - self._exposure_index[right.exposure_ms]) ** 2
            + (self._gain_index[left.gain] - self._gain_index[right.gain]) ** 2
        )


class ContextualRiskNelderMeadPolicy:
    """Keeps an independent safety-aware simplex for each motion/light context."""

    def __init__(
        self,
        config: RiskNelderMeadConfig,
        safety_policy: SafetyPolicy,
        default_cell: SensorCell,
    ) -> None:
        if default_cell not in ALL_CELLS:
            raise ValueError("The default cell must belong to hardware.utils.ALL_CELLS.")
        self.config = config
        self.safety_policy = safety_policy
        self.default_cell = default_cell
        self._controllers: dict[ContextKey, SafetyAwareRiskNelderMead] = {}
        self._brightness_cells: dict[
            ContextKey, tuple[tuple[SensorCell, ...], bool]
        ] = {}
        self._forced_cells: dict[ContextKey, tuple[SensorCell, str]] = {}
        self._issued: dict[
            ContextKey, tuple[SensorCell, SensorCell, str]
        ] = {}
        self.last_update_status = "not_observed"

    def safe_cells(self, context: ContextKey) -> tuple[SensorCell, ...]:
        return tuple(
            sorted(
                self.safety_policy.safe_cells(context),
                key=lambda cell: cell.cell_id,
            )
        )

    def next_cell(self, context: ContextKey) -> SensorCell:
        desired = self._controller(context).next_cell()
        forced = self._forced_cells.pop(context, None)
        if forced is not None:
            selected, operation = forced
        elif context in self._brightness_cells:
            cells, prefer_brighter = self._brightness_cells[context]
            selected = self._nearest_admissible(desired, cells, prefer_brighter)
            operation = (
                self._controller(context).operation
                if selected == desired
                else "brightness_filter"
            )
        else:
            selected = desired
            operation = self._controller(context).operation
        self._issued[context] = (selected, desired, operation)
        return selected

    def operation(self, context: ContextKey) -> str:
        issued = self._issued.get(context)
        return issued[2] if issued is not None else self._controller(context).operation

    def best_cell(self, context: ContextKey) -> SensorCell:
        return self._controller(context).best_cell

    def best_risk(self, context: ContextKey) -> float | None:
        return self._controller(context).best_score

    def observe(
        self,
        context: ContextKey,
        cell: SensorCell,
        score: QScore,
    ) -> bool:
        issued = self._issued.get(context)
        if issued is None or cell != issued[0]:
            raise ValueError(
                "Nelder-Mead observation does not match its pending camera cell."
            )
        try:
            values = (float(score.q), float(score.mu), float(score.uncertainty))
        except (TypeError, ValueError):
            self.last_update_status = "non_finite_score"
            return False
        if not all(math.isfinite(value) for value in values):
            self.last_update_status = "non_finite_score"
            return False
        controller = self._controllers.get(context)
        if (
            controller is None
            or cell != issued[1]
            or issued[2].startswith("brightness_recovery")
        ):
            controller = SafetyAwareRiskNelderMead(
                cell,
                self.safe_cells(context),
                self.config,
            )
            self._controllers[context] = controller
        controller.observe_q(values[0])
        self._issued.pop(context, None)
        self.last_update_status = "updated"
        return True

    def configure_brightness(
        self,
        context: ContextKey,
        current_cell: SensorCell,
        admissible_cells: Sequence[SensorCell],
        prefer_brighter: bool,
        *,
        forced_cell: SensorCell | None = None,
        recovery_unavailable: bool = False,
        reset: bool = False,
    ) -> None:
        hard_safe = self.safe_cells(context)
        allowed = set(admissible_cells)
        cells = tuple(cell for cell in hard_safe if cell in allowed)
        if current_cell in hard_safe and current_cell not in cells:
            cells += (current_cell,)
        if not cells:
            cells = (min(hard_safe, key=lambda cell: cell.cell_id),)
        self._brightness_cells[context] = (cells, prefer_brighter)
        if reset:
            self.reset_for_brightness_change(context)
        if forced_cell is not None:
            if forced_cell not in cells:
                raise ValueError(
                    "A forced brightness recovery cell must be admissible and hard-safe."
                )
            self._forced_cells[context] = (forced_cell, "brightness_recovery")
        elif recovery_unavailable:
            self._forced_cells[context] = (
                current_cell,
                "brightness_recovery_unavailable",
            )
        else:
            self._forced_cells.pop(context, None)

    def reset_for_brightness_change(self, context: ContextKey) -> None:
        context.validate()
        self._controllers.pop(context, None)

    @staticmethod
    def _nearest_admissible(
        desired: SensorCell,
        cells: tuple[SensorCell, ...],
        prefer_brighter: bool,
    ) -> SensorCell:
        desired_point = (
            EXPOSURE_MS_VALUES.index(desired.exposure_ms),
            GAIN_VALUES.index(desired.gain),
        )
        return min(
            cells,
            key=lambda cell: (
                (EXPOSURE_MS_VALUES.index(cell.exposure_ms) - desired_point[0]) ** 2
                + (GAIN_VALUES.index(cell.gain) - desired_point[1]) ** 2,
                -(
                    EXPOSURE_MS_VALUES.index(cell.exposure_ms)
                    + GAIN_VALUES.index(cell.gain)
                )
                if prefer_brighter
                else (
                    EXPOSURE_MS_VALUES.index(cell.exposure_ms)
                    + GAIN_VALUES.index(cell.gain)
                ),
                cell.cell_id,
            ),
        )

    def _controller(self, context: ContextKey) -> SafetyAwareRiskNelderMead:
        context.validate()
        controller = self._controllers.get(context)
        if controller is None:
            controller = SafetyAwareRiskNelderMead(
                self.default_cell,
                self.safe_cells(context),
                self.config,
            )
            self._controllers[context] = controller
        return controller
