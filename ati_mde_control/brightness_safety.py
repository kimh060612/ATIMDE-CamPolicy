from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum
from collections.abc import Iterable, Mapping

import numpy as np

from hardware.utils import EXPOSURE_MS_VALUES, GAIN_VALUES, SensorCell


class BrightnessState(str, Enum):
    SEVERE_OVER = "severe_over"
    OVER = "over"
    GOOD = "good"
    UNDER = "under"
    SEVERE_UNDER = "severe_under"


class BrightnessGuardMode(str, Enum):
    OFF = "off"
    LOG = "log"
    FILTER = "filter"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class BrightnessStats:
    p10: float
    p50: float
    p90: float
    p99: float
    shadow_ratio: float
    luma_clip_ratio: float
    channel_clip_ratio: float
    overexposed_tile_ratio: float


@dataclass(frozen=True)
class BrightnessDecision:
    state: BrightnessState
    raw_stats: BrightnessStats
    smoothed_stats: BrightnessStats
    admissible_cells: tuple[SensorCell, ...]
    recovery_cell: SensorCell | None
    prefer_brighter: bool
    force_recovery: bool
    state_changed: bool


@dataclass(frozen=True)
class BrightnessGuardConfig:
    mode: BrightnessGuardMode = BrightnessGuardMode.OFF
    ema_alpha: float = 0.3
    state_change_frames: int = 2
    tile_grid_size: int = 8
    tile_clip_pixel_ratio: float = 0.05
    over_tile_ratio: float = 0.25
    severe_over_tile_ratio: float = 0.50
    over_luma_clip_ratio: float = 0.01
    severe_over_luma_clip_ratio: float = 0.05
    over_channel_clip_ratio: float = 0.03
    over_p99: float = 248.0
    under_p50: float = 50.0
    severe_under_p50: float = 25.0
    under_shadow_ratio: float = 0.20
    severe_under_shadow_ratio: float = 0.40
    target_p50: float = 110.0
    clipping_value: int = 250
    shadow_value: int = 8
    good_ev_radius: int = 1
    under_max_ev_increase: int = 1
    severe_under_max_ev_increase: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.mode, BrightnessGuardMode):
            try:
                object.__setattr__(self, "mode", BrightnessGuardMode(self.mode))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "brightness_guard.mode must be off, log, filter, or enforce."
                ) from error
        self.validate()

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object] | None
    ) -> BrightnessGuardConfig:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("brightness_guard must be a JSON object.")
        known = {item.name for item in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown brightness_guard settings: {sorted(unknown)}")
        values = dict(payload)
        if "mode" in values and not isinstance(values["mode"], BrightnessGuardMode):
            try:
                values["mode"] = BrightnessGuardMode(str(values["mode"]))
            except (TypeError, ValueError) as error:
                raise ValueError("brightness_guard.mode must be off, log, filter, or enforce.") from error
        return cls(**values)

    def validate(self) -> None:
        if not math.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("brightness_guard.ema_alpha must be in (0, 1].")
        if self.state_change_frames < 1 or self.tile_grid_size < 1:
            raise ValueError("Brightness frame and tile counts must be positive.")
        ratios = (
            self.tile_clip_pixel_ratio,
            self.over_tile_ratio,
            self.severe_over_tile_ratio,
            self.over_luma_clip_ratio,
            self.severe_over_luma_clip_ratio,
            self.over_channel_clip_ratio,
            self.under_shadow_ratio,
            self.severe_under_shadow_ratio,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in ratios):
            raise ValueError("Brightness ratios must be finite values in [0, 1].")
        if self.severe_over_tile_ratio < self.over_tile_ratio:
            raise ValueError("severe_over_tile_ratio must be at least over_tile_ratio.")
        if self.severe_over_luma_clip_ratio < self.over_luma_clip_ratio:
            raise ValueError("severe_over_luma_clip_ratio must be at least over_luma_clip_ratio.")
        if self.severe_under_shadow_ratio < self.under_shadow_ratio:
            raise ValueError("severe_under_shadow_ratio must be at least under_shadow_ratio.")
        percentiles = (
            self.over_p99,
            self.under_p50,
            self.severe_under_p50,
            self.target_p50,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 255 for value in percentiles):
            raise ValueError("Brightness percentile thresholds must be in [0, 255].")
        if self.severe_under_p50 > self.under_p50:
            raise ValueError("severe_under_p50 must not exceed under_p50.")
        if not 0 <= self.shadow_value < self.clipping_value <= 255:
            raise ValueError("Require 0 <= shadow_value < clipping_value <= 255.")
        if min(
            self.good_ev_radius,
            self.under_max_ev_increase,
            self.severe_under_max_ev_increase,
        ) < 0:
            raise ValueError("Brightness EV limits must be non-negative.")


def ev_step(cell: SensorCell) -> int:
    return EXPOSURE_MS_VALUES.index(cell.exposure_ms) + GAIN_VALUES.index(cell.gain)


class BrightnessGuard:
    def __init__(self, config: BrightnessGuardConfig | None = None) -> None:
        self.config = config or BrightnessGuardConfig()
        self.config.validate()
        self._smoothed: BrightnessStats | None = None
        self._state: BrightnessState | None = None
        self._candidate: BrightnessState | None = None
        self._candidate_frames = 0

    def observe(
        self,
        image_bgr: np.ndarray,
        current_cell: SensorCell,
        hard_safe_cells: Iterable[SensorCell],
    ) -> BrightnessDecision:
        hard_safe = tuple(dict.fromkeys(hard_safe_cells))
        if not hard_safe:
            raise ValueError("Brightness guard requires at least one hard-safe cell.")
        raw = self._stats(image_bgr)
        smoothed = self._smooth(raw)
        candidate = self._classify(smoothed)
        if self._classify(raw) is BrightnessState.SEVERE_OVER:
            candidate = BrightnessState.SEVERE_OVER
        state_changed = self._update_state(candidate)
        if self._state is None:
            raise RuntimeError("Brightness guard failed to initialize its state.")
        admissible = self._admissible(self._state, current_cell, hard_safe)
        recovery = (
            self._recovery(current_cell, hard_safe)
            if self._state is BrightnessState.SEVERE_OVER
            else None
        )
        prefer_brighter = (
            self._state in (BrightnessState.UNDER, BrightnessState.SEVERE_UNDER)
            or (
                self._state is BrightnessState.GOOD
                and smoothed.p50 < self.config.target_p50
            )
        )
        return BrightnessDecision(
            state=self._state,
            raw_stats=raw,
            smoothed_stats=smoothed,
            admissible_cells=admissible,
            recovery_cell=recovery,
            prefer_brighter=prefer_brighter,
            force_recovery=(
                self.config.mode is BrightnessGuardMode.ENFORCE
                and self._state is BrightnessState.SEVERE_OVER
                and recovery is not None
            ),
            state_changed=state_changed,
        )

    def _stats(self, image: np.ndarray) -> BrightnessStats:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("Brightness guard expects a non-empty uint8 BGR image.")
        bgr = image.astype(np.float32, copy=False)
        y = 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
        clipping = self.config.clipping_value
        rows = np.array_split(y, min(self.config.tile_grid_size, y.shape[0]), axis=0)
        tile_over = [
            float(np.mean(tile >= clipping)) >= self.config.tile_clip_pixel_ratio
            for row in rows
            for tile in np.array_split(row, min(self.config.tile_grid_size, y.shape[1]), axis=1)
        ]
        p10, p50, p90, p99 = np.percentile(y, (10, 50, 90, 99))
        return BrightnessStats(
            float(p10),
            float(p50),
            float(p90),
            float(p99),
            float(np.mean(y <= self.config.shadow_value)),
            float(np.mean(y >= clipping)),
            float(np.mean(np.max(image, axis=2) >= clipping)),
            float(np.mean(tile_over)),
        )

    def _smooth(self, raw: BrightnessStats) -> BrightnessStats:
        if self._smoothed is None:
            self._smoothed = raw
        else:
            alpha = self.config.ema_alpha
            self._smoothed = BrightnessStats(*(
                alpha * getattr(raw, item.name)
                + (1.0 - alpha) * getattr(self._smoothed, item.name)
                for item in fields(BrightnessStats)
            ))
        return self._smoothed

    def _classify(self, stats: BrightnessStats) -> BrightnessState:
        config = self.config
        if (
            stats.luma_clip_ratio >= config.severe_over_luma_clip_ratio
            or stats.overexposed_tile_ratio >= config.severe_over_tile_ratio
        ):
            return BrightnessState.SEVERE_OVER
        p99_with_evidence = (
            stats.p99 >= config.over_p99
            and (
                stats.luma_clip_ratio >= config.over_luma_clip_ratio
                or stats.channel_clip_ratio >= config.over_channel_clip_ratio
                or stats.overexposed_tile_ratio >= config.over_tile_ratio
            )
        )
        if (
            stats.channel_clip_ratio >= config.over_channel_clip_ratio
            or stats.overexposed_tile_ratio >= config.over_tile_ratio
            or p99_with_evidence
        ):
            return BrightnessState.OVER
        if (
            stats.p50 <= config.severe_under_p50
            or stats.shadow_ratio >= config.severe_under_shadow_ratio
        ):
            return BrightnessState.SEVERE_UNDER
        if (
            stats.p50 <= config.under_p50
            or stats.shadow_ratio >= config.under_shadow_ratio
        ):
            return BrightnessState.UNDER
        return BrightnessState.GOOD

    def _update_state(self, candidate: BrightnessState) -> bool:
        if self._state is None:
            self._state = candidate
            self._candidate = None
            self._candidate_frames = 0
            return False
        if candidate is self._state:
            self._candidate = None
            self._candidate_frames = 0
            return False
        if candidate is BrightnessState.SEVERE_OVER:
            self._state = candidate
            self._candidate = None
            self._candidate_frames = 0
            return True
        if candidate is not self._candidate:
            self._candidate = candidate
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1
        if self._candidate_frames < self.config.state_change_frames:
            return False
        self._state = candidate
        self._candidate = None
        self._candidate_frames = 0
        return True

    def _admissible(
        self,
        state: BrightnessState,
        current: SensorCell,
        hard_safe: tuple[SensorCell, ...],
    ) -> tuple[SensorCell, ...]:
        current_ev = ev_step(current)
        config = self.config
        if state is BrightnessState.SEVERE_OVER:
            allowed = lambda value: value <= current_ev - 1
        elif state is BrightnessState.OVER:
            allowed = lambda value: value <= current_ev
        elif state is BrightnessState.GOOD:
            allowed = lambda value: abs(value - current_ev) <= config.good_ev_radius
        elif state is BrightnessState.UNDER:
            allowed = lambda value: current_ev <= value <= current_ev + config.under_max_ev_increase
        else:
            allowed = lambda value: current_ev <= value <= current_ev + config.severe_under_max_ev_increase
        cells = {cell for cell in hard_safe if allowed(ev_step(cell))}
        cells.add(current)
        return tuple(sorted(cells, key=lambda cell: (ev_step(cell), cell.exposure_ms, cell.gain)))

    @staticmethod
    def _recovery(
        current: SensorCell,
        hard_safe: tuple[SensorCell, ...],
    ) -> SensorCell | None:
        current_point = (
            EXPOSURE_MS_VALUES.index(current.exposure_ms),
            GAIN_VALUES.index(current.gain),
        )
        current_ev = sum(current_point)
        candidates = [cell for cell in hard_safe if ev_step(cell) == current_ev - 1]
        if not candidates:
            candidates = [cell for cell in hard_safe if ev_step(cell) < current_ev]
        if not candidates:
            return None

        def key(cell: SensorCell) -> tuple[int, int, int, int]:
            point = (
                EXPOSURE_MS_VALUES.index(cell.exposure_ms),
                GAIN_VALUES.index(cell.gain),
            )
            distance = (point[0] - current_point[0]) ** 2 + (point[1] - current_point[1]) ** 2
            exposure_reduced = point[0] < current_point[0]
            return (distance, 0 if exposure_reduced else 1, point[0], point[1])

        return min(candidates, key=key)


def brightness_log_values(
    config: BrightnessGuardConfig,
    decision: BrightnessDecision | None,
) -> dict[str, object]:
    values: dict[str, object] = {"brightness_guard_mode": config.mode.value}
    if decision is None:
        return values
    stats = decision.smoothed_stats
    values.update(
        brightness_state=decision.state.value,
        brightness_state_changed=int(decision.state_changed),
        brightness_p10=stats.p10,
        brightness_p50=stats.p50,
        brightness_p90=stats.p90,
        brightness_p99=stats.p99,
        brightness_shadow_ratio=stats.shadow_ratio,
        brightness_luma_clip_ratio=stats.luma_clip_ratio,
        brightness_channel_clip_ratio=stats.channel_clip_ratio,
        brightness_overexposed_tile_ratio=stats.overexposed_tile_ratio,
        brightness_allowed_cell_count=len(decision.admissible_cells),
        brightness_prefer_brighter=int(decision.prefer_brighter),
        brightness_recovery_cell_id=(
            decision.recovery_cell.cell_id if decision.recovery_cell else ""
        ),
        brightness_force_recovery=int(decision.force_recovery),
        brightness_recovery_available=int(decision.recovery_cell is not None),
    )
    return values
