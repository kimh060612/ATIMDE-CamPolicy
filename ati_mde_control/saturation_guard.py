from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hardware.utils import (
    EXPOSURE_MS_VALUES,
    GAIN_VALUES,
    ContextKey,
    SensorCell,
)

from .risk_bandit_policy import RiskBanditDecision, RiskBanditPolicy


_EXPOSURE_INDEX = {value: index for index, value in enumerate(EXPOSURE_MS_VALUES)}
_GAIN_INDEX = {value: index for index, value in enumerate(GAIN_VALUES)}


def ev_index(cell: SensorCell) -> int:
    try:
        return _EXPOSURE_INDEX[cell.exposure_ms] + _GAIN_INDEX[cell.gain]
    except KeyError as error:
        raise ValueError(f"Cell is outside the configured camera grid: {cell}") from error


@dataclass(frozen=True)
class SaturationGuardConfig:
    pixel_clip_threshold: float = 250.0
    soft_clip_ratio: float = 0.80
    secondary_clip_ratio: float = 0.85
    hard_clip_ratio: float = 0.90
    hard_mean_luminance: float = 245.0
    recovery_clip_ratio: float = 0.70
    recovery_consecutive_frames: int = 3
    quarantine_rounds: int = 30
    minimum_ev_drop_stops: int = 1

    def __post_init__(self) -> None:
        floats = (
            self.pixel_clip_threshold,
            self.soft_clip_ratio,
            self.secondary_clip_ratio,
            self.hard_clip_ratio,
            self.hard_mean_luminance,
            self.recovery_clip_ratio,
        )
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("Saturation guard thresholds must be finite.")
        if not 0 < self.pixel_clip_threshold <= 255:
            raise ValueError("Pixel clipping threshold must be in (0, 255].")
        if not 0 <= self.hard_mean_luminance <= 255:
            raise ValueError("Hard mean luminance must be in [0, 255].")
        if not (
            0
            <= self.recovery_clip_ratio
            < self.soft_clip_ratio
            <= self.secondary_clip_ratio
            <= self.hard_clip_ratio
            <= 1
        ):
            raise ValueError(
                "Require 0 <= recovery < soft <= secondary <= hard <= 1."
            )
        if self.recovery_consecutive_frames < 1:
            raise ValueError("Recovery frame count must be positive.")
        if self.quarantine_rounds < 1:
            raise ValueError("Quarantine round count must be positive.")
        if self.minimum_ev_drop_stops < 1:
            raise ValueError("Minimum EV drop must be positive.")

    @classmethod
    def from_args(cls, args) -> "SaturationGuardConfig":
        return cls(
            pixel_clip_threshold=args.sat_pixel_threshold,
            soft_clip_ratio=args.sat_soft_clip_ratio,
            secondary_clip_ratio=args.sat_secondary_clip_ratio,
            hard_clip_ratio=args.sat_hard_clip_ratio,
            hard_mean_luminance=args.sat_hard_mean_luminance,
            recovery_clip_ratio=args.sat_recovery_clip_ratio,
            recovery_consecutive_frames=args.sat_recovery_frames,
            quarantine_rounds=args.sat_quarantine_rounds,
            minimum_ev_drop_stops=args.sat_min_ev_drop_stops,
        )


@dataclass(frozen=True)
class SaturationMetrics:
    mean_luminance: float
    channel_clip_ratio: float
    luminance_clip_ratio: float
    hard_overexposed: bool
    soft_overexposed: bool


@dataclass(frozen=True)
class SaturationObservation:
    metrics: SaturationMetrics
    guard_state: str


@dataclass(frozen=True)
class QuarantineEntry:
    blocked_ev_min: int
    expiry_round: int


@dataclass(frozen=True)
class CandidateFilterResult:
    candidates: tuple[SensorCell, ...]
    candidate_count_before_guard: int
    candidate_count_after_guard: int
    fallback_used: bool
    quarantine_active: bool
    quarantine_expiry_round: int | None


@dataclass
class _ImmediateState:
    hard_active: bool = False
    recovery_count: int = 0


class SaturationGuard:
    def __init__(self, config: SaturationGuardConfig) -> None:
        self.config = config
        self._states: dict[ContextKey, _ImmediateState] = {}
        self._quarantines: dict[ContextKey, QuarantineEntry] = {}

    def measure(self, image: np.ndarray) -> SaturationMetrics:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Saturation guard image must have shape HxWx3.")
        if image.size == 0:
            raise ValueError("Saturation guard image must not be empty.")
        bgr = image.astype(np.float32)
        if not np.all(np.isfinite(bgr)):
            raise ValueError("Saturation guard image must contain only finite values.")

        blue = bgr[..., 0]
        green = bgr[..., 1]
        red = bgr[..., 2]
        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        mean_luminance = float(np.mean(luminance))
        channel_clip_ratio = float(
            np.mean(np.max(bgr, axis=-1) >= self.config.pixel_clip_threshold)
        )
        luminance_clip_ratio = float(
            np.mean(luminance >= self.config.pixel_clip_threshold)
        )
        values = (mean_luminance, channel_clip_ratio, luminance_clip_ratio)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Saturation metrics must be finite.")
        if not all(0 <= value <= 1 for value in values[1:]):
            raise ValueError("Saturation clipping ratios must be in [0, 1].")

        hard = (
            channel_clip_ratio >= self.config.hard_clip_ratio
            or (
                channel_clip_ratio >= self.config.secondary_clip_ratio
                and mean_luminance >= self.config.hard_mean_luminance
            )
        )
        soft = not hard and channel_clip_ratio >= self.config.soft_clip_ratio
        return SaturationMetrics(
            mean_luminance,
            channel_clip_ratio,
            luminance_clip_ratio,
            hard,
            soft,
        )

    def observe(
        self,
        context: ContextKey,
        cell: SensorCell,
        round_index: int,
        image: np.ndarray,
        *,
        setting_effective: bool,
    ) -> SaturationObservation:
        context.validate()
        cell_ev = ev_index(cell)
        metrics = self.measure(image)
        state = self._states.setdefault(context, _ImmediateState())
        if metrics.hard_overexposed:
            state.hard_active = True
            state.recovery_count = 0
            if setting_effective:
                current = self._active_quarantine(context, round_index)
                self._quarantines[context] = QuarantineEntry(
                    min(cell_ev, current.blocked_ev_min) if current else cell_ev,
                    max(round_index + self.config.quarantine_rounds, current.expiry_round)
                    if current
                    else round_index + self.config.quarantine_rounds,
                )
        elif state.hard_active:
            if metrics.channel_clip_ratio <= self.config.recovery_clip_ratio:
                state.recovery_count += 1
                if state.recovery_count >= self.config.recovery_consecutive_frames:
                    state.hard_active = False
                    state.recovery_count = 0
            else:
                state.recovery_count = 0

        guard_state = (
            "hard"
            if state.hard_active
            else "soft"
            if metrics.soft_overexposed
            else "normal"
        )
        return SaturationObservation(metrics, guard_state)

    def filter_candidates(
        self,
        context: ContextKey,
        current_cell: SensorCell,
        round_index: int,
        observation: SaturationObservation,
        base_candidates: Iterable[SensorCell],
    ) -> CandidateFilterResult:
        context.validate()
        current_ev = ev_index(current_cell)
        base = tuple(dict.fromkeys(base_candidates))
        if not base:
            raise RuntimeError("The base safety policy returned no candidate.")
        for cell in base:
            ev_index(cell)

        quarantine = self._active_quarantine(context, round_index)
        candidates = tuple(
            cell
            for cell in base
            if quarantine is None or ev_index(cell) < quarantine.blocked_ev_min
        )
        metrics = observation.metrics
        if metrics.hard_overexposed:
            maximum_ev = current_ev - self.config.minimum_ev_drop_stops
            candidates = tuple(cell for cell in candidates if ev_index(cell) <= maximum_ev)
        elif metrics.soft_overexposed:
            candidates = tuple(cell for cell in candidates if ev_index(cell) <= current_ev)

        fallback_used = not candidates
        if fallback_used:
            candidates = (min(base, key=self._fallback_key),)
        return CandidateFilterResult(
            candidates=candidates,
            candidate_count_before_guard=len(base),
            candidate_count_after_guard=len(candidates),
            fallback_used=fallback_used,
            quarantine_active=quarantine is not None,
            quarantine_expiry_round=(quarantine.expiry_round if quarantine else None),
        )

    def quarantine(self, context: ContextKey, round_index: int) -> QuarantineEntry | None:
        context.validate()
        return self._active_quarantine(context, round_index)

    def _active_quarantine(
        self, context: ContextKey, round_index: int
    ) -> QuarantineEntry | None:
        entry = self._quarantines.get(context)
        if entry is not None and round_index >= entry.expiry_round:
            del self._quarantines[context]
            return None
        return entry

    @staticmethod
    def _fallback_key(cell: SensorCell) -> tuple[int, int, int, str]:
        return ev_index(cell), cell.exposure_ms, cell.gain, cell.cell_id


class SaturationGuardedRiskBanditPolicy(RiskBanditPolicy):
    """Thin candidate-list adapter over the unchanged v1 GP policy."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._candidate_override: tuple[ContextKey, tuple[SensorCell, ...]] | None = None

    def base_safe_cells(self, context: ContextKey) -> tuple[SensorCell, ...]:
        return super().safe_cells(context)

    def safe_cells(self, context: ContextKey) -> tuple[SensorCell, ...]:
        if self._candidate_override is not None and self._candidate_override[0] == context:
            return self._candidate_override[1]
        return self.base_safe_cells(context)

    def select_from_candidates(
        self,
        context: ContextKey,
        current_cell: SensorCell,
        timestamp_ns: int,
        candidates: Iterable[SensorCell],
    ) -> RiskBanditDecision:
        allowed = tuple(sorted(dict.fromkeys(candidates), key=lambda cell: cell.cell_id))
        base_safe = set(self.base_safe_cells(context))
        if not allowed:
            raise ValueError("The GP policy must not receive an empty candidate list.")
        if any(cell not in base_safe for cell in allowed):
            raise ValueError("The GP policy received a cell outside SafetyPolicy.safe_cells().")
        self._candidate_override = (context, allowed)
        try:
            return super().select_action(context, current_cell, timestamp_ns)
        finally:
            self._candidate_override = None
