from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from hardware.utils import ContextKey, SensorCell

from .predictive_saturation_guard import (
    PredictiveSaturationGuard,
    PredictiveSaturationGuardConfig,
)
from .saturation_guard import SaturationGuard, SaturationMetrics, ev_index


@dataclass(frozen=True)
class BidirectionalExposureGuardConfig:
    predictive: PredictiveSaturationGuardConfig = PredictiveSaturationGuardConfig()
    max_ev_step_stops: int = 1
    shadow_pixel_threshold: float = 10.0
    soft_shadow_ratio: float = 0.30
    hard_shadow_ratio: float = 0.50
    soft_under_mean_luminance: float = 60.0
    hard_under_mean_luminance: float = 40.0
    under_recovery_shadow_ratio: float = 0.15
    under_recovery_mean_luminance: float = 80.0
    under_recovery_consecutive_frames: int = 3
    projected_shadow_ratio_limit: float = 0.30

    def __post_init__(self) -> None:
        values = (
            self.shadow_pixel_threshold,
            self.soft_shadow_ratio,
            self.hard_shadow_ratio,
            self.soft_under_mean_luminance,
            self.hard_under_mean_luminance,
            self.under_recovery_shadow_ratio,
            self.under_recovery_mean_luminance,
            self.projected_shadow_ratio_limit,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bidirectional exposure thresholds must be finite.")
        if self.max_ev_step_stops < 1:
            raise ValueError("Maximum EV step must be at least one stop.")
        if not 0 <= self.shadow_pixel_threshold <= 255:
            raise ValueError("Shadow pixel threshold must be in [0, 255].")
        if not (
            0
            <= self.under_recovery_shadow_ratio
            < self.soft_shadow_ratio
            < self.hard_shadow_ratio
            <= 1
        ):
            raise ValueError("Require 0 <= shadow recovery < soft < hard <= 1.")
        if not (
            0
            <= self.hard_under_mean_luminance
            < self.soft_under_mean_luminance
            < self.under_recovery_mean_luminance
            <= 255
        ):
            raise ValueError(
                "Require 0 <= hard mean < soft mean < recovery mean <= 255."
            )
        if not 0 <= self.projected_shadow_ratio_limit <= 1:
            raise ValueError("Projected shadow ratio limit must be in [0, 1].")
        if self.under_recovery_consecutive_frames < 1:
            raise ValueError("Shadow recovery frame count must be positive.")

    @classmethod
    def from_args(cls, args) -> "BidirectionalExposureGuardConfig":
        return cls(
            predictive=PredictiveSaturationGuardConfig.from_args(args),
            max_ev_step_stops=args.exposure_max_ev_step_stops,
            shadow_pixel_threshold=args.shadow_pixel_threshold,
            soft_shadow_ratio=args.shadow_soft_ratio,
            hard_shadow_ratio=args.shadow_hard_ratio,
            soft_under_mean_luminance=args.shadow_soft_mean_luminance,
            hard_under_mean_luminance=args.shadow_hard_mean_luminance,
            under_recovery_shadow_ratio=args.shadow_recovery_ratio,
            under_recovery_mean_luminance=args.shadow_recovery_mean_luminance,
            under_recovery_consecutive_frames=args.shadow_recovery_frames,
            projected_shadow_ratio_limit=args.shadow_projected_ratio_limit,
        )


@dataclass(frozen=True)
class BidirectionalExposureMetrics:
    mean_luminance: float
    channel_clip_ratio: float
    luminance_clip_ratio: float
    shadow_clip_ratio: float
    hard_overexposed: bool
    soft_overexposed: bool
    hard_underexposed: bool
    soft_underexposed: bool


@dataclass(frozen=True)
class BidirectionalExposureObservation:
    metrics: BidirectionalExposureMetrics
    guard_state: str
    overexposure_state: str
    underexposure_state: str
    guard_conflict: bool
    under_recovery_count: int
    over_recovery_count: int
    image: np.ndarray
    luminance: np.ndarray


@dataclass(frozen=True)
class BidirectionalCandidateFilterResult:
    candidates: tuple[SensorCell, ...]
    candidate_count_safety: int
    candidate_count_after_quarantine: int
    candidate_count_after_symmetric_step: int
    candidate_count_after_direction_guard: int
    candidate_count_after_projected_saturation: int
    candidate_count_after_projected_shadow: int
    projected_saturation_ratios: tuple[tuple[SensorCell, float], ...]
    projected_shadow_ratios: tuple[tuple[SensorCell, float], ...]
    ev_step_rejected_cells: tuple[SensorCell, ...]
    direction_rejected_cells: tuple[SensorCell, ...]
    saturation_rejected_cells: tuple[SensorCell, ...]
    shadow_rejected_cells: tuple[SensorCell, ...]
    fallback_used: bool
    fallback_reason: str


@dataclass
class _ExposureState:
    state: str = "normal"
    hard_active: bool = False
    recovery_count: int = 0


class BidirectionalExposureGuard(PredictiveSaturationGuard):
    """Predictive saturation guard with symmetric EV and shadow constraints."""

    def __init__(
        self,
        config: BidirectionalExposureGuardConfig,
        safe_fallback: Callable[[ContextKey], SensorCell] | None = None,
    ) -> None:
        super().__init__(config.predictive)
        self.bidirectional_config = config
        self._meter = SaturationGuard(config.predictive.observed_config)
        self._safe_fallback = safe_fallback
        self._over_states: dict[ContextKey, _ExposureState] = {}
        self._under_states: dict[ContextKey, _ExposureState] = {}

    def measure(self, image: np.ndarray) -> BidirectionalExposureMetrics:
        metrics, _ = self._measure_with_luminance(image)
        return metrics

    def _measure_with_luminance(
        self, image: np.ndarray
    ) -> tuple[BidirectionalExposureMetrics, np.ndarray]:
        saturation = self._meter.measure(image)
        bgr = image.astype(np.float32)
        luminance = 0.2126 * bgr[..., 2] + 0.7152 * bgr[..., 1] + 0.0722 * bgr[..., 0]
        shadow_ratio = float(
            np.mean(luminance <= self.bidirectional_config.shadow_pixel_threshold)
        )
        if not math.isfinite(shadow_ratio) or not 0 <= shadow_ratio <= 1:
            raise ValueError("Shadow clipping ratio must be finite and in [0, 1].")
        hard_under = (
            shadow_ratio >= self.bidirectional_config.hard_shadow_ratio
            and saturation.mean_luminance
            <= self.bidirectional_config.hard_under_mean_luminance
        )
        soft_under = (
            not hard_under
            and shadow_ratio >= self.bidirectional_config.soft_shadow_ratio
            and saturation.mean_luminance
            <= self.bidirectional_config.soft_under_mean_luminance
        )
        return (
            self._metrics(saturation, shadow_ratio, hard_under, soft_under),
            luminance,
        )

    def observe(
        self,
        context: ContextKey,
        cell: SensorCell,
        round_index: int,
        image: np.ndarray,
        *,
        setting_effective: bool,
    ) -> BidirectionalExposureObservation:
        context.validate()
        metrics, luminance = self._measure_with_luminance(image)
        over = self._over_states.setdefault(context, _ExposureState())
        under = self._under_states.setdefault(context, _ExposureState())
        if setting_effective:
            predictive = super().observe(
                context,
                cell,
                round_index,
                image,
                setting_effective=True,
            )
            self._update_over(over, predictive.guard_state, metrics)
            self._update_under(under, metrics)

        conflict = over.state != "normal" and under.state != "normal"
        combined = self._combined_state(over.state, under.state, conflict)
        return BidirectionalExposureObservation(
            metrics=metrics,
            guard_state=combined,
            overexposure_state=over.state,
            underexposure_state=under.state,
            guard_conflict=conflict,
            under_recovery_count=under.recovery_count,
            over_recovery_count=over.recovery_count,
            image=image,
            luminance=luminance,
        )

    def projected_shadow_clip_ratio(
        self,
        image: np.ndarray,
        current_cell: SensorCell,
        candidate: SensorCell,
    ) -> float | None:
        _, luminance = self._measure_with_luminance(image)
        delta_ev = ev_index(candidate) - ev_index(current_cell)
        if delta_ev >= 0:
            return None
        return self._projected_shadow_ratio(luminance, delta_ev)

    def filter_candidates(
        self,
        context: ContextKey,
        current_cell: SensorCell,
        round_index: int,
        observation: BidirectionalExposureObservation,
        base_candidates: Iterable[SensorCell],
    ) -> BidirectionalCandidateFilterResult:
        context.validate()
        current_ev = ev_index(current_cell)
        base = tuple(dict.fromkeys(base_candidates))
        if not base:
            raise RuntimeError("The base safety policy returned no candidate.")
        for cell in base:
            ev_index(cell)

        quarantine = self.quarantine(context, round_index)
        after_quarantine = tuple(
            cell
            for cell in base
            if quarantine is None or ev_index(cell) < quarantine.blocked_ev_min
        )

        after_step = tuple(
            cell
            for cell in after_quarantine
            if abs(ev_index(cell) - current_ev)
            <= self.bidirectional_config.max_ev_step_stops
        )
        step_rejected = self._sorted_difference(after_quarantine, after_step)

        after_direction = tuple(
            cell
            for cell in after_step
            if self._direction_allowed(
                observation.guard_state,
                ev_index(cell) - current_ev,
                cell == current_cell,
            )
        )
        direction_rejected = self._sorted_difference(after_step, after_direction)

        saturation_ratios: list[tuple[SensorCell, float]] = []
        saturation_rejected: list[SensorCell] = []
        after_saturation: list[SensorCell] = []
        for cell in after_direction:
            ratio = self.projected_channel_clip_ratio(
                observation.image, current_cell, cell
            )
            if ratio is None:
                after_saturation.append(cell)
            else:
                saturation_ratios.append((cell, ratio))
                if ratio >= self.config.projected_hard_clip_ratio:
                    saturation_rejected.append(cell)
                else:
                    after_saturation.append(cell)

        shadow_ratios: list[tuple[SensorCell, float]] = []
        shadow_rejected: list[SensorCell] = []
        after_shadow: list[SensorCell] = []
        for cell in after_saturation:
            delta_ev = ev_index(cell) - current_ev
            if delta_ev >= 0:
                after_shadow.append(cell)
                continue
            ratio = self._projected_shadow_ratio(observation.luminance, delta_ev)
            shadow_ratios.append((cell, ratio))
            if ratio >= self.bidirectional_config.projected_shadow_ratio_limit:
                shadow_rejected.append(cell)
            else:
                after_shadow.append(cell)

        fallback_used = not after_shadow
        fallback_reason = ""
        candidates = tuple(after_shadow)
        if fallback_used:
            selected, fallback_reason = self._fallback(
                context, base, current_cell, observation.guard_state
            )
            candidates = (selected,)

        return BidirectionalCandidateFilterResult(
            candidates=candidates,
            candidate_count_safety=len(base),
            candidate_count_after_quarantine=len(after_quarantine),
            candidate_count_after_symmetric_step=len(after_step),
            candidate_count_after_direction_guard=len(after_direction),
            candidate_count_after_projected_saturation=len(after_saturation),
            candidate_count_after_projected_shadow=len(after_shadow),
            projected_saturation_ratios=tuple(saturation_ratios),
            projected_shadow_ratios=tuple(shadow_ratios),
            ev_step_rejected_cells=step_rejected,
            direction_rejected_cells=direction_rejected,
            saturation_rejected_cells=tuple(
                sorted(saturation_rejected, key=lambda cell: cell.cell_id)
            ),
            shadow_rejected_cells=tuple(
                sorted(shadow_rejected, key=lambda cell: cell.cell_id)
            ),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def _update_over(
        self,
        state: _ExposureState,
        predictive_state: str,
        metrics: BidirectionalExposureMetrics,
    ) -> None:
        state.state = (
            "hard_overexposed"
            if predictive_state == "hard"
            else "soft_overexposed" if predictive_state == "soft" else "normal"
        )
        if metrics.hard_overexposed:
            state.hard_active = True
            state.recovery_count = 0
        elif state.state == "hard_overexposed":
            state.recovery_count = (
                state.recovery_count + 1
                if metrics.channel_clip_ratio <= self.config.recovery_clip_ratio
                else 0
            )
        else:
            state.hard_active = False
            state.recovery_count = 0

    def _update_under(
        self, state: _ExposureState, metrics: BidirectionalExposureMetrics
    ) -> None:
        if metrics.hard_underexposed:
            state.hard_active = True
            state.recovery_count = 0
        elif state.hard_active:
            recovered = (
                metrics.shadow_clip_ratio
                <= self.bidirectional_config.under_recovery_shadow_ratio
                and metrics.mean_luminance
                >= self.bidirectional_config.under_recovery_mean_luminance
            )
            state.recovery_count = state.recovery_count + 1 if recovered else 0
            if (
                state.recovery_count
                >= self.bidirectional_config.under_recovery_consecutive_frames
            ):
                state.hard_active = False
                state.recovery_count = 0
        state.state = (
            "hard_underexposed"
            if state.hard_active
            else "soft_underexposed" if metrics.soft_underexposed else "normal"
        )

    @staticmethod
    def _metrics(
        saturation: SaturationMetrics,
        shadow_ratio: float,
        hard_under: bool,
        soft_under: bool,
    ) -> BidirectionalExposureMetrics:
        return BidirectionalExposureMetrics(
            saturation.mean_luminance,
            saturation.channel_clip_ratio,
            saturation.luminance_clip_ratio,
            shadow_ratio,
            saturation.hard_overexposed,
            saturation.soft_overexposed,
            hard_under,
            soft_under,
        )

    @staticmethod
    def _combined_state(over: str, under: str, conflict: bool) -> str:
        if conflict:
            return "guard_conflict"
        if over == "hard_overexposed" or under == "hard_underexposed":
            return over if over == "hard_overexposed" else under
        if over != "normal":
            return over
        if under != "normal":
            return under
        return "normal"

    @staticmethod
    def _direction_allowed(state: str, delta_ev: int, is_current: bool) -> bool:
        if state == "guard_conflict":
            return is_current
        if state == "hard_overexposed":
            return delta_ev == -1
        if state == "soft_overexposed":
            return -1 <= delta_ev <= 0
        if state == "hard_underexposed":
            return delta_ev == 1
        if state == "soft_underexposed":
            return 0 <= delta_ev <= 1
        return True

    def _projected_shadow_ratio(self, luminance: np.ndarray, delta_ev: int) -> float:
        ratio = float(
            np.mean(
                luminance * (2.0**delta_ev)
                <= self.bidirectional_config.shadow_pixel_threshold
            )
        )
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError(
                "Projected shadow clipping ratio must be finite and in [0, 1]."
            )
        return ratio

    def _fallback(
        self,
        context: ContextKey,
        base: tuple[SensorCell, ...],
        current_cell: SensorCell,
        state: str,
    ) -> tuple[SensorCell, str]:
        current_ev = ev_index(current_cell)
        if state == "guard_conflict":
            if current_cell in base:
                return current_cell, "guard_conflict"
            return (
                self._default_fallback(context, base),
                "guard_conflict_current_unsafe",
            )
        if state == "hard_underexposed":
            brighter = tuple(cell for cell in base if ev_index(cell) > current_ev)
            if brighter:
                return (
                    min(
                        brighter, key=lambda cell: self._direction_key(cell, current_ev)
                    ),
                    "hard_under_fallback",
                )
            if current_cell in base:
                return current_cell, "underexposure_unrecoverable_within_safety_policy"
            return (
                self._default_fallback(context, base),
                "underexposure_unrecoverable_within_safety_policy",
            )
        if state == "hard_overexposed":
            darker = tuple(cell for cell in base if ev_index(cell) < current_ev)
            if darker:
                return (
                    min(darker, key=lambda cell: self._direction_key(cell, current_ev)),
                    "hard_over_fallback",
                )
            if current_cell in base:
                return current_cell, "overexposure_unrecoverable_within_safety_policy"
            return (
                self._default_fallback(context, base),
                "overexposure_unrecoverable_within_safety_policy",
            )
        if current_cell in base:
            return current_cell, "hold_current_safe_cell"
        return self._default_fallback(context, base), "current_cell_unsafe"

    def _default_fallback(
        self, context: ContextKey, base: tuple[SensorCell, ...]
    ) -> SensorCell:
        if self._safe_fallback is not None:
            cell = self._safe_fallback(context)
            if cell not in base:
                raise RuntimeError(
                    "Default fallback returned a cell outside SafetyPolicy."
                )
            return cell
        return min(
            base,
            key=lambda cell: (
                ev_index(cell),
                cell.exposure_ms,
                cell.gain,
                cell.cell_id,
            ),
        )

    @staticmethod
    def _direction_key(cell: SensorCell, current_ev: int) -> tuple[int, int, int, str]:
        return (
            abs(ev_index(cell) - current_ev),
            cell.exposure_ms,
            cell.gain,
            cell.cell_id,
        )

    @staticmethod
    def _sorted_difference(
        before: tuple[SensorCell, ...], after: tuple[SensorCell, ...]
    ) -> tuple[SensorCell, ...]:
        allowed = set(after)
        return tuple(
            sorted(
                (cell for cell in before if cell not in allowed),
                key=lambda cell: cell.cell_id,
            )
        )
