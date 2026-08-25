from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hardware.utils import LIGHT_STATE_COUNT, ContextKey, SensorCell

from .saturation_guard import (
    QuarantineEntry,
    SaturationGuard,
    SaturationGuardConfig,
    SaturationMetrics,
    ev_index,
)


@dataclass(frozen=True)
class PredictiveSaturationGuardConfig:
    pixel_clip_threshold: float = 250.0
    soft_clip_ratio: float = 0.60
    secondary_clip_ratio: float = 0.85
    hard_clip_ratio: float = 0.89
    hard_mean_luminance: float = 243.0
    recovery_clip_ratio: float = 0.50
    recovery_consecutive_frames: int = 3
    quarantine_rounds: int = 60
    minimum_ev_drop_stops: int = 1
    max_upward_ev_stops: int = 1
    projected_pixel_clip_threshold: float = 250.0
    projected_hard_clip_ratio: float = 0.90

    def __post_init__(self) -> None:
        SaturationGuardConfig(
            pixel_clip_threshold=self.pixel_clip_threshold,
            soft_clip_ratio=self.soft_clip_ratio,
            secondary_clip_ratio=self.secondary_clip_ratio,
            hard_clip_ratio=self.hard_clip_ratio,
            hard_mean_luminance=self.hard_mean_luminance,
            recovery_clip_ratio=self.recovery_clip_ratio,
            recovery_consecutive_frames=self.recovery_consecutive_frames,
            quarantine_rounds=self.quarantine_rounds,
            minimum_ev_drop_stops=self.minimum_ev_drop_stops,
        )
        projected = (
            self.projected_pixel_clip_threshold,
            self.projected_hard_clip_ratio,
        )
        if not all(math.isfinite(value) for value in projected):
            raise ValueError("Projected saturation thresholds must be finite.")
        if not 0 < self.projected_pixel_clip_threshold <= 255:
            raise ValueError("Projected pixel threshold must be in (0, 255].")
        if not 0 < self.projected_hard_clip_ratio <= 1:
            raise ValueError("Projected hard clipping ratio must be in (0, 1].")
        if self.max_upward_ev_stops < 0:
            raise ValueError("Maximum upward EV stops must be non-negative.")

    @property
    def observed_config(self) -> SaturationGuardConfig:
        return SaturationGuardConfig(
            pixel_clip_threshold=self.pixel_clip_threshold,
            soft_clip_ratio=self.soft_clip_ratio,
            secondary_clip_ratio=self.secondary_clip_ratio,
            hard_clip_ratio=self.hard_clip_ratio,
            hard_mean_luminance=self.hard_mean_luminance,
            recovery_clip_ratio=self.recovery_clip_ratio,
            recovery_consecutive_frames=self.recovery_consecutive_frames,
            quarantine_rounds=self.quarantine_rounds,
            minimum_ev_drop_stops=self.minimum_ev_drop_stops,
        )

    @classmethod
    def from_args(cls, args) -> "PredictiveSaturationGuardConfig":
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
            max_upward_ev_stops=args.sat_max_upward_ev_stops,
            projected_pixel_clip_threshold=args.sat_projected_pixel_threshold,
            projected_hard_clip_ratio=args.sat_projected_hard_clip_ratio,
        )


@dataclass(frozen=True)
class PredictiveSaturationObservation:
    metrics: SaturationMetrics
    guard_state: str
    current_max_channel: np.ndarray


@dataclass(frozen=True)
class PredictiveCandidateFilterResult:
    candidates: tuple[SensorCell, ...]
    candidate_count_safety: int
    candidate_count_after_quarantine: int
    candidate_count_after_observed_guard: int
    candidate_count_after_ev_step: int
    candidate_count_after_projection: int
    quarantine_active: bool
    blocked_ev_min: int | None
    quarantine_expiry_round: int | None
    projected_clip_ratios: tuple[tuple[SensorCell, float], ...]
    projected_rejected_cells: tuple[SensorCell, ...]
    ev_step_rejected_cells: tuple[SensorCell, ...]
    fallback_used: bool
    fallback_reason: str


class PredictiveSaturationGuard:
    """Reactive statistic reuse plus light-scoped predictive constraints."""

    def __init__(self, config: PredictiveSaturationGuardConfig) -> None:
        self.config = config
        self._reactive = SaturationGuard(config.observed_config)
        self._quarantines: dict[int, QuarantineEntry] = {}

    def observe(
        self,
        context: ContextKey,
        cell: SensorCell,
        round_index: int,
        image: np.ndarray,
        *,
        setting_effective: bool,
    ) -> PredictiveSaturationObservation:
        context.validate()
        observation = self._reactive.observe(
            context,
            cell,
            round_index,
            image,
            setting_effective=False,
        )
        current_max_channel = np.max(image.astype(np.float32), axis=-1)
        if observation.metrics.hard_overexposed and setting_effective:
            current = self._active_quarantine(context.light_state, round_index)
            blocked_ev = ev_index(cell)
            self._quarantines[context.light_state] = QuarantineEntry(
                min(blocked_ev, current.blocked_ev_min) if current else blocked_ev,
                max(round_index + self.config.quarantine_rounds, current.expiry_round)
                if current
                else round_index + self.config.quarantine_rounds,
            )
        return PredictiveSaturationObservation(
            observation.metrics,
            observation.guard_state,
            current_max_channel,
        )

    def projected_channel_clip_ratio(
        self,
        image: np.ndarray,
        current_cell: SensorCell,
        candidate: SensorCell,
    ) -> float | None:
        self._reactive.measure(image)
        delta_ev = ev_index(candidate) - ev_index(current_cell)
        if delta_ev <= 0:
            return None
        return self._projected_ratio(
            np.max(image.astype(np.float32), axis=-1), delta_ev
        )

    def filter_candidates(
        self,
        context: ContextKey,
        current_cell: SensorCell,
        round_index: int,
        observation: PredictiveSaturationObservation,
        base_candidates: Iterable[SensorCell],
    ) -> PredictiveCandidateFilterResult:
        context.validate()
        current_ev = ev_index(current_cell)
        base = tuple(dict.fromkeys(base_candidates))
        if not base:
            raise RuntimeError("The base safety policy returned no candidate.")
        for cell in base:
            ev_index(cell)

        quarantine = self._active_quarantine(context.light_state, round_index)
        after_quarantine = tuple(
            cell
            for cell in base
            if quarantine is None or ev_index(cell) < quarantine.blocked_ev_min
        )
        metrics = observation.metrics
        if metrics.hard_overexposed:
            observed_ceiling = current_ev - self.config.minimum_ev_drop_stops
            after_observed = tuple(
                cell for cell in after_quarantine if ev_index(cell) <= observed_ceiling
            )
        elif metrics.soft_overexposed:
            after_observed = tuple(
                cell for cell in after_quarantine if ev_index(cell) <= current_ev
            )
        else:
            after_observed = after_quarantine

        upward_ceiling = current_ev + self.config.max_upward_ev_stops
        after_ev_step = tuple(
            cell for cell in after_observed if ev_index(cell) <= upward_ceiling
        )
        ev_step_rejected = tuple(
            sorted(
                (cell for cell in after_observed if ev_index(cell) > upward_ceiling),
                key=lambda cell: cell.cell_id,
            )
        )

        projected_ratios: list[tuple[SensorCell, float]] = []
        projected_rejected: list[SensorCell] = []
        after_projection: list[SensorCell] = []
        for cell in after_ev_step:
            delta_ev = ev_index(cell) - current_ev
            if delta_ev <= 0:
                after_projection.append(cell)
                continue
            ratio = self._projected_ratio(observation.current_max_channel, delta_ev)
            projected_ratios.append((cell, ratio))
            if ratio >= self.config.projected_hard_clip_ratio:
                projected_rejected.append(cell)
            else:
                after_projection.append(cell)

        fallback_used = not after_projection
        fallback_reason = ""
        candidates = tuple(after_projection)
        if fallback_used:
            fallback_reason = (
                "quarantine_empty"
                if not after_quarantine
                else "observed_guard_empty"
                if not after_observed
                else "ev_step_empty"
                if not after_ev_step
                else "projection_empty"
            )
            candidates = (
                self._fallback(base, current_cell, metrics.hard_overexposed),
            )

        return PredictiveCandidateFilterResult(
            candidates=candidates,
            candidate_count_safety=len(base),
            candidate_count_after_quarantine=len(after_quarantine),
            candidate_count_after_observed_guard=len(after_observed),
            candidate_count_after_ev_step=len(after_ev_step),
            candidate_count_after_projection=len(after_projection),
            quarantine_active=quarantine is not None,
            blocked_ev_min=quarantine.blocked_ev_min if quarantine else None,
            quarantine_expiry_round=quarantine.expiry_round if quarantine else None,
            projected_clip_ratios=tuple(projected_ratios),
            projected_rejected_cells=tuple(
                sorted(projected_rejected, key=lambda cell: cell.cell_id)
            ),
            ev_step_rejected_cells=ev_step_rejected,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    def quarantine(
        self, context_or_light_state: ContextKey | int, round_index: int
    ) -> QuarantineEntry | None:
        light_state = (
            context_or_light_state.light_state
            if isinstance(context_or_light_state, ContextKey)
            else int(context_or_light_state)
        )
        return self._active_quarantine(light_state, round_index)

    def _active_quarantine(
        self, light_state: int, round_index: int
    ) -> QuarantineEntry | None:
        if not 0 <= light_state < LIGHT_STATE_COUNT:
            raise ValueError(f"light_state must be in [0, {LIGHT_STATE_COUNT - 1}].")
        entry = self._quarantines.get(light_state)
        if entry is not None and round_index >= entry.expiry_round:
            del self._quarantines[light_state]
            return None
        return entry

    def _projected_ratio(self, current_max_channel: np.ndarray, delta_ev: int) -> float:
        scale = 2.0 ** delta_ev
        ratio = float(
            np.mean(
                current_max_channel * scale
                >= self.config.projected_pixel_clip_threshold
            )
        )
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("Projected clipping ratio must be finite and in [0, 1].")
        return ratio

    @staticmethod
    def _fallback(
        base: tuple[SensorCell, ...],
        current_cell: SensorCell,
        hard_overexposed: bool,
    ) -> SensorCell:
        current_ev = ev_index(current_cell)
        pool = tuple(cell for cell in base if ev_index(cell) < current_ev)
        if not pool:
            pool = base
        if hard_overexposed:
            pool = tuple(cell for cell in pool if ev_index(cell) <= current_ev)
            if not pool:
                raise RuntimeError(
                    "No base-safe fallback can avoid an upward EV move after hard clipping."
                )
        return min(
            pool,
            key=lambda cell: (
                ev_index(cell),
                cell.exposure_ms,
                cell.gain,
                cell.cell_id,
            ),
        )
