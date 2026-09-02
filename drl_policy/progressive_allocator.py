"""Paper-inspired progressive exposure allocation.

This operationalizes the progressive allocation example from Zhang et al.,
ICRA 2024. The paper does not publish every interval boundary or hardware
quantization rule, so this is not an exact reproduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


DEFAULT_MILESTONES: tuple[tuple[float, float], ...] = (
    (5_000.0, 5.0),
    (10_000.0, 10.0),
    (20_000.0, 20.0),
)


@dataclass(frozen=True)
class ExposureAllocation:
    exposure_time_us: float
    gain_db: float
    requested_exposure: float
    realized_exposure: float
    was_clipped: bool
    stage: str


def _grid_maximum(minimum: float, maximum: float, step: float) -> float:
    return minimum + math.floor((maximum - minimum) / step + 1e-12) * step


def _quantize_nearest(
    value: float, minimum: float, maximum: float, step: float
) -> float:
    maximum = _grid_maximum(minimum, maximum, step)
    index = math.floor((value - minimum) / step + 0.5)
    return min(max(minimum + index * step, minimum), maximum)


def _quantize_up(
    value: float, minimum: float, maximum: float, step: float
) -> float:
    maximum = _grid_maximum(minimum, maximum, step)
    index = math.ceil((value - minimum) / step - 1e-12)
    return min(max(minimum + index * step, minimum), maximum)


def allocate_progressive_exposure(
    target_exposure: float,
    *,
    t_min_us: float,
    t_max_us: float,
    gain_min_db: float,
    gain_max_db: float,
    gain_step_db: float,
    milestones: Sequence[tuple[float, float]] = DEFAULT_MILESTONES,
    t_step_us: float = 1.0,
) -> ExposureAllocation:
    """Split composite exposure ``e=t*10**(g/20)`` into quantized time/gain."""
    values = (
        target_exposure,
        t_min_us,
        t_max_us,
        gain_min_db,
        gain_max_db,
        gain_step_db,
        t_step_us,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Exposure allocation inputs must be finite.")
    if target_exposure <= 0 or t_min_us <= 0 or t_max_us < t_min_us:
        raise ValueError("Require positive exposure and 0 < t_min_us <= t_max_us.")
    if gain_max_db < gain_min_db or gain_step_db <= 0 or t_step_us <= 0:
        raise ValueError("Gain bounds and quantization steps are invalid.")
    if not milestones:
        raise ValueError("At least one milestone is required.")

    milestone_values = tuple((float(t), float(g)) for t, g in milestones)
    if any(not math.isfinite(t) or not math.isfinite(g) or t <= 0 for t, g in milestone_values):
        raise ValueError("Milestone exposure times must be positive and finite.")
    thresholds = tuple(t * 10.0 ** (g / 20.0) for t, g in milestone_values)
    if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("Milestone composite exposures must be strictly increasing.")

    if target_exposure < thresholds[0]:
        gain_db = gain_min_db
        stage = "below_first"
    else:
        for index, upper in enumerate(thresholds[1:]):
            if target_exposure < upper:
                gain_db = milestone_values[index][1]
                stage = f"interval_{index}"
                break
        else:
            gain_db = milestone_values[-1][1]
            stage = "above_last"

    t_max_grid = _grid_maximum(t_min_us, t_max_us, t_step_us)
    gain_max_grid = _grid_maximum(gain_min_db, gain_max_db, gain_step_db)
    gain_db = min(max(gain_db, gain_min_db), gain_max_grid)
    required_gain_db = 20.0 * math.log10(target_exposure / t_max_grid)
    if required_gain_db > gain_db:
        gain_db = _quantize_up(
            required_gain_db, gain_min_db, gain_max_grid, gain_step_db
        )
        stage += "_gain_ramp"
    else:
        gain_db = _quantize_nearest(
            gain_db, gain_min_db, gain_max_grid, gain_step_db
        )

    exposure_time_us = _quantize_nearest(
        target_exposure / 10.0 ** (gain_db / 20.0),
        t_min_us,
        t_max_grid,
        t_step_us,
    )
    realized_exposure = exposure_time_us * 10.0 ** (gain_db / 20.0)
    feasible_min = t_min_us * 10.0 ** (gain_min_db / 20.0)
    feasible_max = t_max_grid * 10.0 ** (gain_max_grid / 20.0)
    was_clipped = target_exposure < feasible_min or target_exposure > feasible_max
    if was_clipped:
        stage += "_clipped"

    return ExposureAllocation(
        exposure_time_us=exposure_time_us,
        gain_db=gain_db,
        requested_exposure=target_exposure,
        realized_exposure=realized_exposure,
        was_clipped=was_clipped,
        stage=stage,
    )
