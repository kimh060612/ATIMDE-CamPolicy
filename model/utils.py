import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image

LIGHT_LEVELS = ("normal", "dim", "dark")
MOTION_LEVELS = ("fast", "slow", "stop", "rotate", "spin")

@dataclass(frozen=True)
class CameraParameterRange:
    """
    한 physical camera model에 대해 train/inference에서 공통으로 사용하는
    exposure/gain 허용 범위입니다.

    이 값은 dataset에서 자동 추정하기보다 camera API/실험 설계에서 정한
    고정 범위를 명시하는 것을 권장합니다.
    """
    exposure_min: float
    exposure_max: float
    gain_min: float
    gain_max: float

    def __post_init__(self) -> None:
        if not self.exposure_max > self.exposure_min:
            raise ValueError("exposure_max must be greater than exposure_min.")
        if not self.gain_max > self.gain_min:
            raise ValueError("gain_max must be greater than gain_min.")

def _normalize_camera_value(
    value: torch.Tensor,
    minimum: float,
    maximum: float,
    *,
    scale: Literal["linear", "log"],
    output_range: Literal["zero_one", "minus_one_one"],
    clip: bool,
) -> torch.Tensor:
    value = value.to(dtype=torch.float32)
    if scale == "log":
        value = torch.log(value.clamp_min(torch.finfo(value.dtype).tiny))
        minimum = math.log(minimum)
        maximum = math.log(maximum)

    normalized = (value - minimum) / (maximum - minimum)
    if clip:
        normalized = normalized.clamp(0.0, 1.0)
    if output_range == "minus_one_one":
        normalized = normalized.mul(2.0).sub(1.0)
    return normalized

def _validate_camera_parameter_normalization(
    parameter_range: CameraParameterRange,
    scale: Literal["linear", "log"],
    output_range: Literal["zero_one", "minus_one_one"],
) -> None:
    if scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'.")
    if output_range not in {"zero_one", "minus_one_one"}:
        raise ValueError("output_range must be 'zero_one' or 'minus_one_one'.")
    if scale == "log":
        if parameter_range.exposure_min <= 0:
            raise ValueError("Log normalization requires exposure_min > 0.")
        if parameter_range.gain_min <= 0:
            raise ValueError("Log normalization requires gain_min > 0.")

def normalize_camera_parameters(
    exposure: torch.Tensor,
    gain: torch.Tensor,
    parameter_range: CameraParameterRange,
    *,
    scale: Literal["linear", "log"] = "linear",
    output_range: Literal["zero_one", "minus_one_one"] = "zero_one",
    clip: bool = True,
) -> torch.Tensor:
    if exposure.shape != gain.shape:
        raise ValueError("exposure and gain must have the same shape.")

    _validate_camera_parameter_normalization(parameter_range, scale, output_range)
    exposure_norm = _normalize_camera_value(
        exposure,
        parameter_range.exposure_min,
        parameter_range.exposure_max,
        scale=scale,
        output_range=output_range,
        clip=clip,
    )
    gain_norm = _normalize_camera_value(
        gain,
        parameter_range.gain_min,
        parameter_range.gain_max,
        scale=scale,
        output_range=output_range,
        clip=clip,
    )
    return torch.stack([exposure_norm, gain_norm], dim=-1)