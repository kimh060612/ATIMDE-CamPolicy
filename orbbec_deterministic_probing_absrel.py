#!/usr/bin/env python3
"""
Deterministic exposure/gain probing for Orbbec Gemini 336/336L.

Algorithm
---------
For each (motion_state, light_state) context:

1. Initialization
   - Capture every safe exposure/gain cell exactly once.
   - Four different cells are captured sequentially per probing cycle.
   - With the default 5 x 4 grid, initialization takes five cycles.

2. Tracking
   - Capture the current best cell every cycle.
   - If a challenger needs confirmation, capture it again.
   - Fill the remaining slots with deterministic round-robin exploration.
   - Update each cell score with an exponential moving average (EMA).
   - Promote a challenger only after it beats the current best by a margin
     for a configurable number of confirmations.

3. Context isolation
   - A cycle is committed only when motion/light remains unchanged for all
     captures in that cycle.

`compute_q_score()` is implemented with Depth Anything V2 and the aligned
Orbbec depth frame as ground truth. The returned q is AbsRel (lower is better).

Notes
-----
- This is sequential parameter switching, not a hardware-synchronous burst.
- Color auto exposure is disabled before manual exposure/gain control.
- Color and depth are captured as one frameset; depth is software-aligned to color.
- The default q is metric Depth Anything V2 AbsRel against aligned depth GT.
- Exposure values are converted from milliseconds to the device property
  using `--exposure-value-per-ms` (default: 1000, i.e. microseconds).
- By default, motion is classified from ROS 2 `/imu` and `/odom` using
  `motion_classifier.assign_motion_label()`; lighting is selected with
  `--lighting-state normal|dim|dark`.
"""

from __future__ import annotations

import argparse
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

from motion_classifier import assign_motion_label
from model.model_camerror import MODEL_IDS, CameraInducedErrorModel

try:
    from pyorbbecsdk import (  # type: ignore
        AlignFilter,
        Config,
        OBError,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBPermissionType,
        OBPropertyID,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )
except ImportError as exc:  # pragma: no cover - depends on target machine
    raise SystemExit(
        "pyorbbecsdk is not installed. For SDK v2, install the official "
        "package/build appropriate for your platform (the PyPI package is "
        "typically named pyorbbecsdk2 but imports as pyorbbecsdk)."
    ) from exc


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


class DepthAnythingAbsRelScorer:
    """GPU-accelerated Depth Anything V2 + AbsRel evaluator.

    The default checkpoint is the indoor metric Small model, which predicts
    metric depth and can be compared directly with the RGB-D ground truth.

    For a relative Depth Anything V2 checkpoint, select one of the alignment
    modes:
      - ``scale_shift_depth``: fit s * prediction + t to metric depth.
      - ``scale_shift_inverse``: fit s * prediction + t to inverse GT depth,
        then invert the aligned prediction.
    """

    VALID_MODES = {"metric", "scale_shift_depth", "scale_shift_inverse"}

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        precision: str,
        score_mode: str,
        min_depth_m: float,
        max_depth_m: float,
        min_valid_pixels: int,
        local_files_only: bool,
    ) -> None:
        if score_mode not in self.VALID_MODES:
            raise ValueError(
                f"score_mode must be one of {sorted(self.VALID_MODES)}, "
                f"got {score_mode!r}."
            )
        if precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be 'fp16' or 'fp32'.")
        if not 0 < min_depth_m < max_depth_m:
            raise ValueError("Require 0 < min_depth_m < max_depth_m.")
        if min_valid_pixels < 1:
            raise ValueError("min_valid_pixels must be positive.")

        try:
            import torch
            import torch.nn.functional as torch_f
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "Depth Anything scoring requires torch, transformers, and PIL. "
                "Install a JetPack-compatible PyTorch build and "
                "`pip install transformers>=4.45 pillow safetensors`."
            ) from exc

        self.torch = torch
        self.torch_f = torch_f
        self.model_id = model_id
        self.score_mode = score_mode
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.min_valid_pixels = int(min_valid_pixels)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device!r} was requested, but torch.cuda.is_available() is False."
            )
        self.device = torch.device(device)

        self.use_fp16 = precision == "fp16" and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True

        print(f"[DepthAnything] loading processor: {model_id}")
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )

        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        # Loading weights directly in fp16 reduces Jetson GPU memory usage.
        if self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        print(
            f"[DepthAnything] loading model: {model_id} "
            f"device={self.device} precision={'fp16' if self.use_fp16 else 'fp32'} "
            f"score_mode={score_mode}"
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()
        self.model.to(self.device)

    def _fit_scale_shift(
        self,
        prediction: Any,
        target: Any,
        mask: Any,
    ) -> tuple[Any, Any]:
        """Least-squares fit of s * prediction + t to target on valid pixels."""
        torch = self.torch
        p = prediction[mask].float()
        g = target[mask].float()
        n = torch.tensor(float(p.numel()), device=p.device, dtype=torch.float32)

        a00 = torch.sum(p * p)
        a01 = torch.sum(p)
        a11 = n
        b0 = torch.sum(p * g)
        b1 = torch.sum(g)
        determinant = a00 * a11 - a01 * a01

        if torch.abs(determinant).item() < 1e-12:
            raise ValueError("Scale-shift alignment is singular for this frame.")

        scale = (a11 * b0 - a01 * b1) / determinant
        shift = (-a01 * b0 + a00 * b1) / determinant
        return scale, shift

    def __call__(
        self,
        image_bgr: np.ndarray,
        ground_truth_depth_m: np.ndarray,
    ) -> QScore:
        torch = self.torch
        torch_f = self.torch_f

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got {image_bgr.shape}.")
        if ground_truth_depth_m.ndim != 2:
            raise ValueError(
                f"Expected HxW depth map, got {ground_truth_depth_m.shape}."
            )
        if image_bgr.shape[:2] != ground_truth_depth_m.shape:
            raise ValueError(
                "Color and aligned depth dimensions differ: "
                f"{image_bgr.shape[:2]} vs {ground_truth_depth_m.shape}."
            )

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )

        start = time.perf_counter()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.use_fp16,
                ):
                    outputs = self.model(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values=pixel_values)

            predicted = outputs.predicted_depth
            if predicted.ndim == 3:
                predicted = predicted.unsqueeze(1)
            predicted = torch_f.interpolate(
                predicted.float(),
                size=ground_truth_depth_m.shape,
                mode="bicubic",
                align_corners=False,
            )[0, 0]

            gt = torch.from_numpy(
                np.ascontiguousarray(ground_truth_depth_m, dtype=np.float32)
            ).to(self.device, non_blocking=True)

            valid = (
                torch.isfinite(gt)
                & (gt >= self.min_depth_m)
                & (gt <= self.max_depth_m)
                & torch.isfinite(predicted)
            )
            valid_count = int(valid.sum().item())
            if valid_count < self.min_valid_pixels:
                raise ValueError(
                    f"Only {valid_count} valid GT pixels; require at least "
                    f"{self.min_valid_pixels}."
                )

            scale_value = 1.0
            shift_value = 0.0
            if self.score_mode == "metric":
                aligned_depth = predicted
            elif self.score_mode == "scale_shift_depth":
                scale, shift = self._fit_scale_shift(predicted, gt, valid)
                aligned_depth = scale * predicted + shift
                scale_value = float(scale.item())
                shift_value = float(shift.item())
            else:
                inverse_gt = torch.reciprocal(torch.clamp(gt, min=1e-4))
                scale, shift = self._fit_scale_shift(predicted, inverse_gt, valid)
                aligned_inverse = scale * predicted + shift
                # Invalid/non-positive inverse depth cannot be converted to depth.
                valid = valid & torch.isfinite(aligned_inverse) & (aligned_inverse > 1e-6)
                valid_count = int(valid.sum().item())
                if valid_count < self.min_valid_pixels:
                    raise ValueError(
                        f"Only {valid_count} valid pixels remain after inverse-depth "
                        "alignment."
                    )
                aligned_depth = torch.reciprocal(torch.clamp(aligned_inverse, min=1e-6))
                scale_value = float(scale.item())
                shift_value = float(shift.item())

            valid = valid & torch.isfinite(aligned_depth) & (aligned_depth > 0)
            valid_count = int(valid.sum().item())
            if valid_count < self.min_valid_pixels:
                raise ValueError(
                    f"Only {valid_count} valid prediction/GT pixels remain."
                )

            abs_rel = torch.mean(
                torch.abs(aligned_depth[valid] - gt[valid])
                / torch.clamp(gt[valid], min=1e-6)
            )
            q = float(abs_rel.item())

        inference_ms = (time.perf_counter() - start) * 1000.0
        return QScore(
            q=q,
            uncertainty=None,
            mu=q,
            extra={
                "abs_rel": q,
                "valid_depth_pixels": float(valid_count),
                "valid_depth_ratio": float(valid_count / ground_truth_depth_m.size),
                "depth_alignment_scale": scale_value,
                "depth_alignment_shift": shift_value,
                "depth_inference_ms": float(inference_ms),
            },
        )


_Q_SCORER: Optional[DepthAnythingAbsRelScorer] = None


def configure_q_score_model(
    *,
    model_id: str,
    device: str,
    precision: str,
    score_mode: str,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_pixels: int,
    local_files_only: bool,
) -> None:
    """Load Depth Anything once before the probing loop."""
    global _Q_SCORER
    _Q_SCORER = DepthAnythingAbsRelScorer(
        model_id=model_id,
        device=device,
        precision=precision,
        score_mode=score_mode,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        min_valid_pixels=min_valid_pixels,
        local_files_only=local_files_only,
    )


def compute_q_score(
    image_bgr: np.ndarray,
    ground_truth_depth_m: np.ndarray,
    *,
    exposure_ms: int,
    gain: int,
    motion_state: int,
    light_state: int,
) -> Union[float, QScore]:
    """Return Depth Anything V2 AbsRel against the aligned RGB-D ground truth.

    ``exposure_ms``, ``gain``, ``motion_state``, and ``light_state`` are kept in
    the signature so this function remains interchangeable with the learned
    structural-risk scorer used by the final system.
    """
    del exposure_ms, gain, motion_state, light_state
    if _Q_SCORER is None:
        raise RuntimeError(
            "Q-score model is not configured. Call configure_q_score_model() first."
        )
    return _Q_SCORER(image_bgr, ground_truth_depth_m)


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


class ContextProvider:
    def get(self) -> ContextKey:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FixedContextProvider(ContextProvider):
    def __init__(self, context: ContextKey) -> None:
        context.validate()
        self._context = context

    def get(self) -> ContextKey:
        return self._context


class JsonContextProvider(ContextProvider):
    """Read motion/light from a JSON file that may be updated externally.

    Expected content:

        {"motion_state": 0, "light_state": 0}

    Invalid/transient partial writes fall back to the last valid context.
    External writers should still prefer atomic file replacement.
    """

    def __init__(self, path: Path, fallback: ContextKey) -> None:
        fallback.validate()
        self.path = path
        self.last_valid = fallback

    def get(self) -> ContextKey:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            context = ContextKey(
                motion_state=int(payload["motion_state"]),
                light_state=int(payload["light_state"]),
            )
            context.validate()
            self.last_valid = context
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        return self.last_valid


class RosMotionContextProvider(ContextProvider):
    """Classify motion from ROS 2 IMU and wheel-odometry topics.

    This is an embedded rclpy node, so the script does not need to be installed
    as a ROS 2 package.  The caller must still source a ROS 2 environment that
    provides rclpy, sensor_msgs, and nav_msgs.
    """

    def __init__(
        self,
        *,
        light_state: int,
        imu_topic: str,
        odom_topic: str,
        sensor_timeout_sec: float,
    ) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Imu
        except ImportError as exc:
            raise RuntimeError(
                "ROS motion input requires rclpy, sensor_msgs, and nav_msgs. "
                "Source the ROS 2 installation/workspace before running this script."
            ) from exc

        if sensor_timeout_sec <= 0:
            raise ValueError("sensor_timeout_sec must be positive.")
        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node("orbbec_motion_context")
        self._light_state = light_state
        self._sensor_timeout_sec = sensor_timeout_sec
        self._linear_speed: Optional[float] = None
        self._linear_acceleration: Optional[float] = None
        self._angular_speed: Optional[float] = None
        self._imu_received_at: Optional[float] = None
        self._odom_received_at: Optional[float] = None
        self._last_context = ContextKey(0, light_state)
        self._warned_waiting = False

        def on_imu(message: Any) -> None:
            # Use planar acceleration to avoid treating gravity on the IMU Z
            # axis as vehicle acceleration; yaw rate is rotation about Z.
            self._linear_acceleration = math.hypot(
                message.linear_acceleration.x,
                message.linear_acceleration.y,
            )
            self._angular_speed = abs(float(message.angular_velocity.z))
            self._imu_received_at = time.monotonic()

        def on_odom(message: Any) -> None:
            twist = message.twist.twist
            self._linear_speed = math.hypot(twist.linear.x, twist.linear.y)
            self._odom_received_at = time.monotonic()

        self._imu_subscription = self._node.create_subscription(
            Imu, imu_topic, on_imu, qos_profile_sensor_data
        )
        self._odom_subscription = self._node.create_subscription(
            Odometry, odom_topic, on_odom, qos_profile_sensor_data
        )
        print(
            f"[ROS] motion input imu={imu_topic} odom={odom_topic} "
            f"light_state={light_state}"
        )

    def get(self) -> ContextKey:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        now = time.monotonic()
        fresh = (
            self._imu_received_at is not None
            and self._odom_received_at is not None
            and now - self._imu_received_at <= self._sensor_timeout_sec
            and now - self._odom_received_at <= self._sensor_timeout_sec
        )
        if not fresh:
            if not self._warned_waiting:
                print("[WARNING] Waiting for fresh /imu and /odom data; using stop.")
                self._warned_waiting = True
            self._last_context = ContextKey(0, self._light_state)
            return self._last_context

        label = assign_motion_label(
            float(self._linear_speed),
            float(self._linear_acceleration),
            float(self._angular_speed),
        )
        self._last_context = ContextKey(
            MOTION_LABEL_TO_STATE[label], self._light_state
        )
        self._warned_waiting = False
        return self._last_context

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


@dataclass
class SafetyPolicy:
    max_exposure_ms_by_motion: tuple[int, ...] = (32, 32, 32, 32, 32)
    allowed_gains_by_light: tuple[tuple[int, ...], ...] = (
        GAIN_VALUES,
        GAIN_VALUES,
        GAIN_VALUES,
    )
    disabled_cells_by_context: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Optional[Path]) -> "SafetyPolicy":
        if path is None:
            return cls()

        payload = json.loads(path.read_text(encoding="utf-8"))

        max_exp_raw = payload.get(
            "max_exposure_ms_by_motion", [32] * MOTION_STATE_COUNT
        )
        if len(max_exp_raw) != MOTION_STATE_COUNT:
            raise ValueError(
                "max_exposure_ms_by_motion must contain exactly 5 values."
            )
        max_exp = tuple(int(value) for value in max_exp_raw)

        gains_raw = payload.get(
            "allowed_gains_by_light",
            [list(GAIN_VALUES) for _ in range(LIGHT_STATE_COUNT)],
        )
        if len(gains_raw) != LIGHT_STATE_COUNT:
            raise ValueError(
                "allowed_gains_by_light must contain exactly 3 lists."
            )
        allowed_gains = tuple(
            tuple(int(value) for value in gains) for gains in gains_raw
        )

        disabled_payload = payload.get("disabled_cells_by_context", {})
        disabled: dict[str, set[str]] = {}
        for context_key, entries in disabled_payload.items():
            values: set[str] = set()
            for entry in entries:
                if isinstance(entry, str):
                    cell_id = entry
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    cell_id = SensorCell(int(entry[0]), int(entry[1])).cell_id
                else:
                    raise ValueError(
                        f"Invalid disabled cell entry for {context_key}: {entry}"
                    )
                if cell_id not in CELL_BY_ID:
                    raise ValueError(f"Unknown cell id in safety config: {cell_id}")
                values.add(cell_id)
            disabled[str(context_key)] = values

        return cls(
            max_exposure_ms_by_motion=max_exp,
            allowed_gains_by_light=allowed_gains,
            disabled_cells_by_context=disabled,
        )

    def safe_cells(self, context: ContextKey) -> list[SensorCell]:
        context.validate()
        max_exposure = self.max_exposure_ms_by_motion[context.motion_state]
        allowed_gains = set(self.allowed_gains_by_light[context.light_state])
        disabled = self.disabled_cells_by_context.get(context.table_key, set())

        safe = [
            cell
            for cell in ALL_CELLS
            if cell.exposure_ms <= max_exposure
            and cell.gain in allowed_gains
            and cell.cell_id not in disabled
        ]
        if not safe:
            raise RuntimeError(f"No safe cells for context {context.table_key}.")
        return safe


class OrbbecColorCamera:
    def __init__(
        self,
        *,
        exposure_value_per_ms: float,
        settle_frames: int,
        frame_timeout_ms: int,
        warmup_frames: int,
        disable_awb: bool,
        strict_property_grid: bool,
    ) -> None:
        if exposure_value_per_ms <= 0:
            raise ValueError("exposure_value_per_ms must be positive.")
        if settle_frames < 0 or warmup_frames < 0:
            raise ValueError("Frame counts must be non-negative.")

        self.exposure_value_per_ms = exposure_value_per_ms
        self.settle_frames = settle_frames
        self.frame_timeout_ms = frame_timeout_ms
        self.pipeline = Pipeline()
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        config = Config()
        try:
            color_profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            try:
                color_profile = color_profiles.get_video_stream_profile(
                    0, 0, OBFormat.RGB, 0
                )
            except (AttributeError, OBError, RuntimeError):
                color_profile = color_profiles.get_default_video_stream_profile()
            config.enable_stream(color_profile)

            depth_profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            depth_profile = depth_profiles.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )
        except (AttributeError, OBError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to configure RGB-D streams: {exc}") from exc

        try:
            self.pipeline.enable_frame_sync()
        except (AttributeError, OBError, RuntimeError) as exc:
            print(f"[WARNING] Could not enable frame sync: {exc}", file=sys.stderr)

        self.pipeline.start(config)
        self.device = self.pipeline.get_device()

        self._warmup(warmup_frames)
        self._require_write_support(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL)
        self._require_write_support(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
        self._require_write_support(OBPropertyID.OB_PROP_COLOR_GAIN_INT)

        self.device.set_bool_property(
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False
        )

        if disable_awb and self._is_write_supported(
            OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL
        ):
            self.device.set_bool_property(
                OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, False
            )

        self.exposure_range = self._get_property_range(
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
        )
        self.gain_range = self._get_property_range(
            OBPropertyID.OB_PROP_COLOR_GAIN_INT
        )

        self._validate_grid(strict=strict_property_grid)

    def close(self) -> None:
        self.pipeline.stop()

    def _warmup(self, count: int) -> None:
        for _ in range(count):
            self.pipeline.wait_for_frames(self.frame_timeout_ms)

    def _is_write_supported(self, property_id: Any) -> bool:
        try:
            return bool(
                self.device.is_property_supported(
                    property_id, OBPermissionType.PERMISSION_WRITE
                )
            )
        except OBError:
            return False

    def _require_write_support(self, property_id: Any) -> None:
        if not self._is_write_supported(property_id):
            raise RuntimeError(f"Device does not support writing property {property_id}.")

    @staticmethod
    def _range_field(range_obj: Any, *names: str) -> Optional[int]:
        for name in names:
            if hasattr(range_obj, name):
                value = getattr(range_obj, name)
                value = value() if callable(value) else value
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _get_property_range(self, property_id: Any) -> Optional[dict[str, int]]:
        try:
            range_obj = self.device.get_int_property_range(property_id)
        except (AttributeError, OBError):
            return None

        result = {
            "min": self._range_field(range_obj, "min", "get_min"),
            "max": self._range_field(range_obj, "max", "get_max"),
            "step": self._range_field(range_obj, "step", "get_step"),
            "current": self._range_field(range_obj, "cur", "current", "get_current"),
            "default": self._range_field(range_obj, "def", "default", "get_default"),
        }
        return {key: value for key, value in result.items() if value is not None}

    def exposure_to_raw(self, exposure_ms: int) -> int:
        return int(round(exposure_ms * self.exposure_value_per_ms))

    @staticmethod
    def _validate_property_value(
        value: int,
        property_range: Optional[dict[str, int]],
        *,
        label: str,
        strict: bool,
    ) -> None:
        if not property_range:
            return

        minimum = property_range.get("min")
        maximum = property_range.get("max")
        step = property_range.get("step", 1)

        errors: list[str] = []
        if minimum is not None and value < minimum:
            errors.append(f"below min {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"above max {maximum}")
        if minimum is not None and step not in (None, 0, 1):
            if (value - minimum) % step != 0:
                errors.append(f"not aligned to step {step} from min {minimum}")

        if errors:
            message = f"{label} value {value} is " + ", ".join(errors)
            if strict:
                raise ValueError(message)
            print(f"[WARNING] {message}", file=sys.stderr)

    def _validate_grid(self, *, strict: bool) -> None:
        for exposure_ms in EXPOSURE_MS_VALUES:
            self._validate_property_value(
                self.exposure_to_raw(exposure_ms),
                self.exposure_range,
                label=f"exposure {exposure_ms} ms",
                strict=strict,
            )
        for gain in GAIN_VALUES:
            self._validate_property_value(
                gain,
                self.gain_range,
                label="gain",
                strict=strict,
            )

        print(f"[Camera] exposure range: {self.exposure_range}")
        print(f"[Camera] gain range: {self.gain_range}")

    def apply_cell(self, cell: SensorCell) -> tuple[int, Optional[int], Optional[int]]:
        exposure_raw = self.exposure_to_raw(cell.exposure_ms)
        self.device.set_int_property(
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_raw
        )
        self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, cell.gain)

        for _ in range(self.settle_frames):
            self.pipeline.wait_for_frames(self.frame_timeout_ms)

        actual_exposure: Optional[int] = None
        actual_gain: Optional[int] = None
        try:
            actual_exposure = int(
                self.device.get_int_property(
                    OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
                )
            )
        except (AttributeError, OBError, TypeError, ValueError):
            pass
        try:
            actual_gain = int(
                self.device.get_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT)
            )
        except (AttributeError, OBError, TypeError, ValueError):
            pass

        return exposure_raw, actual_exposure, actual_gain

    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture one synchronized RGB-D frameset and align depth to color.

        Returns
        -------
        image_bgr:
            HxWx3 uint8 OpenCV image.
        depth_m:
            HxW float32 depth in metres, aligned to the color view. Invalid
            Orbbec depth values remain zero.
        """
        deadline = time.monotonic() + max(1.0, self.frame_timeout_ms / 1000.0 * 3)
        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
            if frames is None:
                continue
            try:
                aligned = self.align_filter.process(frames)
            except (AttributeError, OBError, RuntimeError):
                aligned = None
            if aligned is None:
                continue

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            image = frame_to_bgr_image(color_frame)
            if image is None or image.size == 0:
                continue

            try:
                depth = np.frombuffer(
                    depth_frame.get_data(), dtype=np.uint16
                ).reshape(
                    int(depth_frame.get_height()),
                    int(depth_frame.get_width()),
                )
            except (TypeError, ValueError):
                continue

            # Orbbec's official examples multiply uint16 samples by
            # get_depth_scale() to obtain millimetres.
            depth_m = (
                depth.astype(np.float32)
                * float(depth_frame.get_depth_scale())
                / 1000.0
            )

            if depth_m.shape != image.shape[:2]:
                # This should not normally be needed after D2C alignment, but
                # nearest-neighbour preserves invalid zeros and metric samples.
                depth_m = cv2.resize(
                    depth_m,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            return image, np.ascontiguousarray(depth_m, dtype=np.float32)

        raise TimeoutError("Timed out waiting for a valid aligned RGB-D frame.")


def frame_to_bgr_image(frame: Any) -> Optional[np.ndarray]:
    """Convert common Orbbec color frame formats to OpenCV BGR."""

    width = int(frame.get_width())
    height = int(frame.get_height())
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data()).reshape(-1)

    if color_format == OBFormat.RGB:
        image = data.reshape(height, width, 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return data.reshape(height, width, 3).copy()
    if color_format == OBFormat.YUYV:
        image = data.reshape(height, width, 2)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if hasattr(OBFormat, "UYVY") and color_format == OBFormat.UYVY:
        image = data.reshape(height, width, 2)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.I420:
        image = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
    if hasattr(OBFormat, "NV12") and color_format == OBFormat.NV12:
        image = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12)
    if hasattr(OBFormat, "NV21") and color_format == OBFormat.NV21:
        image = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21)

    raise RuntimeError(f"Unsupported Orbbec color format: {color_format}")


def normalize_q_score(result: Union[float, QScore]) -> QScore:
    if isinstance(result, QScore):
        score = result
    elif isinstance(result, (float, int, np.floating, np.integer)):
        score = QScore(q=float(result))
    else:
        raise TypeError(
            "compute_q_score() must return float or QScore, "
            f"got {type(result).__name__}."
        )

    if not math.isfinite(score.q):
        raise ValueError(f"q must be finite, got {score.q}")
    if score.uncertainty is not None and not math.isfinite(score.uncertainty):
        raise ValueError(
            f"uncertainty must be finite or None, got {score.uncertainty}"
        )
    if score.mu is not None and not math.isfinite(score.mu):
        raise ValueError(f"mu must be finite or None, got {score.mu}")
    return score


class ExperimentLogger:
    FIELDNAMES = [
        "timestamp_ns",
        "cycle_index",
        "slot_index",
        "phase",
        "motion_state",
        "motion_label",
        "light_state",
        "lighting_label",
        "cell_id",
        "exposure_ms",
        "gain",
        "requested_exposure_raw",
        "actual_exposure_raw",
        "actual_gain",
        "camera_parameter_ms",
        "mde_inference_ms",
        "control_decision_delay_ms",
        "delay_warning",
        "delay_reasons",
        "q",
        "uncertainty",
        "mu",
        "ema_before",
        "ema_after",
        "count_after",
        "best_before",
        "best_after",
        "challenger_after",
        "challenger_streak_after",
        "cycle_committed",
        "discard_reason",
        "offload_requested",
        "image_path",
        "depth_path",
        "extra_json",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class DeterministicProbingController:
    def __init__(
        self,
        *,
        camera: OrbbecColorCamera,
        context_provider: ContextProvider,
        safety_policy: SafetyPolicy,
        output_dir: Path,
        ema_alpha: float,
        switch_margin: float,
        challenger_confirmations: int,
        cells_per_cycle: int,
        save_images: bool,
        save_depth: bool,
        rotate_capture_order: bool,
        offload_q_threshold: Optional[float],
        offload_uncertainty_threshold: Optional[float],
        offload_command: Optional[str],
        state_path: Path,
        logger: ExperimentLogger,
        camera_parameter_warn_ms: float,
        mde_inference_warn_ms: float,
        control_decision_warn_ms: float,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if switch_margin < 0:
            raise ValueError("switch_margin must be non-negative.")
        if challenger_confirmations < 1:
            raise ValueError("challenger_confirmations must be >= 1.")
        if cells_per_cycle < 1:
            raise ValueError("cells_per_cycle must be >= 1.")
        if min(
            camera_parameter_warn_ms,
            mde_inference_warn_ms,
            control_decision_warn_ms,
        ) <= 0:
            raise ValueError("Delay warning thresholds must be positive.")

        self.camera = camera
        self.context_provider = context_provider
        self.safety_policy = safety_policy
        self.output_dir = output_dir
        self.ema_alpha = ema_alpha
        self.switch_margin = switch_margin
        self.challenger_confirmations = challenger_confirmations
        self.cells_per_cycle = cells_per_cycle
        self.save_images = save_images
        self.save_depth = save_depth
        self.rotate_capture_order = rotate_capture_order
        self.offload_q_threshold = offload_q_threshold
        self.offload_uncertainty_threshold = offload_uncertainty_threshold
        self.offload_command = offload_command
        self.state_path = state_path
        self.logger = logger
        self.camera_parameter_warn_ms = camera_parameter_warn_ms
        self.mde_inference_warn_ms = mde_inference_warn_ms
        self.control_decision_warn_ms = control_decision_warn_ms
        self.context_states: dict[str, ContextState] = self._load_state()
        self.cycle_index = 0

    def _load_state(self) -> dict[str, ContextState]:
        if not self.state_path.exists():
            return {}

        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            states: dict[str, ContextState] = {}
            for key, raw_state in payload.get("contexts", {}).items():
                cells = {
                    cell.cell_id: CellStats() for cell in ALL_CELLS
                }
                for cell_id, raw_stats in raw_state.get("cells", {}).items():
                    if cell_id in cells:
                        cells[cell_id] = CellStats(**raw_stats)
                states[key] = ContextState(
                    cells=cells,
                    best_cell_id=raw_state.get("best_cell_id"),
                    challenger_cell_id=raw_state.get("challenger_cell_id"),
                    challenger_streak=int(raw_state.get("challenger_streak", 0)),
                    round_robin_pointer=int(raw_state.get("round_robin_pointer", 0)),
                    committed_cycles=int(raw_state.get("committed_cycles", 0)),
                )
            print(f"[State] loaded {len(states)} context tables from {self.state_path}")
            return states
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"[WARNING] Failed to load state file: {exc}", file=sys.stderr)
            return {}

    def _save_state(self) -> None:
        payload: dict[str, Any] = {"version": 1, "contexts": {}}
        for key, state in self.context_states.items():
            payload["contexts"][key] = {
                "cells": {
                    cell_id: asdict(stats)
                    for cell_id, stats in state.cells.items()
                },
                "best_cell_id": state.best_cell_id,
                "challenger_cell_id": state.challenger_cell_id,
                "challenger_streak": state.challenger_streak,
                "round_robin_pointer": state.round_robin_pointer,
                "committed_cycles": state.committed_cycles,
            }

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temp_path, self.state_path)

    def _state_for(self, context: ContextKey) -> ContextState:
        if context.table_key not in self.context_states:
            self.context_states[context.table_key] = ContextState()
        return self.context_states[context.table_key]

    def _next_round_robin_cells(
        self,
        state: ContextState,
        safe_cells: Sequence[SensorCell],
        *,
        count: int,
        excluded_ids: set[str],
    ) -> list[SensorCell]:
        if count <= 0:
            return []

        ordered = list(safe_cells)
        selected: list[SensorCell] = []
        attempts = 0
        max_attempts = max(1, len(ordered) * 3)

        while len(selected) < count and attempts < max_attempts:
            index = state.round_robin_pointer % len(ordered)
            state.round_robin_pointer = (state.round_robin_pointer + 1) % len(ordered)
            cell = ordered[index]
            attempts += 1
            if cell.cell_id in excluded_ids:
                continue
            if cell in selected:
                continue
            selected.append(cell)
            excluded_ids.add(cell.cell_id)

        return selected

    def _select_cells(
        self,
        state: ContextState,
        safe_cells: Sequence[SensorCell],
    ) -> tuple[str, list[SensorCell]]:
        unobserved = state.safe_unobserved(safe_cells)
        if unobserved:
            selected = unobserved[: self.cells_per_cycle]
            return "initialization", selected

        if state.best_cell_id not in {cell.cell_id for cell in safe_cells}:
            state.best_cell_id = state.best_from_scores(safe_cells).cell_id
            state.challenger_cell_id = None
            state.challenger_streak = 0

        assert state.best_cell_id is not None
        best = CELL_BY_ID[state.best_cell_id]
        selected = [best]
        excluded = {best.cell_id}

        # Confirm the current challenger in the immediately following cycle.
        if (
            state.challenger_cell_id is not None
            and state.challenger_streak > 0
            and state.challenger_cell_id not in excluded
            and state.challenger_cell_id in {cell.cell_id for cell in safe_cells}
            and len(selected) < self.cells_per_cycle
        ):
            challenger = CELL_BY_ID[state.challenger_cell_id]
            selected.append(challenger)
            excluded.add(challenger.cell_id)

        selected.extend(
            self._next_round_robin_cells(
                state,
                safe_cells,
                count=self.cells_per_cycle - len(selected),
                excluded_ids=excluded,
            )
        )

        if self.rotate_capture_order and len(selected) > 1:
            shift = state.committed_cycles % len(selected)
            selected = selected[shift:] + selected[:shift]

        return "tracking", selected

    def _save_image(
        self,
        image: np.ndarray,
        *,
        context: ContextKey,
        cell: SensorCell,
        slot_index: int,
    ) -> Optional[str]:
        if not self.save_images:
            return None

        directory = (
            self.output_dir
            / "images"
            / f"motion_{context.motion_state}"
            / f"light_{context.light_state}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"cycle_{self.cycle_index:08d}_slot_{slot_index}_"
            f"{cell.cell_id}_{time.time_ns()}.png"
        )
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to save image: {path}")
        return str(path)

    def _save_depth(
        self,
        depth_m: np.ndarray,
        *,
        context: ContextKey,
        cell: SensorCell,
        slot_index: int,
    ) -> Optional[str]:
        if not self.save_depth:
            return None

        directory = (
            self.output_dir
            / "depth_gt"
            / f"motion_{context.motion_state}"
            / f"light_{context.light_state}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"cycle_{self.cycle_index:08d}_slot_{slot_index}_"
            f"{cell.cell_id}_{time.time_ns()}.png"
        )
        depth_mm = np.clip(
            np.rint(depth_m * 1000.0), 0, np.iinfo(np.uint16).max
        ).astype(np.uint16)
        if not cv2.imwrite(
            str(path),
            depth_mm,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise OSError(f"Failed to save depth image: {path}")
        return str(path)

    def _capture_cell(
        self,
        cell: SensorCell,
        *,
        context: ContextKey,
        slot_index: int,
    ) -> RawCaptureObservation:
        """Capture and save RGB-D only; DNN inference happens after the cycle.

        Keeping inference out of this method minimizes the time gap between the
        four parameterized captures.
        """
        parameter_start = time.perf_counter()
        requested_raw, actual_raw, actual_gain = self.camera.apply_cell(cell)
        camera_parameter_ms = (time.perf_counter() - parameter_start) * 1000.0
        image, depth_gt_m = self.camera.capture_rgbd()
        capture_time_ns = time.time_ns()
        image_path = self._save_image(
            image, context=context, cell=cell, slot_index=slot_index
        )
        depth_path = self._save_depth(
            depth_gt_m, context=context, cell=cell, slot_index=slot_index
        )
        return RawCaptureObservation(
            cell=cell,
            image_bgr=image,
            ground_truth_depth_m=depth_gt_m,
            image_path=image_path,
            depth_path=depth_path,
            requested_exposure_raw=requested_raw,
            actual_exposure_raw=actual_raw,
            actual_gain=actual_gain,
            capture_time_ns=capture_time_ns,
            camera_parameter_ms=camera_parameter_ms,
        )

    @staticmethod
    def _score_capture(
        capture: RawCaptureObservation,
        *,
        context: ContextKey,
    ) -> CaptureObservation:
        score = normalize_q_score(
            compute_q_score(
                capture.image_bgr,
                capture.ground_truth_depth_m,
                exposure_ms=capture.cell.exposure_ms,
                gain=capture.cell.gain,
                motion_state=context.motion_state,
                light_state=context.light_state,
            )
        )
        return CaptureObservation(
            cell=capture.cell,
            score=score,
            image_bgr=capture.image_bgr,
            ground_truth_depth_m=capture.ground_truth_depth_m,
            image_path=capture.image_path,
            depth_path=capture.depth_path,
            requested_exposure_raw=capture.requested_exposure_raw,
            actual_exposure_raw=capture.actual_exposure_raw,
            actual_gain=capture.actual_gain,
            capture_time_ns=capture.capture_time_ns,
            camera_parameter_ms=capture.camera_parameter_ms,
        )

    def _update_challenger(
        self,
        state: ContextState,
        safe_cells: Sequence[SensorCell],
        observed_ids: set[str],
    ) -> None:
        if state.best_cell_id is None:
            state.best_cell_id = state.best_from_scores(safe_cells).cell_id
            state.challenger_cell_id = None
            state.challenger_streak = 0
            return

        best_stats = state.cells[state.best_cell_id]
        if best_stats.ema_score is None:
            state.best_cell_id = state.best_from_scores(safe_cells).cell_id
            best_stats = state.cells[state.best_cell_id]

        candidate_cells = [
            cell
            for cell in safe_cells
            if cell.cell_id != state.best_cell_id
            and cell.cell_id in observed_ids
            and state.cells[cell.cell_id].ema_score is not None
        ]
        if not candidate_cells:
            return

        candidate = min(
            candidate_cells,
            key=lambda cell: (
                float(state.cells[cell.cell_id].ema_score),
                ALL_CELLS.index(cell),
            ),
        )
        candidate_score = float(state.cells[candidate.cell_id].ema_score)
        best_score = float(best_stats.ema_score)

        if candidate_score < best_score - self.switch_margin:
            if state.challenger_cell_id == candidate.cell_id:
                state.challenger_streak += 1
            else:
                state.challenger_cell_id = candidate.cell_id
                state.challenger_streak = 1

            if state.challenger_streak >= self.challenger_confirmations:
                old_best = state.best_cell_id
                state.best_cell_id = candidate.cell_id
                state.challenger_cell_id = None
                state.challenger_streak = 0
                print(
                    f"[Switch] {old_best} -> {state.best_cell_id} "
                    f"({candidate_score:.6f} < {best_score:.6f} - "
                    f"{self.switch_margin:.6f})"
                )
        else:
            # Reset only when the pending challenger was actually re-observed.
            if (
                state.challenger_cell_id is not None
                and state.challenger_cell_id in observed_ids
            ):
                state.challenger_cell_id = None
                state.challenger_streak = 0

    def _should_offload(self, state: ContextState) -> tuple[bool, str]:
        if state.best_cell_id is None:
            return False, ""
        stats = state.cells[state.best_cell_id]

        if (
            self.offload_q_threshold is not None
            and stats.ema_score is not None
            and stats.ema_score > self.offload_q_threshold
        ):
            return True, "best_q_above_threshold"

        if (
            self.offload_uncertainty_threshold is not None
            and stats.last_uncertainty is not None
            and stats.last_uncertainty > self.offload_uncertainty_threshold
        ):
            return True, "best_uncertainty_above_threshold"

        return False, ""

    def _invoke_offload(
        self,
        *,
        context: ContextKey,
        state: ContextState,
        reason: str,
    ) -> None:
        print(
            f"[OFFLOAD] context={context.table_key}, "
            f"best={state.best_cell_id}, reason={reason}"
        )
        if not self.offload_command:
            return

        env = os.environ.copy()
        env.update(
            {
                "ATI_MOTION_STATE": str(context.motion_state),
                "ATI_LIGHT_STATE": str(context.light_state),
                "ATI_BEST_CELL": state.best_cell_id or "",
                "ATI_OFFLOAD_REASON": reason,
            }
        )
        subprocess.run(
            shlex.split(self.offload_command),
            env=env,
            check=False,
        )

    def run_cycle(self) -> None:
        context = self.context_provider.get()
        context.validate()
        state = self._state_for(context)
        safe_cells = self.safety_policy.safe_cells(context)

        if len(safe_cells) < self.cells_per_cycle:
            print(
                f"[WARNING] Context {context.table_key} has only "
                f"{len(safe_cells)} safe cells; capturing all of them."
            )

        phase, selected = self._select_cells(state, safe_cells)
        if not selected:
            raise RuntimeError("Cell selection returned an empty list.")

        print(
            f"\n[Cycle {self.cycle_index}] context={context.table_key} "
            f"phase={phase} selected={[cell.cell_id for cell in selected]} "
            f"best={state.best_cell_id} challenger={state.challenger_cell_id}"
        )

        best_before = state.best_cell_id
        raw_captures: list[RawCaptureObservation] = []
        discard_reason = ""

        for slot_index, cell in enumerate(selected):
            current_context = self.context_provider.get()
            if current_context != context:
                discard_reason = (
                    f"context_changed_before_slot_{slot_index}:"
                    f"{context.table_key}->{current_context.table_key}"
                )
                break

            capture = self._capture_cell(
                cell,
                context=context,
                slot_index=slot_index,
            )
            raw_captures.append(capture)
            print(
                f"  captured slot={slot_index} cell={cell.cell_id} "
                f"at={capture.capture_time_ns}"
            )

            current_context = self.context_provider.get()
            if current_context != context:
                discard_reason = (
                    f"context_changed_after_slot_{slot_index}:"
                    f"{context.table_key}->{current_context.table_key}"
                )
                break

        # Run Depth Anything only after all parameterized captures have been
        # acquired, so GPU inference does not widen the inter-capture gap.
        observations = [
            self._score_capture(capture, context=context)
            for capture in raw_captures
        ]
        for slot_index, observation in enumerate(observations):
            print(
                f"  scored slot={slot_index} cell={observation.cell.cell_id} "
                f"q={observation.score.q:.6f} "
                f"inference_ms={observation.score.extra.get('depth_inference_ms')}"
            )

        cycle_committed = (
            len(observations) == len(selected) and not discard_reason
        )
        ema_before: dict[str, Optional[float]] = {
            obs.cell.cell_id: state.cells[obs.cell.cell_id].ema_score
            for obs in observations
        }

        if cycle_committed:
            for obs in observations:
                state.cells[obs.cell.cell_id].update(
                    obs.score,
                    alpha=self.ema_alpha,
                    cycle_index=self.cycle_index,
                )

            if state.initialization_complete(safe_cells):
                if state.best_cell_id is None:
                    state.best_cell_id = state.best_from_scores(safe_cells).cell_id
                self._update_challenger(
                    state,
                    safe_cells,
                    observed_ids={obs.cell.cell_id for obs in observations},
                )

            state.committed_cycles += 1
            self._save_state()
        else:
            print(f"[Discard] {discard_reason}")

        offload_requested = False
        offload_reason = ""
        if cycle_committed and state.initialization_complete(safe_cells):
            offload_requested, offload_reason = self._should_offload(state)
            if offload_requested:
                self._invoke_offload(
                    context=context,
                    state=state,
                    reason=offload_reason,
                )

        best_after = state.best_cell_id
        decision_time_ns = time.time_ns()
        for slot_index, obs in enumerate(observations):
            stats = state.cells[obs.cell.cell_id]
            mde_inference_ms = float(
                obs.score.extra.get("depth_inference_ms", 0.0)
            )
            control_decision_delay_ms = max(
                0.0, (decision_time_ns - obs.capture_time_ns) / 1_000_000.0
            )
            delay_reasons: list[str] = []
            if obs.camera_parameter_ms > self.camera_parameter_warn_ms:
                delay_reasons.append("camera_parameter")
            if mde_inference_ms > self.mde_inference_warn_ms:
                delay_reasons.append("mde_inference")
            if control_decision_delay_ms > self.control_decision_warn_ms:
                delay_reasons.append("control_decision")
            if delay_reasons:
                print(
                    f"[LATENCY WARNING] cycle={self.cycle_index} slot={slot_index} "
                    f"reasons={','.join(delay_reasons)} "
                    f"camera_parameter_ms={obs.camera_parameter_ms:.2f} "
                    f"mde_inference_ms={mde_inference_ms:.2f} "
                    f"control_decision_delay_ms={control_decision_delay_ms:.2f}"
                )
            self.logger.write(
                {
                    "timestamp_ns": obs.capture_time_ns,
                    "cycle_index": self.cycle_index,
                    "slot_index": slot_index,
                    "phase": phase,
                    "motion_state": context.motion_state,
                    "motion_label": MOTION_STATE_TO_LABEL[context.motion_state],
                    "light_state": context.light_state,
                    "lighting_label": LIGHT_STATE_TO_LABEL[context.light_state],
                    "cell_id": obs.cell.cell_id,
                    "exposure_ms": obs.cell.exposure_ms,
                    "gain": obs.cell.gain,
                    "requested_exposure_raw": obs.requested_exposure_raw,
                    "actual_exposure_raw": obs.actual_exposure_raw,
                    "actual_gain": obs.actual_gain,
                    "camera_parameter_ms": obs.camera_parameter_ms,
                    "mde_inference_ms": mde_inference_ms,
                    "control_decision_delay_ms": control_decision_delay_ms,
                    "delay_warning": int(bool(delay_reasons)),
                    "delay_reasons": ",".join(delay_reasons),
                    "q": obs.score.q,
                    "uncertainty": obs.score.uncertainty,
                    "mu": obs.score.mu,
                    "ema_before": ema_before[obs.cell.cell_id],
                    "ema_after": stats.ema_score if cycle_committed else ema_before[obs.cell.cell_id],
                    "count_after": stats.count,
                    "best_before": best_before,
                    "best_after": best_after,
                    "challenger_after": state.challenger_cell_id,
                    "challenger_streak_after": state.challenger_streak,
                    "cycle_committed": int(cycle_committed),
                    "discard_reason": discard_reason,
                    "offload_requested": int(offload_requested),
                    "image_path": obs.image_path,
                    "depth_path": obs.depth_path,
                    "extra_json": json.dumps(obs.score.extra, sort_keys=True),
                }
            )

        # Leave the camera at the selected best operating cell between cycles.
        if cycle_committed and state.best_cell_id is not None:
            best_cell = CELL_BY_ID[state.best_cell_id]
            self.camera.apply_cell(best_cell)

        self.cycle_index += 1


def build_context_provider(args: argparse.Namespace) -> ContextProvider:
    light_state = LIGHT_LABEL_TO_STATE[args.lighting_state]
    fallback = ContextKey(args.motion_state, light_state)
    fallback.validate()
    if args.context_file is not None:
        return JsonContextProvider(Path(args.context_file), fallback)
    if args.motion_source == "ros":
        return RosMotionContextProvider(
            light_state=light_state,
            imu_topic=args.imu_topic,
            odom_topic=args.odom_topic,
            sensor_timeout_sec=args.ros_sensor_timeout_sec,
        )
    return FixedContextProvider(fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic best + round-robin probing on Orbbec color camera."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("probing_output"))
    parser.add_argument("--context-file", type=Path, default=None)
    parser.add_argument(
        "--motion-source",
        choices=("ros", "fixed"),
        default="ros",
        help="Classify /imu and /odom data, or use --motion-state.",
    )
    parser.add_argument("--motion-state", type=int, default=0)
    parser.add_argument(
        "--lighting-state",
        choices=tuple(LIGHT_LABEL_TO_STATE),
        default="normal",
        help="Manually selected lighting state.",
    )
    parser.add_argument("--imu-topic", type=str, default="/imu")
    parser.add_argument("--odom-topic", type=str, default="/odom")
    parser.add_argument("--ros-sensor-timeout-sec", type=float, default=1.0)
    parser.add_argument("--safety-config", type=Path, default=None)

    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--switch-margin", type=float, default=0.0)
    parser.add_argument("--challenger-confirmations", type=int, default=2)
    parser.add_argument("--cells-per-cycle", type=int, default=4)
    parser.add_argument("--cycle-interval-ms", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--rotate-capture-order", action="store_true")
    parser.add_argument(
        "--camera-parameter-warn-ms", type=float, default=50.0,
        help="Warn when applying exposure/gain exceeds this duration.",
    )
    parser.add_argument(
        "--mde-inference-warn-ms", type=float, default=100.0,
        help="Warn when one MDE inference exceeds this duration.",
    )
    parser.add_argument(
        "--control-decision-warn-ms", type=float, default=500.0,
        help="Warn when capture-to-control-decision latency exceeds this duration.",
    )

    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--exposure-value-per-ms",
        type=float,
        default=1000.0,
        help="Raw OB exposure property units per millisecond. Gemini 330-series "
        "is commonly used with 1000 (microseconds); verify on your firmware.",
    )
    parser.add_argument("--disable-awb", action="store_true")
    parser.add_argument(
        "--allow-unsupported-grid-values",
        action="store_true",
        help="Warn instead of failing when requested grid values disagree with "
        "the device-reported property range/step.",
    )

    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--no-save-depth", action="store_true")
    parser.add_argument(
        "--depth-model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face Depth Anything V2 checkpoint.",
    )
    parser.add_argument(
        "--depth-device",
        type=str,
        default="auto",
        help="Torch device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--depth-precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="FP16 is recommended on Jetson CUDA devices.",
    )
    parser.add_argument(
        "--depth-score-mode",
        choices=("metric", "scale_shift_depth", "scale_shift_inverse"),
        default="metric",
        help="Use metric for metric checkpoints. Relative checkpoints require "
        "one of the scale-shift modes before AbsRel is meaningful.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=1000)
    parser.add_argument(
        "--depth-model-local-files-only",
        action="store_true",
        help="Do not access Hugging Face; load an already cached model only.",
    )
    parser.add_argument("--offload-q-threshold", type=float, default=None)
    parser.add_argument(
        "--offload-uncertainty-threshold", type=float, default=None
    )
    parser.add_argument(
        "--offload-command",
        type=str,
        default=None,
        help="Optional command executed when offloading is requested. Context is "
        "provided through ATI_* environment variables.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete the saved controller state before starting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_path = args.output_dir / "controller_state.json"
    if args.reset_state and state_path.exists():
        state_path.unlink()

    context_provider = build_context_provider(args)
    safety_policy = SafetyPolicy.from_json(args.safety_config)
    logger = ExperimentLogger(args.output_dir / "probing_log.csv")

    camera: Optional[OrbbecColorCamera] = None
    try:
        configure_q_score_model(
            model_id=args.depth_model_id,
            device=args.depth_device,
            precision=args.depth_precision,
            score_mode=args.depth_score_mode,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_valid_pixels=args.min_valid_depth_pixels,
            local_files_only=args.depth_model_local_files_only,
        )

        camera = OrbbecColorCamera(
            exposure_value_per_ms=args.exposure_value_per_ms,
            settle_frames=args.settle_frames,
            frame_timeout_ms=args.frame_timeout_ms,
            warmup_frames=args.warmup_frames,
            disable_awb=args.disable_awb,
            strict_property_grid=not args.allow_unsupported_grid_values,
        )

        controller = DeterministicProbingController(
            camera=camera,
            context_provider=context_provider,
            safety_policy=safety_policy,
            output_dir=args.output_dir,
            ema_alpha=args.ema_alpha,
            switch_margin=args.switch_margin,
            challenger_confirmations=args.challenger_confirmations,
            cells_per_cycle=args.cells_per_cycle,
            save_images=not args.no_save_images,
            save_depth=not args.no_save_depth,
            rotate_capture_order=args.rotate_capture_order,
            offload_q_threshold=args.offload_q_threshold,
            offload_uncertainty_threshold=args.offload_uncertainty_threshold,
            offload_command=args.offload_command,
            state_path=state_path,
            logger=logger,
            camera_parameter_warn_ms=args.camera_parameter_warn_ms,
            mde_inference_warn_ms=args.mde_inference_warn_ms,
            control_decision_warn_ms=args.control_decision_warn_ms,
        )

        print("[Start] Press Ctrl-C to stop.")
        while args.max_cycles <= 0 or controller.cycle_index < args.max_cycles:
            cycle_start = time.monotonic()
            controller.run_cycle()

            if args.cycle_interval_ms > 0:
                elapsed = time.monotonic() - cycle_start
                remaining = args.cycle_interval_ms / 1000.0 - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")
    except NotImplementedError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2
    except (OBError, RuntimeError, ValueError, OSError, TimeoutError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()
        context_provider.close()
        if camera is not None:
            camera.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
