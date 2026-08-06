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
from hardware.utils import *

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
except ImportError:  # pragma: no cover - depends on target machine
    AlignFilter = Config = Pipeline = None  # type: ignore
    OBError = RuntimeError  # type: ignore


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

QUEUE_DRAIN_TIMEOUT_MS = 5
MAX_QUEUE_DRAIN_FRAMES = 16


@dataclass(frozen=True)
class FrameMetadata:
    color_frame_number: Optional[int]
    depth_frame_number: Optional[int]
    color_timestamp_us: Optional[int]
    depth_timestamp_us: Optional[int]


def _optional_int(frame: Any, method_name: str) -> Optional[int]:
    method = getattr(frame, method_name, None)
    if not callable(method):
        return None
    try:
        return int(method())
    except (AttributeError, OBError, RuntimeError, TypeError, ValueError):
        return None


def _timestamp_us(frame: Any) -> Optional[int]:
    timestamp = _optional_int(frame, "get_timestamp_us")
    if timestamp is not None:
        return timestamp
    timestamp_ms = _optional_int(frame, "get_timestamp")
    return timestamp_ms * 1000 if timestamp_ms is not None else None


def frame_metadata(frames: Any) -> Optional[FrameMetadata]:
    try:
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
    except (AttributeError, OBError, RuntimeError):
        return None
    if color is None or depth is None:
        return None
    return FrameMetadata(
        _optional_int(color, "get_frame_number"),
        _optional_int(depth, "get_frame_number"),
        _timestamp_us(color),
        _timestamp_us(depth),
    )


def _stream_is_fresh(
    previous_number: Optional[int],
    previous_timestamp_us: Optional[int],
    number: Optional[int],
    timestamp_us: Optional[int],
) -> bool:
    if number is None and timestamp_us is None:
        return False
    if previous_number is not None and (number is None or number <= previous_number):
        return False
    if (
        previous_timestamp_us is not None
        and (timestamp_us is None or timestamp_us <= previous_timestamp_us)
    ):
        return False
    return True


def metadata_is_fresh(previous: Optional[FrameMetadata], current: FrameMetadata) -> bool:
    if previous is None:
        previous = FrameMetadata(None, None, None, None)
    return _stream_is_fresh(
        previous.color_frame_number,
        previous.color_timestamp_us,
        current.color_frame_number,
        current.color_timestamp_us,
    ) and _stream_is_fresh(
        previous.depth_frame_number,
        previous.depth_timestamp_us,
        current.depth_frame_number,
        current.depth_timestamp_us,
    )


def _metadata_max(previous: Optional[FrameMetadata], current: FrameMetadata) -> FrameMetadata:
    if previous is None:
        return current

    def maximum(left: Optional[int], right: Optional[int]) -> Optional[int]:
        values = [value for value in (left, right) if value is not None]
        return max(values) if values else None

    return FrameMetadata(
        maximum(previous.color_frame_number, current.color_frame_number),
        maximum(previous.depth_frame_number, current.depth_frame_number),
        maximum(previous.color_timestamp_us, current.color_timestamp_us),
        maximum(previous.depth_timestamp_us, current.depth_timestamp_us),
    )

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
        if Pipeline is None:
            raise RuntimeError(
                "pyorbbecsdk is not installed. Install the official SDK v2 package "
                "(typically pyorbbecsdk2, imported as pyorbbecsdk)."
            )
        if exposure_value_per_ms <= 0:
            raise ValueError("exposure_value_per_ms must be positive.")
        if settle_frames < 0 or warmup_frames < 0:
            raise ValueError("Frame counts must be non-negative.")

        self.exposure_value_per_ms = exposure_value_per_ms
        self.settle_frames = settle_frames
        self.frame_timeout_ms = frame_timeout_ms
        self.frame_sequence: Optional[int] = None
        self.color_frame_number: Optional[int] = None
        self.depth_frame_number: Optional[int] = None
        self.color_timestamp_us: Optional[int] = None
        self.depth_timestamp_us: Optional[int] = None
        self.setting_effective = False
        self.sensor_settle_ms = 0.0
        self._last_metadata: Optional[FrameMetadata] = None
        self._capture_safe = True
        self._readback_matches = False
        self._settled_frames = 0
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
            frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
            metadata = frame_metadata(frames) if frames is not None else None
            if metadata is not None:
                self._last_metadata = _metadata_max(self._last_metadata, metadata)

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
        self.setting_effective = False
        self._settled_frames = 0
        self._capture_safe = self._drain_pending_frames()
        exposure_raw = self.exposure_to_raw(cell.exposure_ms)
        started = time.perf_counter()
        self.device.set_int_property(
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_raw
        )
        self.device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, cell.gain)

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

        self._readback_matches = (
            actual_exposure == exposure_raw and actual_gain == cell.gain
        )
        if self._capture_safe:
            # Close the small drain-to-command race before counting settling frames.
            self._capture_safe = self._drain_pending_frames()
        if self._capture_safe:
            for _ in range(self.settle_frames):
                _, metadata = self._wait_for_fresh_frameset()
                self._last_metadata = metadata
                self._settled_frames += 1
        self.sensor_settle_ms = (time.perf_counter() - started) * 1000.0

        return exposure_raw, actual_exposure, actual_gain

    def _drain_pending_frames(self) -> bool:
        for _ in range(MAX_QUEUE_DRAIN_FRAMES):
            frames = self.pipeline.wait_for_frames(QUEUE_DRAIN_TIMEOUT_MS)
            if frames is None:
                return True
            metadata = frame_metadata(frames)
            if metadata is not None:
                self._last_metadata = _metadata_max(self._last_metadata, metadata)
        return False

    def _wait_for_fresh_frameset(self) -> tuple[Any, FrameMetadata]:
        deadline = time.monotonic() + max(1.0, self.frame_timeout_ms / 1000.0 * 3)
        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
            if frames is None:
                continue
            metadata = frame_metadata(frames)
            if metadata is not None and metadata_is_fresh(self._last_metadata, metadata):
                return frames, metadata
        raise TimeoutError("Timed out waiting for a fresh RGB-D frameset.")

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
        if not self._capture_safe:
            raise RuntimeError(
                "Frame queue remained non-empty at the bounded drain limit; "
                "refusing an unprovably fresh capture."
            )
        deadline = time.monotonic() + max(1.0, self.frame_timeout_ms / 1000.0 * 3)
        while time.monotonic() < deadline:
            frames, metadata = self._wait_for_fresh_frameset()
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

            self._last_metadata = metadata
            self.color_frame_number = metadata.color_frame_number
            self.depth_frame_number = metadata.depth_frame_number
            self.color_timestamp_us = metadata.color_timestamp_us
            self.depth_timestamp_us = metadata.depth_timestamp_us
            self.frame_sequence = metadata.color_frame_number
            self.setting_effective = (
                self._readback_matches
                and self._capture_safe
                and self._settled_frames == self.settle_frames
                and metadata_is_fresh(None, metadata)
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
