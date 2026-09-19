#!/usr/bin/env python3
"""Capture with Orbbec color auto-exposure and evaluate synchronized MDE."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from orbbec_iqa_control import DepthAnythingV2Small, _evaluate_saved_prediction

try:
    from pyorbbecsdk import (  # type: ignore
        AlignFilter,
        Config,
        OBError,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBFrameMetadataType,
        OBPermissionType,
        OBPropertyID,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )
except ImportError:  # pragma: no cover - requires the camera machine
    AlignFilter = Config = Pipeline = None  # type: ignore
    OBFrameMetadataType = OBPermissionType = None  # type: ignore
    OBError = RuntimeError  # type: ignore


FIXED_COLOR_WHITE_BALANCE = 4600
FIXED_COLOR_BRIGHTNESS = 0
FIXED_COLOR_GAMMA = 300
FIXED_COLOR_SATURATION = 64
FIXED_COLOR_SHARPNESS = 50


METRIC_FIELDS = (
    "depth_alignment",
    "alignment_scale",
    "alignment_shift",
    "abs_rel",
    "rmse",
    "mae",
    "a1",
    "a2",
    "a3",
    "valid_depth_pixels",
)
CSV_FIELDS = (
    "record_type",
    "frame_index",
    "timestamp_ns",
    "auto_exposure",
    "auto_exposure_source",
    "actual_exposure_raw",
    "exposure_ms",
    "exposure_source",
    "actual_gain",
    "gain_source",
    "device_exposure_raw",
    "device_gain",
    "color_frame_number",
    "depth_frame_number",
    "color_timestamp_us",
    "depth_timestamp_us",
    "rgbd_timestamp_gap_us",
    "capture_cycle_ms",
    "image_path",
    "gt_depth_path",
    "raw_pred_depth_path",
    "pred_depth_path",
    "mde_inference_ms",
    *METRIC_FIELDS,
    "mde_error",
    "evaluation_error",
)


def _frame_value(frame: Any, method_name: str) -> int | None:
    try:
        return int(getattr(frame, method_name)())
    except (AttributeError, OBError, RuntimeError, TypeError, ValueError):
        return None


def _timestamp_us(frame: Any) -> int | None:
    value = _frame_value(frame, "get_timestamp_us")
    if value is not None:
        return value
    value = _frame_value(frame, "get_timestamp")
    return value * 1000 if value is not None else None


def _frame_to_bgr(frame: Any) -> np.ndarray:
    width, height = int(frame.get_width()), int(frame.get_height())
    frame_format = frame.get_format()
    data = np.asanyarray(frame.get_data()).reshape(-1)
    if frame_format == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if frame_format == OBFormat.BGR:
        return data.reshape(height, width, 3).copy()
    if frame_format == OBFormat.YUYV:
        return cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2
        )
    if frame_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            return image
    raise RuntimeError(f"Unsupported Orbbec color format: {frame_format}")


class DefaultOrbbecCamera:
    """RGB-D capture with color AE and per-frame camera metadata."""

    def __init__(
        self,
        *,
        frame_timeout_ms: int,
        settle_frames: int,
        settle_timeout: float,
        exposure_value_per_ms: float,
    ) -> None:
        if Pipeline is None:
            raise RuntimeError("pyorbbecsdk is not installed.")
        self.frame_timeout_ms = frame_timeout_ms
        self.exposure_value_per_ms = exposure_value_per_ms
        self.frame_auto_exposure: bool | None = None
        self.frame_exposure_raw: int | None = None
        self.frame_gain: int | None = None
        self.auto_exposure_source = ""
        self.exposure_source = ""
        self.gain_source = ""
        self.device_exposure_raw: int | None = None
        self.device_gain: int | None = None
        self.pipeline = Pipeline()
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        config = Config()
        color_profile = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_default_video_stream_profile()
        depth_profile = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_default_video_stream_profile()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )
        try:
            self.pipeline.enable_frame_sync()
        except (AttributeError, OBError, RuntimeError) as error:
            print(f"[WARNING] Could not enable frame sync: {error}", file=sys.stderr)
        self.pipeline.start(config)
        try:
            self.device = self.pipeline.get_device()
            self._configure_fixed_color_appearance()
            self._enable_auto_exposure_and_settle(
                settle_frames=settle_frames,
                settle_timeout=settle_timeout,
            )
        except BaseException:
            try:
                self.pipeline.stop()
            except (AttributeError, OBError, RuntimeError):
                pass
            raise

    def close(self) -> None:
        self.pipeline.stop()

    def _set_fixed_bool_property(
        self,
        property_id: Any,
        value: bool,
        *,
        label: str,
        required: bool,
    ) -> bool:
        try:
            writable = bool(
                self.device.is_property_supported(
                    property_id, OBPermissionType.PERMISSION_WRITE
                )
            )
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError) as error:
            if required:
                raise RuntimeError(f"Could not query {label} support: {error}") from error
            print(f"[WARNING] Could not query {label} support: {error}", file=sys.stderr)
            return False
        if not writable:
            if required:
                raise RuntimeError(f"Device does not support writing {label}.")
            print(f"[WARNING] Device does not support writing {label}.", file=sys.stderr)
            return False

        try:
            self.device.set_bool_property(property_id, value)
            actual = bool(self.device.get_bool_property(property_id))
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Could not configure {label}: {error}") from error
        if actual != value:
            raise RuntimeError(
                f"{label} readback mismatch: requested={value}, actual={actual}."
            )
        print(f"[Camera] {label}={actual}")
        return True

    def _set_fixed_int_property(
        self,
        property_id: Any,
        value: int,
        *,
        label: str,
        required: bool,
    ) -> bool:
        try:
            writable = bool(
                self.device.is_property_supported(
                    property_id, OBPermissionType.PERMISSION_WRITE
                )
            )
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError) as error:
            if required:
                raise RuntimeError(f"Could not query {label} support: {error}") from error
            print(f"[WARNING] Could not query {label} support: {error}", file=sys.stderr)
            return False
        if not writable:
            if required:
                raise RuntimeError(f"Device does not support writing {label}.")
            print(f"[WARNING] Device does not support writing {label}.", file=sys.stderr)
            return False

        try:
            self.device.set_int_property(property_id, value)
            actual = int(self.device.get_int_property(property_id))
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Could not configure {label}: {error}") from error
        if actual != value:
            raise RuntimeError(
                f"{label} readback mismatch: requested={value}, actual={actual}."
            )
        print(f"[Camera] {label}={actual}")
        return True

    def _configure_fixed_color_appearance(self) -> None:
        """Remove automatic/non-deterministic color appearance changes."""

        self._set_fixed_bool_property(
            OBPropertyID.OB_PROP_COLOR_HDR_BOOL,
            False,
            label="color_hdr",
            required=False,
        )
        self._set_fixed_bool_property(
            OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
            False,
            label="color_auto_white_balance",
            required=True,
        )
        self._set_fixed_int_property(
            OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT,
            FIXED_COLOR_WHITE_BALANCE,
            label="color_white_balance",
            required=True,
        )
        for property_id, value, label in (
            (
                OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT,
                FIXED_COLOR_BRIGHTNESS,
                "color_brightness",
            ),
            (
                OBPropertyID.OB_PROP_COLOR_GAMMA_INT,
                FIXED_COLOR_GAMMA,
                "color_gamma",
            ),
            (
                OBPropertyID.OB_PROP_COLOR_SATURATION_INT,
                FIXED_COLOR_SATURATION,
                "color_saturation",
            ),
            (
                OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT,
                FIXED_COLOR_SHARPNESS,
                "color_sharpness",
            ),
        ):
            self._set_fixed_int_property(
                property_id,
                value,
                label=label,
                required=False,
            )

    def _enable_auto_exposure_and_settle(
        self, *, settle_frames: int, settle_timeout: float
    ) -> None:
        property_id = OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
        try:
            writable = bool(
                self.device.is_property_supported(
                    property_id, OBPermissionType.PERMISSION_WRITE
                )
            )
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Could not query color auto-exposure support: {error}"
            ) from error
        if not writable:
            raise RuntimeError("Device does not support enabling color auto-exposure.")

        self.device.set_bool_property(property_id, True)
        try:
            enabled = bool(self.device.get_bool_property(property_id))
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Could not verify color auto-exposure: {error}") from error
        if not enabled:
            raise RuntimeError("The camera did not enable color auto-exposure.")

        deadline = time.monotonic() + settle_timeout
        settled = 0
        while settled < settle_frames:
            remaining_ms = int(math.ceil((deadline - time.monotonic()) * 1000.0))
            if remaining_ms <= 0:
                raise TimeoutError(
                    f"Timed out after {settled}/{settle_frames} AE settling frames."
                )
            frames = self.pipeline.wait_for_frames(remaining_ms)
            if frames is not None:
                settled += 1

    @staticmethod
    def _metadata_value(frame: Any, metadata_type: Any) -> int | None:
        try:
            if frame is None or not frame.has_metadata(metadata_type):
                return None
            return int(frame.get_metadata_value(metadata_type))
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError):
            return None

    def _record_frame_settings(self, color_frame: Any) -> None:
        auto_exposure = self._metadata_value(
            color_frame, OBFrameMetadataType.AUTO_EXPOSURE
        )
        self.frame_auto_exposure = (
            bool(auto_exposure) if auto_exposure is not None else None
        )
        self.frame_exposure_raw = self._metadata_value(
            color_frame, OBFrameMetadataType.EXPOSURE
        )
        self.frame_gain = self._metadata_value(
            color_frame, OBFrameMetadataType.GAIN
        )

    def read_settings(self) -> tuple[bool | None, int | None, int | None]:
        try:
            device_auto_exposure = bool(
                self.device.get_bool_property(
                    OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
                )
            )
        except (AttributeError, OBError, RuntimeError, TypeError, ValueError):
            device_auto_exposure = None

        def read(property_id: Any) -> int | None:
            try:
                return int(self.device.get_int_property(property_id))
            except (AttributeError, OBError, RuntimeError, TypeError, ValueError):
                return None

        self.device_exposure_raw = read(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)
        self.device_gain = read(OBPropertyID.OB_PROP_COLOR_GAIN_INT)
        auto_exposure = (
            self.frame_auto_exposure
            if self.frame_auto_exposure is not None
            else device_auto_exposure
        )
        exposure_raw = (
            self.frame_exposure_raw
            if self.frame_exposure_raw is not None
            else self.device_exposure_raw
        )
        gain = self.frame_gain if self.frame_gain is not None else self.device_gain
        self.auto_exposure_source = (
            "frame_metadata"
            if self.frame_auto_exposure is not None
            else "device_property" if device_auto_exposure is not None else ""
        )
        self.exposure_source = (
            "frame_metadata"
            if self.frame_exposure_raw is not None
            else "device_property" if self.device_exposure_raw is not None else ""
        )
        self.gain_source = (
            "frame_metadata"
            if self.frame_gain is not None
            else "device_property" if self.device_gain is not None else ""
        )
        return auto_exposure, exposure_raw, gain

    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + max(1.0, self.frame_timeout_ms / 1000.0 * 3)
        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
            source_color_frame = (
                frames.get_color_frame() if frames is not None else None
            )
            aligned = self.align_filter.process(frames) if frames is not None else None
            if aligned is None:
                continue
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            metadata_frame = (
                source_color_frame
                if source_color_frame is not None
                else color_frame
            )
            self._record_frame_settings(metadata_frame)
            image = _frame_to_bgr(color_frame)
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                int(depth_frame.get_height()), int(depth_frame.get_width())
            )
            depth_m = (
                depth.astype(np.float32) * float(depth_frame.get_depth_scale()) / 1000.0
            )
            if depth_m.shape != image.shape[:2]:
                depth_m = cv2.resize(
                    depth_m,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            self.color_frame_number = _frame_value(color_frame, "get_frame_number")
            self.depth_frame_number = _frame_value(depth_frame, "get_frame_number")
            self.color_timestamp_us = _timestamp_us(color_frame)
            self.depth_timestamp_us = _timestamp_us(depth_frame)
            return image, np.ascontiguousarray(depth_m, dtype=np.float32)
        raise TimeoutError("Timed out waiting for an aligned RGB-D frame.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orbbec color auto-exposure capture and MDE evaluation."
    )
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("ae_control_output"))
    parser.add_argument(
        "--camera-settle-frames",
        "--warmup-frames",
        dest="camera_settle_frames",
        type=int,
        default=30,
        help="Frames discarded after explicitly enabling color auto-exposure.",
    )
    parser.add_argument(
        "--camera-settle-timeout",
        type=float,
        default=10.0,
        help="Total timeout in seconds for AE settling frames.",
    )
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument("--capture-interval-ms", type=float, default=0.0)
    parser.add_argument("--exposure-value-per-ms", type=float, default=10.0)
    parser.add_argument("--depth-device", default="cuda")
    parser.add_argument("--depth-precision", choices=("fp16", "fp32"), default="fp32")
    parser.add_argument(
        "--depth-alignment",
        choices=("scale_shift_depth", "scale_shift_inverse"),
        default="scale_shift_inverse",
    )
    parser.add_argument("--min-depth-m", type=float, default=1e-3)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=10000)
    parser.add_argument("--depth-model-local-files-only", action="store_true")
    args = parser.parse_args()
    if args.num_frames < 1 or args.frame_timeout_ms < 1:
        parser.error("Frame count and timeout must be positive.")
    if args.camera_settle_frames < 0:
        parser.error("--camera-settle-frames must be non-negative.")
    if (
        not math.isfinite(args.camera_settle_timeout)
        or args.camera_settle_timeout <= 0
    ):
        parser.error("--camera-settle-timeout must be finite and positive.")
    if not math.isfinite(args.capture_interval_ms) or args.capture_interval_ms < 0:
        parser.error("--capture-interval-ms must be finite and non-negative.")
    if (
        not math.isfinite(args.exposure_value_per_ms)
        or args.exposure_value_per_ms <= 0
    ):
        parser.error("--exposure-value-per-ms must be finite and positive.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        parser.error("Require 0 < --min-depth-m < --max-depth-m.")
    if args.min_valid_depth_pixels < 1:
        parser.error("--min-valid-depth-pixels must be positive.")
    return args


def _empty_row() -> dict[str, Any]:
    return {field: "" for field in CSV_FIELDS}


def _save_capture(
    output_dir: Path,
    frame_index: int,
    image: np.ndarray,
    depth_m: np.ndarray,
    camera: DefaultOrbbecCamera,
    auto_exposure: bool | None,
    exposure_raw: int | None,
    gain: int | None,
) -> dict[str, Any]:
    timestamp_ns = time.time_ns()
    stem = f"frame_{frame_index:05d}_{timestamp_ns}"
    image_path = output_dir / "images" / f"{stem}.png"
    gt_depth_path = output_dir / "depth_gt" / f"{stem}.npy"
    raw_pred_depth_path = output_dir / "depth_pred_raw" / f"{stem}.npy"
    pred_depth_path = output_dir / "depth_pred" / f"{stem}.npy"
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"Failed to save RGB image: {image_path}")
    np.save(gt_depth_path, depth_m)
    color_timestamp = camera.color_timestamp_us
    depth_timestamp = camera.depth_timestamp_us
    row = _empty_row()
    row.update(
        record_type="capture",
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        auto_exposure=int(auto_exposure) if auto_exposure is not None else "",
        auto_exposure_source=getattr(camera, "auto_exposure_source", ""),
        actual_exposure_raw=exposure_raw if exposure_raw is not None else "",
        exposure_ms=(
            exposure_raw / camera.exposure_value_per_ms
            if exposure_raw is not None
            else ""
        ),
        exposure_source=getattr(camera, "exposure_source", ""),
        actual_gain=gain if gain is not None else "",
        gain_source=getattr(camera, "gain_source", ""),
        device_exposure_raw=(
            getattr(camera, "device_exposure_raw", None)
            if getattr(camera, "device_exposure_raw", None) is not None
            else ""
        ),
        device_gain=(
            getattr(camera, "device_gain", None)
            if getattr(camera, "device_gain", None) is not None
            else ""
        ),
        color_frame_number=camera.color_frame_number,
        depth_frame_number=camera.depth_frame_number,
        color_timestamp_us=color_timestamp,
        depth_timestamp_us=depth_timestamp,
        rgbd_timestamp_gap_us=(
            abs(color_timestamp - depth_timestamp)
            if color_timestamp is not None and depth_timestamp is not None
            else ""
        ),
        image_path=str(image_path),
        gt_depth_path=str(gt_depth_path),
        raw_pred_depth_path=str(raw_pred_depth_path),
        pred_depth_path=str(pred_depth_path),
    )
    return row


def _write_report(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    report_path = output_dir / "ae_control_report.csv"
    temporary_path = report_path.with_suffix(".csv.tmp")
    evaluated = [row for row in rows if row["abs_rel"] != ""]
    summary = _empty_row()
    summary.update(record_type="summary", frame_index=len(rows))
    for field in (
        "auto_exposure",
        "actual_exposure_raw",
        "exposure_ms",
        "actual_gain",
        "rgbd_timestamp_gap_us",
        "capture_cycle_ms",
        "mde_inference_ms",
        "abs_rel",
        "rmse",
        "mae",
        "a1",
        "a2",
        "a3",
    ):
        source = evaluated if field in METRIC_FIELDS else rows
        values = [float(row[field]) for row in source if row[field] != ""]
        summary[field] = float(np.mean(values)) if values else ""
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)
    os.replace(temporary_path, report_path)
    return report_path


def _run(args: argparse.Namespace) -> tuple[Path, int, int]:
    output_dir = args.output_dir.resolve()
    for name in ("images", "depth_gt", "depth_pred_raw", "depth_pred"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    predictor = DepthAnythingV2Small(args)
    rows: list[dict[str, Any]] = []
    run_error: BaseException | None = None
    camera: DefaultOrbbecCamera | None = None
    try:
        frame_timeout_ms = getattr(
            args,
            "frame_timeout_ms",
            max(1, int(round(args.camera_settle_timeout * 1000.0))),
        )
        camera = DefaultOrbbecCamera(
            frame_timeout_ms=frame_timeout_ms,
            settle_frames=args.camera_settle_frames,
            settle_timeout=args.camera_settle_timeout,
            exposure_value_per_ms=args.exposure_value_per_ms,
        )
        for frame_index in range(args.num_frames):
            started = time.perf_counter()
            image, depth_m = camera.capture_rgbd()
            auto_exposure, exposure_raw, gain = camera.read_settings()
            row = _save_capture(
                output_dir,
                frame_index,
                image,
                depth_m,
                camera,
                auto_exposure,
                exposure_raw,
                gain,
            )
            try:
                row["mde_inference_ms"] = predictor.infer(
                    Path(row["image_path"]), Path(row["raw_pred_depth_path"])
                )
            except (OSError, RuntimeError, ValueError) as error:
                row["mde_error"] = str(error)
            if not row["mde_error"]:
                try:
                    row.update(_evaluate_saved_prediction(row, args))
                except (OSError, RuntimeError, ValueError) as error:
                    row["evaluation_error"] = str(error)
            row["capture_cycle_ms"] = (time.perf_counter() - started) * 1000.0
            rows.append(row)
            result = (
                f"AbsRel={row['abs_rel']:.6f}"
                if row["abs_rel"] != ""
                else f"error={row['mde_error'] or row['evaluation_error']}"
            )
            print(
                f"[Frame] {frame_index + 1:03d}/{args.num_frames} "
                f"AE={row['auto_exposure']} E={row['exposure_ms']}ms "
                f"G={row['actual_gain']} source={row['exposure_source']} {result}"
            )
            remaining = args.capture_interval_ms / 1000.0 - (
                time.perf_counter() - started
            )
            if remaining > 0:
                time.sleep(remaining)
    except (KeyboardInterrupt, OSError, RuntimeError, TimeoutError, ValueError) as error:
        run_error = error
    finally:
        if camera is not None:
            camera.close()
    report = _write_report(output_dir, rows)
    evaluated_count = sum(row["abs_rel"] != "" for row in rows)
    if run_error is not None:
        print(
            f"[WARNING] Capture stopped after {len(rows)} frames: {run_error}",
            file=sys.stderr,
        )
    return report, len(rows), evaluated_count


def main() -> int:
    try:
        args = _parse_args()
        report, captured, evaluated = _run(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[Done] captured={captured} evaluated={evaluated} report={report}")
    return 0 if captured == args.num_frames and evaluated == captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
