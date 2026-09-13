#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import orbbec_ae_control as source


WIDTH = 640
HEIGHT = 480
FPS = 15
SDKPipeline = source.Pipeline
LATENCY_CSV_FIELDS = (
    "record_type",
    "frame_index",
    "timestamp_ns",
    "target_width",
    "target_height",
    "target_fps",
    "actual_width",
    "actual_height",
    "color_timestamp_us",
    "depth_timestamp_us",
    "rgbd_timestamp_gap_us",
    "capture_ms",
    "mde_inference_ms",
    "mde_total_ms",
    "capture_cycle_ms",
    "output_interval_ms",
    "output_fps",
    "device_frame_interval_ms",
    "device_fps",
    "final_fps",
    "latency_from_final_fps_ms",
    "mde_error",
)


def _required_profile(profiles: Any, formats: tuple[Any, ...], label: str) -> Any:
    last_error: Exception | None = None
    for frame_format in formats:
        try:
            return profiles.get_video_stream_profile(WIDTH, HEIGHT, frame_format, FPS)
        except (AttributeError, source.OBError, RuntimeError) as error:
            last_error = error
    raise RuntimeError(
        f"Required {label} profile {WIDTH}x{HEIGHT}@{FPS} is unavailable."
    ) from last_error


class FixedRateProfiles:
    def __init__(self, profiles: Any, sensor_type: Any) -> None:
        self.profiles = profiles
        self.sensor_type = sensor_type

    def get_default_video_stream_profile(self) -> Any:
        if self.sensor_type == source.OBSensorType.COLOR_SENSOR:
            formats = (
                source.OBFormat.RGB,
                source.OBFormat.BGR,
                source.OBFormat.YUYV,
                source.OBFormat.MJPG,
            )
            label = "color"
        else:
            formats = (source.OBFormat.Y16,)
            label = "depth"
        return _required_profile(self.profiles, formats, label)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.profiles, name)


class FixedRatePipeline:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if SDKPipeline is None:
            raise RuntimeError("pyorbbecsdk is not installed.")
        self.pipeline = SDKPipeline(*args, **kwargs)

    def get_stream_profile_list(self, sensor_type: Any) -> FixedRateProfiles:
        return FixedRateProfiles(
            self.pipeline.get_stream_profile_list(sensor_type), sensor_type
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipeline, name)


class FixedRateOrbbecCamera(source.DefaultOrbbecCamera):
    measurements: list[dict[str, Any]] = []

    def __init__(
        self,
        *args: Any,
        settle_frames: int = 0,
        settle_timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        if settle_frames < 0:
            raise ValueError("settle_frames must be non-negative.")
        if settle_timeout is not None and settle_timeout < 0:
            raise ValueError("settle_timeout must be non-negative.")
        super().__init__(*args, **kwargs)
        type(self).measurements = []
        print(f"[Camera] color={WIDTH}x{HEIGHT}@{FPS} depth={WIDTH}x{HEIGHT}@{FPS}")

    def capture_rgbd(self):
        started = time.perf_counter()
        image, depth = super().capture_rgbd()
        capture_ms = (time.perf_counter() - started) * 1000.0
        if image.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError(
                f"Camera returned {image.shape[1]}x{image.shape[0]}, "
                f"expected {WIDTH}x{HEIGHT}."
            )
        type(self).measurements.append(
            {
                "capture_ms": capture_ms,
                "actual_width": image.shape[1],
                "actual_height": image.shape[0],
            }
        )
        return image, depth


class TimedDepthPredictor(source.DepthAnythingV2Small):
    measurements: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        type(self).measurements = []

    def infer(self, image_path: Path, output_path: Path) -> float:
        started = time.perf_counter()
        try:
            inference_ms = super().infer(image_path, output_path)
            type(self).measurements.append(
                {
                    "mde_inference_ms": inference_ms,
                    "mde_total_ms": (time.perf_counter() - started) * 1000.0,
                    "mde_error": "",
                }
            )
            return inference_ms
        except Exception as error:
            type(self).measurements.append(
                {
                    "mde_inference_ms": "",
                    "mde_total_ms": (time.perf_counter() - started) * 1000.0,
                    "mde_error": f"{type(error).__name__}: {error}",
                }
            )
            raise


def _optional_number(value: str, conversion):
    return conversion(value) if value not in ("", None) else None


def write_latency_report(
    output_dir: Path,
    source_report: Path,
    camera_measurements: list[dict[str, Any]],
    mde_measurements: list[dict[str, Any]],
) -> Path:
    with source_report.open(newline="", encoding="utf-8") as file:
        captures = [
            row for row in csv.DictReader(file) if row["record_type"] == "capture"
        ]

    rows: list[dict[str, Any]] = []
    previous_timestamp_ns: int | None = None
    previous_color_timestamp_us: int | None = None
    for index, capture in enumerate(captures):
        timestamp_ns = int(capture["timestamp_ns"])
        color_timestamp_us = _optional_number(capture["color_timestamp_us"], int)
        output_interval_ms = (
            (timestamp_ns - previous_timestamp_ns) / 1_000_000.0
            if previous_timestamp_ns is not None
            else None
        )
        device_interval_ms = (
            (color_timestamp_us - previous_color_timestamp_us) / 1000.0
            if color_timestamp_us is not None
            and previous_color_timestamp_us is not None
            else None
        )
        measured = (
            camera_measurements[index] if index < len(camera_measurements) else {}
        )
        mde = mde_measurements[index] if index < len(mde_measurements) else {}
        row = {field: "" for field in LATENCY_CSV_FIELDS}
        row.update(
            record_type="capture",
            frame_index=capture["frame_index"],
            timestamp_ns=timestamp_ns,
            target_width=WIDTH,
            target_height=HEIGHT,
            target_fps=FPS,
            actual_width=measured.get("actual_width", ""),
            actual_height=measured.get("actual_height", ""),
            color_timestamp_us=capture["color_timestamp_us"],
            depth_timestamp_us=capture["depth_timestamp_us"],
            rgbd_timestamp_gap_us=capture["rgbd_timestamp_gap_us"],
            capture_ms=measured.get("capture_ms", ""),
            mde_inference_ms=capture["mde_inference_ms"],
            mde_total_ms=mde.get("mde_total_ms", ""),
            capture_cycle_ms=capture["capture_cycle_ms"],
            output_interval_ms=output_interval_ms,
            output_fps=(
                1000.0 / output_interval_ms
                if output_interval_ms is not None and output_interval_ms > 0
                else ""
            ),
            device_frame_interval_ms=device_interval_ms,
            device_fps=(
                1000.0 / device_interval_ms
                if device_interval_ms is not None and device_interval_ms > 0
                else ""
            ),
            mde_error=capture["mde_error"] or mde.get("mde_error", ""),
        )
        rows.append(row)
        previous_timestamp_ns = timestamp_ns
        previous_color_timestamp_us = color_timestamp_us

    summary = {field: "" for field in LATENCY_CSV_FIELDS}
    summary.update(
        record_type="summary",
        frame_index=len(rows),
        target_width=WIDTH,
        target_height=HEIGHT,
        target_fps=FPS,
    )
    if len(rows) > 1:
        duration_s = (rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"]) / 1e9
        if duration_s > 0:
            final_fps = (len(rows) - 1) / duration_s
            summary.update(
                final_fps=final_fps,
                latency_from_final_fps_ms=1000.0 / final_fps,
            )

    path = output_dir / "ae_latency.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LATENCY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)
    os.replace(temporary, path)
    return path


def main() -> int:
    args = source._parse_args()
    source.Pipeline = FixedRatePipeline
    source.DefaultOrbbecCamera = FixedRateOrbbecCamera
    source.DepthAnythingV2Small = TimedDepthPredictor
    try:
        report, captured, evaluated = source._run(args)
        latency_report = write_latency_report(
            args.output_dir.resolve(),
            report,
            FixedRateOrbbecCamera.measurements,
            TimedDepthPredictor.measurements,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(
        f"[Done] captured={captured} evaluated={evaluated} "
        f"report={report} latency={latency_report}"
    )
    return 0 if captured == args.num_frames and evaluated == captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
