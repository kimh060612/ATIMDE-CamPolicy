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

import hardware.sensor as sensor
from orbbec_iqa_control import DepthAnythingV2Small, _evaluate_saved_prediction


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
    "actual_exposure_raw",
    "exposure_ms",
    "actual_gain",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orbbec built-in auto-exposure capture and MDE evaluation."
    )
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("ae_control_output"))
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--ae-settle-frames",
        type=int,
        default=30,
        help="RGB-D frames discarded after enabling color auto-exposure.",
    )
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument("--capture-interval-ms", type=float, default=0.0)
    parser.add_argument("--exposure-value-per-ms", type=float, default=10.0)
    parser.add_argument("--disable-awb", action="store_true")

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

    if args.num_frames < 1:
        parser.error("--num-frames must be positive.")
    if args.warmup_frames < 0 or args.ae_settle_frames < 0:
        parser.error("Camera frame counts must be non-negative.")
    if args.frame_timeout_ms < 1:
        parser.error("--frame-timeout-ms must be positive.")
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


def _enable_auto_exposure(camera: sensor.OrbbecColorCamera) -> None:
    property_id = sensor.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
    camera.device.set_bool_property(property_id, True)
    try:
        enabled = bool(camera.device.get_bool_property(property_id))
    except (AttributeError, sensor.OBError, RuntimeError, TypeError, ValueError):
        enabled = True  # Some devices expose write-only AE control.
    if not enabled:
        raise RuntimeError("The camera did not enable color auto-exposure.")


def _read_int_property(camera: sensor.OrbbecColorCamera, property_id: Any) -> int | None:
    try:
        return int(camera.device.get_int_property(property_id))
    except (AttributeError, sensor.OBError, RuntimeError, TypeError, ValueError):
        return None


def _save_capture(
    output_dir: Path,
    frame_index: int,
    image: np.ndarray,
    depth_m: np.ndarray,
    camera: sensor.OrbbecColorCamera,
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
    np.save(gt_depth_path, np.ascontiguousarray(depth_m, dtype=np.float32))

    color_timestamp = camera.color_timestamp_us
    depth_timestamp = camera.depth_timestamp_us
    gap = (
        abs(color_timestamp - depth_timestamp)
        if color_timestamp is not None and depth_timestamp is not None
        else ""
    )
    row = _empty_row()
    row.update(
        record_type="capture",
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        auto_exposure=1,
        actual_exposure_raw=exposure_raw if exposure_raw is not None else "",
        exposure_ms=(
            exposure_raw / camera.exposure_value_per_ms
            if exposure_raw is not None
            else ""
        ),
        actual_gain=gain if gain is not None else "",
        color_frame_number=camera.color_frame_number,
        depth_frame_number=camera.depth_frame_number,
        color_timestamp_us=color_timestamp,
        depth_timestamp_us=depth_timestamp,
        rgbd_timestamp_gap_us=gap,
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
    summary.update(
        record_type="summary",
        frame_index=len(rows),
        auto_exposure=1,
    )
    for field in (
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
    camera: sensor.OrbbecColorCamera | None = None
    try:
        camera = sensor.OrbbecColorCamera(
            exposure_value_per_ms=args.exposure_value_per_ms,
            settle_frames=0,
            frame_timeout_ms=args.frame_timeout_ms,
            warmup_frames=args.warmup_frames,
            disable_awb=args.disable_awb,
            strict_property_grid=False,
        )
        _enable_auto_exposure(camera)
        for _ in range(args.ae_settle_frames):
            camera.capture_rgbd()

        for frame_index in range(args.num_frames):
            cycle_started = time.perf_counter()
            image, depth_m = camera.capture_rgbd()
            exposure_raw = _read_int_property(
                camera, sensor.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
            )
            gain = _read_int_property(
                camera, sensor.OBPropertyID.OB_PROP_COLOR_GAIN_INT
            )
            row = _save_capture(
                output_dir, frame_index, image, depth_m, camera, exposure_raw, gain
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
            row["capture_cycle_ms"] = (time.perf_counter() - cycle_started) * 1000.0
            rows.append(row)
            result = (
                f"AbsRel={row['abs_rel']:.6f}"
                if row["abs_rel"] != ""
                else f"error={row['mde_error'] or row['evaluation_error']}"
            )
            print(
                f"[Frame] {frame_index + 1:03d}/{args.num_frames} "
                f"E={row['exposure_ms']}ms G={row['actual_gain']} {result}"
            )
            remaining = args.capture_interval_ms / 1000.0 - (
                time.perf_counter() - cycle_started
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
