#!/usr/bin/env python3
"""Capture fixed-camera-parameter RGB-D frames and evaluate MDE accuracy."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from hardware.sensor import OrbbecColorCamera
from hardware.utils import ContextKey, SensorCell
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
    "exposure_ms",
    "gain",
    "requested_exposure_raw",
    "actual_exposure_raw",
    "actual_gain",
    "setting_effective",
    "image_path",
    "gt_depth_path",
    "raw_pred_depth_path",
    "pred_depth_path",
    "mde_inference_ms",
    *METRIC_FIELDS,
    "mde_error",
    "evaluation_error",
)


class _FixedContext:
    def __init__(self) -> None:
        self.context = ContextKey(0, 0)

    def get(self) -> ContextKey:
        return self.context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MDE on RGB-D frames captured at fixed exposure and gain."
    )
    parser.add_argument(
        "--exposure-ms",
        "--exposure-time",
        dest="exposure_ms",
        type=int,
        default=32,
        help="Fixed color exposure time in milliseconds (default: 32).",
    )
    parser.add_argument("--gain", type=int, default=25, help="Fixed color gain (default: 25).")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("fixed_camparam_output"))

    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument("--exposure-value-per-ms", type=float, default=10.0)
    parser.add_argument("--disable-awb", action="store_true")
    parser.add_argument("--allow-unsupported-grid-values", action="store_true")

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

    if args.exposure_ms <= 0:
        parser.error("--exposure-ms must be positive.")
    if args.gain < 0:
        parser.error("--gain must be non-negative.")
    if args.num_frames < 1:
        parser.error("--num-frames must be positive.")
    if args.settle_frames < 0 or args.warmup_frames < 0:
        parser.error("Camera frame counts must be non-negative.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        parser.error("Require 0 < --min-depth-m < --max-depth-m.")
    return args


def _empty_row() -> dict[str, Any]:
    return {field: "" for field in CSV_FIELDS}


def _save_capture(output_dir: Path, frame: Any, cell: SensorCell) -> dict[str, Any]:
    stem = f"frame_{frame.capture_index:05d}_{frame.timestamp_ns}"
    image_path = output_dir / "images" / f"{stem}.png"
    gt_depth_path = output_dir / "depth_gt" / f"{stem}.npy"
    raw_pred_depth_path = output_dir / "depth_pred_raw" / f"{stem}.npy"
    pred_depth_path = output_dir / "depth_pred" / f"{stem}.npy"

    if not cv2.imwrite(str(image_path), frame.image):
        raise OSError(f"Failed to save RGB image: {image_path}")
    np.save(gt_depth_path, np.ascontiguousarray(frame.depth_m, dtype=np.float32))

    row = _empty_row()
    row.update(
        record_type="capture",
        frame_index=frame.capture_index,
        timestamp_ns=frame.timestamp_ns,
        exposure_ms=cell.exposure_ms,
        gain=cell.gain,
        requested_exposure_raw=frame.requested_exposure_raw,
        actual_exposure_raw=frame.actual_exposure_raw,
        actual_gain=frame.actual_gain,
        setting_effective=int(frame.setting_effective),
        image_path=str(image_path),
        gt_depth_path=str(gt_depth_path),
        raw_pred_depth_path=str(raw_pred_depth_path),
        pred_depth_path=str(pred_depth_path),
    )
    return row


def _write_report(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    report_path = output_dir / "fixed_camparam_report.csv"
    temporary_path = report_path.with_suffix(".csv.tmp")
    evaluated = [row for row in rows if row["abs_rel"] != ""]
    summary = _empty_row()
    summary.update(
        record_type="summary",
        frame_index=len(rows),
        exposure_ms=rows[0]["exposure_ms"] if rows else "",
        gain=rows[0]["gain"] if rows else "",
    )
    for field in ("abs_rel", "rmse", "mae", "a1", "a2", "a3"):
        values = [float(row[field]) for row in evaluated]
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

    cell = SensorCell(args.exposure_ms, args.gain)
    context = _FixedContext()
    rows: list[dict[str, Any]] = []
    capture_error: BaseException | None = None
    camera: OrbbecColorCamera | None = None

    try:
        camera = OrbbecColorCamera(
            exposure_value_per_ms=args.exposure_value_per_ms,
            settle_frames=args.settle_frames,
            frame_timeout_ms=args.frame_timeout_ms,
            warmup_frames=args.warmup_frames,
            disable_awb=args.disable_awb,
            strict_property_grid=not args.allow_unsupported_grid_values,
        )
        runner = CaptureRunner(camera, context, max_pair_gap_ms=100.0)
        for frame_index in range(args.num_frames):
            frame = runner.capture(cell, context.get(), "fixed", frame_index)
            if not frame.setting_effective:
                raise RuntimeError(
                    f"Camera setting was not verified for frame {frame_index}: {cell}"
                )
            rows.append(_save_capture(output_dir, frame, cell))
            print(
                f"[Capture] {frame_index + 1:03d}/{args.num_frames} "
                f"E={cell.exposure_ms}ms G={cell.gain}"
            )
    except (KeyboardInterrupt, OSError, RuntimeError, TimeoutError, ValueError) as error:
        capture_error = error
    finally:
        if camera is not None:
            camera.close()

    # Match orbbec_iqa_control.py: DA-V2 inference, then scale/shift alignment
    # against aligned Orbbec depth and the same MDE metrics.
    if rows:
        try:
            predictor = DepthAnythingV2Small(args)
            for index, row in enumerate(rows, 1):
                try:
                    row["mde_inference_ms"] = predictor.infer(
                        Path(row["image_path"]), Path(row["raw_pred_depth_path"])
                    )
                    print(f"[MDE] {index:03d}/{len(rows)}")
                except (OSError, RuntimeError, ValueError) as error:
                    row["mde_error"] = str(error)
        except (OSError, RuntimeError, ValueError) as error:
            for row in rows:
                row["mde_error"] = str(error)

    for index, row in enumerate(rows, 1):
        if row["mde_error"]:
            continue
        try:
            row.update(_evaluate_saved_prediction(row, args))
            print(
                f"[Evaluate] {index:03d}/{len(rows)} "
                f"AbsRel={row['abs_rel']:.6f}"
            )
        except (OSError, RuntimeError, ValueError) as error:
            row["evaluation_error"] = str(error)

    report = _write_report(output_dir, rows)
    evaluated_count = sum(row["abs_rel"] != "" for row in rows)
    if capture_error is not None:
        print(
            f"[WARNING] Capture stopped after {len(rows)} frames: {capture_error}",
            file=sys.stderr,
        )
    return report, len(rows), evaluated_count


def main() -> int:
    args = _parse_args()
    try:
        report, captured, evaluated = _run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[Done] captured={captured} evaluated={evaluated} report={report}")
    return 0 if captured == args.num_frames and evaluated == captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
