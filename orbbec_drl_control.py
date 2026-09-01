#!/usr/bin/env python3
"""Run the pretrained DRL exposure policy on an Orbbec color camera."""

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
import torch

from drl_policy.controller import DRLExposureController, exposure_for_ev
from drl_policy.log import Log
from hardware.sensor import OrbbecColorCamera
from hardware.utils import SensorCell
from orbbec_iqa_control import DepthAnythingV2Small, _evaluate_saved_prediction


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "drl_policy" / "actor_drl_feat_10000.pth"
CSV_FIELDS = (
    "record_type",
    "frame_index",
    "evaluated_frames",
    "timestamp_ns",
    "actor_action",
    "ev",
    "exposure_ms",
    "gain",
    "requested_exposure_raw",
    "actual_exposure_raw",
    "actual_gain",
    "color_frame_number",
    "depth_frame_number",
    "color_timestamp_us",
    "depth_timestamp_us",
    "setting_effective",
    "sensor_settle_ms",
    "control_cycle_ms",
    "image_path",
    "gt_depth_path",
    "raw_pred_depth_path",
    "pred_depth_path",
    "mde_inference_ms",
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
    "mde_error",
    "evaluation_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DRL exposure-only control for an Orbbec color camera."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--initial-exposure-ms", type=float, default=10.0)
    parser.add_argument("--min-exposure-ms", type=float, default=0.05)
    parser.add_argument("--max-exposure-ms", type=float, default=2000.0)
    parser.add_argument(
        "--gain", type=int, default=16, help="Fixed gain; the policy never changes it."
    )
    parser.add_argument("--exposure-value-per-ms", type=float, default=10.0)
    parser.add_argument("--settle-frames", type=int, default=4)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument("--disable-awb", action="store_true")
    parser.add_argument(
        "--max-frames", type=int, default=200, help="0 runs until Ctrl-C."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("drl_control_output")
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=ROOT / "drl_policy" / "orbbec_drl.log",
    )
    parser.add_argument("--depth-device", default="cuda")
    parser.add_argument(
        "--depth-precision", choices=("fp16", "fp32"), default="fp32"
    )
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

    finite_positive = (
        args.initial_exposure_ms,
        args.min_exposure_ms,
        args.max_exposure_ms,
        args.exposure_value_per_ms,
    )
    if not all(math.isfinite(value) and value > 0 for value in finite_positive):
        parser.error(
            "Exposure values and --exposure-value-per-ms must be finite and positive."
        )
    if not args.min_exposure_ms <= args.initial_exposure_ms <= args.max_exposure_ms:
        parser.error("Require min exposure <= initial exposure <= max exposure.")
    if args.gain < 0 or args.settle_frames < 0 or args.warmup_frames < 0:
        parser.error("Gain and frame counts must be non-negative.")
    if args.frame_timeout_ms < 1 or args.max_frames < 0:
        parser.error("Frame timeout must be positive and max frames non-negative.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        parser.error("Require 0 < --min-depth-m < --max-depth-m.")
    if args.min_valid_depth_pixels < 1:
        parser.error("--min-valid-depth-pixels must be positive.")
    return args


def resolve_device(name: str) -> torch.device:
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def exposure_raw_limits(
    args: argparse.Namespace, camera: OrbbecColorCamera
) -> tuple[int, int, int]:
    scale = args.exposure_value_per_ms
    sensor_range = camera.exposure_range or {}
    origin = sensor_range.get("min", 0)
    step = max(1, sensor_range.get("step", 1))
    lower = max(math.ceil(args.min_exposure_ms * scale), sensor_range.get("min", 1))
    upper = min(
        math.floor(args.max_exposure_ms * scale),
        sensor_range.get("max", math.inf),
    )
    lower = origin + math.ceil((lower - origin) / step) * step
    upper = origin + math.floor((upper - origin) / step) * step
    if lower > upper:
        raise ValueError("Configured exposure limits do not overlap the camera exposure range.")
    return int(lower), int(upper), step


def quantize_exposure_ms(
    exposure_ms: float, scale: float, limits: tuple[int, int, int]
) -> float:
    lower, upper, step = limits
    raw = lower + round((exposure_ms * scale - lower) / step) * step
    return min(max(raw, lower), upper) / scale


def empty_row() -> dict[str, Any]:
    return {field: "" for field in CSV_FIELDS}


def save_capture(
    output_dir: Path,
    frame_index: int,
    image: np.ndarray,
    depth_m: np.ndarray,
    camera: OrbbecColorCamera,
    actor_action: float,
    ev: float,
    exposure_ms: float,
    gain: int,
    requested_raw: int,
    actual_raw: int | None,
    actual_gain: int | None,
    control_cycle_ms: float,
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

    row = empty_row()
    row.update(
        record_type="capture",
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        actor_action=actor_action,
        ev=ev,
        exposure_ms=exposure_ms,
        gain=gain,
        requested_exposure_raw=requested_raw,
        actual_exposure_raw=actual_raw,
        actual_gain=actual_gain,
        color_frame_number=camera.color_frame_number,
        depth_frame_number=camera.depth_frame_number,
        color_timestamp_us=camera.color_timestamp_us,
        depth_timestamp_us=camera.depth_timestamp_us,
        setting_effective=int(camera.setting_effective),
        sensor_settle_ms=camera.sensor_settle_ms,
        control_cycle_ms=control_cycle_ms,
        image_path=str(image_path),
        gt_depth_path=str(gt_depth_path),
        raw_pred_depth_path=str(raw_pred_depth_path),
        pred_depth_path=str(pred_depth_path),
    )
    return row


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    report_path = output_dir / "drl_control_report.csv"
    temporary_path = report_path.with_suffix(".csv.tmp")
    evaluated = [row for row in rows if row["abs_rel"] != ""]
    summary = empty_row()
    summary.update(
        record_type="summary",
        frame_index=len(rows),
        evaluated_frames=len(evaluated),
    )
    for field in (
        "ev",
        "exposure_ms",
        "control_cycle_ms",
        "mde_inference_ms",
        "abs_rel",
        "rmse",
        "mae",
        "a1",
        "a2",
        "a3",
    ):
        source = evaluated if field in {
            "abs_rel",
            "rmse",
            "mae",
            "a1",
            "a2",
            "a3",
        } else rows
        values = [float(row[field]) for row in source if row[field] != ""]
        summary[field] = float(np.mean(values)) if values else ""

    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)
    os.replace(temporary_path, report_path)
    return report_path


def run(args: argparse.Namespace) -> tuple[Path, int, int]:
    output_dir = args.output_dir.resolve()
    for name in ("images", "depth_gt", "depth_pred_raw", "depth_pred"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    controller = DRLExposureController(args.checkpoint.resolve(), device)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = Log(str(args.log_file))
    rows: list[dict[str, Any]] = []
    capture_error: BaseException | None = None
    camera: OrbbecColorCamera | None = None

    try:
        predictor = DepthAnythingV2Small(args)
        camera = OrbbecColorCamera(
            exposure_value_per_ms=args.exposure_value_per_ms,
            settle_frames=args.settle_frames,
            frame_timeout_ms=args.frame_timeout_ms,
            warmup_frames=args.warmup_frames,
            disable_awb=args.disable_awb,
            strict_property_grid=False,
        )
        limits = exposure_raw_limits(args, camera)
        minimum_ms = limits[0] / args.exposure_value_per_ms
        maximum_ms = limits[1] / args.exposure_value_per_ms
        exposure_ms = quantize_exposure_ms(
            args.initial_exposure_ms, args.exposure_value_per_ms, limits
        )

        _, actual_raw, _ = camera.apply_cell(SensorCell(exposure_ms, args.gain))
        image, _ = camera.capture_rgbd()
        if not camera.setting_effective:
            raise RuntimeError("The initial camera exposure could not be verified.")
        if actual_raw is not None:
            exposure_ms = actual_raw / args.exposure_value_per_ms
        controller.observe(image)

        logger.add_log(
            f"[Start] checkpoint={args.checkpoint.resolve()} device={device} "
            f"exposure_ms={exposure_ms:.6f} gain={args.gain}"
        )
        logger.save_buffer_to_file()
        frame_index = 0

        while args.max_frames == 0 or frame_index < args.max_frames:
            started = time.perf_counter()
            ev, actor_action = controller.action()
            target_ms = exposure_for_ev(exposure_ms, ev, minimum_ms, maximum_ms)
            target_ms = quantize_exposure_ms(target_ms, args.exposure_value_per_ms, limits)
            requested_raw, actual_raw, actual_gain = camera.apply_cell(
                SensorCell(target_ms, args.gain)
            )
            image, depth_m = camera.capture_rgbd()
            if not camera.setting_effective:
                raise RuntimeError(
                    f"Camera setting was not verified at frame {frame_index}."
                )
            exposure_ms = (
                actual_raw / args.exposure_value_per_ms
                if actual_raw is not None
                else requested_raw / args.exposure_value_per_ms
            )
            controller.observe(image)
            inference_cycle_ms = (time.perf_counter() - started) * 1000.0
            row = save_capture(
                output_dir,
                frame_index,
                image,
                depth_m,
                camera,
                actor_action,
                ev,
                exposure_ms,
                args.gain,
                requested_raw,
                actual_raw,
                actual_gain,
                inference_cycle_ms,
            )
            rows.append(row)
            try:
                row["mde_inference_ms"] = predictor.infer(
                    Path(row["image_path"]), Path(row["raw_pred_depth_path"])
                )
                print(f"[MDE] {frame_index + 1:03d}")
            except (OSError, RuntimeError, ValueError) as error:
                row["mde_error"] = str(error)
            line = (
                f"[Frame {frame_index:06d}] actor={actor_action:+.6f} EV={ev:+.6f} "
                f"exposure_ms={exposure_ms:.6f} requested_raw={requested_raw} "
                f"actual_raw={actual_raw} gain={actual_gain} cycle_ms={inference_cycle_ms:.3f}"
            )
            print(line)
            logger.add_log(line)
            logger.save_buffer_to_file()
            frame_index += 1
    except (KeyboardInterrupt, OSError, RuntimeError, TimeoutError, ValueError) as error:
        capture_error = error
    finally:
        if camera is not None:
            camera.close()

    for index, row in enumerate(rows, 1):
        if row["mde_error"]:
            continue
        try:
            row.update(_evaluate_saved_prediction(row, args))
            print(f"[Evaluate] {index:03d}/{len(rows)} AbsRel={row['abs_rel']:.6f}")
        except (OSError, RuntimeError, ValueError) as error:
            row["evaluation_error"] = str(error)

    report = write_report(output_dir, rows)
    evaluated_count = sum(row["abs_rel"] != "" for row in rows)
    if capture_error is not None:
        print(
            f"[WARNING] Capture stopped after {len(rows)} frames: {capture_error}",
            file=sys.stderr,
        )
    return report, len(rows), evaluated_count


def main() -> int:
    try:
        args = parse_args()
        report, captured, evaluated = run(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[Done] captured={captured} evaluated={evaluated} report={report}")
    expected = captured if args.max_frames == 0 else args.max_frames
    return 0 if captured == expected and evaluated == captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
