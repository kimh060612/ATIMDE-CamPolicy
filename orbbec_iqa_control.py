#!/usr/bin/env python3
"""Noise-aware exposure/gain control and MDE evaluation on an Orbbec RGB-D camera."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.evaluation import _fit_scale_shift
from hardware.sensor import OrbbecColorCamera
from hardware.utils import ContextKey, SensorCell
from iqa_control.noise_aware_iqa_control import (
    ControlSetting,
    IQAResult,
    NoiseAwareNelderMead,
    noise_aware_iqa,
)


DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
CSV_FIELDS = (
    "record_type", "run_mode", "frame_index", "timestamp_ns", "operation",
    "exposure_ms", "gain", "requested_exposure_raw", "actual_exposure_raw",
    "actual_gain", "color_frame_number", "depth_frame_number",
    "color_timestamp_us", "depth_timestamp_us", "setting_effective",
    "sensor_settle_ms", "camera_parameter_ms", "iqa_ms", "iqa_score",
    "iqa_gradient", "iqa_entropy", "iqa_noise", "mean_intensity",
    "best_exposure_ms", "best_gain", "best_iqa_score", "nm_iteration",
    "image_path", "gt_depth_path", "raw_pred_depth_path", "pred_depth_path",
    "mde_inference_ms", "depth_alignment", "alignment_scale",
    "alignment_shift", "abs_rel", "rmse", "mae", "a1", "a2", "a3",
    "valid_depth_pixels", "mde_error", "evaluation_error",
)


class _FixedContext:
    def __init__(self) -> None:
        self.context = ContextKey(0, 0)

    def get(self) -> ContextKey:
        return self.context


class DepthAnythingV2Small:
    """Thin inference-only adapter around the already implemented DA-V2 loader."""

    def __init__(self, args: argparse.Namespace) -> None:
        from orbbec_deterministic_probing_absrel import DepthAnythingAbsRelScorer

        self.backend = DepthAnythingAbsRelScorer(
            model_id=DEPTH_MODEL_ID,
            device=args.depth_device,
            precision=args.depth_precision,
            score_mode=args.depth_alignment,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_valid_pixels=args.min_valid_depth_pixels,
            local_files_only=args.depth_model_local_files_only,
        )

    def predict(
        self, image: np.ndarray, target_size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, float]:
        """Return the raw relative-depth prediction through the shared path."""
        backend = self.backend
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = backend.processor(images=rgb, return_tensors="pt")["pixel_values"].to(
            device=backend.device, dtype=backend.dtype, non_blocking=True
        )
        if backend.device.type == "cuda":
            torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(
                device_type=backend.device.type,
                dtype=torch.float16,
                enabled=backend.use_fp16,
            ):
                prediction = backend.model(pixel_values=pixels).predicted_depth
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            prediction = backend.torch_f.interpolate(
                prediction.float(),
                size=target_size or image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        if backend.device.type == "cuda":
            torch.cuda.synchronize(backend.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return (
            np.ascontiguousarray(prediction.detach().cpu().numpy(), dtype=np.float32),
            inference_ms,
        )

    def infer(self, image_path: Path, output_path: Path) -> float:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Failed to read RGB image: {image_path}")
        prediction, inference_ms = self.predict(image, image.shape[:2])
        np.save(output_path, prediction)
        return inference_ms


def evaluate_depth_arrays(
    prediction_np: np.ndarray,
    target_np: np.ndarray,
    *,
    alignment: str,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_depth_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shared CPU scale/shift alignment and metrics for both experiments."""
    if prediction_np.shape != target_np.shape:
        prediction_np = cv2.resize(
            prediction_np,
            (target_np.shape[1], target_np.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    target = torch.from_numpy(np.ascontiguousarray(target_np))
    prediction = torch.from_numpy(np.ascontiguousarray(prediction_np))
    valid = (
        torch.isfinite(target)
        & (target >= min_depth_m)
        & (target <= max_depth_m)
        & torch.isfinite(prediction)
    )
    if int(valid.sum()) < min_valid_depth_pixels:
        raise ValueError("Not enough valid depth pixels for scale/shift alignment.")

    if alignment == "scale_shift_depth":
        scale, shift = _fit_scale_shift(prediction, target, valid)
        aligned = scale * prediction + shift
    elif alignment == "scale_shift_inverse":
        inverse_target = torch.reciprocal(target.clamp_min(1e-6))
        scale, shift = _fit_scale_shift(prediction, inverse_target, valid)
        aligned_inverse = scale * prediction + shift
        valid &= torch.isfinite(aligned_inverse) & (aligned_inverse > 1e-6)
        aligned = torch.reciprocal(aligned_inverse.clamp_min(1e-6))
    else:
        raise ValueError(f"Unsupported scale/shift alignment: {alignment}")

    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Scale/shift alignment produced a non-positive scale.")
    valid &= torch.isfinite(aligned) & (aligned > 1e-6)
    count = int(valid.sum())
    if count < min_valid_depth_pixels:
        raise ValueError("Not enough valid pixels remain after scale/shift alignment.")
    predicted, expected = aligned[valid], target[valid]
    difference = predicted - expected
    ratio = torch.maximum(predicted / expected, expected / predicted)
    metrics = {
        "depth_alignment": alignment,
        "alignment_scale": float(scale),
        "alignment_shift": float(shift),
        "abs_rel": float(torch.mean(torch.abs(difference) / expected)),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "mae": float(torch.mean(torch.abs(difference))),
        "a1": float(torch.mean((ratio < 1.25).float())),
        "a2": float(torch.mean((ratio < 1.25**2).float())),
        "a3": float(torch.mean((ratio < 1.25**3).float())),
        "valid_depth_pixels": count,
    }
    return np.ascontiguousarray(aligned.numpy(), dtype=np.float32), metrics


def _evaluate_saved_prediction(
    row: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    target = np.load(row["gt_depth_path"], allow_pickle=False).astype(np.float32)
    prediction = np.load(row["raw_pred_depth_path"], allow_pickle=False).astype(np.float32)
    aligned, metrics = evaluate_depth_arrays(
        prediction,
        target,
        alignment=args.depth_alignment,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_valid_depth_pixels=args.min_valid_depth_pixels,
    )
    np.save(row["pred_depth_path"], aligned)
    return metrics


def _empty_row() -> dict[str, Any]:
    return {field: "" for field in CSV_FIELDS}


def _save_capture(
    output_dir: Path,
    frame: Any,
    setting: ControlSetting,
    operation: str,
    iqa: IQAResult,
    iqa_ms: float,
    controller: NoiseAwareNelderMead,
    mode: int,
) -> dict[str, Any]:
    stem = f"frame_{frame.capture_index:05d}_{frame.cell.cell_id}_{frame.timestamp_ns}"
    image_path = output_dir / "images" / f"{stem}.png"
    gt_path = output_dir / "depth_gt" / f"{stem}.npy"
    raw_prediction_path = output_dir / "depth_pred_raw" / f"{stem}.npy"
    prediction_path = output_dir / "depth_pred" / f"{stem}.npy"
    if not cv2.imwrite(str(image_path), frame.image):
        raise OSError(f"Failed to save RGB image: {image_path}")
    np.save(gt_path, np.ascontiguousarray(frame.depth_m, dtype=np.float32))
    best = controller.best_setting
    row = _empty_row()
    row.update(
        record_type="capture",
        run_mode=mode,
        frame_index=frame.capture_index,
        timestamp_ns=frame.timestamp_ns,
        operation=operation,
        exposure_ms=setting.exposure_ms,
        gain=setting.gain,
        requested_exposure_raw=frame.requested_exposure_raw,
        actual_exposure_raw=frame.actual_exposure_raw,
        actual_gain=frame.actual_gain,
        color_frame_number=frame.color_frame_number,
        depth_frame_number=frame.depth_frame_number,
        color_timestamp_us=frame.color_timestamp_us,
        depth_timestamp_us=frame.depth_timestamp_us,
        setting_effective=int(frame.setting_effective),
        sensor_settle_ms=frame.sensor_settle_ms,
        camera_parameter_ms=frame.camera_parameter_ms,
        iqa_ms=iqa_ms,
        iqa_score=iqa.score,
        iqa_gradient=iqa.gradient,
        iqa_entropy=iqa.entropy,
        iqa_noise=iqa.noise,
        mean_intensity=iqa.mean_intensity,
        best_exposure_ms=best.exposure_ms,
        best_gain=best.gain,
        best_iqa_score=controller.best_score,
        nm_iteration=controller.iteration_count,
        image_path=str(image_path),
        gt_depth_path=str(gt_path),
        raw_pred_depth_path=str(raw_prediction_path),
        pred_depth_path=str(prediction_path),
    )
    return row


def _write_report(output_dir: Path, rows: list[dict[str, Any]], mode: int) -> Path:
    path = output_dir / "iqa_control_report.csv"
    temporary = path.with_suffix(".csv.tmp")
    evaluated = [row for row in rows if row["abs_rel"] != ""]
    summary = _empty_row()
    summary.update(
        record_type="summary",
        run_mode=mode,
        frame_index=len(rows),
        operation=f"evaluated={len(evaluated)}",
    )
    for field in ("iqa_score", "abs_rel", "rmse", "mae", "a1", "a2", "a3"):
        values = [float(row[field]) for row in (rows if field == "iqa_score" else evaluated)]
        summary[field] = float(np.mean(values)) if values else ""
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)
    os.replace(temporary, path)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noise-aware Nelder-Mead exposure control for Orbbec RGB-D cameras."
    )
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-frames", type=int, default=360)
    parser.add_argument("--output-dir", type=Path, default=Path("iqa_control_output"))
    parser.add_argument("--initial-exposure-ms", type=int, default=8)
    parser.add_argument("--initial-gain", type=int, default=64)
    parser.add_argument("--exposure-min-ms", type=int, default=4)
    parser.add_argument("--exposure-max-ms", type=int, default=67)
    parser.add_argument("--exposure-step-ms", type=int, default=3)
    parser.add_argument("--gain-min", type=int, default=16)
    parser.add_argument("--gain-max", type=int, default=128)
    parser.add_argument("--gain-step", type=int, default=4)
    parser.add_argument("--simplex-restart-frames", type=int, default=60)
    parser.add_argument("--simplex-tolerance", type=float, default=0.02)
    parser.add_argument("--iqa-resize-factor", type=float, default=1.0)
    parser.add_argument("--capture-interval-ms", type=float, default=0.0)

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

    if args.num_frames < 1:
        parser.error("--num-frames must be positive.")
    if args.exposure_min_ms > args.initial_exposure_ms or args.initial_exposure_ms > args.exposure_max_ms:
        parser.error("Initial exposure must be inside the exposure bounds.")
    if args.gain_min > args.initial_gain or args.initial_gain > args.gain_max:
        parser.error("Initial gain must be inside the gain bounds.")
    if args.exposure_step_ms < 1 or args.gain_step < 1:
        parser.error("Exposure/gain steps must be positive.")
    if args.settle_frames < 0 or args.warmup_frames < 0:
        parser.error("Camera frame counts must be non-negative.")
    if not math.isfinite(args.capture_interval_ms) or args.capture_interval_ms < 0:
        parser.error("--capture-interval-ms must be finite and non-negative.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        parser.error("Require 0 < --min-depth-m < --max-depth-m.")
    return args


def _run(args: argparse.Namespace) -> tuple[Path, int, int]:
    output_dir = args.output_dir.resolve()
    for name in ("images", "depth_gt", "depth_pred_raw", "depth_pred"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    controller = NoiseAwareNelderMead(
        ControlSetting(args.initial_exposure_ms, args.initial_gain),
        exposure_bounds=(args.exposure_min_ms, args.exposure_max_ms),
        gain_bounds=(args.gain_min, args.gain_max),
        exposure_step=args.exposure_step_ms,
        gain_step=args.gain_step,
        restart_interval=args.simplex_restart_frames,
        simplex_tolerance=args.simplex_tolerance,
    )
    context_provider = _FixedContext()
    predictor = DepthAnythingV2Small(args) if args.mode == 2 else None
    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="depth-anything")
        if args.mode == 2
        else None
    )
    jobs: list[tuple[Future[float], dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    run_error: BaseException | None = None
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
        capture_runner = CaptureRunner(camera, context_provider, max_pair_gap_ms=100.0)
        for frame_index in range(args.num_frames):
            cycle_started = time.monotonic()
            setting = controller.next_setting()
            operation = controller.operation
            frame = capture_runner.capture(
                SensorCell(setting.exposure_ms, setting.gain),
                context_provider.get(),
                operation,
                frame_index,
            )
            if not frame.setting_effective:
                raise RuntimeError(
                    f"Camera setting was not verified for frame {frame_index}: {setting}"
                )
            iqa_started = time.perf_counter()
            iqa = noise_aware_iqa(frame.image, args.iqa_resize_factor)
            iqa_ms = (time.perf_counter() - iqa_started) * 1000.0
            controller.observe(iqa)
            row = _save_capture(
                output_dir, frame, setting, operation, iqa, iqa_ms, controller, args.mode
            )
            rows.append(row)
            if executor is not None and predictor is not None:
                future = executor.submit(
                    predictor.infer,
                    Path(row["image_path"]),
                    Path(row["raw_pred_depth_path"]),
                )
                jobs.append((future, row))
            best = controller.best_setting
            print(
                f"[Capture] {frame_index + 1:03d}/{args.num_frames} "
                f"op={operation} E={setting.exposure_ms}ms G={setting.gain} "
                f"IQA={iqa.score:.5f} best=E{best.exposure_ms} G{best.gain}"
            )
            remaining = args.capture_interval_ms / 1000.0 - (
                time.monotonic() - cycle_started
            )
            if remaining > 0:
                time.sleep(remaining)
    except (KeyboardInterrupt, OSError, RuntimeError, TimeoutError, ValueError) as error:
        run_error = error
    finally:
        if camera is not None:
            camera.close()

    if executor is not None:
        executor.shutdown(wait=True)
        for future, row in jobs:
            try:
                row["mde_inference_ms"] = future.result()
            except (OSError, RuntimeError, ValueError) as error:
                row["mde_error"] = str(error)
    elif rows:
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
            print(f"[Evaluate] {index:03d}/{len(rows)} AbsRel={row['abs_rel']:.6f}")
        except (OSError, RuntimeError, ValueError) as error:
            row["evaluation_error"] = str(error)

    report = _write_report(output_dir, rows, args.mode)
    evaluated_count = sum(row["abs_rel"] != "" for row in rows)
    if run_error is not None:
        print(f"[WARNING] Capture stopped after {len(rows)} frames: {run_error}", file=sys.stderr)
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
