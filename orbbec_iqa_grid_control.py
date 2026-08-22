#!/usr/bin/env python3
"""Noise-aware IQA control restricted to the Orbbec 4 x 4 sensor grid."""

from __future__ import annotations

import argparse
import math
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from hardware.sensor import OrbbecColorCamera
from hardware.utils import EXPOSURE_MS_VALUES, GAIN_VALUES, SensorCell
from iqa_control.noise_aware_iqa_control import (
    ControlSetting,
    NoiseAwareNelderMead,
    noise_aware_iqa,
)
from orbbec_iqa_control import (
    DepthAnythingV2Small,
    _FixedContext,
    _evaluate_saved_prediction,
    _save_capture,
    _write_report,
)


class GridNoiseAwareNelderMead(NoiseAwareNelderMead):
    """The existing IQA Nelder-Mead controller projected onto the 4 x 4 grid."""

    _GRID = (
        np.asarray(EXPOSURE_MS_VALUES, dtype=np.float64),
        np.asarray(GAIN_VALUES, dtype=np.float64),
    )

    def __init__(
        self,
        initial: ControlSetting,
        *,
        restart_interval: int,
        simplex_tolerance: float,
    ) -> None:
        super().__init__(
            initial,
            exposure_bounds=(EXPOSURE_MS_VALUES[0], EXPOSURE_MS_VALUES[-1]),
            gain_bounds=(GAIN_VALUES[0], GAIN_VALUES[-1]),
            exposure_step=1,
            gain_step=1,
            restart_interval=restart_interval,
            simplex_tolerance=simplex_tolerance,
        )

    def _quantize(self, point: np.ndarray) -> np.ndarray:
        requested = np.asarray(point, dtype=np.float64)
        return np.asarray(
            [
                values[np.abs(values - requested[axis]).argmin()]
                for axis, values in enumerate(self._GRID)
            ],
            dtype=np.float64,
        )

    def _make_initial_candidates(
        self, anchor: np.ndarray, mean_intensity: float
    ) -> list[np.ndarray]:
        intensity = float(np.clip(mean_intensity / 255.0, 0.0, 1.0))
        h = -intensity / 1.7 if intensity >= 0.5 else 1.7 * (1.0 - intensity)
        candidates: list[np.ndarray] = []
        for axis, values in enumerate(self._GRID):
            point = anchor.copy()
            point[axis] *= 1.0 + h
            point = self._quantize(point)
            if np.array_equal(point, anchor):
                anchor_index = int(np.abs(values - anchor[axis]).argmin())
                direction = 1 if h >= 0 else -1
                candidate_index = anchor_index + direction
                if not 0 <= candidate_index < len(values):
                    candidate_index = anchor_index - direction
                point[axis] = values[candidate_index]
            candidates.append(point)
        return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Noise-aware Nelder-Mead exposure control on the fixed Orbbec "
            "4 x 4 exposure/gain grid."
        )
    )
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-frames", type=int, default=360)
    parser.add_argument("--output-dir", type=Path, default=Path("iqa_control_output"))
    parser.add_argument(
        "--initial-exposure-ms", type=int, choices=EXPOSURE_MS_VALUES, default=8
    )
    parser.add_argument("--initial-gain", type=int, choices=GAIN_VALUES, default=64)
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
    if args.simplex_restart_frames < 3:
        parser.error("--simplex-restart-frames must be at least 3.")
    if not math.isfinite(args.simplex_tolerance) or args.simplex_tolerance < 0:
        parser.error("--simplex-tolerance must be finite and non-negative.")
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

    controller = GridNoiseAwareNelderMead(
        ControlSetting(args.initial_exposure_ms, args.initial_gain),
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
        print(
            f"[WARNING] Capture stopped after {len(rows)} frames: {run_error}",
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
