#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

import orbbec_drl_control as source


LATENCY_CSV_FIELDS = (
    "frame_index",
    "timestamp_ns",
    "exposure_ms",
    "gain",
    "camera_switched",
    "mde_inference_ms",
    "mde_total_ms",
    "camera_apply_latency_ms",
    "camera_switching_latency_ms",
    "policy_decision_ms",
    "control_cycle_ms",
    "mde_error",
)


class LatencyLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "drl_policy_latency.csv"
        self.rows: list[dict[str, Any]] = []

    def record(self, values: dict[str, Any]) -> None:
        self.rows.append({field: values.get(field, "") for field in LATENCY_CSV_FIELDS})

    def update_last(self, **values: Any) -> None:
        if self.rows:
            self.rows[-1].update(values)

    def write(self) -> Path:
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LATENCY_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)
        return self.path


def install_instrumentation(logger: LatencyLogger) -> None:
    base_controller = source.DRLExposureController
    base_camera = source.OrbbecColorCamera
    base_predictor = source.DepthAnythingV2Small
    save_capture = source.save_capture

    class TimedController(base_controller):
        last_decision_ms: float | None = None

        def action(self):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            try:
                return super().action()
            finally:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                TimedController.last_decision_ms = (
                    time.perf_counter() - started
                ) * 1000.0

    class TimedCamera(base_camera):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.latency_active_cell = None
            self.last_apply_ms: float | None = None
            self.last_apply_switched = False

        def apply_cell(self, cell):
            previous = self.latency_active_cell
            started = time.perf_counter()
            try:
                result = super().apply_cell(cell)
                self.latency_active_cell = cell
                return result
            finally:
                self.last_apply_ms = (time.perf_counter() - started) * 1000.0
                self.last_apply_switched = previous is not None and previous != cell

    class TimedPredictor(base_predictor):
        def infer(self, image_path, output_path):
            started = time.perf_counter()
            try:
                inference_ms = super().infer(image_path, output_path)
                logger.update_last(
                    mde_inference_ms=inference_ms,
                    mde_total_ms=(time.perf_counter() - started) * 1000.0,
                )
                return inference_ms
            except Exception as error:
                logger.update_last(
                    mde_total_ms=(time.perf_counter() - started) * 1000.0,
                    mde_error=f"{type(error).__name__}: {error}",
                )
                raise

    def timed_save_capture(
        output_dir,
        frame_index,
        image,
        depth_m,
        camera,
        actor_action,
        ev,
        allocation,
        gain_raw,
        requested_raw,
        actual_raw,
        actual_gain,
        control_cycle_ms,
    ):
        row = save_capture(
            output_dir,
            frame_index,
            image,
            depth_m,
            camera,
            actor_action,
            ev,
            allocation,
            gain_raw,
            requested_raw,
            actual_raw,
            actual_gain,
            control_cycle_ms,
        )
        logger.record(
            {
                "frame_index": frame_index,
                "timestamp_ns": row["timestamp_ns"],
                "exposure_ms": allocation.exposure_time_us / 1000.0,
                "gain": gain_raw,
                "camera_switched": int(camera.last_apply_switched),
                "camera_apply_latency_ms": camera.last_apply_ms,
                "camera_switching_latency_ms": (
                    camera.last_apply_ms if camera.last_apply_switched else None
                ),
                "policy_decision_ms": TimedController.last_decision_ms,
                "control_cycle_ms": control_cycle_ms,
            }
        )
        return row

    source.DRLExposureController = TimedController
    source.OrbbecColorCamera = TimedCamera
    source.DepthAnythingV2Small = TimedPredictor
    source.save_capture = timed_save_capture


def main() -> int:
    args = source.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = LatencyLogger(output_dir)
    latency_write_failed = False
    try:
        install_instrumentation(logger)
        report, captured, evaluated = source.run(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    finally:
        try:
            print(f"[Latency] wrote {logger.write()}")
        except OSError as error:
            print(f"[ERROR] latency CSV write failed: {error}", file=sys.stderr)
            latency_write_failed = True
    print(f"[Done] captured={captured} evaluated={evaluated} report={report}")
    expected = captured if args.max_frames == 0 else args.max_frames
    return (
        0
        if not latency_write_failed and captured == expected and evaluated == captured
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
