#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.sensor import OrbbecColorCamera
from hardware.utils import SensorCell


MAX_OBSERVED_FRAMES = 10
CSV_FIELDS = (
    "transition", "repetition", "frame_offset", "frame_number",
    "device_timestamp_us", "requested_exposure_raw", "readback_exposure_raw",
    "requested_gain", "readback_gain", "mean_luminance", "setting_effective",
)
TRANSITIONS = (
    (SensorCell(4, 64), SensorCell(8, 64)),
    (SensorCell(8, 64), SensorCell(4, 64)),
    (SensorCell(8, 32), SensorCell(8, 64)),
    (SensorCell(8, 64), SensorCell(8, 32)),
)


def stable_offset(luminances: list[float]) -> int | None:
    if len(luminances) < 3:
        return None
    plateau = float(np.median(luminances[-3:]))
    tolerance = max(1.0, abs(plateau) * 0.05)
    for offset in range(len(luminances)):
        if all(abs(value - plateau) <= tolerance for value in luminances[offset:]):
            return offset
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Orbbec exposure/gain settling.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    stable_offsets: list[int] = []
    latencies_ms: list[float] = []
    direction_matches = 0
    valid_repetitions = 0
    stale_outputs = 0
    previous_frame_number: int | None = None
    previous_timestamp_us: int | None = None
    total = args.repetitions * len(TRANSITIONS)
    camera = None
    try:
        camera = OrbbecColorCamera(
            exposure_value_per_ms=10.0,
            settle_frames=0,
            frame_timeout_ms=1000,
            warmup_frames=30,
            disable_awb=True,
            strict_property_grid=False,
        )
        for repetition in range(args.repetitions):
            for source, target in TRANSITIONS:
                camera.apply_cell(source)
                baseline_image, _ = camera.capture_rgbd()
                baseline = float(np.mean(cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)))
                baseline_timestamp = camera.color_timestamp_us

                requested, readback_exposure, readback_gain = camera.apply_cell(target)
                transition = f"{source.cell_id}->{target.cell_id}"
                luminances: list[float] = []
                timestamps: list[int | None] = []
                effective: list[bool] = []
                for offset in range(MAX_OBSERVED_FRAMES):
                    image, _ = camera.capture_rgbd()
                    luminance = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
                    luminances.append(luminance)
                    timestamps.append(camera.color_timestamp_us)
                    effective.append(camera.setting_effective)
                    stale_outputs += int(
                        camera.color_frame_number is None
                        or camera.color_timestamp_us is None
                        or (
                            previous_frame_number is not None
                            and camera.color_frame_number <= previous_frame_number
                        )
                        or (
                            previous_timestamp_us is not None
                            and camera.color_timestamp_us <= previous_timestamp_us
                        )
                    )
                    previous_frame_number = camera.color_frame_number
                    previous_timestamp_us = camera.color_timestamp_us
                    rows.append({
                        "transition": transition,
                        "repetition": repetition,
                        "frame_offset": offset,
                        "frame_number": camera.color_frame_number,
                        "device_timestamp_us": camera.color_timestamp_us,
                        "requested_exposure_raw": requested,
                        "readback_exposure_raw": readback_exposure,
                        "requested_gain": target.gain,
                        "readback_gain": readback_gain,
                        "mean_luminance": luminance,
                        "setting_effective": int(camera.setting_effective),
                    })

                offset = stable_offset(luminances)
                expected_sign = 1 if target.exposure_ms * target.gain > source.exposure_ms * source.gain else -1
                direction_ok = expected_sign * (float(np.median(luminances[-3:])) - baseline) > 0
                direction_matches += int(direction_ok)
                valid = (
                    offset is not None
                    and all(effective)
                    and baseline_timestamp is not None
                    and timestamps[offset] is not None
                    and timestamps[offset] > baseline_timestamp
                )
                if valid:
                    valid_repetitions += 1
                    stable_offsets.append(offset)
                    latencies_ms.append((timestamps[offset] - baseline_timestamp) / 1000.0)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[ERROR] calibration failed: {error}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.close()

    with (args.output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    percentile = lambda values, q: float(np.percentile(values, q)) if values else None
    stable_p99 = percentile(stable_offsets, 99)
    summary = {
        "recommended_settle_frames": int(math.ceil(stable_p99)) if stable_p99 is not None else None,
        "direction_match_rate": direction_matches / total,
        "valid_repetition_rate": valid_repetitions / total,
        "sensor_effect_latency_p50_ms": percentile(latencies_ms, 50),
        "sensor_effect_latency_p95_ms": percentile(latencies_ms, 95),
        "sensor_effect_latency_p99_ms": percentile(latencies_ms, 99),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(json.dumps(summary, indent=2))
    passed = (
        summary["direction_match_rate"] >= 0.95
        and summary["valid_repetition_rate"] >= 0.99
        and stale_outputs == 0
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
