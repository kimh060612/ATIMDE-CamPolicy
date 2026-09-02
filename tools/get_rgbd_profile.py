#!/usr/bin/env python3
"""List same-FPS RGB-D profiles advertised as D2C-compatible by OrbbecSDK."""

from __future__ import annotations

import argparse
import sys
from typing import Any


def profile_values(profile: Any) -> tuple[int, int, int, str]:
    video = profile.as_video_stream_profile()
    fmt = str(video.get_format()).rsplit(".", 1)[-1]
    return video.get_width(), video.get_height(), video.get_fps(), fmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show same-FPS color/depth pairs supported for Orbbec D2C alignment."
    )
    parser.add_argument("--fps", type=int, help="show only this FPS (for example: 10)")
    parser.add_argument(
        "--align",
        choices=("hw", "sw", "both"),
        default="both",
        help="D2C alignment mode to query (default: both)",
    )
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        from pyorbbecsdk import OBAlignMode, OBError, OBSensorType, Pipeline
    except ImportError:
        print(
            "[ERROR] pyorbbecsdk is not installed. Install pyorbbecsdk2 first.",
            file=sys.stderr,
        )
        return 1

    try:
        pipeline = Pipeline()
        info = pipeline.get_device().get_device_info()
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    except (OBError, RuntimeError) as error:
        print(f"[ERROR] Could not open an Orbbec camera: {error}", file=sys.stderr)
        return 1

    modes = []
    if args.align in ("hw", "both"):
        modes.append(("HW", OBAlignMode.HW_MODE))
    if args.align in ("sw", "both"):
        modes.append(("SW", OBAlignMode.SW_MODE))

    rows: set[tuple[Any, ...]] = set()
    for index in range(len(color_profiles)):
        color = color_profiles[index]
        color_values = profile_values(color)
        if args.fps is not None and color_values[2] != args.fps:
            continue
        for mode_name, mode in modes:
            try:
                depth_profiles = pipeline.get_d2c_depth_profile_list(color, mode)
            except (OBError, RuntimeError):
                continue
            for depth_index in range(len(depth_profiles)):
                depth_values = profile_values(depth_profiles[depth_index])
                if depth_values[2] == color_values[2]:
                    rows.add((mode_name, *color_values, *depth_values))

    print(f"Device: {info.get_name()}  S/N: {info.get_serial_number()}")
    print("ALIGN  COLOR                     DEPTH")
    print("-----  ------------------------  ------------------------")
    for mode, cw, ch, cfps, cfmt, dw, dh, dfps, dfmt in sorted(rows):
        print(
            f"{mode:<5}  {cw}x{ch}@{cfps:<3} {cfmt:<10}  "
            f"{dw}x{dh}@{dfps:<3} {dfmt}"
        )

    if not rows:
        suffix = f" at {args.fps} FPS" if args.fps is not None else ""
        print(f"No same-FPS D2C-compatible RGB-D profiles found{suffix}.")
        return 2
    print(f"\n{len(rows)} compatible profile(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
