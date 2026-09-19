#!/usr/bin/env python3
from __future__ import annotations

import queue
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import orbbec_drl_control as control
from orbbec_ati_risk_bandit_lap_record import (
    LAP_END_ADDRESS,
    LAP_END_PATH,
    LapCaptureLogger,
    LapEndHandler,
    apply_lap_events,
)


def run(args) -> tuple[Path, int, int]:
    import hardware.sensor as sensor

    sensor.RGBD_WIDTH = 640
    sensor.RGBD_HEIGHT = 480
    sensor.RGBD_FPS = 15

    events: queue.SimpleQueue[tuple[int, str]] = queue.SimpleQueue()
    lap_logger = LapCaptureLogger(args.output_dir.resolve())
    original_save_capture = control.save_capture

    def save_capture_by_lap(_output_dir, *capture_args, **capture_kwargs):
        apply_lap_events(events, lap_logger)
        return original_save_capture(
            lap_logger.image_dir.parent, *capture_args, **capture_kwargs
        )

    LapEndHandler.events = events
    server = ThreadingHTTPServer(LAP_END_ADDRESS, LapEndHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    control.save_capture = save_capture_by_lap
    print(
        f"[Lap] 640x480@15; "
        f"listening=http://{LAP_END_ADDRESS[0]}:{LAP_END_ADDRESS[1]}{LAP_END_PATH}"
    )
    try:
        return control.run(args)
    finally:
        control.save_capture = original_save_capture
        server.shutdown()
        server.server_close()


def main() -> int:
    try:
        args = control.parse_args()
        report, captured, evaluated = run(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[Done] captured={captured} evaluated={evaluated} report={report}")
    expected = captured if args.max_frames == 0 else args.max_frames
    return 0 if captured == expected and evaluated == captured else 1


if __name__ == "__main__":
    raise SystemExit(main())
