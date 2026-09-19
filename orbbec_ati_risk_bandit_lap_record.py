#!/usr/bin/env python3
from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ati_mde_control.logging import CaptureLogger
from orbbec_ati_risk_bandit_bidirectional_exposure_sync import (
    METHOD_NAME,
    build_experiment as build_control_experiment,
    parse_args,
)
from orbbec_deterministic_probing_modelv1 import FairDepthEvaluator


LAP_END_ADDRESS = ("localhost", 3000)
LAP_END_PATH = "/lap-end"


class LapCaptureLogger(CaptureLogger):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.rows = []
        self.current_lap = 1
        self._select_lap(self.current_lap)

    def _select_lap(self, lap: int) -> None:
        lap_dir = self.output_dir / f"lap_{lap:04d}"
        self.image_dir = lap_dir / "images"
        self.depth_dir = lap_dir / "depth_gt"
        for name in ("images", "depth_gt", "depth_pred", "depth_pred_raw"):
            (lap_dir / name).mkdir(parents=True, exist_ok=True)

    def finish_lap(self, lap: int, mode: str) -> None:
        if lap < self.current_lap:
            print(f"[Lap] duplicate notification ignored: lap={lap}")
            return
        if lap != self.current_lap:
            raise ValueError(
                f"Expected lap-end for lap={self.current_lap}, received lap={lap}."
            )
        print(f"[Lap] complete lap={lap} mode={mode}")
        self.current_lap += 1
        self._select_lap(self.current_lap)


class LapDepthEvaluator(FairDepthEvaluator):
    def evaluate_rows(self, rows) -> None:
        super().evaluate_rows(rows)
        for row in rows:
            lap_dir = Path(row["image_path"]).parent.parent
            filename = Path(row["depth_path"]).name
            for name in ("depth_pred", "depth_pred_raw"):
                source = self.config.output_dir / name / filename
                if source.exists():
                    source.replace(lap_dir / name / filename)


class LapEndHandler(BaseHTTPRequestHandler):
    events: queue.SimpleQueue[tuple[int, str]]

    def do_POST(self) -> None:
        if self.path != LAP_END_PATH:
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("Content-Length must be between 1 and 1024.")
            payload = json.loads(self.rfile.read(length))
            lap = payload["lap"]
            mode = payload["mode"]
            if type(lap) is not int or lap < 1:
                raise ValueError("lap must be a positive integer.")
            if not isinstance(mode, str) or not mode.strip():
                raise ValueError("mode must be a non-empty string.")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._respond(400, {"error": str(error)})
            return
        self.events.put((lap, mode))
        self._respond(200, {"status": "accepted", "lap": lap})

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def build_experiment(args):
    import hardware.sensor as sensor

    sensor.RGBD_WIDTH = 640
    sensor.RGBD_HEIGHT = 480
    sensor.RGBD_FPS = 15
    experiment = build_control_experiment(args)
    experiment.logger = LapCaptureLogger(experiment.config.output_dir)
    experiment.evaluator = LapDepthEvaluator(
        experiment.predictor,
        experiment.config,
        args.evaluation_precision,
    )
    return experiment


def apply_lap_events(
    events: queue.SimpleQueue[tuple[int, str]], logger: LapCaptureLogger
) -> None:
    while True:
        try:
            lap, mode = events.get_nowait()
        except queue.Empty:
            return
        logger.finish_lap(lap, mode)


def main(argv: Sequence[str] | None = None) -> int:
    experiment = None
    server = None
    exit_code = 0
    events: queue.SimpleQueue[tuple[int, str]] = queue.SimpleQueue()
    try:
        args = parse_args(argv)
        LapEndHandler.events = events
        server = ThreadingHTTPServer(LAP_END_ADDRESS, LapEndHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        experiment = build_experiment(args)
        print(
            f"[Start] {METHOD_NAME}; 640x480@15; "
            f"lap-end=http://{LAP_END_ADDRESS[0]}:{LAP_END_ADDRESS[1]}{LAP_END_PATH}; "
            f"capture rounds={args.max_rounds}; press Ctrl-C to stop early."
        )
        while experiment.round_index < args.max_rounds:
            apply_lap_events(events, experiment.logger)
            started = time.monotonic()
            experiment.run_round()
            apply_lap_events(events, experiment.logger)
            remaining = args.round_interval_ms / 1000.0 - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[Stop] interrupted; finalizing captured frames.")
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        print(f"\n[ERROR] {error}")
        exit_code = 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if experiment is not None:
            try:
                apply_lap_events(events, experiment.logger)
                experiment.finalize()
            except (OSError, RuntimeError, ValueError) as error:
                print(f"[ERROR] result finalization failed: {error}")
                exit_code = 1
            try:
                experiment.capture_runner.camera.close()
                experiment.capture_runner.context_provider.close()
            except Exception as error:
                print(f"[ERROR] shutdown failed: {error}")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
