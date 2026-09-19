#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hardware.utils import SensorCell
from orbbec_ati_risk_bandit_bidirectional_exposure_sync import (
    METHOD_NAME,
    parse_args,
)
from orbbec_deterministic_probing_modelv1 import FairDepthEvaluator


LAP_END_ADDRESS = ("localhost", 3000)
LAP_END_PATH = "/lap-end"
REPORT_FIELDS = (
    "record_type",
    "lap",
    "mode",
    "frame_index",
    "frame_count",
    "timestamp_ns",
    "color_timestamp_us",
    "depth_timestamp_us",
    "capture_fps",
    "duration_sec",
    "cell_id",
    "exposure_ms",
    "gain",
    "requested_exposure_raw",
    "actual_exposure_raw",
    "actual_gain",
    "setting_effective",
    "image_path",
    "depth_path",
    "raw_pred_depth_path",
    "pred_depth_path",
    "abs_rel",
    "a1",
    "valid_depth_pixels",
    "evaluation_inference_ms",
    "evaluation_error",
)


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    lap: int
    timestamp_ns: int
    image: np.ndarray
    depth_m: np.ndarray
    cell: SensorCell
    requested_exposure_raw: int
    actual_exposure_raw: int | None
    actual_gain: int | None
    color_frame_number: int | None
    depth_frame_number: int | None
    color_timestamp_us: int | None
    depth_timestamp_us: int | None
    setting_effective: bool
    sensor_settle_ms: float


class FullRateLogger:
    """Persist every camera frame; control-round logging is intentionally a no-op."""

    _STOP = object()

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.lap_modes: dict[int, str] = {}
        # ponytail: unbounded queue preserves every frame; monitor disk backlog for very long runs.
        self._pending: queue.SimpleQueue[FramePacket | object] = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._closed = False
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()

    def finish_lap(self, lap: int, mode: str) -> None:
        self.lap_modes[lap] = mode
        print(f"[Lap] complete lap={lap} mode={mode}")

    def submit(self, packet: FramePacket) -> None:
        self._raise_if_failed()
        self._pending.put(packet)

    def record(self, *_args, **_kwargs) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending.put(self._STOP)
        self._writer.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Full-rate frame writer failed: {self._error}") from self._error

    def _write_loop(self) -> None:
        try:
            while True:
                packet = self._pending.get()
                if packet is self._STOP:
                    return
                self._save(packet)
        except BaseException as error:
            self._error = error

    def _save(self, packet: FramePacket) -> None:
        lap_dir = self.output_dir / f"lap_{packet.lap:04d}"
        for name in ("images", "depth_gt", "depth_pred", "depth_pred_raw"):
            (lap_dir / name).mkdir(parents=True, exist_ok=True)
        stem = f"frame_{packet.frame_index:07d}_{packet.timestamp_ns}"
        image_path = lap_dir / "images" / f"{stem}.png"
        depth_path = lap_dir / "depth_gt" / f"{stem}.npy"
        raw_pred_path = lap_dir / "depth_pred_raw" / f"{stem}.npy"
        pred_path = lap_dir / "depth_pred" / f"{stem}.npy"
        if not cv2.imwrite(str(image_path), packet.image):
            raise OSError(f"Failed to save RGB image: {image_path}")
        np.save(depth_path, np.ascontiguousarray(packet.depth_m, dtype=np.float32))
        self.rows.append(
            {
                "record_type": "capture",
                "lap": packet.lap,
                "mode": "",
                "round_index": packet.frame_index,
                "capture_index": packet.frame_index,
                "frame_index": packet.frame_index,
                "timestamp_ns": packet.timestamp_ns,
                "color_timestamp_us": packet.color_timestamp_us,
                "depth_timestamp_us": packet.depth_timestamp_us,
                "cell_id": packet.cell.cell_id,
                "exposure_ms": packet.cell.exposure_ms,
                "gain": packet.cell.gain,
                "requested_exposure_raw": packet.requested_exposure_raw,
                "actual_exposure_raw": packet.actual_exposure_raw,
                "actual_gain": packet.actual_gain,
                "setting_effective": int(packet.setting_effective),
                "sensor_settle_ms": packet.sensor_settle_ms,
                "output_delivered": 1,
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "raw_pred_depth_path": str(raw_pred_path),
                "pred_depth_path": str(pred_path),
                "abs_rel": "",
                "a1": "",
                "valid_depth_pixels": "",
                "evaluation_inference_ms": "",
                "evaluation_error": "",
            }
        )

    def write(self) -> Path:
        self.close()
        for lap in sorted({int(row["lap"]) for row in self.rows}):
            lap_rows = [row for row in self.rows if row["lap"] == lap]
            self._write_report(
                self.output_dir / f"lap_{lap:04d}" / "metrics.csv",
                lap_rows,
                lap,
            )
        path = self.output_dir / "lap_record_report.csv"
        self._write_report(path, self.rows, None)
        return path

    def _write_report(
        self, path: Path, rows: list[dict[str, Any]], lap: int | None
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        summary = {field: "" for field in REPORT_FIELDS}
        summary.update(
            record_type="summary",
            lap="" if lap is None else lap,
            mode="" if lap is None else self.lap_modes.get(lap, ""),
            frame_count=len(rows),
        )
        timestamps = [
            int(row["color_timestamp_us"])
            for row in rows
            if row.get("color_timestamp_us") is not None
        ]
        if len(timestamps) != len(rows):
            timestamps = [int(row["timestamp_ns"]) // 1000 for row in rows]
        if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
            duration = (timestamps[-1] - timestamps[0]) / 1_000_000.0
            summary["duration_sec"] = duration
            summary["capture_fps"] = (len(timestamps) - 1) / duration
        for field in ("abs_rel", "a1", "evaluation_inference_ms"):
            values = [float(row[field]) for row in rows if row.get(field, "") != ""]
            summary[field] = float(np.mean(values)) if values else ""

        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for row in rows:
                output = {field: row.get(field, "") for field in REPORT_FIELDS}
                output["mode"] = self.lap_modes.get(int(row["lap"]), "")
                writer.writerow(output)
            writer.writerow(summary)
        os.replace(temporary, path)


class LapCaptureLogger:
    """Directory selector kept for the DRL lap recorder."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.current_lap = 1
        self._select_lap()

    def _select_lap(self) -> None:
        lap_dir = self.output_dir / f"lap_{self.current_lap:04d}"
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
        self._select_lap()


def apply_lap_events(
    events: queue.SimpleQueue[tuple[int, str]], logger: LapCaptureLogger
) -> None:
    while True:
        try:
            lap, mode = events.get_nowait()
        except queue.Empty:
            return
        logger.finish_lap(lap, mode)


def apply_cell_without_frame_drain(
    camera: Any, cell: SensorCell
) -> tuple[int, int | None, int | None]:
    """Apply controls without consuming frames from the continuous stream."""
    from pyorbbecsdk import OBError, OBPropertyID

    exposure_raw = camera.exposure_to_raw(cell.exposure_ms)
    camera.device.set_int_property(
        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure_raw
    )
    camera.device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, cell.gain)

    try:
        actual_exposure = int(
            camera.device.get_int_property(
                OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
            )
        )
    except (AttributeError, OBError, TypeError, ValueError):
        actual_exposure = None
    try:
        actual_gain = int(
            camera.device.get_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT)
        )
    except (AttributeError, OBError, TypeError, ValueError):
        actual_gain = None

    camera._capture_safe = True
    camera._readback_matches = (
        actual_exposure == exposure_raw and actual_gain == cell.gain
    )
    camera._settled_frames = camera.settle_frames
    camera._pending_cell = cell
    camera._verified_active_cell = None
    camera._actual_exposure = actual_exposure
    camera._actual_gain = actual_gain
    return exposure_raw, actual_exposure, actual_gain


class FullRateCamera:
    """Capture every 15 FPS frame while control consumes only settled latest frames."""

    def __init__(
        self,
        camera_kwargs: dict[str, Any],
        logger: FullRateLogger,
        lap_events: queue.SimpleQueue[tuple[int, str]],
    ) -> None:
        self.camera_kwargs = camera_kwargs
        self.logger = logger
        self.lap_events = lap_events
        self.exposure_value_per_ms = float(camera_kwargs["exposure_value_per_ms"])
        self.frame_timeout_ms = int(camera_kwargs["frame_timeout_ms"])
        self._operation_timeout_sec = max(
            2.0,
            self.frame_timeout_ms
            / 1000.0
            * max(4, int(camera_kwargs["settle_frames"]) + 2),
        )
        self.color_frame_number = None
        self.depth_frame_number = None
        self.color_timestamp_us = None
        self.depth_timestamp_us = None
        self.setting_effective = False
        self.sensor_settle_ms = 0.0
        self._actual_exposure: int | None = None
        self._actual_gain: int | None = None
        self._active_cell: SensorCell | None = None
        self._readback_matches = False
        self._settle_remaining = 0
        self._current_lap = 1
        self._frame_index = 0
        self._commands: queue.Queue[tuple[SensorCell, queue.Queue[Any]]] = queue.Queue()
        self._controller_frames: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="orbbec-camera-owner",
            daemon=True,
        )
        self._camera_thread.start()
        self._ready.wait()
        self._raise_if_failed()

    def apply_cell(self, cell: SensorCell) -> tuple[int, int | None, int | None]:
        if self._closed:
            raise RuntimeError("Camera is closed.")
        self._raise_if_failed()
        reply: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._commands.put((cell, reply))
        deadline = time.monotonic() + self._operation_timeout_sec
        while time.monotonic() < deadline:
            try:
                result = reply.get(timeout=0.1)
            except queue.Empty:
                self._raise_if_failed()
                continue
            if isinstance(result, BaseException):
                raise result
            return result
        self._raise_if_failed()
        raise TimeoutError("Timed out waiting for the camera control command.")

    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + self._operation_timeout_sec
        while time.monotonic() < deadline:
            self._raise_if_failed()
            try:
                packet = self._controller_frames.get(timeout=0.1)
            except queue.Empty:
                continue
            self.color_frame_number = packet.color_frame_number
            self.depth_frame_number = packet.depth_frame_number
            self.color_timestamp_us = packet.color_timestamp_us
            self.depth_timestamp_us = packet.depth_timestamp_us
            self.setting_effective = packet.setting_effective
            self.sensor_settle_ms = packet.sensor_settle_ms
            return packet.image, packet.depth_m
        self._raise_if_failed()
        raise TimeoutError("Timed out waiting for a settled full-rate RGB-D frame.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._camera_thread.join(timeout=self._operation_timeout_sec)
        if self._camera_thread.is_alive():
            raise RuntimeError("Camera owner thread did not stop.")
        self._raise_if_failed()

    def _camera_loop(self) -> None:
        camera = None
        try:
            from hardware.sensor import OrbbecColorCamera

            camera = OrbbecColorCamera(**self.camera_kwargs)
            self._ready.set()
            while not self._stop.is_set():
                self._apply_lap_events()
                self._apply_control_commands(camera)
                image, depth_m = camera.capture_rgbd()
                cell = self._active_cell
                if cell is None:
                    continue
                effective = self._readback_matches and self._settle_remaining == 0
                if self._settle_remaining > 0:
                    self._settle_remaining -= 1
                packet = FramePacket(
                    frame_index=self._frame_index,
                    lap=self._current_lap,
                    timestamp_ns=time.time_ns(),
                    image=image,
                    depth_m=depth_m,
                    cell=cell,
                    requested_exposure_raw=camera.exposure_to_raw(cell.exposure_ms),
                    actual_exposure_raw=self._actual_exposure,
                    actual_gain=self._actual_gain,
                    color_frame_number=camera.color_frame_number,
                    depth_frame_number=camera.depth_frame_number,
                    color_timestamp_us=camera.color_timestamp_us,
                    depth_timestamp_us=camera.depth_timestamp_us,
                    setting_effective=effective,
                    sensor_settle_ms=self.sensor_settle_ms,
                )
                self._frame_index += 1
                self.logger.submit(packet)
                if packet.setting_effective:
                    self._offer_to_controller(packet)
            self._apply_lap_events()
        except BaseException as error:
            if self._error is None:
                self._error = error
            self._stop.set()
        finally:
            if camera is not None:
                try:
                    camera.close()
                except BaseException as error:
                    if self._error is None:
                        self._error = error
            self._ready.set()

    def _apply_control_commands(self, camera: Any) -> None:
        while True:
            try:
                cell, reply = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                requested = camera.exposure_to_raw(cell.exposure_ms)
                if cell == self._active_cell and self._readback_matches:
                    reply.put((requested, self._actual_exposure, self._actual_gain))
                    continue

                started = time.perf_counter()
                requested, actual_exposure, actual_gain = (
                    apply_cell_without_frame_drain(camera, cell)
                )
                self._active_cell = cell
                self._actual_exposure = actual_exposure
                self._actual_gain = actual_gain
                self._readback_matches = (
                    actual_exposure == requested and actual_gain == cell.gain
                )
                self._settle_remaining = camera.settle_frames
                self.sensor_settle_ms = (time.perf_counter() - started) * 1000.0
                self._clear_controller_frames()
                reply.put((requested, actual_exposure, actual_gain))
            except BaseException as error:
                reply.put(error)

    def _apply_lap_events(self) -> None:
        while True:
            try:
                lap, mode = self.lap_events.get_nowait()
            except queue.Empty:
                return
            if lap < self._current_lap:
                print(f"[Lap] duplicate notification ignored: lap={lap}")
                continue
            if lap != self._current_lap:
                raise ValueError(
                    f"Expected lap-end for lap={self._current_lap}, received lap={lap}."
                )
            self.logger.finish_lap(lap, mode)
            self._current_lap += 1

    def _offer_to_controller(self, packet: FramePacket) -> None:
        try:
            self._controller_frames.put_nowait(packet)
        except queue.Full:
            try:
                self._controller_frames.get_nowait()
            except queue.Empty:
                pass
            self._controller_frames.put_nowait(packet)

    def _clear_controller_frames(self) -> None:
        while True:
            try:
                self._controller_frames.get_nowait()
            except queue.Empty:
                return

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Full-rate camera capture failed: {self._error}") from self._error


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


def build_experiment(args, events):
    from ati_mde_control.bidirectional_exposure_guard import (
        BidirectionalExposureGuard,
        BidirectionalExposureGuardConfig,
    )
    from ati_mde_control.capture_runner import CaptureRunner
    from ati_mde_control.config import ExperimentConfig, SafetyPolicy
    from ati_mde_control.context import build_context_provider
    from ati_mde_control.predictor import CameraErrorPredictor
    from ati_mde_control.risk_bandit_bidirectional_exposure_sync_experiment import (
        RiskBanditBidirectionalExposureSyncExperiment,
    )
    from ati_mde_control.risk_bandit_policy import RiskBanditConfig
    from ati_mde_control.saturation_guard import SaturationGuardedRiskBanditPolicy
    import hardware.sensor as sensor

    sensor.RGBD_WIDTH = 640
    sensor.RGBD_HEIGHT = 480
    sensor.RGBD_FPS = 15

    config = ExperimentConfig.from_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("depth_pred_raw", "depth_pred"):
        (config.output_dir / name).mkdir(parents=True, exist_ok=True)

    context_provider = build_context_provider(args)
    predictor = CameraErrorPredictor(
        config.checkpoint_path,
        config.model_size,
        config.device,
        config.precision,
        config.q_uncertainty_weight,
        config.local_files_only,
    )
    policy = SaturationGuardedRiskBanditPolicy(
        RiskBanditConfig.from_args(args),
        SafetyPolicy.from_json(config.safety_path),
        config.default_cell,
    )
    guard = BidirectionalExposureGuard(
        BidirectionalExposureGuardConfig.from_args(args), policy.safe_fallback
    )
    logger = FullRateLogger(config.output_dir)
    try:
        camera = FullRateCamera(
            {
                "exposure_value_per_ms": args.exposure_value_per_ms,
                "settle_frames": args.settle_frames,
                "frame_timeout_ms": args.frame_timeout_ms,
                "warmup_frames": args.warmup_frames,
                "disable_awb": args.disable_awb,
                "strict_property_grid": not args.allow_unsupported_grid_values,
            },
            logger,
            events,
        )
    except BaseException:
        logger.close()
        context_provider.close()
        raise
    capture_runner = CaptureRunner(
        camera, context_provider, config.max_pair_capture_gap_ms
    )
    experiment = RiskBanditBidirectionalExposureSyncExperiment(
        config,
        capture_runner,
        predictor,
        policy,
        logger,
        LapDepthEvaluator(predictor, config, args.evaluation_precision),
        guard,
    )
    return experiment, camera, logger


def main(argv: Sequence[str] | None = None) -> int:
    experiment = None
    camera = None
    logger = None
    server = None
    exit_code = 0
    events: queue.SimpleQueue[tuple[int, str]] = queue.SimpleQueue()
    try:
        args = parse_args(argv)
        LapEndHandler.events = events
        server = ThreadingHTTPServer(LAP_END_ADDRESS, LapEndHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        experiment, camera, logger = build_experiment(args, events)
        print(
            f"[Start] {METHOD_NAME}; continuous capture=640x480@15; "
            f"lap-end=http://{LAP_END_ADDRESS[0]}:{LAP_END_ADDRESS[1]}{LAP_END_PATH}; "
            f"control rounds={args.max_rounds}; press Ctrl-C to stop early."
        )
        while experiment.round_index < args.max_rounds:
            started = time.monotonic()
            experiment.run_round()
            remaining = args.round_interval_ms / 1000.0 - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[Stop] interrupted; finishing queued frames and MDE evaluation.")
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        print(f"\n[ERROR] {error}")
        exit_code = 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if camera is not None:
            try:
                camera.close()
            except (OSError, RuntimeError, ValueError, TimeoutError) as error:
                print(f"[ERROR] camera shutdown failed: {error}")
                exit_code = 1
        if logger is not None:
            try:
                logger.close()
            except (OSError, RuntimeError, ValueError) as error:
                print(f"[ERROR] frame writer shutdown failed: {error}")
                exit_code = 1
        if experiment is not None:
            try:
                experiment.finalize()
            except (OSError, RuntimeError, ValueError) as error:
                print(f"[ERROR] result finalization failed: {error}")
                exit_code = 1
            try:
                experiment.capture_runner.context_provider.close()
            except Exception as error:
                print(f"[ERROR] context shutdown failed: {error}")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
