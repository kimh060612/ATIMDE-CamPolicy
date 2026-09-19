#!/usr/bin/env python3
from __future__ import annotations

import queue
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import orbbec_drl_control as control
from hardware.utils import SensorCell
from orbbec_ati_risk_bandit_lap_record import (
    LAP_END_ADDRESS,
    LAP_END_PATH,
    FramePacket,
    FullRateCamera,
    FullRateLogger,
    LapDepthEvaluator,
    LapEndHandler,
)


class DRLSensorCell(SensorCell):
    @property
    def cell_id(self) -> str:
        return f"E{float(self.exposure_ms):.3f}_G{self.gain:03d}"


class DRLFullRateCamera(FullRateCamera):
    exposure_range: dict[str, int] | None
    gain_range: dict[str, int] | None

    def apply_cell(self, cell: SensorCell) -> tuple[int, int | None, int | None]:
        return super().apply_cell(DRLSensorCell(cell.exposure_ms, cell.gain))

    def _camera_loop(self) -> None:
        camera = None
        try:
            from hardware.sensor import OrbbecColorCamera

            camera = OrbbecColorCamera(**self.camera_kwargs)
            self.exposure_range = camera.exposure_range
            self.gain_range = camera.gain_range
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


def run(args) -> tuple[Path, int, int]:
    import hardware.sensor as sensor

    sensor.RGBD_WIDTH = 640
    sensor.RGBD_HEIGHT = 480
    sensor.RGBD_FPS = 15

    events: queue.SimpleQueue[tuple[int, str]] = queue.SimpleQueue()
    server = ThreadingHTTPServer(LAP_END_ADDRESS, LapEndHandler)
    original_camera = control.OrbbecColorCamera
    lap_logger = None
    server_started = False
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="lap-end-server",
        daemon=True,
    )

    try:
        lap_logger = FullRateLogger(args.output_dir.resolve())

        def camera_factory(**camera_kwargs):
            return DRLFullRateCamera(camera_kwargs, lap_logger, events)

        LapEndHandler.events = events
        control.OrbbecColorCamera = camera_factory
        server_thread.start()
        server_started = True
        print(
            f"[Lap] continuous capture=640x480@15; "
            f"listening=http://{LAP_END_ADDRESS[0]}:{LAP_END_ADDRESS[1]}{LAP_END_PATH}"
        )
        return control.run(args)
    finally:
        control.OrbbecColorCamera = original_camera
        try:
            if server_started:
                server.shutdown()
            server.server_close()
            if server_started:
                server_thread.join()
        finally:
            if lap_logger is not None:
                try:
                    lap_logger.close()
                    config = SimpleNamespace(
                        device=args.depth_device,
                        evaluation_alignment=args.depth_alignment,
                        min_depth_m=args.min_depth_m,
                        max_depth_m=args.max_depth_m,
                        min_valid_depth_pixels=args.min_valid_depth_pixels,
                        local_files_only=args.depth_model_local_files_only,
                        output_dir=args.output_dir.resolve(),
                    )
                    LapDepthEvaluator(None, config, args.depth_precision).evaluate_rows(
                        lap_logger.rows
                    )
                    if lap_logger.rows and not any(
                        row["abs_rel"] != "" for row in lap_logger.rows
                    ):
                        raise RuntimeError(
                            "AbsRel/A1 evaluation failed for every full-rate frame."
                        )
                finally:
                    print(f"[Lap] wrote {lap_logger.write()}")


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
