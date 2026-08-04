from __future__ import annotations

import math
import time
from typing import Protocol

import numpy as np

from hardware.utils import ContextKey, SensorCell

from .context import ContextProvider
from .types import CapturePair, CapturedFrame


class Camera(Protocol):
    exposure_value_per_ms: float

    def apply_cell(self, cell: SensorCell) -> tuple[int, int | None, int | None]: ...
    def capture_rgbd(self) -> tuple[np.ndarray, np.ndarray]: ...
    def close(self) -> None: ...


class CaptureRunner:
    def __init__(self, camera: Camera, context_provider: ContextProvider, max_pair_gap_ms: float) -> None:
        if not math.isfinite(max_pair_gap_ms) or max_pair_gap_ms <= 0:
            raise ValueError("Maximum pair capture gap must be positive.")
        self.camera = camera
        self.context_provider = context_provider
        self.max_pair_gap_ms = max_pair_gap_ms
        self.next_capture_index = 0

    def context_state(self) -> tuple[ContextKey, bool]:
        context = self.context_provider.get()
        return context, bool(getattr(self.context_provider, "is_stable", True))

    def capture(
        self,
        cell: SensorCell,
        latched_context: ContextKey,
        role: str,
        round_index: int,
        *,
        apply_cell: bool = True,
    ) -> CapturedFrame:
        started = time.perf_counter()
        if apply_cell:
            requested, actual_exposure, actual_gain = self.camera.apply_cell(cell)
        else:
            requested = round(cell.exposure_ms * self.camera.exposure_value_per_ms)
            actual_exposure = actual_gain = None
        parameter_ms = (time.perf_counter() - started) * 1000.0
        image, depth = self.camera.capture_rgbd()
        timestamp_ns = time.time_ns()
        capture_context, stable = self.context_state()
        exposure_us = cell.exposure_ms * 1000.0
        if actual_exposure is not None:
            exposure_us = actual_exposure / self.camera.exposure_value_per_ms * 1000.0
        frame = CapturedFrame(
            round_index=round_index,
            capture_index=self.next_capture_index,
            role=role,
            cell=cell,
            latched_context=latched_context,
            capture_context=capture_context,
            context_stable=stable,
            timestamp_ns=timestamp_ns,
            image=image,
            depth_m=depth,
            exposure_us=float(exposure_us),
            requested_exposure_raw=int(requested),
            actual_exposure_raw=actual_exposure,
            actual_gain=actual_gain,
            camera_parameter_ms=parameter_ms,
            freshness_token=getattr(self.camera, "frame_sequence", None),
        )
        self.next_capture_index += 1
        return frame

    def capture_pair(
        self,
        current: SensorCell,
        challenger: SensorCell,
        context: ContextKey,
        round_index: int,
    ) -> CapturePair:
        current_capture = self.capture(current, context, "initial", round_index)
        challenger_capture = self.capture(challenger, context, "challenger", round_index)
        gap_ms = (challenger_capture.timestamp_ns - current_capture.timestamp_ns) / 1_000_000.0
        fresh = (
            current_capture.freshness_token is None
            or challenger_capture.freshness_token is None
            or current_capture.freshness_token != challenger_capture.freshness_token
        )
        valid = (
            current_capture.context_stable
            and challenger_capture.context_stable
            and current_capture.capture_context == context
            and challenger_capture.capture_context == context
            and gap_ms <= self.max_pair_gap_ms
            and fresh
        )
        reason = "" if valid else (
            "stale_challenger" if not fresh else
            "capture_gap" if gap_ms > self.max_pair_gap_ms else
            "capture_context"
        )
        return CapturePair(current_capture, challenger_capture, gap_ms, valid, reason)
