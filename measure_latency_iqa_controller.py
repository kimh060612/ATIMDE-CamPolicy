#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import orbbec_iqa_control as source


LATENCY_CSV_FIELDS = (
    "frame_index",
    "timestamp_ns",
    "operation",
    "exposure_ms",
    "gain",
    "camera_switched",
    "sensor_settle_ms",
    "camera_apply_latency_ms",
    "camera_switching_latency_ms",
    "capture_total_ms",
    "iqa_total_ms",
    "iqa_gpu_ms",
    "iqa_compute_gpu_ms",
    "controller_next_setting_ms",
    "controller_observe_ms",
    "controller_computation_ms",
)


class LatencyLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "iqa_controller_latency.csv"
        self.rows: list[dict[str, Any]] = []

    def record(self, values: dict[str, Any]) -> None:
        self.rows.append({field: values.get(field, "") for field in LATENCY_CSV_FIELDS})

    def write(self) -> Path:
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LATENCY_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)
        return self.path


class GPUNoiseAwareIQA:
    """GPU implementation of the original gradient + entropy - noise metric."""

    def __init__(self, device_name: str) -> None:
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPU IQA requires an available CUDA device.")
        self.sobel_x = torch.tensor(
            ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)),
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        self.sobel_y = self.sobel_x.transpose(-1, -2).contiguous()
        self.laplacian = torch.tensor(
            ((1, -2, 1), (-2, 4, -2), (1, -2, 1)),
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        self.gpu_ms: float | None = None
        self.compute_gpu_ms: float | None = None

    @staticmethod
    def _convolve(image: torch.Tensor, kernel: torch.Tensor, mode: str) -> torch.Tensor:
        padded = F.pad(image[None, None], (1, 1, 1, 1), mode=mode)
        return F.conv2d(padded, kernel)[0, 0]

    def _noise_level(self, channel: torch.Tensor) -> torch.Tensor:
        grad_x = self._convolve(channel, self.sobel_x, "reflect")
        grad_y = self._convolve(channel, self.sobel_y, "reflect")
        magnitude = torch.sqrt(grad_x.square() + grad_y.square())
        flat = magnitude.flatten()
        threshold = flat.kthvalue(int(0.10 * (flat.numel() - 1)) + 1).values
        reliable = (magnitude <= threshold) & (channel >= 15.0) & (channel <= 235.0)
        laplacian = self._convolve(channel, self.laplacian, "constant").abs()
        count = reliable.sum()
        fallback = count < channel.numel() * 0.0001
        absolute_sum = torch.where(fallback, laplacian.sum(), laplacian[reliable].sum())
        fallback_count = max((channel.shape[0] - 2) * (channel.shape[1] - 2), 1)
        denominator = torch.where(
            fallback, count.new_tensor(fallback_count), count
        ).float()
        return math.sqrt(math.pi / 2.0) * absolute_sum / (6.0 * denominator)

    def __call__(
        self, image_bgr: np.ndarray, resize_factor: float = 1.0
    ) -> source.IQAResult:
        if image_bgr.ndim not in (2, 3) or image_bgr.size == 0:
            raise ValueError(
                f"Expected a non-empty grayscale/BGR image, got {image_bgr.shape}."
            )
        if not math.isfinite(resize_factor) or resize_factor <= 0:
            raise ValueError("resize_factor must be finite and positive.")
        if image_bgr.ndim == 3 and image_bgr.shape[2] != 3:
            raise ValueError(f"Expected one or three channels, got {image_bgr.shape}.")

        self.gpu_ms = self.compute_gpu_ms = None
        gpu_started = torch.cuda.Event(enable_timing=True)
        transfer_finished = torch.cuda.Event(enable_timing=True)
        compute_finished = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        stream = torch.cuda.current_stream(self.device)
        gpu_started.record(stream)
        image = torch.from_numpy(np.ascontiguousarray(image_bgr)).to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        if image.ndim == 2:
            image = image.unsqueeze(-1)
        image = image.permute(2, 0, 1)
        transfer_finished.record(stream)

        if resize_factor != 1.0:
            height = max(1, round(image.shape[1] * resize_factor))
            width = max(1, round(image.shape[2] * resize_factor))
            if resize_factor < 1.0:
                image = F.interpolate(
                    image.unsqueeze(0), size=(height, width), mode="area"
                )[0]
            else:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]

        if image.shape[0] == 1:
            gray = image[0]
            channels = (gray,)
        else:
            blue, green, red = image
            gray = torch.round(0.114 * blue + 0.587 * green + 0.299 * red).clamp(0, 255)
            channels = (blue, green, red)

        grad_x = self._convolve(gray, self.sobel_x, "reflect")
        grad_y = self._convolve(gray, self.sobel_y, "reflect")
        normalized = torch.sqrt(grad_x.square() + grad_y.square()) / math.sqrt(
            2.0 * 16.0 * 255.0**2
        )
        gamma, lambda_value = 0.06, 1_000.0
        mapped = torch.where(
            normalized >= gamma,
            torch.log(lambda_value * (normalized - gamma) + 1.0)
            / math.log(lambda_value * (1.0 - gamma) + 1.0),
            0.0,
        )
        grid_size = min(10, mapped.shape[0], mapped.shape[1])
        grid_means = F.adaptive_avg_pool2d(
            mapped[None, None], (grid_size, grid_size)
        ).flatten()
        spatial_std = (
            grid_means.std(unbiased=True)
            if grid_means.numel() > 1
            else grid_means.new_zeros(())
        )
        gradient = torch.where(
            spatial_std > 1e-12,
            grid_means.mean() / spatial_std,
            spatial_std.new_zeros(()),
        )

        histogram = torch.bincount(gray.to(torch.int64).flatten(), minlength=256)
        probabilities = histogram[histogram > 0].float() / gray.numel()
        entropy = -0.125 * (probabilities * torch.log2(probabilities)).sum()
        channel_noise = tuple(self._noise_level(channel) for channel in channels)
        noise = (
            (channel_noise[0] + 2.0 * channel_noise[1] + channel_noise[2]) / 4.0
            if len(channel_noise) == 3
            else channel_noise[0]
        )
        score = 0.8 * gradient + 0.6 * entropy - 0.4 * noise
        values = torch.stack((score, gradient, entropy, noise, gray.mean()))

        compute_finished.record(stream)
        torch.cuda.synchronize(self.device)
        self.gpu_ms = gpu_started.elapsed_time(compute_finished)
        self.compute_gpu_ms = transfer_finished.elapsed_time(compute_finished)
        score_value, gradient_value, entropy_value, noise_value, mean_value = (
            float(value) for value in values.cpu()
        )
        if not all(
            math.isfinite(value)
            for value in (score_value, gradient_value, entropy_value, noise_value)
        ):
            raise ValueError("The GPU IQA metric produced a non-finite value.")
        return source.IQAResult(
            score_value,
            gradient_value,
            entropy_value,
            noise_value,
            mean_value,
        )


def install_instrumentation(logger: LatencyLogger, gpu_iqa: GPUNoiseAwareIQA) -> None:
    base_controller = source.NoiseAwareNelderMead
    base_capture_runner = source.CaptureRunner
    save_capture = source._save_capture
    previous_setting: source.ControlSetting | None = None

    class TimedController(base_controller):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.next_setting_ms: float | None = None
            self.observe_ms: float | None = None

        def next_setting(self):
            started = time.perf_counter()
            try:
                return super().next_setting()
            finally:
                self.next_setting_ms = (time.perf_counter() - started) * 1000.0

        def observe(self, result):
            started = time.perf_counter()
            try:
                return super().observe(result)
            finally:
                self.observe_ms = (time.perf_counter() - started) * 1000.0

    class TimedCaptureRunner(base_capture_runner):
        last_capture_ms: float | None = None

        def capture(self, *args: Any, **kwargs: Any):
            started = time.perf_counter()
            try:
                return super().capture(*args, **kwargs)
            finally:
                TimedCaptureRunner.last_capture_ms = (
                    time.perf_counter() - started
                ) * 1000.0

    def timed_save_capture(
        output_dir,
        frame,
        setting,
        operation,
        iqa,
        iqa_ms,
        controller,
        mode,
    ):
        nonlocal previous_setting
        row = save_capture(
            output_dir,
            frame,
            setting,
            operation,
            iqa,
            iqa_ms,
            controller,
            mode,
        )
        switched = previous_setting is not None and setting != previous_setting
        controller_times = [
            value
            for value in (controller.next_setting_ms, controller.observe_ms)
            if value is not None
        ]
        logger.record(
            {
                "frame_index": frame.capture_index,
                "timestamp_ns": frame.timestamp_ns,
                "operation": operation,
                "exposure_ms": setting.exposure_ms,
                "gain": setting.gain,
                "camera_switched": int(switched),
                "sensor_settle_ms": frame.sensor_settle_ms,
                "camera_apply_latency_ms": frame.camera_parameter_ms,
                "camera_switching_latency_ms": (
                    frame.camera_parameter_ms if switched else None
                ),
                "capture_total_ms": TimedCaptureRunner.last_capture_ms,
                "iqa_total_ms": iqa_ms,
                "iqa_gpu_ms": gpu_iqa.gpu_ms,
                "iqa_compute_gpu_ms": gpu_iqa.compute_gpu_ms,
                "controller_next_setting_ms": controller.next_setting_ms,
                "controller_observe_ms": controller.observe_ms,
                "controller_computation_ms": (
                    sum(controller_times) if controller_times else None
                ),
            }
        )
        previous_setting = setting
        return row

    source.noise_aware_iqa = gpu_iqa
    source.NoiseAwareNelderMead = TimedController
    source.CaptureRunner = TimedCaptureRunner
    source._save_capture = timed_save_capture


def main() -> int:
    args = source._parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = LatencyLogger(output_dir)
    latency_write_failed = False
    try:
        install_instrumentation(logger, GPUNoiseAwareIQA(args.depth_device))
        report, captured, evaluated = source._run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    finally:
        try:
            print(f"[Latency] wrote {logger.write()}")
        except OSError as error:
            print(f"[ERROR] latency CSV write failed: {error}", file=sys.stderr)
            latency_write_failed = True
    print(f"[Done] captured={captured} evaluated={evaluated} report={report}")
    return (
        0
        if not latency_write_failed
        and captured == args.num_frames
        and evaluated == captured
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
