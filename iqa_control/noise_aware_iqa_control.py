"""Noise-aware image quality and incremental Nelder-Mead exposure control.

This is a NumPy/OpenCV port of the metric and two-parameter optimizer from
Shin et al., "Camera Exposure Control for Robust Robot Vision with
Noise-Aware Image Quality Assessment" (IROS 2019).  The controller maximizes
the paper's image-quality score by minimizing its negative value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class IQAResult:
    score: float
    gradient: float
    entropy: float
    noise: float
    mean_intensity: float


@dataclass(frozen=True)
class ControlSetting:
    exposure_ms: int
    gain: int


@dataclass(frozen=True)
class _Vertex:
    point: np.ndarray
    objective: float


def _noise_level(channel: np.ndarray) -> float:
    image = channel.astype(np.float64, copy=False)
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    flat = magnitude.reshape(-1)
    threshold = np.partition(flat, int(0.10 * (flat.size - 1)))[
        int(0.10 * (flat.size - 1))
    ]
    reliable = (
        (magnitude <= threshold)
        & (image >= 15.0)
        & (image <= 235.0)
    )
    kernel = np.array(((1, -2, 1), (-2, 4, -2), (1, -2, 1)), np.float64)
    laplacian = cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)
    count = int(reliable.sum())
    height, width = image.shape
    if count < height * width * 0.0001:
        count = max((height - 2) * (width - 2), 1)
        absolute_sum = float(np.abs(laplacian).sum())
    else:
        absolute_sum = float(np.abs(laplacian[reliable]).sum())
    return math.sqrt(math.pi / 2.0) * absolute_sum / (6.0 * count)


def noise_aware_iqa(image_bgr: np.ndarray, resize_factor: float = 1.0) -> IQAResult:
    """Evaluate the paper's gradient + entropy - noise image-quality metric."""
    if image_bgr.ndim not in (2, 3) or image_bgr.size == 0:
        raise ValueError(f"Expected a non-empty grayscale/BGR image, got {image_bgr.shape}.")
    if not math.isfinite(resize_factor) or resize_factor <= 0:
        raise ValueError("resize_factor must be finite and positive.")

    image = image_bgr
    if resize_factor != 1.0:
        image = cv2.resize(
            image_bgr,
            None,
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_AREA if resize_factor < 1.0 else cv2.INTER_LINEAR,
        )
    if image.ndim == 2:
        gray = image
        channels = (image,)
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = cv2.split(image)
    else:
        raise ValueError(f"Expected one or three channels, got {image.shape}.")

    gray_float = gray.astype(np.float64, copy=False)
    grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    normalized = cv2.magnitude(grad_x, grad_y) / math.sqrt(
        16.0 * 255.0**2 + 16.0 * 255.0**2
    )
    gamma, lambda_value = 0.06, 1_000.0
    mapped = np.zeros_like(normalized)
    selected = normalized >= gamma
    mapped[selected] = np.log(lambda_value * (normalized[selected] - gamma) + 1.0)
    mapped[selected] /= math.log(lambda_value * (1.0 - gamma) + 1.0)

    grid_size = min(10, mapped.shape[0], mapped.shape[1])
    grid_means = np.array(
        [
            float(tile.mean())
            for row in np.array_split(mapped, grid_size, axis=0)
            for tile in np.array_split(row, grid_size, axis=1)
        ],
        dtype=np.float64,
    )
    spatial_std = float(grid_means.std(ddof=1)) if grid_means.size > 1 else 0.0
    gradient = float(grid_means.mean() / spatial_std) if spatial_std > 1e-12 else 0.0

    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    entropy = 0.125 * float(-(probabilities * np.log2(probabilities)).sum())

    channel_noise = [_noise_level(channel) for channel in channels]
    if len(channel_noise) == 3:
        # OpenCV is BGR; the paper weights the green Bayer channel twice.
        noise = (channel_noise[0] + 2.0 * channel_noise[1] + channel_noise[2]) / 4.0
    else:
        noise = channel_noise[0]

    # Paper/official MATLAB hyperparameters: alpha=.4, beta=.4, Kg=2.
    score = 0.5 * gradient + 0.5 * entropy - 0.4 * noise # 0.4 * 2.0, 0.6, 0.4
    values = (score, gradient, entropy, noise)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"The IQA metric produced a non-finite value: {values}.")
    return IQAResult(
        score=float(score),
        gradient=float(gradient),
        entropy=float(entropy),
        noise=float(noise),
        mean_intensity=float(gray_float.mean()),
    )


class NoiseAwareNelderMead:
    """One-image-at-a-time Nelder-Mead controller for exposure and gain.

    ``next_setting`` supplies the parameters for the next capture and
    ``observe`` feeds its IQA result back into the optimizer.  Quantization is
    applied before every capture so the simplex always represents settings
    that were actually sent to the camera.
    """

    def __init__(
        self,
        initial: ControlSetting,
        *,
        exposure_bounds: tuple[int, int],
        gain_bounds: tuple[int, int],
        exposure_step: int = 1,
        gain_step: int = 1,
        restart_interval: int = 60,
        simplex_tolerance: float = 0.02,
    ) -> None:
        if exposure_step < 1 or gain_step < 1 or restart_interval < 3:
            raise ValueError("Steps must be positive and restart_interval must be at least 3.")
        if exposure_bounds[0] > exposure_bounds[1] or gain_bounds[0] > gain_bounds[1]:
            raise ValueError("Invalid exposure/gain bounds.")
        if not math.isfinite(simplex_tolerance) or simplex_tolerance < 0:
            raise ValueError("simplex_tolerance must be finite and non-negative.")

        self._minimum = np.array((exposure_bounds[0], gain_bounds[0]), np.float64)
        self._maximum = np.array((exposure_bounds[1], gain_bounds[1]), np.float64)
        self._step = np.array((exposure_step, gain_step), np.float64)
        if np.any(self._maximum - self._minimum < self._step):
            raise ValueError("Each control range must contain at least two settings.")
        self.restart_interval = restart_interval
        self.simplex_tolerance = simplex_tolerance
        self.simplex: list[_Vertex] = []
        self.evaluation_count = 0
        self.iteration_count = 0
        self._evaluations_since_restart = 0
        self._phase = "initial_anchor"
        self._operation = "initial_anchor"
        self._pending = self._quantize(np.array((initial.exposure_ms, initial.gain)))
        self._initial_candidates: list[np.ndarray] = []
        self._reflection: _Vertex | None = None
        self._shrink_vertices: list[_Vertex] = []
        self._last_intensity = 127.5

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def best_setting(self) -> ControlSetting:
        point = min(self.simplex, key=lambda vertex: vertex.objective).point if self.simplex else self._pending
        return self._setting(point)

    @property
    def best_score(self) -> float | None:
        return -min(vertex.objective for vertex in self.simplex) if self.simplex else None

    def next_setting(self) -> ControlSetting:
        return self._setting(self._pending)

    def observe(self, result: IQAResult) -> None:
        """Consume the IQA result for the most recent ``next_setting``."""
        vertex = _Vertex(self._pending.copy(), -float(result.score))
        self.evaluation_count += 1
        self._evaluations_since_restart += 1
        self._last_intensity = result.mean_intensity

        if self._phase == "initial_anchor":
            self.simplex = [vertex]
            self._initial_candidates = self._make_initial_candidates(
                vertex.point, result.mean_intensity
            )
            self._schedule(self._initial_candidates[0], "initial_exposure", "initial_1")
        elif self._phase == "initial_1":
            self.simplex.append(vertex)
            self._schedule(self._initial_candidates[1], "initial_gain", "initial_2")
        elif self._phase == "initial_2":
            self.simplex.append(vertex)
            self._complete_iteration(count_iteration=False)
        elif self._phase == "reflect":
            self._handle_reflection(vertex)
        elif self._phase == "expand":
            assert self._reflection is not None
            self.simplex[-1] = vertex if vertex.objective < self._reflection.objective else self._reflection
            self._complete_iteration()
        elif self._phase == "contract_out":
            assert self._reflection is not None
            if vertex.objective <= self._reflection.objective:
                self.simplex[-1] = vertex
                self._complete_iteration()
            else:
                self._begin_shrink()
        elif self._phase == "contract_in":
            if vertex.objective < self.simplex[-1].objective:
                self.simplex[-1] = vertex
                self._complete_iteration()
            else:
                self._begin_shrink()
        elif self._phase == "shrink_1":
            self._shrink_vertices = [self.simplex[0], vertex]
            second = self._quantize(
                self.simplex[0].point + 0.5 * (self.simplex[2].point - self.simplex[0].point)
            )
            self._schedule(second, "shrink", "shrink_2")
        elif self._phase == "shrink_2":
            self.simplex = self._shrink_vertices + [vertex]
            self._complete_iteration()
        else:  # pragma: no cover - protects persistent hardware runs
            raise RuntimeError(f"Unknown Nelder-Mead phase: {self._phase}")

    def _handle_reflection(self, reflected: _Vertex) -> None:
        self.simplex.sort(key=lambda vertex: vertex.objective)
        best, second, worst = self.simplex
        centroid = 0.5 * (best.point + second.point)
        if best.objective <= reflected.objective < second.objective:
            self.simplex[-1] = reflected
            self._complete_iteration()
        elif reflected.objective < best.objective:
            self._reflection = reflected
            expanded = self._quantize(centroid + 2.0 * (centroid - worst.point))
            self._schedule(expanded, "expand", "expand")
        elif second.objective <= reflected.objective < worst.objective:
            self._reflection = reflected
            contracted = self._quantize(centroid + 0.5 * (centroid - worst.point))
            self._schedule(contracted, "contract_out", "contract_out")
        else:
            contracted = self._quantize(centroid + 0.5 * (worst.point - centroid))
            self._schedule(contracted, "contract_in", "contract_in")

    def _begin_shrink(self) -> None:
        self.simplex.sort(key=lambda vertex: vertex.objective)
        first = self._quantize(
            self.simplex[0].point + 0.5 * (self.simplex[1].point - self.simplex[0].point)
        )
        self._schedule(first, "shrink", "shrink_1")

    def _complete_iteration(self, *, count_iteration: bool = True) -> None:
        self.simplex.sort(key=lambda vertex: vertex.objective)
        if count_iteration:
            self.iteration_count += 1
        ranges = self._maximum - self._minimum
        diameter = max(
            float(np.linalg.norm((vertex.point - self.simplex[0].point) / ranges))
            for vertex in self.simplex[1:]
        )
        if (
            self._evaluations_since_restart >= self.restart_interval
            or diameter <= self.simplex_tolerance
        ):
            anchor = self.simplex[0].point.copy()
            self.simplex = []
            self._evaluations_since_restart = 0
            self._schedule(anchor, "restart_anchor", "initial_anchor")
            return

        centroid = 0.5 * (self.simplex[0].point + self.simplex[1].point)
        reflected = self._quantize(2.0 * centroid - self.simplex[2].point)
        self._schedule(reflected, "reflect", "reflect")

    def _make_initial_candidates(
        self, anchor: np.ndarray, mean_intensity: float
    ) -> list[np.ndarray]:
        intensity = float(np.clip(mean_intensity / 255.0, 0.0, 1.0))
        h = -intensity / 1.7 if intensity >= 0.5 else 1.7 * (1.0 - intensity)
        candidates: list[np.ndarray] = []
        for axis in range(2):
            point = anchor.copy()
            point[axis] *= 1.0 + h
            point = self._quantize(point)
            if np.array_equal(point, anchor):
                direction = 1.0 if h >= 0 else -1.0
                point[axis] = anchor[axis] + direction * self._step[axis]
                point = self._quantize(point)
                if np.array_equal(point, anchor):
                    point[axis] = anchor[axis] - direction * self._step[axis]
                    point = self._quantize(point)
            candidates.append(point)
        return candidates

    def _schedule(self, point: np.ndarray, operation: str, phase: str) -> None:
        self._pending = self._quantize(point)
        self._operation = operation
        self._phase = phase

    def _quantize(self, point: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(point, dtype=np.float64), self._minimum, self._maximum)
        quantized = self._minimum + np.rint((clipped - self._minimum) / self._step) * self._step
        return np.clip(quantized, self._minimum, self._maximum)

    @staticmethod
    def _setting(point: np.ndarray) -> ControlSetting:
        return ControlSetting(int(round(point[0])), int(round(point[1])))
