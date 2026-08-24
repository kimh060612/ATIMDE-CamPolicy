from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hardware.utils import ALL_CELLS, ContextKey, QScore, SensorCell

from .config import SafetyPolicy


METHOD_NAME = "Delay-Aware Risk-Sensitive Time-Varying Contextual GP Bandit"


@dataclass(frozen=True)
class RiskBanditConfig:
    window_size: int = 48
    exploration_beta: float = 1.0
    switch_penalty: float = 0.005
    exposure_length_scale: float = 0.35
    gain_length_scale: float = 0.35
    motion_cross_correlation: float = 0.20
    light_cross_correlation: float = 0.20
    temporal_scale_sec: float = 5.0
    observation_noise: float = 0.10
    target_scale_floor: float = 0.01
    jitter: float = 1e-6
    max_cholesky_attempts: int = 4

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("Bandit window size must be positive.")
        if self.max_cholesky_attempts < 1:
            raise ValueError("Cholesky attempt count must be positive.")
        positive = (
            self.exposure_length_scale,
            self.gain_length_scale,
            self.temporal_scale_sec,
            self.observation_noise,
            self.target_scale_floor,
            self.jitter,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Bandit scales, noise, target floor, and jitter must be finite and positive.")
        non_negative = (self.exploration_beta, self.switch_penalty)
        if any(not math.isfinite(value) or value < 0 for value in non_negative):
            raise ValueError("Bandit exploration beta and switch penalty must be finite and non-negative.")
        correlations = (
            self.motion_cross_correlation,
            self.light_cross_correlation,
        )
        if any(not math.isfinite(value) or not 0 <= value < 1 for value in correlations):
            raise ValueError("Bandit context cross-correlations must be in [0, 1).")

    @classmethod
    def from_args(cls, args) -> "RiskBanditConfig":
        return cls(
            window_size=args.bandit_window_size,
            exploration_beta=args.bandit_exploration_beta,
            switch_penalty=args.bandit_switch_penalty,
            exposure_length_scale=args.bandit_exposure_length_scale,
            gain_length_scale=args.bandit_gain_length_scale,
            motion_cross_correlation=args.bandit_motion_cross_correlation,
            light_cross_correlation=args.bandit_light_cross_correlation,
            temporal_scale_sec=args.bandit_temporal_scale_sec,
            observation_noise=args.bandit_observation_noise,
            target_scale_floor=args.bandit_target_scale_floor,
            jitter=args.bandit_jitter,
        )


@dataclass(frozen=True)
class RiskObservation:
    context: ContextKey
    cell: SensorCell
    timestamp_ns: int
    q: float


@dataclass(frozen=True)
class CandidatePosterior:
    cell: SensorCell
    mean_risk: float
    epistemic_std: float
    acquisition: float


@dataclass(frozen=True)
class RiskBanditDecision:
    selected_cell: SensorCell
    candidates: tuple[CandidatePosterior, ...]
    status: str


class GPNumericalError(RuntimeError):
    pass


class RiskBanditPolicy:
    """Small exact-GP policy for the delay-aware risk-bandit execution path."""

    def __init__(
        self,
        config: RiskBanditConfig,
        safety_policy: SafetyPolicy,
        default_cell: SensorCell,
    ) -> None:
        if default_cell not in ALL_CELLS:
            raise ValueError("The default cell must belong to hardware.utils.ALL_CELLS.")
        self.config = config
        self.safety_policy = safety_policy
        self.default_cell = default_cell
        self._history: deque[RiskObservation] = deque(maxlen=config.window_size)
        self.last_update_status = "not_observed"

        exposures = tuple(dict.fromkeys(cell.exposure_ms for cell in ALL_CELLS))
        gains = tuple(dict.fromkeys(cell.gain for cell in ALL_CELLS))
        self._exposure_feature = {
            value: index / max(1, len(exposures) - 1)
            for index, value in enumerate(exposures)
        }
        self._gain_feature = {
            value: index / max(1, len(gains) - 1)
            for index, value in enumerate(gains)
        }

    @property
    def history(self) -> tuple[RiskObservation, ...]:
        return tuple(self._history)

    def add_observation(
        self,
        context: ContextKey,
        cell: SensorCell,
        timestamp_ns: int,
        score: QScore,
    ) -> bool:
        context.validate()
        if cell not in ALL_CELLS:
            raise ValueError("Risk observations must use hardware.utils.ALL_CELLS.")
        try:
            values = (float(score.q), float(score.mu), float(score.uncertainty))
        except (TypeError, ValueError):
            self.last_update_status = "non_finite_score"
            return False
        if not all(math.isfinite(value) for value in values):
            self.last_update_status = "non_finite_score"
            return False
        self._history.append(RiskObservation(context, cell, int(timestamp_ns), values[0]))
        self.last_update_status = "updated"
        return True

    def safe_cells(self, context: ContextKey) -> tuple[SensorCell, ...]:
        return tuple(sorted(self.safety_policy.safe_cells(context), key=lambda cell: cell.cell_id))

    def safe_fallback(self, context: ContextKey) -> SensorCell:
        safe = self.safe_cells(context)
        default_e, default_g = self._action_feature(self.default_cell)
        return min(
            safe,
            key=lambda cell: (
                (self._action_feature(cell)[0] - default_e) ** 2
                + (self._action_feature(cell)[1] - default_g) ** 2,
                cell.cell_id,
            ),
        )

    def select_action(
        self,
        context: ContextKey,
        current_cell: SensorCell,
        timestamp_ns: int,
    ) -> RiskBanditDecision:
        safe = self.safe_cells(context)
        fallback = current_cell if current_cell in safe else self.safe_fallback(context)
        if not self._history:
            return RiskBanditDecision(fallback, (), "no_history")
        try:
            posterior = self.posterior_predictions(context, safe, timestamp_ns)
            candidates = tuple(
                CandidatePosterior(
                    cell=item.cell,
                    mean_risk=item.mean_risk,
                    epistemic_std=item.epistemic_std,
                    acquisition=(
                        item.mean_risk
                        - self.config.exploration_beta * item.epistemic_std
                        + self.config.switch_penalty * (item.cell != current_cell)
                    ),
                )
                for item in posterior
            )
            if not all(
                math.isfinite(value)
                for item in candidates
                for value in (item.mean_risk, item.epistemic_std, item.acquisition)
            ):
                raise GPNumericalError("GP posterior returned a non-finite value.")
            selected = min(
                candidates,
                key=lambda item: (
                    item.acquisition,
                    0 if item.cell == current_cell else 1,
                    item.cell.cell_id,
                ),
            ).cell
            return RiskBanditDecision(selected, candidates, "selected")
        except (GPNumericalError, np.linalg.LinAlgError, FloatingPointError):
            return RiskBanditDecision(fallback, (), "gp_numerical_fallback")

    def posterior_predictions(
        self,
        context: ContextKey,
        cells: Iterable[SensorCell],
        timestamp_ns: int,
    ) -> tuple[CandidatePosterior, ...]:
        observations = self.history
        if not observations:
            raise ValueError("A GP posterior requires at least one observation.")
        query_cells = tuple(cells)
        if not query_cells:
            return ()

        targets = np.asarray([item.q for item in observations], dtype=np.float64)
        target_center = float(np.mean(targets))
        target_scale = max(float(np.std(targets)), self.config.target_scale_floor)
        normalized_targets = (targets - target_center) / target_scale

        covariance = self._observation_kernel(observations)
        covariance.flat[:: len(observations) + 1] += self.config.observation_noise
        cholesky = self._cholesky(covariance)
        alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, normalized_targets))

        query_contexts = (context,) * len(query_cells)
        query_times = (int(timestamp_ns),) * len(query_cells)
        cross_covariance = self._cross_kernel(
            tuple(item.context for item in observations),
            tuple(item.cell for item in observations),
            tuple(item.timestamp_ns for item in observations),
            query_contexts,
            query_cells,
            query_times,
        )
        normalized_mean = cross_covariance.T @ alpha
        projected = np.linalg.solve(cholesky, cross_covariance)
        normalized_variance = np.maximum(0.0, 1.0 - np.sum(projected * projected, axis=0))
        means = target_center + target_scale * normalized_mean
        standard_deviations = target_scale * np.sqrt(normalized_variance)
        return tuple(
            CandidatePosterior(cell, float(mean), float(std), math.nan)
            for cell, mean, std in zip(query_cells, means, standard_deviations)
        )

    def _action_feature(self, cell: SensorCell) -> tuple[float, float]:
        try:
            return self._exposure_feature[cell.exposure_ms], self._gain_feature[cell.gain]
        except KeyError as error:
            raise ValueError(f"Cell is outside hardware.utils.ALL_CELLS: {cell}") from error

    def _observation_kernel(self, observations: tuple[RiskObservation, ...]) -> np.ndarray:
        return self._cross_kernel(
            tuple(item.context for item in observations),
            tuple(item.cell for item in observations),
            tuple(item.timestamp_ns for item in observations),
            tuple(item.context for item in observations),
            tuple(item.cell for item in observations),
            tuple(item.timestamp_ns for item in observations),
        )

    def _cross_kernel(
        self,
        left_contexts: tuple[ContextKey, ...],
        left_cells: tuple[SensorCell, ...],
        left_times_ns: tuple[int, ...],
        right_contexts: tuple[ContextKey, ...],
        right_cells: tuple[SensorCell, ...],
        right_times_ns: tuple[int, ...],
    ) -> np.ndarray:
        left_action = np.asarray([self._action_feature(cell) for cell in left_cells])
        right_action = np.asarray([self._action_feature(cell) for cell in right_cells])
        exposure_delta = (
            left_action[:, None, 0] - right_action[None, :, 0]
        ) / self.config.exposure_length_scale
        gain_delta = (
            left_action[:, None, 1] - right_action[None, :, 1]
        ) / self.config.gain_length_scale
        action_kernel = np.exp(-0.5 * (exposure_delta ** 2 + gain_delta ** 2))

        left_motion = np.asarray([item.motion_state for item in left_contexts])
        right_motion = np.asarray([item.motion_state for item in right_contexts])
        left_light = np.asarray([item.light_state for item in left_contexts])
        right_light = np.asarray([item.light_state for item in right_contexts])
        motion_kernel = np.where(
            left_motion[:, None] == right_motion[None, :],
            1.0,
            self.config.motion_cross_correlation,
        )
        light_kernel = np.where(
            left_light[:, None] == right_light[None, :],
            1.0,
            self.config.light_cross_correlation,
        )

        left_time = np.asarray(left_times_ns, dtype=np.int64)
        right_time = np.asarray(right_times_ns, dtype=np.int64)
        time_delta_sec = np.abs(
            left_time[:, None] - right_time[None, :]
        ).astype(np.float64) / 1_000_000_000.0
        time_kernel = np.exp(-time_delta_sec / self.config.temporal_scale_sec)
        return action_kernel * motion_kernel * light_kernel * time_kernel

    def _cholesky(self, covariance: np.ndarray) -> np.ndarray:
        identity = np.eye(covariance.shape[0], dtype=np.float64)
        jitter = self.config.jitter
        last_error: np.linalg.LinAlgError | None = None
        for _ in range(self.config.max_cholesky_attempts):
            try:
                return np.linalg.cholesky(covariance + jitter * identity)
            except np.linalg.LinAlgError as error:
                last_error = error
                jitter *= 10.0
        raise GPNumericalError(
            f"GP Cholesky failed after {self.config.max_cholesky_attempts} attempts."
        ) from last_error
