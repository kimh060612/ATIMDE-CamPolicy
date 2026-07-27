from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from hardware.utils import (
    ALL_CELLS,
    CELL_BY_ID,
    GAIN_VALUES,
    LIGHT_STATE_COUNT,
    MOTION_STATE_COUNT,
    ContextKey,
    ContextState,
    QScore,
    SensorCell,
)

SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
NEAR_ZERO_STD = 1e-12


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT_TWO))


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / SQRT_TWO_PI


def probability_of_improvement(
    candidate_mean: float,
    candidate_variance: float,
    best_mean: float,
    best_variance: float,
    practical_margin: float,
) -> float:
    combined_variance = max(candidate_variance + best_variance, 0.0)
    difference = best_mean - practical_margin - candidate_mean
    standard_deviation = math.sqrt(combined_variance)
    if standard_deviation <= NEAR_ZERO_STD:
        if difference > 0.0:
            return 1.0
        if difference < 0.0:
            return 0.0
        return 0.5
    return normal_cdf(difference / standard_deviation)


def expected_improvement(
    candidate_mean: float,
    candidate_variance: float,
    best_mean: float,
    best_variance: float,
    practical_margin: float,
) -> float:
    combined_variance = max(candidate_variance + best_variance, 0.0)
    difference = best_mean - practical_margin - candidate_mean
    standard_deviation = math.sqrt(combined_variance)
    if standard_deviation <= NEAR_ZERO_STD:
        return max(difference, 0.0)
    z_value = difference / standard_deviation
    return (
        difference * normal_cdf(z_value)
        + standard_deviation * normal_pdf(z_value)
    )


@dataclass
class SafetyPolicy:
    max_exposure_ms_by_motion: tuple[int, ...] = (32, 32, 32, 32, 32)
    allowed_gains_by_light: tuple[tuple[int, ...], ...] = (
        GAIN_VALUES,
        GAIN_VALUES,
        GAIN_VALUES,
    )
    disabled_cells_by_context: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Optional[Path]) -> "SafetyPolicy":
        if path is None:
            return cls()

        payload = json.loads(path.read_text(encoding="utf-8"))
        max_exposure = tuple(
            int(value)
            for value in payload.get(
                "max_exposure_ms_by_motion", [32] * MOTION_STATE_COUNT
            )
        )
        if len(max_exposure) != MOTION_STATE_COUNT:
            raise ValueError(
                f"max_exposure_ms_by_motion must contain {MOTION_STATE_COUNT} values."
            )

        allowed_gains = tuple(
            tuple(int(value) for value in gains)
            for gains in payload.get(
                "allowed_gains_by_light",
                [list(GAIN_VALUES) for _ in range(LIGHT_STATE_COUNT)],
            )
        )
        if len(allowed_gains) != LIGHT_STATE_COUNT:
            raise ValueError(
                f"allowed_gains_by_light must contain {LIGHT_STATE_COUNT} lists."
            )

        disabled: dict[str, set[str]] = {}
        for context_key, entries in payload.get(
            "disabled_cells_by_context", {}
        ).items():
            cell_ids: set[str] = set()
            for entry in entries:
                if isinstance(entry, str):
                    cell_id = entry
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    cell_id = SensorCell(int(entry[0]), int(entry[1])).cell_id
                else:
                    raise ValueError(
                        f"Invalid disabled cell for context {context_key}: {entry!r}"
                    )
                if cell_id not in CELL_BY_ID:
                    raise ValueError(f"Unknown disabled cell: {cell_id}")
                cell_ids.add(cell_id)
            disabled[str(context_key)] = cell_ids

        return cls(
            max_exposure_ms_by_motion=max_exposure,
            allowed_gains_by_light=allowed_gains,
            disabled_cells_by_context=disabled,
        )

    def safe_cells(self, context: ContextKey) -> list[SensorCell]:
        context.validate()
        max_exposure = self.max_exposure_ms_by_motion[context.motion_state]
        allowed_gains = set(self.allowed_gains_by_light[context.light_state])
        disabled = self.disabled_cells_by_context.get(context.table_key, set())
        cells = [
            cell
            for cell in ALL_CELLS
            if cell.exposure_ms <= max_exposure
            and cell.gain in allowed_gains
            and cell.cell_id not in disabled
        ]
        if not cells:
            raise RuntimeError(f"No safe camera cell for context {context.table_key}.")
        return cells


@dataclass(frozen=True)
class PolicyDecision:
    action: Literal["use", "probe", "offload"]
    reason: str
    selected_cell: SensorCell
    score: QScore


@dataclass(frozen=True)
class BeliefUpdate:
    belief_mean: float
    belief_variance: float
    scene_epoch: int
    scene_change_z: Optional[float]
    scene_change_detected: bool


class ATIMDECameraProbingController:
    """Deterministic camera policy driven by predicted bias and uncertainty.

    A low predicted camera error is used directly. A confidently high error
    means that changing exposure/gain is worth probing. Ambiguous or uncertain
    frames are offloaded. One probing phase is allowed per capture round.
    """

    def __init__(
        self,
        safety_policy: SafetyPolicy,
        *,
        cells_per_probe: int = 4,
        min_initial_probes: int = 0,
        ema_alpha: float = 0.3,
        switch_margin: float = 0.01,
        low_bias_threshold: float = 0.12,
        high_bias_threshold: float = 0.20,
        probe_std_threshold: float = 0.08,
        ei_practical_margin: float = 0.01,
        ei_probe_cost: float = 0.001,
        confidence_z: float = 1.96,
        good_enough_probability: float = 0.95,
        scene_change_z_threshold: float = 3.0,
        scene_change_confirmations: int = 2,
        scene_variance_inflation: float = 4.0,
        scene_reset_variance: float = 0.01,
        unobserved_prior_mean: Optional[float] = None,
        unobserved_prior_variance: float = 0.04,
        belief_variance_floor: float = 1e-6,
        offload_command: Optional[str] = None,
    ) -> None:
        if cells_per_probe < 1:
            raise ValueError("cells_per_probe must be positive.")
        if min_initial_probes < 0:
            raise ValueError("min_initial_probes must be non-negative.")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        if switch_margin < 0.0:
            raise ValueError("switch_margin must be non-negative.")
        if not 0.0 <= low_bias_threshold < high_bias_threshold <= 0.5:
            raise ValueError(
                "Require 0 <= low_bias_threshold < high_bias_threshold <= 0.5."
            )
        if not 0.001 <= probe_std_threshold <= 0.5:
            raise ValueError("probe_std_threshold must be in [0.001, 0.5].")
        if ei_practical_margin < 0.0:
            raise ValueError("ei_practical_margin must be non-negative.")
        if ei_probe_cost < 0.0:
            raise ValueError("ei_probe_cost must be non-negative.")
        if confidence_z < 0.0:
            raise ValueError("confidence_z must be non-negative.")
        if not 0.0 < good_enough_probability <= 1.0:
            raise ValueError("good_enough_probability must be in (0, 1].")
        if scene_change_z_threshold <= 0.0:
            raise ValueError("scene_change_z_threshold must be positive.")
        if scene_change_confirmations < 1:
            raise ValueError("scene_change_confirmations must be positive.")
        if scene_variance_inflation < 1.0:
            raise ValueError("scene_variance_inflation must be at least 1.")
        if scene_reset_variance <= 0.0:
            raise ValueError("scene_reset_variance must be positive.")
        if unobserved_prior_mean is None:
            unobserved_prior_mean = high_bias_threshold
        if not math.isfinite(unobserved_prior_mean):
            raise ValueError("unobserved_prior_mean must be finite.")
        if unobserved_prior_variance <= 0.0:
            raise ValueError("unobserved_prior_variance must be positive.")
        if belief_variance_floor <= 0.0:
            raise ValueError("belief_variance_floor must be positive.")
        if not all(
            math.isfinite(value)
            for value in (
                ei_practical_margin,
                ei_probe_cost,
                confidence_z,
                good_enough_probability,
                scene_change_z_threshold,
                scene_variance_inflation,
                scene_reset_variance,
                unobserved_prior_variance,
                belief_variance_floor,
            )
        ):
            raise ValueError("Sequential probing parameters must be finite.")

        self.safety_policy = safety_policy
        self.cells_per_probe = cells_per_probe
        self.min_initial_probes = min_initial_probes
        self.ema_alpha = ema_alpha
        self.switch_margin = switch_margin
        self.low_bias_threshold = low_bias_threshold
        self.high_bias_threshold = high_bias_threshold
        self.probe_std_threshold = probe_std_threshold
        self.ei_practical_margin = ei_practical_margin
        self.ei_probe_cost = ei_probe_cost
        self.confidence_z = confidence_z
        self.good_enough_probability = good_enough_probability
        self.scene_change_z_threshold = scene_change_z_threshold
        self.scene_change_confirmations = scene_change_confirmations
        self.scene_variance_inflation = scene_variance_inflation
        self.scene_reset_variance = scene_reset_variance
        self.unobserved_prior_mean = float(unobserved_prior_mean)
        self.unobserved_prior_variance = unobserved_prior_variance
        self.belief_variance_floor = belief_variance_floor
        self.offload_command = offload_command
        self.context_states: dict[str, ContextState] = {}

    @staticmethod
    def _bias(score: QScore) -> float:
        if score.mu is None or not math.isfinite(score.mu):
            raise ValueError("QScore.mu must contain the finite camera bias.")
        return float(score.mu)

    @staticmethod
    def _std(score: QScore) -> float:
        if score.uncertainty is None or not math.isfinite(score.uncertainty):
            raise ValueError("QScore.uncertainty must contain the finite model std.")
        standard_deviation = float(score.uncertainty)
        if standard_deviation < 0.0:
            raise ValueError("QScore.uncertainty must be non-negative.")
        return standard_deviation

    def _state_for(self, context: ContextKey) -> ContextState:
        return self.context_states.setdefault(context.table_key, ContextState())

    def observe(
        self, context: ContextKey, cell: SensorCell, score: QScore
    ) -> BeliefUpdate:
        context.validate()
        state = self._state_for(context)
        safe_cells = self.safety_policy.safe_cells(context)
        if cell not in safe_cells:
            raise ValueError(
                f"Cannot observe unsafe cell {cell.cell_id} for {context.table_key}."
            )

        stats = state.cells[cell.cell_id]
        observation_variance = max(
            self._std(score) ** 2,
            self.belief_variance_floor,
        )
        camera_bias = self._bias(score)
        scene_change_z: Optional[float] = None
        scene_change_detected = False
        if stats.belief_mean is not None and stats.belief_variance is not None:
            innovation_variance = max(
                stats.belief_variance + observation_variance,
                self.belief_variance_floor,
            )
            scene_change_z = abs(camera_bias - stats.belief_mean) / math.sqrt(
                innovation_variance
            )
            if scene_change_z >= self.scene_change_z_threshold:
                state.scene_change_streak += 1
            else:
                state.scene_change_streak = 0

            if state.scene_change_streak >= self.scene_change_confirmations:
                state.scene_epoch += 1
                state.scene_change_streak = 0
                scene_change_detected = True
                for safe_cell in safe_cells:
                    safe_stats = state.cells[safe_cell.cell_id]
                    if (
                        safe_stats.belief_mean is not None
                        and safe_stats.belief_variance is not None
                    ):
                        safe_stats.belief_variance = max(
                            safe_stats.belief_variance
                            * self.scene_variance_inflation,
                            self.scene_reset_variance,
                        )

        stats.update(
            score,
            alpha=self.ema_alpha,
            cycle_index=state.committed_cycles,
            observation_variance=observation_variance,
            variance_floor=self.belief_variance_floor,
            scene_epoch=state.scene_epoch,
        )
        assert stats.belief_mean is not None
        assert stats.belief_variance is not None
        return BeliefUpdate(
            belief_mean=stats.belief_mean,
            belief_variance=stats.belief_variance,
            scene_epoch=state.scene_epoch,
            scene_change_z=scene_change_z,
            scene_change_detected=scene_change_detected,
        )

    def decide(self, cell: SensorCell, score: QScore) -> PolicyDecision:
        bias = self._bias(score)
        std = self._std(score)
        if bias <= self.low_bias_threshold:
            return PolicyDecision("use", "camera_bias_low", cell, score)
        if bias >= self.high_bias_threshold and std <= self.probe_std_threshold:
            return PolicyDecision(
                "probe", "camera_bias_high_uncertainty_low", cell, score
            )
        return PolicyDecision("offload", "camera_error_uncertain", cell, score)

    def cell_for_context(
        self, context: ContextKey, fallback: SensorCell
    ) -> SensorCell:
        safe_cells = self.safety_policy.safe_cells(context)
        state = self._state_for(context)
        if state.best_cell_id in {cell.cell_id for cell in safe_cells}:
            return CELL_BY_ID[state.best_cell_id]
        if fallback in safe_cells:
            return fallback
        return safe_cells[0]

    def belief(
        self, context: ContextKey, cell: SensorCell
    ) -> tuple[float, float]:
        state = self._state_for(context)
        stats = state.cells[cell.cell_id]
        if stats.belief_mean is None or stats.belief_variance is None:
            return self.unobserved_prior_mean, self.unobserved_prior_variance
        return stats.belief_mean, stats.belief_variance

    def posterior_best(self, context: ContextKey) -> SensorCell:
        safe_cells = self.safety_policy.safe_cells(context)
        return min(
            safe_cells,
            key=lambda cell: (
                self.belief(context, cell)[0],
                ALL_CELLS.index(cell),
            ),
        )

    def observed_in_current_scene(
        self, context: ContextKey, cell: SensorCell
    ) -> bool:
        state = self._state_for(context)
        return (
            state.cells[cell.cell_id].last_scene_epoch
            == state.scene_epoch
        )

    def minimum_initial_probes_remaining(self, context: ContextKey) -> int:
        safe_cells = self.safety_policy.safe_cells(context)
        # The initial frame already observes one cell. Require up to
        # min_initial_probes additional distinct safe-cell observations.
        target_observed_cells = min(
            len(safe_cells),
            self.min_initial_probes + 1,
        )
        observed_cells = sum(
            self.observed_in_current_scene(context, cell)
            for cell in safe_cells
        )
        return max(target_observed_cells - observed_cells, 0)

    def select_next_probe_cell(
        self,
        context: ContextKey,
        current_best: SensorCell,
        probed_this_round: set[str],
    ) -> tuple[Optional[SensorCell], float, float]:
        state = self._state_for(context)
        safe_cells = self.safety_policy.safe_cells(context)
        if current_best not in safe_cells:
            raise ValueError(
                f"Current best {current_best.cell_id} is unsafe for "
                f"{context.table_key}."
            )

        # After a scene change, validate the warm-start posterior best before
        # considering any other acquisition.
        if (
            state.cells[current_best.cell_id].last_scene_epoch
            != state.scene_epoch
        ):
            return current_best, 0.0, 0.0

        best_mean, best_variance = self.belief(context, current_best)
        selected_cell: Optional[SensorCell] = None
        selected_probability = 0.0
        selected_improvement = -1.0
        for candidate in safe_cells:
            if (
                candidate == current_best
                or candidate.cell_id in probed_this_round
            ):
                continue
            candidate_mean, candidate_variance = self.belief(context, candidate)
            candidate_probability = probability_of_improvement(
                candidate_mean,
                candidate_variance,
                best_mean,
                best_variance,
                self.ei_practical_margin,
            )
            candidate_improvement = expected_improvement(
                candidate_mean,
                candidate_variance,
                best_mean,
                best_variance,
                self.ei_practical_margin,
            )
            # safe_cells follows ALL_CELLS order, so strict comparison gives
            # deterministic tie-breaking without a separate round-robin state.
            if candidate_improvement > selected_improvement:
                selected_cell = candidate
                selected_probability = candidate_probability
                selected_improvement = candidate_improvement

        if selected_cell is None:
            return None, 0.0, 0.0
        return selected_cell, selected_probability, selected_improvement

    def best_good_enough_probability(
        self, context: ContextKey, best_cell: SensorCell
    ) -> float:
        best_mean, best_variance = self.belief(context, best_cell)
        standard_deviation = math.sqrt(max(best_variance, 0.0))
        difference = self.low_bias_threshold - best_mean
        if standard_deviation <= NEAR_ZERO_STD:
            if difference > 0.0:
                return 1.0
            if difference < 0.0:
                return 0.0
            return 0.5
        return normal_cdf(difference / standard_deviation)

    def best_is_confidently_separated(
        self, context: ContextKey, best_cell: SensorCell
    ) -> bool:
        if not self.observed_in_current_scene(context, best_cell):
            return False
        best_mean, best_variance = self.belief(context, best_cell)
        best_ucb = best_mean + self.confidence_z * math.sqrt(best_variance)
        competitors = [
            cell
            for cell in self.safety_policy.safe_cells(context)
            if cell != best_cell
        ]
        if not competitors:
            return True
        minimum_competitor_lcb = min(
            self.belief(context, cell)[0]
            - self.confidence_z * math.sqrt(self.belief(context, cell)[1])
            for cell in competitors
        )
        return (
            best_ucb + self.switch_margin
            < minimum_competitor_lcb
        )

    def score_for_cell(self, context: ContextKey, cell: SensorCell) -> QScore:
        state = self._state_for(context)
        stats = state.cells[cell.cell_id]
        belief_mean, belief_variance = self.belief(context, cell)
        q_value = (
            float(stats.last_q)
            if stats.last_q is not None and math.isfinite(stats.last_q)
            else belief_mean
        )
        return QScore(
            q=q_value,
            uncertainty=math.sqrt(belief_variance),
            mu=belief_mean,
        )

    def complete_probe(
        self,
        context: ContextKey,
        *,
        action: Literal["use", "offload"],
        reason: str,
    ) -> PolicyDecision:
        best_cell = self.posterior_best(context)
        if action == "use" and not self.observed_in_current_scene(
            context, best_cell
        ):
            raise RuntimeError(
                "Cannot use a cell that has not been observed in the current scene."
            )
        state = self._state_for(context)
        if self.observed_in_current_scene(context, best_cell):
            state.best_cell_id = best_cell.cell_id
        state.committed_cycles += 1
        return PolicyDecision(
            action,
            reason,
            best_cell,
            self.score_for_cell(context, best_cell),
        )

    def remember_used_cell(self, context: ContextKey, cell: SensorCell) -> None:
        if not self.observed_in_current_scene(context, cell):
            raise RuntimeError(
                "Cannot remember a cell that was not observed in the current scene."
            )
        state = self._state_for(context)
        state.best_cell_id = cell.cell_id
        state.committed_cycles += 1

    def invoke_offload(self, context: ContextKey, decision: PolicyDecision) -> None:
        print(
            f"[OFFLOAD] context={context.table_key} cell={decision.selected_cell.cell_id} "
            f"reason={decision.reason} q={decision.score.q:.6f}"
        )
        if not self.offload_command:
            return

        environment = os.environ.copy()
        environment.update(
            {
                "ATI_MOTION_STATE": str(context.motion_state),
                "ATI_LIGHT_STATE": str(context.light_state),
                "ATI_BEST_CELL": decision.selected_cell.cell_id,
                "ATI_OFFLOAD_REASON": decision.reason,
                "ATI_CAMERA_BIAS": str(self._bias(decision.score)),
                "ATI_CAMERA_STD": str(self._std(decision.score)),
                "ATI_Q_VALUE": str(decision.score.q),
            }
        )
        subprocess.run(
            shlex.split(self.offload_command),
            env=environment,
            check=False,
        )
