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
PROBABILITY_EPSILON = 1e-12
GRID_DIVERSE_CELL_IDS = (
    "E04_G016",
    "E04_G128",
    "E32_G016",
    "E32_G128",
    "E08_G032",
    "E08_G064",
    "E16_G032",
    "E16_G064",
    "E04_G032",
    "E04_G064",
    "E32_G032",
    "E32_G064",
    "E08_G016",
    "E08_G128",
    "E16_G016",
    "E16_G128",
)
GRID_DIVERSE_RANK = {
    cell_id: rank for rank, cell_id in enumerate(GRID_DIVERSE_CELL_IDS)
}


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / SQRT_TWO))


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
    action: Literal["use", "offload"]
    reason: str
    selected_cell: SensorCell
    score: QScore


@dataclass(frozen=True)
class ObservationUpdate:
    probability_good: float
    success_score_before: float
    success_score_after: float


class ATIMDECameraProbingController:
    """Motion-light-conditioned controller that searches for a good-enough frame."""

    def __init__(
        self,
        safety_policy: SafetyPolicy,
        *,
        accept_threshold: float = 0.11,
        accept_probability: float = 0.90,
        required_bad_frames: int = 2,
        success_ema_alpha: float = 0.3,
        challenger_cooldown_rounds: int = 5,
        offload_command: Optional[str] = None,
    ) -> None:
        if not math.isfinite(accept_threshold):
            raise ValueError("accept_threshold must be finite.")
        if not 0.0 < accept_probability <= 1.0:
            raise ValueError("accept_probability must be in (0, 1].")
        if required_bad_frames < 1:
            raise ValueError("required_bad_frames must be positive.")
        if not 0.0 < success_ema_alpha <= 1.0:
            raise ValueError("success_ema_alpha must be in (0, 1].")
        if challenger_cooldown_rounds < 0:
            raise ValueError("challenger_cooldown_rounds must be non-negative.")

        self.safety_policy = safety_policy
        self.accept_threshold = accept_threshold
        self.accept_probability = accept_probability
        self.required_bad_frames = required_bad_frames
        self.success_ema_alpha = success_ema_alpha
        self.challenger_cooldown_rounds = challenger_cooldown_rounds
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
        context.validate()
        return self.context_states.setdefault(context.table_key, ContextState())

    def probability_good(self, score: QScore) -> float:
        return normal_cdf(
            (self.accept_threshold - self._bias(score))
            / max(self._std(score), PROBABILITY_EPSILON)
        )

    def is_acceptable(self, probability_good: float) -> bool:
        return probability_good >= self.accept_probability

    def observe(
        self, context: ContextKey, cell: SensorCell, score: QScore
    ) -> ObservationUpdate:
        if cell not in self.safety_policy.safe_cells(context):
            raise ValueError(
                f"Cannot observe unsafe cell {cell.cell_id} for {context.table_key}."
            )
        probability_good = self.probability_good(score)
        stats = self._state_for(context).cells[cell.cell_id]
        before = stats.success_score
        stats.update(probability_good, alpha=self.success_ema_alpha)
        return ObservationUpdate(probability_good, before, stats.success_score)

    def start_round(self, context: ContextKey) -> None:
        for stats in self._state_for(context).cells.values():
            stats.cooldown = max(stats.cooldown - 1, 0)

    def cell_for_context(
        self, context: ContextKey, fallback: SensorCell
    ) -> SensorCell:
        state = self._state_for(context)
        safe_cells = self.safety_policy.safe_cells(context)
        safe_ids = {cell.cell_id for cell in safe_cells}
        if state.active_cell_id in safe_ids:
            return CELL_BY_ID[state.active_cell_id]
        selected = fallback if fallback in safe_cells else safe_cells[0]
        state.active_cell_id = selected.cell_id
        return selected

    def bridge_cell(
        self, contexts: tuple[ContextKey, ...], fallback: SensorCell
    ) -> SensorCell:
        contexts = tuple(dict.fromkeys(contexts))
        if not contexts:
            raise ValueError("At least one context is required for a bridge cell.")
        common_cells = [
            cell
            for cell in ALL_CELLS
            if all(
                cell in self.safety_policy.safe_cells(context)
                for context in contexts
            )
        ]
        if not common_cells:
            raise RuntimeError("No camera cell is safe across the transition.")

        common_ids = {cell.cell_id for cell in common_cells}
        preferred_ids = [
            self._state_for(context).active_cell_id
            for context in reversed(contexts)
        ] + [fallback.cell_id]
        for cell_id in preferred_ids:
            if cell_id in common_ids:
                return CELL_BY_ID[cell_id]
        return min(
            common_cells,
            key=lambda cell: GRID_DIVERSE_RANK[cell.cell_id],
        )

    def record_current_result(
        self,
        context: ContextKey,
        current: SensorCell,
        probability_good: float,
    ) -> bool:
        state = self._state_for(context)
        state.active_cell_id = current.cell_id
        if self.is_acceptable(probability_good):
            state.consecutive_bad_frames = 0
            return False
        state.consecutive_bad_frames += 1
        return state.consecutive_bad_frames >= self.required_bad_frames

    def select_challenger(
        self, context: ContextKey, current: SensorCell
    ) -> Optional[SensorCell]:
        state = self._state_for(context)
        candidates = [
            cell
            for cell in self.safety_policy.safe_cells(context)
            if cell != current and state.cells[cell.cell_id].cooldown == 0
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cell: (
                -state.cells[cell.cell_id].success_score,
                GRID_DIVERSE_RANK[cell.cell_id],
            ),
        )

    def resolve_challenger(
        self,
        context: ContextKey,
        current: SensorCell,
        challenger: SensorCell,
        challenger_probability_good: float,
    ) -> SensorCell:
        state = self._state_for(context)
        if self.is_acceptable(challenger_probability_good):
            state.active_cell_id = challenger.cell_id
            state.consecutive_bad_frames = 0
            state.cells[challenger.cell_id].cooldown = 0
            return challenger
        state.active_cell_id = current.cell_id
        state.cells[challenger.cell_id].cooldown = self.challenger_cooldown_rounds
        return current

    def success_score(self, context: ContextKey, cell: SensorCell) -> float:
        return self._state_for(context).cells[cell.cell_id].success_score

    def challenger_cooldown(self, context: ContextKey, cell: SensorCell) -> int:
        return self._state_for(context).cells[cell.cell_id].cooldown

    def consecutive_bad_frames(self, context: ContextKey) -> int:
        return self._state_for(context).consecutive_bad_frames

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
