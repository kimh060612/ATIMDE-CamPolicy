from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

from hardware.utils import (
    ALL_CELLS,
    CELL_BY_ID,
    GAIN_VALUES,
    LIGHT_STATE_COUNT,
    MOTION_STATE_COUNT,
    CellStats,
    ContextKey,
    ContextState,
    QScore,
    SensorCell,
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
        ema_alpha: float = 0.3,
        switch_margin: float = 0.01,
        low_bias_threshold: float = 0.12,
        high_bias_threshold: float = 0.20,
        probe_std_threshold: float = 0.08,
        offload_command: Optional[str] = None,
    ) -> None:
        if cells_per_probe < 1:
            raise ValueError("cells_per_probe must be positive.")
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

        self.safety_policy = safety_policy
        self.cells_per_probe = cells_per_probe
        self.ema_alpha = ema_alpha
        self.switch_margin = switch_margin
        self.low_bias_threshold = low_bias_threshold
        self.high_bias_threshold = high_bias_threshold
        self.probe_std_threshold = probe_std_threshold
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
        return float(score.uncertainty)

    def _state_for(self, context: ContextKey) -> ContextState:
        return self.context_states.setdefault(context.table_key, ContextState())

    def observe(self, context: ContextKey, cell: SensorCell, score: QScore) -> None:
        context.validate()
        state = self._state_for(context)
        state.cells[cell.cell_id].update(
            score,
            alpha=self.ema_alpha,
            cycle_index=state.committed_cycles,
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

    def select_probe_cells(
        self, context: ContextKey, current_cell: SensorCell
    ) -> list[SensorCell]:
        state = self._state_for(context)
        safe_cells = [
            cell
            for cell in self.safety_policy.safe_cells(context)
            if cell != current_cell
        ]
        if not safe_cells:
            return []

        unobserved = [
            cell for cell in safe_cells if state.cells[cell.cell_id].count == 0
        ]
        selected = unobserved[: self.cells_per_probe]
        selected_ids = {cell.cell_id for cell in selected}
        while len(selected) < min(self.cells_per_probe, len(safe_cells)):
            cell = safe_cells[state.round_robin_pointer % len(safe_cells)]
            state.round_robin_pointer = (
                state.round_robin_pointer + 1
            ) % len(safe_cells)
            if cell.cell_id not in selected_ids:
                selected.append(cell)
                selected_ids.add(cell.cell_id)
        return selected

    def resolve_probe(
        self,
        context: ContextKey,
        current_cell: SensorCell,
        observations: Sequence[tuple[SensorCell, QScore]],
    ) -> PolicyDecision:
        if not observations:
            raise ValueError("At least one probe observation is required.")

        current_score = next(
            (score for cell, score in observations if cell == current_cell), None
        )
        best_cell, best_score = min(observations, key=lambda item: item[1].q)
        if (
            current_score is not None
            and best_cell != current_cell
            and best_score.q > current_score.q - self.switch_margin
        ):
            best_cell, best_score = current_cell, current_score

        state = self._state_for(context)
        state.best_cell_id = best_cell.cell_id
        state.committed_cycles += 1
        if self._bias(best_score) <= self.low_bias_threshold:
            return PolicyDecision(
                "use", "probe_found_low_camera_bias", best_cell, best_score
            )
        return PolicyDecision("offload", "probe_exhausted", best_cell, best_score)

    def remember_used_cell(self, context: ContextKey, cell: SensorCell) -> None:
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
