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
    EdgeStats,
    QScore,
    SensorCell,
)

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


PairStatus = Literal[
    "challenger_won",
    "current_won",
    "ambiguous",
    "invalid_pair",
    "not_probed",
]


@dataclass(frozen=True)
class PairwiseResult:
    selected_cell: SensorCell
    status: PairStatus
    delta_mu: float
    pair_std: float
    effective_margin: float
    confidence: float
    edge_ema_before: float
    edge_ema_after: float


class ATIMDECameraProbingController:
    """Context-specific active cell with one same-round pairwise-mu challenger."""

    def __init__(
        self,
        safety_policy: SafetyPolicy,
        *,
        probe_trigger_threshold: float = 0.11,
        switch_margin: float = 0.01,
        challenger_cooldown_rounds: int = 5,
        pair_uncertainty_weight: float = 0.25,
        reference_pair_std: float = 0.03,
        edge_ema_alpha: float = 0.3,
        offload_uncertainty_weight: float = 1.0,
        offload_threshold: float = 0.15,
        ambiguous_offload_threshold: float = 0.11,
        offload_command: Optional[str] = None,
    ) -> None:
        if not math.isfinite(probe_trigger_threshold):
            raise ValueError("probe_trigger_threshold must be finite.")
        if not math.isfinite(switch_margin) or switch_margin < 0.0:
            raise ValueError("switch_margin must be finite and non-negative.")
        if challenger_cooldown_rounds < 0:
            raise ValueError("challenger_cooldown_rounds must be non-negative.")
        if (
            not math.isfinite(pair_uncertainty_weight)
            or pair_uncertainty_weight < 0.0
        ):
            raise ValueError(
                "pair_uncertainty_weight must be finite and non-negative."
            )
        if not math.isfinite(reference_pair_std) or reference_pair_std <= 0.0:
            raise ValueError("reference_pair_std must be finite and positive.")
        if not 0.0 < edge_ema_alpha <= 1.0:
            raise ValueError("edge_ema_alpha must be in (0, 1].")
        if (
            not math.isfinite(offload_uncertainty_weight)
            or offload_uncertainty_weight < 0.0
        ):
            raise ValueError(
                "offload_uncertainty_weight must be finite and non-negative."
            )
        if not math.isfinite(offload_threshold):
            raise ValueError("offload_threshold must be finite.")
        if not math.isfinite(ambiguous_offload_threshold):
            raise ValueError("ambiguous_offload_threshold must be finite.")

        self.safety_policy = safety_policy
        self.probe_trigger_threshold = probe_trigger_threshold
        self.switch_margin = switch_margin
        self.challenger_cooldown_rounds = challenger_cooldown_rounds
        self.pair_uncertainty_weight = pair_uncertainty_weight
        self.reference_pair_std = reference_pair_std
        self.edge_ema_alpha = edge_ema_alpha
        self.offload_uncertainty_weight = offload_uncertainty_weight
        self.offload_threshold = offload_threshold
        self.ambiguous_offload_threshold = ambiguous_offload_threshold
        self.offload_command = offload_command
        self.context_states: dict[str, ContextState] = {}

    @staticmethod
    def _finite_mu(mu: float) -> float:
        if not math.isfinite(mu):
            raise ValueError("Observed mu must be finite.")
        return float(mu)

    @staticmethod
    def _finite_std(std: float) -> float:
        if not math.isfinite(std) or std < 0.0:
            raise ValueError("Observed std must be finite and non-negative.")
        return float(std)

    def _state_for(self, context: ContextKey) -> ContextState:
        context.validate()
        return self.context_states.setdefault(context.table_key, ContextState())

    def cell_for_context(
        self, context: ContextKey, fallback: SensorCell
    ) -> SensorCell:
        state = self._state_for(context)
        safe_cells = self.safety_policy.safe_cells(context)
        if state.active_cell_id is None:
            selected = fallback if fallback in safe_cells else safe_cells[0]
            state.active_cell_id = selected.cell_id
            return selected
        if state.active_cell_id not in {cell.cell_id for cell in safe_cells}:
            raise RuntimeError(
                f"Committed cell {state.active_cell_id} is unsafe for "
                f"{context.table_key}."
            )
        return CELL_BY_ID[state.active_cell_id]

    def committed_cell_for_context(
        self, context: ContextKey, fallback: SensorCell
    ) -> SensorCell:
        safe_cells = self.safety_policy.safe_cells(context)
        state = self.context_states.get(context.table_key)
        if state is not None and state.active_cell_id in {
            cell.cell_id for cell in safe_cells
        }:
            return CELL_BY_ID[state.active_cell_id]
        return fallback if fallback in safe_cells else safe_cells[0]

    def record_current_result(
        self,
        context: ContextKey,
        current: SensorCell,
        current_mu: float,
    ) -> bool:
        state = self._state_for(context)
        if state.active_cell_id != current.cell_id:
            raise RuntimeError("Current cell is not the committed context cell.")
        return self._finite_mu(current_mu) > self.probe_trigger_threshold

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
        unobserved = [
            cell
            for cell in candidates
            if state.edges.get(
                (current.cell_id, cell.cell_id), EdgeStats()
            ).comparison_count == 0
        ]
        if unobserved:
            return min(
                unobserved,
                key=lambda cell: GRID_DIVERSE_RANK[cell.cell_id],
            )
        return min(
            candidates,
            key=lambda cell: (
                -state.edges[(current.cell_id, cell.cell_id)].ema_improvement,
                GRID_DIVERSE_RANK[cell.cell_id],
            ),
        )

    def resolve_challenger(
        self,
        context: ContextKey,
        current: SensorCell,
        current_mu: float,
        current_std: float,
        challenger: SensorCell,
        challenger_mu: float,
        challenger_std: float,
    ) -> PairwiseResult:
        state = self._state_for(context)
        if state.active_cell_id != current.cell_id:
            raise RuntimeError("Current cell is not the committed context cell.")
        current_mu = self._finite_mu(current_mu)
        challenger_mu = self._finite_mu(challenger_mu)
        current_std = self._finite_std(current_std)
        challenger_std = self._finite_std(challenger_std)
        delta_mu = current_mu - challenger_mu
        pair_std = math.hypot(current_std, challenger_std)
        effective_margin = (
            self.switch_margin + self.pair_uncertainty_weight * pair_std
        )
        if delta_mu > effective_margin:
            status: PairStatus = "challenger_won"
        elif delta_mu < -effective_margin:
            status = "current_won"
        else:
            status = "ambiguous"

        confidence = 1.0 / (1.0 + pair_std / self.reference_pair_std)
        edge = state.edges.setdefault(
            (current.cell_id, challenger.cell_id), EdgeStats()
        )
        edge_ema_before = edge.ema_improvement
        effective_alpha = self.edge_ema_alpha * confidence
        edge.ema_improvement = (
            (1.0 - effective_alpha) * edge.ema_improvement
            + effective_alpha * delta_mu
        )
        edge.comparison_count += 1
        edge.ambiguous_count += int(status == "ambiguous")
        edge.challenger_win_count += int(status == "challenger_won")

        selected = current
        if status == "challenger_won":
            state.active_cell_id = challenger.cell_id
            selected = challenger
        elif status == "current_won":
            state.cells[challenger.cell_id].cooldown = (
                self.challenger_cooldown_rounds
            )
        return PairwiseResult(
            selected,
            status,
            delta_mu,
            pair_std,
            effective_margin,
            confidence,
            edge_ema_before,
            edge.ema_improvement,
        )

    def evaluate_offload(
        self,
        selected_mu: float,
        selected_std: float,
        pair_status: PairStatus,
    ) -> tuple[bool, float]:
        selected_mu = self._finite_mu(selected_mu)
        selected_std = self._finite_std(selected_std)
        risk = selected_mu + self.offload_uncertainty_weight * selected_std
        should_offload = risk > self.offload_threshold or (
            pair_status == "ambiguous"
            and selected_mu > self.ambiguous_offload_threshold
        )
        return should_offload, risk

    def complete_round(self, context: ContextKey) -> None:
        for stats in self._state_for(context).cells.values():
            stats.cooldown = max(stats.cooldown - 1, 0)

    def active_cell_id(self, context: ContextKey) -> Optional[str]:
        state = self.context_states.get(context.table_key)
        return state.active_cell_id if state is not None else None

    def challenger_cooldown(self, context: ContextKey, cell: SensorCell) -> int:
        return self._state_for(context).cells[cell.cell_id].cooldown

    def invoke_offload(self, context: ContextKey, decision: PolicyDecision) -> None:
        print(
            f"[OFFLOAD] context={context.table_key} "
            f"cell={decision.selected_cell.cell_id} "
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
                "ATI_CAMERA_BIAS": str(decision.score.mu),
                "ATI_CAMERA_STD": str(decision.score.uncertainty),
                "ATI_Q_VALUE": str(decision.score.q),
            }
        )
        subprocess.run(
            shlex.split(self.offload_command),
            env=environment,
            check=False,
        )
