from __future__ import annotations

from dataclasses import dataclass, field

from .types import SearchAxis, SearchDirection


@dataclass
class LocalEdgeState:
    valid_count: int = 0
    invalid_count: int = 0
    ambiguous_count: int = 0
    invalid_cooldown: int = 0
    ambiguous_cooldown: int = 0
    last_delta_mu: float | None = None
    last_observed_round: int | None = None


@dataclass
class ContextState:
    active_cell_id: str | None = None
    search_axis: SearchAxis = SearchAxis.EXPOSURE
    search_direction: SearchDirection | None = None
    exposure_negative_tested: bool = False
    exposure_positive_tested: bool = False
    gain_negative_tested: bool = False
    gain_positive_tested: bool = False
    cycle_had_switch: bool = False
    search_cycle_complete: bool = False
    probe_pending: bool = True
    rounds_since_valid_probe: int = 0
    consecutive_invalid_pairs: int = 0
    force_current_only_rounds: int = 0
    pending_switch_from_id: str | None = None
    pending_switch_to_id: str | None = None
    pending_switch_wins: int = 0
    pending_switch_started_round: int | None = None
    pending_switch_axis: SearchAxis | None = None
    pending_switch_direction: SearchDirection | None = None
    rollback_cell_id: str | None = None
    rollback_verification_pending: bool = False
    post_switch_dwell_remaining: int = 0
    rollback_verification_started_round: int | None = None
    exposure_recheck: bool = False
    local_edges: dict[tuple[str, str], LocalEdgeState] = field(default_factory=dict)

    def clear_pending_switch(self) -> None:
        self.pending_switch_from_id = None
        self.pending_switch_to_id = None
        self.pending_switch_wins = 0
        self.pending_switch_started_round = None
        self.pending_switch_axis = None
        self.pending_switch_direction = None

    def clear_rollback(self) -> None:
        self.rollback_cell_id = None
        self.rollback_verification_pending = False
        self.post_switch_dwell_remaining = 0
        self.rollback_verification_started_round = None


class StateStore:
    def __init__(self, initial_axis: SearchAxis = SearchAxis.EXPOSURE) -> None:
        self.initial_axis = initial_axis
        self.contexts: dict[str, ContextState] = {}

    def get(self, context_key: str) -> ContextState:
        if context_key not in self.contexts:
            self.contexts[context_key] = ContextState(search_axis=self.initial_axis)
        return self.contexts[context_key]
