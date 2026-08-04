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
    pending: bool = False
    pending_weighted_delta_sum: float = 0.0
    pending_precision_sum: float = 0.0
    pending_observation_count: int = 0
    pending_started_round: int | None = None


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
    exposure_recheck: bool = False
    pending_edge_from_id: str | None = None
    pending_edge_to_id: str | None = None
    pending_axis: SearchAxis | None = None
    pending_direction: SearchDirection | None = None
    local_edges: dict[tuple[str, str], LocalEdgeState] = field(default_factory=dict)


def clear_pending(state: ContextState) -> None:
    for edge in state.local_edges.values():
        if edge.pending:
            edge.pending = False
            edge.pending_weighted_delta_sum = 0.0
            edge.pending_precision_sum = 0.0
            edge.pending_observation_count = 0
            edge.pending_started_round = None
    state.pending_edge_from_id = None
    state.pending_edge_to_id = None
    state.pending_axis = None
    state.pending_direction = None


class StateStore:
    def __init__(self, initial_axis: SearchAxis = SearchAxis.EXPOSURE) -> None:
        self.initial_axis = initial_axis
        self.contexts: dict[str, ContextState] = {}

    def get(self, context_key: str) -> ContextState:
        if context_key not in self.contexts:
            self.contexts[context_key] = ContextState(search_axis=self.initial_axis)
        return self.contexts[context_key]
