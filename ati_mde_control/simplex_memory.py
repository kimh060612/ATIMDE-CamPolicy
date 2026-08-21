from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from hardware.utils import ContextKey, SensorCell

from .geometric_search import (
    GeometricProposalKind,
    GeometricSearchState,
    GeometricVertex,
    IndexPoint,
)


@dataclass(frozen=True)
class SimplexSnapshot:
    """Immutable geometric trajectory captured at a control boundary."""

    source_context: ContextKey
    vertices: tuple[GeometricVertex, ...]
    tested_cell_ids: frozenset[str]
    initialization_queue: tuple[tuple[GeometricProposalKind, SensorCell], ...]
    deferred_kind: GeometricProposalKind | None
    deferred_target: IndexPoint | None
    deferred_replace_cell_id: str | None
    episode_id: int
    cycle_had_switch: bool

    @classmethod
    def from_state(
        cls, context: ContextKey, state: GeometricSearchState
    ) -> SimplexSnapshot:
        return cls(
            source_context=context,
            vertices=tuple(state.vertices),
            tested_cell_ids=frozenset(state.tested_cell_ids),
            initialization_queue=tuple(state.initialization_queue),
            deferred_kind=state.deferred_kind,
            deferred_target=state.deferred_target,
            deferred_replace_cell_id=state.deferred_replace_cell_id,
            episode_id=state.episode_id,
            cycle_had_switch=state.cycle_had_switch,
        )


class SimplexMemory:
    """In-process context cache for geometric simplex trajectories."""

    def __init__(self) -> None:
        self._entries: dict[ContextKey, tuple[int, SimplexSnapshot]] = {}
        self._sequence = 0

    def update(self, context: ContextKey, state: GeometricSearchState) -> None:
        """Store a detached snapshot when the state contains observations."""

        context.validate()
        if not state.vertices:
            return
        self._sequence += 1
        self._entries[context] = (
            self._sequence,
            SimplexSnapshot.from_state(context, state),
        )

    def exact(self, context: ContextKey) -> SimplexSnapshot | None:
        context.validate()
        entry = self._entries.get(context)
        return None if entry is None else entry[1]

    def discard(self, context: ContextKey) -> None:
        context.validate()
        self._entries.pop(context, None)

    @staticmethod
    def vertices_expired(
        vertices: Iterable[GeometricVertex],
        current_round: int,
        ttl_rounds: int,
    ) -> bool:
        """Return whether any observed score reached the configured TTL."""

        if ttl_rounds <= 0:
            return False
        return any(
            current_round - vertex.observed_round >= ttl_rounds
            for vertex in vertices
        )

    def recall_candidates(
        self,
        context: ContextKey,
        current_round: int = 0,
        ttl_rounds: int = 0,
    ) -> tuple[SimplexSnapshot, ...]:
        """Return fresh exact context first, then fresh same-motion entries."""

        context.validate()
        if ttl_rounds < 0:
            raise ValueError("Simplex memory TTL must be non-negative.")
        candidates: list[tuple[ContextKey, int, SimplexSnapshot]] = []
        exact = self._entries.get(context)
        if exact is not None:
            candidates.append((context, exact[0], exact[1]))
        fallbacks = sorted(
            (
                (stored_context, entry[0], entry[1])
                for stored_context, entry in self._entries.items()
                if stored_context != context
                and stored_context.motion_state == context.motion_state
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        candidates.extend(fallbacks)

        fresh: list[SimplexSnapshot] = []
        for stored_context, _, snapshot in candidates:
            if self.vertices_expired(
                snapshot.vertices, current_round, ttl_rounds
            ):
                self.discard(stored_context)
                continue
            fresh.append(snapshot)
        return tuple(fresh)
