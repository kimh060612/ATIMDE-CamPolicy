from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import Iterable, Sequence

from hardware.utils import EXPOSURE_MS_VALUES, GAIN_VALUES, SensorCell

from .state import ContextState


IndexPoint = tuple[float, float]


class GeometricProposalKind(str, Enum):
    BRIGHTNESS_RECOVERY = "brightness_recovery"
    INITIAL_EXPOSURE = "initial_exposure"
    INITIAL_GAIN = "initial_gain"
    REFLECTION = "reflection"
    EXPANSION = "expansion"
    CONTRACTION = "contraction"


@dataclass(frozen=True)
class GeometricVertex:
    cell: SensorCell
    mu: float
    uncertainty: float
    observed_round: int = 0


@dataclass(frozen=True)
class GeometricProposal:
    cell: SensorCell
    kind: GeometricProposalKind
    target: IndexPoint
    centroid: IndexPoint | None = None
    replace_cell_id: str | None = None


@dataclass
class GeometricSearchState(ContextState):
    """Short-lived simplex state for one context.

    The inherited fields keep the existing capture logger compatible.  Local
    edge history is diagnostic only; geometric proposals use only ``vertices``
    and ``tested_cell_ids``, both of which are cleared at every episode reset.
    """

    search_cycle_complete: bool = True
    probe_pending: bool = False
    episode_active: bool = False
    episode_id: int = 0
    vertices: list[GeometricVertex] = field(default_factory=list)
    tested_cell_ids: set[str] = field(default_factory=set)
    initialization_queue: list[tuple[GeometricProposalKind, SensorCell]] = field(
        default_factory=list
    )
    pending_proposal: GeometricProposal | None = None
    deferred_kind: GeometricProposalKind | None = None
    deferred_target: IndexPoint | None = None
    deferred_replace_cell_id: str | None = None
    last_geometric_reason: str = "current_only"
    wait_for_periodic_reprobe: bool = False


def cell_to_point(cell: SensorCell) -> tuple[int, int]:
    return (
        EXPOSURE_MS_VALUES.index(cell.exposure_ms),
        GAIN_VALUES.index(cell.gain),
    )


def symmetric_axis_candidates(
    current: SensorCell,
    axis: str,
    safe_cells: Iterable[SensorCell],
    tested_cell_ids: set[str],
    positive_first: bool,
) -> list[SensorCell]:
    """Return safe, untested ±1 neighbors in caller-selected order.

    No feasible-span or score prior is used, and an unselected opposite
    neighbor remains untested.
    """

    safe = set(safe_cells)
    if current not in safe:
        raise ValueError(f"Current cell {current.cell_id} is not safe.")
    if axis == "exposure":
        values = EXPOSURE_MS_VALUES
        current_index = values.index(current.exposure_ms)
    elif axis == "gain":
        values = GAIN_VALUES
        current_index = values.index(current.gain)
    else:
        raise ValueError(f"Unknown geometric axis: {axis}")

    directions = (1, -1) if positive_first else (-1, 1)
    candidates: list[SensorCell] = []
    for direction in directions:
        index = current_index + direction
        if not 0 <= index < len(values):
            continue
        cell = (
            SensorCell(values[index], current.gain)
            if axis == "exposure"
            else SensorCell(current.exposure_ms, values[index])
        )
        if cell in safe and cell.cell_id not in tested_cell_ids:
            candidates.append(cell)
    return candidates


def project_safe_untested(
    target: IndexPoint,
    safe_cells: Iterable[SensorCell],
    tested_cell_ids: set[str],
) -> SensorCell | None:
    """Project a continuous point to the nearest safe, untested grid cell."""

    best_cell: SensorCell | None = None
    best_key: tuple[float, int, int] = (inf, 0, 0)
    for cell in safe_cells:
        if cell.cell_id in tested_cell_ids:
            continue
        exposure_index, gain_index = cell_to_point(cell)
        distance = (
            (exposure_index - target[0]) ** 2
            + (gain_index - target[1]) ** 2
        )
        key = (distance, exposure_index, gain_index)
        if key < best_key:
            best_key = key
            best_cell = cell
    return best_cell


def ranked_vertices(
    vertices: Sequence[GeometricVertex],
) -> tuple[GeometricVertex, GeometricVertex, GeometricVertex]:
    if len(vertices) != 3:
        raise ValueError("A 2D simplex requires exactly three vertices.")
    ranked = sorted(vertices, key=lambda vertex: (vertex.mu, cell_to_point(vertex.cell)))
    return ranked[0], ranked[1], ranked[2]


def reflection_geometry(
    vertices: Sequence[GeometricVertex],
) -> tuple[IndexPoint, IndexPoint, GeometricVertex]:
    best, second, worst = ranked_vertices(vertices)
    best_point = cell_to_point(best.cell)
    second_point = cell_to_point(second.cell)
    worst_point = cell_to_point(worst.cell)
    centroid = (
        (best_point[0] + second_point[0]) / 2.0,
        (best_point[1] + second_point[1]) / 2.0,
    )
    reflected = (
        2.0 * centroid[0] - worst_point[0],
        2.0 * centroid[1] - worst_point[1],
    )
    return centroid, reflected, worst


def expansion_target(centroid: IndexPoint, reflected: SensorCell) -> IndexPoint:
    reflected_point = cell_to_point(reflected)
    return (
        centroid[0] + 2.0 * (reflected_point[0] - centroid[0]),
        centroid[1] + 2.0 * (reflected_point[1] - centroid[1]),
    )


def contraction_target(centroid: IndexPoint, worst: SensorCell) -> IndexPoint:
    worst_point = cell_to_point(worst)
    return (
        centroid[0] + 0.5 * (worst_point[0] - centroid[0]),
        centroid[1] + 0.5 * (worst_point[1] - centroid[1]),
    )
