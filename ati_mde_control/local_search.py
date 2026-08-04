from __future__ import annotations

from collections.abc import Collection

from hardware.utils import EXPOSURE_MS_VALUES, GAIN_VALUES, SensorCell

from .state import ContextState
from .types import SearchAxis, SearchDirection


def neighbor(
    cell: SensorCell, axis: SearchAxis, direction: SearchDirection
) -> SensorCell | None:
    values = EXPOSURE_MS_VALUES if axis is SearchAxis.EXPOSURE else GAIN_VALUES
    value = cell.exposure_ms if axis is SearchAxis.EXPOSURE else cell.gain
    index = values.index(value) + direction.value
    if not 0 <= index < len(values):
        return None
    return (
        SensorCell(values[index], cell.gain)
        if axis is SearchAxis.EXPOSURE
        else SensorCell(cell.exposure_ms, values[index])
    )


def direction_tested(state: ContextState, direction: SearchDirection) -> bool:
    return bool(
        getattr(state, f"{state.search_axis.value}_{'negative' if direction.value < 0 else 'positive'}_tested")
    )


def mark_direction_tested(
    state: ContextState, direction: SearchDirection, tested: bool = True
) -> None:
    setattr(
        state,
        f"{state.search_axis.value}_{'negative' if direction.value < 0 else 'positive'}_tested",
        tested,
    )


def reset_axis(state: ContextState, axis: SearchAxis) -> None:
    state.search_axis = axis
    state.search_direction = None
    setattr(state, f"{axis.value}_negative_tested", False)
    setattr(state, f"{axis.value}_positive_tested", False)


def restart_cycle(state: ContextState, initial_axis: SearchAxis) -> None:
    state.exposure_negative_tested = False
    state.exposure_positive_tested = False
    state.gain_negative_tested = False
    state.gain_positive_tested = False
    state.cycle_had_switch = False
    state.search_cycle_complete = False
    state.exposure_recheck = False
    state.search_axis = initial_axis
    state.search_direction = None
    state.probe_pending = True


def _advance_phase(state: ContextState, initial_axis: SearchAxis) -> bool:
    if state.search_axis is SearchAxis.EXPOSURE and not state.exposure_recheck:
        reset_axis(state, SearchAxis.GAIN)
        return True
    if state.search_axis is SearchAxis.GAIN:
        state.exposure_recheck = True
        reset_axis(state, SearchAxis.EXPOSURE)
        return True
    if state.cycle_had_switch:
        restart_cycle(state, initial_axis)
        return True
    state.search_cycle_complete = True
    state.probe_pending = False
    state.search_direction = None
    return False


def select_challenger(
    state: ContextState,
    current: SensorCell,
    safe_cells: Collection[SensorCell],
    initial_axis: SearchAxis = SearchAxis.EXPOSURE,
) -> SensorCell | None:
    safe = set(safe_cells)
    while not state.search_cycle_complete:
        directions = (
            (state.search_direction,)
            if state.search_direction is not None
            else (SearchDirection.NEGATIVE, SearchDirection.POSITIVE)
        )
        cooling = False
        for direction in directions:
            if direction_tested(state, direction):
                continue
            candidate = neighbor(current, state.search_axis, direction)
            if candidate is None or candidate not in safe:
                mark_direction_tested(state, direction)
                continue
            edge = state.local_edges.get((current.cell_id, candidate.cell_id))
            if edge and (edge.invalid_cooldown or edge.ambiguous_cooldown):
                cooling = True
                continue
            state.search_direction = direction
            return candidate
        if cooling:
            state.search_direction = None
            return None
        if not _advance_phase(state, initial_axis):
            return None
    return None
