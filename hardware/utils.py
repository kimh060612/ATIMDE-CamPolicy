from dataclasses import dataclass, field
from typing import Optional

EXPOSURE_MS_VALUES: tuple[int, ...] = (4, 8, 16, 32)
GAIN_VALUES: tuple[int, ...] = (16, 32, 64, 128)
MOTION_STATE_COUNT = 5
LIGHT_STATE_COUNT = 3

@dataclass(frozen=True, order=True)
class ContextKey:
    motion_state: int
    light_state: int

    def validate(self) -> None:
        if not 0 <= self.motion_state < MOTION_STATE_COUNT:
            raise ValueError(
                f"motion_state must be in [0, {MOTION_STATE_COUNT - 1}], "
                f"got {self.motion_state}."
            )
        if not 0 <= self.light_state < LIGHT_STATE_COUNT:
            raise ValueError(
                f"light_state must be in [0, {LIGHT_STATE_COUNT - 1}], "
                f"got {self.light_state}."
            )

    @property
    def table_key(self) -> str:
        return f"{self.motion_state},{self.light_state}"


@dataclass(frozen=True, order=True)
class SensorCell:
    exposure_ms: int
    gain: int

    @property
    def cell_id(self) -> str:
        return f"E{self.exposure_ms:02d}_G{self.gain:03d}"


ALL_CELLS: tuple[SensorCell, ...] = tuple(
    SensorCell(exposure_ms=e, gain=g)
    for e in EXPOSURE_MS_VALUES
    for g in GAIN_VALUES
)
CELL_BY_ID: dict[str, SensorCell] = {cell.cell_id: cell for cell in ALL_CELLS}


@dataclass(frozen=True)
class QScore:
    """Output of `compute_q_score`.

    Parameters
    ----------
    q:
        Logged analysis score, typically `mu + beta * sigma`. The satisficing
        controller does not use it for cell switching.
    uncertainty:
        Optional predictive uncertainty. When present, it can trigger an
        offload decision through `--offload-uncertainty-threshold`.
    mu:
        Optional predicted mean, logged for later analysis.
    extra:
        Optional scalar metadata to append to the JSON metadata field.
    """

    q: float
    uncertainty: Optional[float] = None
    mu: Optional[float] = None
    extra: dict[str, float] = field(default_factory=dict)

@dataclass
class CellStats:
    success_score: float = 0.5
    observation_count: int = 0
    last_probability_good: Optional[float] = None
    cooldown: int = 0

    def update(self, probability_good: float, *, alpha: float) -> None:
        self.success_score = (
            (1.0 - alpha) * self.success_score + alpha * probability_good
        )
        self.observation_count += 1
        self.last_probability_good = probability_good


@dataclass
class ContextState:
    active_cell_id: Optional[str] = None
    consecutive_bad_frames: int = 0
    cells: dict[str, CellStats] = field(
        default_factory=lambda: {
            cell.cell_id: CellStats() for cell in ALL_CELLS
        }
    )
