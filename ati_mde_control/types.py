from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from hardware.utils import ContextKey, QScore, SensorCell


class SearchAxis(str, Enum):
    EXPOSURE = "exposure"
    GAIN = "gain"


class SearchDirection(int, Enum):
    NEGATIVE = -1
    POSITIVE = 1


class PairStatus(str, Enum):
    CHALLENGER_WON = "challenger_won"
    CURRENT_WON = "current_won"
    AMBIGUOUS = "ambiguous"
    INVALID_PAIR = "invalid_pair"
    NOT_PROBED = "not_probed"


class PairMode(str, Enum):
    NORMAL_SEARCH = "normal_search"
    PENDING_RECHECK = "pending_recheck"


class SwitchEvent(str, Enum):
    NONE = "none"
    IMMEDIATE_COMMIT = "immediate_commit"
    PENDING_STARTED = "pending_started"
    PENDING_UPDATED = "pending_updated"
    PENDING_COMMITTED = "pending_committed"
    PENDING_REJECTED = "pending_rejected"
    PENDING_EXHAUSTED = "pending_exhausted"
    PENDING_TIMEOUT = "pending_timeout"


@dataclass(frozen=True)
class CapturedFrame:
    round_index: int
    capture_index: int
    role: str
    cell: SensorCell
    latched_context: ContextKey
    capture_context: ContextKey
    context_stable: bool
    timestamp_ns: int
    image: np.ndarray
    depth_m: np.ndarray
    exposure_us: float
    requested_exposure_raw: int
    actual_exposure_raw: int | None
    actual_gain: int | None
    camera_parameter_ms: float
    freshness_token: Any = None


@dataclass(frozen=True)
class CapturePair:
    current: CapturedFrame
    challenger: CapturedFrame
    gap_ms: float
    valid: bool
    invalid_reason: str = ""


@dataclass(frozen=True)
class PairwiseDecision:
    status: PairStatus
    selected_cell: SensorCell
    delta_mu: float
    pair_std: float
    effective_margin: float
    pair_mode: PairMode = PairMode.NORMAL_SEARCH
    switch_event: SwitchEvent = SwitchEvent.NONE
    pending_observation_count: int = 0
    pending_weighted_delta_sum: float = 0.0
    pending_precision_sum: float = 0.0
    aggregated_delta_mu: float | None = None
    aggregated_pair_std: float | None = None
    aggregated_effective_margin: float | None = None
    pending_age_rounds: int | None = None


@dataclass(frozen=True)
class RoundResult:
    initial: CapturedFrame
    initial_score: QScore | None
    challenger: CapturedFrame | None = None
    challenger_score: QScore | None = None
    decision: PairwiseDecision | None = None
    output_delivered: bool = True
    offload_requested: bool = False
    reason: str = ""
