from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from hardware.utils import (
    ALL_CELLS,
    CELL_BY_ID,
    EXPOSURE_MS_VALUES,
    GAIN_VALUES,
    LIGHT_STATE_COUNT,
    MOTION_STATE_COUNT,
    ContextKey,
    SensorCell,
)

from .types import SearchAxis


@dataclass(frozen=True)
class SafetyPolicy:
    max_exposure_ms_by_motion: tuple[int, ...] = (32,) * MOTION_STATE_COUNT
    allowed_gains_by_light: tuple[tuple[int, ...], ...] = (GAIN_VALUES,) * LIGHT_STATE_COUNT
    disabled_cells_by_context: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path | None) -> "SafetyPolicy":
        if path is None:
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        exposures = tuple(int(value) for value in payload.get(
            "max_exposure_ms_by_motion", [32] * MOTION_STATE_COUNT
        ))
        gains = tuple(tuple(int(value) for value in values) for values in payload.get(
            "allowed_gains_by_light", [list(GAIN_VALUES) for _ in range(LIGHT_STATE_COUNT)]
        ))
        if len(exposures) != MOTION_STATE_COUNT or len(gains) != LIGHT_STATE_COUNT:
            raise ValueError("Safety configuration has the wrong context dimensions.")
        disabled: dict[str, frozenset[str]] = {}
        for key, entries in payload.get("disabled_cells_by_context", {}).items():
            ids = frozenset(
                entry if isinstance(entry, str) else SensorCell(*map(int, entry)).cell_id
                for entry in entries
            )
            unknown = ids - CELL_BY_ID.keys()
            if unknown:
                raise ValueError(f"Unknown disabled cells: {sorted(unknown)}")
            disabled[str(key)] = ids
        return cls(exposures, gains, disabled)

    def safe_cells(self, context: ContextKey) -> list[SensorCell]:
        context.validate()
        disabled = self.disabled_cells_by_context.get(context.table_key, frozenset())
        cells = [
            cell for cell in ALL_CELLS
            if cell.exposure_ms <= self.max_exposure_ms_by_motion[context.motion_state]
            and cell.gain in self.allowed_gains_by_light[context.light_state]
            and cell.cell_id not in disabled
        ]
        if not cells:
            raise RuntimeError(f"No safe camera cell for {context.table_key}.")
        return cells


@dataclass(frozen=True)
class PolicyConfig:
    probe_trigger_threshold: float = 0.11
    switch_margin: float = 0.01
    pair_uncertainty_weight: float = 0.25
    invalid_edge_cooldown_rounds: int = 4
    ambiguous_edge_cooldown_rounds: int = 3
    max_consecutive_invalid_pairs: int = 3
    recovery_current_only_rounds: int = 2
    periodic_reprobe_interval: int = 30
    initial_search_axis: SearchAxis = SearchAxis.EXPOSURE
    offload_uncertainty_weight: float = 1.0
    offload_threshold: float = 0.15
    ambiguous_offload_threshold: float = 0.11
    offload_command: str | None = None

    def __post_init__(self) -> None:
        if self.switch_margin < 0 or self.pair_uncertainty_weight < 0:
            raise ValueError("Switch margin and uncertainty weight must be non-negative.")
        if min(
            self.invalid_edge_cooldown_rounds,
            self.ambiguous_edge_cooldown_rounds,
            self.max_consecutive_invalid_pairs,
            self.recovery_current_only_rounds,
            self.periodic_reprobe_interval,
        ) < 1:
            raise ValueError("Cooldown, recovery, and periodic counts must be positive.")
        values = (
            self.probe_trigger_threshold,
            self.switch_margin,
            self.pair_uncertainty_weight,
            self.offload_uncertainty_weight,
            self.offload_threshold,
            self.ambiguous_offload_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Policy thresholds and weights must be finite.")


@dataclass(frozen=True)
class ExperimentConfig:
    checkpoint_path: Path
    output_dir: Path
    model_size: str
    device: str
    precision: str
    local_files_only: bool
    q_uncertainty_weight: float
    policy: PolicyConfig
    safety_path: Path | None
    default_cell: SensorCell
    max_rounds: int
    round_interval_ms: float
    max_pair_capture_gap_ms: float
    camera_parameter_warn_ms: float
    mde_inference_warn_ms: float
    control_decision_warn_ms: float
    evaluation_alignment: str
    min_depth_m: float
    max_depth_m: float
    min_valid_depth_pixels: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        return cls(
            checkpoint_path=args.camera_error_checkpoint,
            output_dir=args.output_dir,
            model_size=args.model_size,
            device=args.device,
            precision=args.precision,
            local_files_only=args.local_files_only,
            q_uncertainty_weight=args.q_uncertainty_weight,
            policy=PolicyConfig(
                probe_trigger_threshold=args.probe_trigger_threshold,
                switch_margin=args.switch_margin,
                pair_uncertainty_weight=args.pair_uncertainty_weight,
                invalid_edge_cooldown_rounds=args.invalid_edge_cooldown_rounds,
                ambiguous_edge_cooldown_rounds=args.ambiguous_edge_cooldown_rounds,
                max_consecutive_invalid_pairs=args.max_consecutive_invalid_pairs,
                recovery_current_only_rounds=args.recovery_current_only_rounds,
                periodic_reprobe_interval=args.periodic_reprobe_interval,
                initial_search_axis=SearchAxis(args.initial_search_axis),
                offload_uncertainty_weight=args.offload_uncertainty_weight,
                offload_threshold=args.offload_threshold,
                ambiguous_offload_threshold=args.ambiguous_offload_threshold,
                offload_command=args.offload_command,
            ),
            safety_path=args.safety_config,
            default_cell=SensorCell(args.initial_exposure_ms, args.initial_gain),
            max_rounds=args.max_rounds,
            round_interval_ms=args.round_interval_ms,
            max_pair_capture_gap_ms=args.max_pair_capture_gap_ms,
            camera_parameter_warn_ms=args.camera_parameter_warn_ms,
            mde_inference_warn_ms=args.mde_inference_warn_ms,
            control_decision_warn_ms=args.control_decision_warn_ms,
            evaluation_alignment=args.evaluation_alignment,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_valid_depth_pixels=args.min_valid_depth_pixels,
        )


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-error-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("probing_modelv1_output"))
    parser.add_argument("--model-size", choices=("small", "base"), default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--q-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--probe-trigger-threshold", type=float, default=0.11)
    parser.add_argument("--switch-margin", type=float, default=0.01)
    parser.add_argument("--pair-uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--invalid-edge-cooldown-rounds", type=int, default=4)
    parser.add_argument("--ambiguous-edge-cooldown-rounds", type=int, default=3)
    parser.add_argument("--max-consecutive-invalid-pairs", type=int, default=3)
    parser.add_argument("--recovery-current-only-rounds", type=int, default=2)
    parser.add_argument("--periodic-reprobe-interval", type=int, default=30)
    parser.add_argument("--initial-search-axis", choices=("exposure", "gain"), default="exposure")
    parser.add_argument("--offload-uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--offload-threshold", type=float, default=0.15)
    parser.add_argument("--ambiguous-offload-threshold", type=float, default=0.11)
    parser.add_argument("--offload-command")
    parser.add_argument("--safety-config", type=Path)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--round-interval-ms", type=float, default=0.0)
    parser.add_argument("--initial-exposure-ms", type=int, choices=EXPOSURE_MS_VALUES, default=32)
    parser.add_argument("--initial-gain", type=int, choices=GAIN_VALUES, default=64)
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--motion-source", choices=("ros", "fixed"), default="ros")
    parser.add_argument("--motion-state", type=int, choices=range(5), default=2)
    parser.add_argument("--lighting-state", choices=("normal", "dim", "dark"), default="normal")
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--ros-sensor-timeout-sec", type=float, default=1.0)
    parser.add_argument("--context-debounce-ms", type=float, default=250.0)
    parser.add_argument("--context-debounce-samples", type=int, default=3)
    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument("--exposure-value-per-ms", type=float, default=10.0)
    parser.add_argument("--disable-awb", action="store_true")
    parser.add_argument("--allow-unsupported-grid-values", action="store_true")
    parser.add_argument("--camera-parameter-warn-ms", type=float, default=50.0)
    parser.add_argument("--mde-inference-warn-ms", type=float, default=100.0)
    parser.add_argument("--control-decision-warn-ms", type=float, default=500.0)
    parser.add_argument("--max-pair-capture-gap-ms", type=float, default=100.0)
    parser.add_argument("--evaluation-alignment", choices=("metric", "scale_shift_depth", "scale_shift_inverse"), default="scale_shift_inverse")
    parser.add_argument("--min-depth-m", type=float, default=1e-3)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=10000)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "invalid_edge_cooldown_rounds", "ambiguous_edge_cooldown_rounds",
        "max_consecutive_invalid_pairs", "recovery_current_only_rounds",
        "periodic_reprobe_interval", "context_debounce_samples", "min_valid_depth_pixels",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("Cooldown, recovery, periodic, debounce, and pixel counts must be positive.")
    if not 1 <= args.max_rounds <= 400:
        parser.error("--max-rounds must be between 1 and 400.")
    if args.round_interval_ms < 0 or args.context_debounce_ms < 0:
        parser.error("Round interval and context debounce must be non-negative.")
    if not 0 < args.min_depth_m < args.max_depth_m:
        parser.error("Require 0 < --min-depth-m < --max-depth-m.")
    if not math.isfinite(args.max_pair_capture_gap_ms) or args.max_pair_capture_gap_ms <= 0:
        parser.error("--max-pair-capture-gap-ms must be finite and positive.")
