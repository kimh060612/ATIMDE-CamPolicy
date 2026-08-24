#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import (
    ExperimentConfig,
    SafetyPolicy,
    configure_parser,
    validate_args,
)
from ati_mde_control.context import build_context_provider
from ati_mde_control.evaluation import DepthEvaluator
from ati_mde_control.geometric_search_policy import GeometricSearchPolicy
from ati_mde_control.geometric_search_update_experiment import (
    GeometricSearchUpdateExperiment,
)
from ati_mde_control.logging import CaptureLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometric pairwise Orbbec camera control with current-score reuse."
    )
    configure_parser(parser)
    parser.add_argument(
        "--simplex-memory-ttl-rounds",
        type=int,
        default=0,
        help=(
            "Discard a remembered simplex when any vertex score is this many "
            "control rounds old; 0 disables expiry."
        ),
    )
    args = parser.parse_args()
    validate_args(parser, args)
    if args.simplex_memory_ttl_rounds < 0:
        parser.error("--simplex-memory-ttl-rounds must be non-negative.")
    return args


def build_experiment(args: argparse.Namespace) -> GeometricSearchUpdateExperiment:
    from ati_mde_control.predictor import CameraErrorPredictor
    from hardware.sensor import OrbbecColorCamera

    config = ExperimentConfig.from_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    context_provider = build_context_provider(args)
    camera = OrbbecColorCamera(
        exposure_value_per_ms=args.exposure_value_per_ms,
        settle_frames=args.settle_frames,
        frame_timeout_ms=args.frame_timeout_ms,
        warmup_frames=args.warmup_frames,
        disable_awb=args.disable_awb,
        strict_property_grid=not args.allow_unsupported_grid_values,
    )
    predictor = CameraErrorPredictor(
        config.checkpoint_path,
        config.model_size,
        config.device,
        config.precision,
        config.q_uncertainty_weight,
        config.local_files_only,
    )
    safety = SafetyPolicy.from_json(config.safety_path)
    policy = GeometricSearchPolicy(
        config.policy,
        safety,
        simplex_memory_ttl_rounds=args.simplex_memory_ttl_rounds,
    )
    capture_runner = CaptureRunner(
        camera, context_provider, config.max_pair_capture_gap_ms
    )
    logger = CaptureLogger(config.output_dir)
    evaluator = DepthEvaluator(predictor, config)
    return GeometricSearchUpdateExperiment(
        config,
        capture_runner,
        predictor,
        policy,
        logger,
        evaluator,
        brightness_guard_config=safety.brightness_guard,
    )


def main() -> int:
    experiment = None
    exit_code = 0
    try:
        args = parse_args()
        experiment = build_experiment(args)
        print(f"[Start] capture rounds={args.max_rounds}; press Ctrl-C to stop early.")
        while experiment.round_index < args.max_rounds:
            started = time.monotonic()
            experiment.run_round()
            remaining = args.round_interval_ms / 1000.0 - (
                time.monotonic() - started
            )
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[Stop] interrupted; finalizing captured frames.")
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        print(f"\n[ERROR] {error}")
        exit_code = 1
    finally:
        if experiment is not None:
            try:
                experiment.finalize()
            except (OSError, RuntimeError, ValueError) as error:
                print(f"[ERROR] result finalization failed: {error}")
                exit_code = 1
            try:
                experiment.capture_runner.camera.close()
                experiment.capture_runner.context_provider.close()
            except Exception as error:
                print(f"[ERROR] shutdown failed: {error}")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
