#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from pathlib import Path

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import (
    ExperimentConfig,
    SafetyPolicy,
    configure_parser,
    validate_args,
)
from ati_mde_control.context import build_context_provider
from ati_mde_control.logging import CaptureLogger
from ati_mde_control.nelder_mead_experiment import RiskNelderMeadExperiment
from ati_mde_control.nelder_mead_policy import (
    ContextualRiskNelderMeadPolicy,
    RiskNelderMeadConfig,
)
from orbbec_deterministic_probing_modelv1 import FairDepthEvaluator


DEFAULT_SAFETY_CONFIG = (
    Path(__file__).resolve().parent / "config" / "safety_envelop.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safety-aware Nelder-Mead camera control minimizing the trained "
            "camera prediction risk q = mu + alpha * sigma."
        )
    )
    configure_parser(parser)
    parser.add_argument("--simplex-restart-frames", type=int, default=60)
    parser.add_argument("--simplex-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--evaluation-precision",
        choices=("fp16", "fp32"),
        default="fp32",
    )
    parser.set_defaults(
        safety_config=DEFAULT_SAFETY_CONFIG,
        evaluation_alignment="scale_shift_inverse",
        min_depth_m=1e-3,
        min_valid_depth_pixels=10000,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.simplex_restart_frames < 3:
        parser.error("--simplex-restart-frames must be at least 3.")
    if not math.isfinite(args.simplex_tolerance) or args.simplex_tolerance < 0:
        parser.error("--simplex-tolerance must be finite and non-negative.")
    if args.evaluation_alignment == "metric":
        parser.error(
            "Fair DA-V2 Small comparison requires scale_shift_depth or "
            "scale_shift_inverse alignment."
        )
    return args


def build_experiment(args: argparse.Namespace) -> RiskNelderMeadExperiment:
    from ati_mde_control.full_depth_predictor import CameraErrorFullDepthPredictor
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
    predictor = CameraErrorFullDepthPredictor(
        config.checkpoint_path,
        config.model_size,
        config.device,
        config.precision,
        config.q_uncertainty_weight,
        config.local_files_only,
    )
    policy = ContextualRiskNelderMeadPolicy(
        RiskNelderMeadConfig(
            restart_frames=args.simplex_restart_frames,
            simplex_tolerance=args.simplex_tolerance,
        ),
        SafetyPolicy.from_json(config.safety_path),
        config.default_cell,
    )
    capture_runner = CaptureRunner(
        camera,
        context_provider,
        config.max_pair_capture_gap_ms,
    )
    logger = CaptureLogger(config.output_dir)
    evaluator = FairDepthEvaluator(
        predictor,
        config,
        args.evaluation_precision,
    )
    return RiskNelderMeadExperiment(
        config,
        capture_runner,
        predictor,
        policy,
        logger,
        evaluator,
    )


def main(argv: Sequence[str] | None = None) -> int:
    experiment: RiskNelderMeadExperiment | None = None
    exit_code = 0
    try:
        args = parse_args(argv)
        experiment = build_experiment(args)
        print(
            f"[Start] safety-aware risk Nelder-Mead rounds={args.max_rounds}; "
            "press Ctrl-C to stop early."
        )
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
