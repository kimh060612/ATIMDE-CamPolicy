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
from ati_mde_control.risk_bandit_experiment import RiskBanditExperiment
from ati_mde_control.risk_bandit_policy import (
    METHOD_NAME,
    RiskBanditConfig,
    RiskBanditPolicy,
)
from orbbec_deterministic_probing_modelv1 import FairDepthEvaluator


DEFAULT_SAFETY_CONFIG = (
    Path(__file__).resolve().parent / "config" / "safety_envelop.json"
)


def configure_bandit_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bandit-window-size", type=int, default=48)
    parser.add_argument("--bandit-exploration-beta", type=float, default=1.0)
    parser.add_argument("--bandit-switch-penalty", type=float, default=0.005)
    parser.add_argument("--bandit-exposure-length-scale", type=float, default=0.35)
    parser.add_argument("--bandit-gain-length-scale", type=float, default=0.35)
    parser.add_argument("--bandit-motion-cross-correlation", type=float, default=0.20)
    parser.add_argument("--bandit-light-cross-correlation", type=float, default=0.20)
    parser.add_argument("--bandit-temporal-scale-sec", type=float, default=5.0)
    parser.add_argument("--bandit-observation-noise", type=float, default=0.10)
    parser.add_argument("--bandit-target-scale-floor", type=float, default=0.01)
    parser.add_argument("--bandit-jitter", type=float, default=1e-6)
    parser.add_argument(
        "--evaluation-precision",
        choices=("fp16", "fp32"),
        default="fp32",
        help="Precision of the independent DA-V2 Small evaluation model.",
    )


def validate_bandit_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.bandit_window_size < 1:
        parser.error("--bandit-window-size must be a positive integer.")
    positive_names = (
        "bandit_exposure_length_scale",
        "bandit_gain_length_scale",
        "bandit_temporal_scale_sec",
        "bandit_observation_noise",
        "bandit_target_scale_floor",
        "bandit_jitter",
    )
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0
        for name in positive_names
    ):
        parser.error(
            "Bandit length/temporal scales, observation noise, target scale "
            "floor, and jitter must be finite and positive."
        )
    non_negative_names = ("bandit_exploration_beta", "bandit_switch_penalty")
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) < 0
        for name in non_negative_names
    ):
        parser.error(
            "Bandit exploration beta and switch penalty must be finite and non-negative."
        )
    correlation_names = (
        "bandit_motion_cross_correlation",
        "bandit_light_cross_correlation",
    )
    if any(
        not math.isfinite(getattr(args, name))
        or not 0 <= getattr(args, name) < 1
        for name in correlation_names
    ):
        parser.error("Bandit context cross-correlations must be in [0, 1).")
    if args.evaluation_alignment == "metric":
        parser.error(
            "Fair DA-V2 Small comparison requires scale_shift_depth or "
            "scale_shift_inverse alignment."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=METHOD_NAME)
    configure_parser(parser)
    configure_bandit_parser(parser)
    parser.set_defaults(
        safety_config=DEFAULT_SAFETY_CONFIG,
        evaluation_alignment="scale_shift_inverse",
        min_depth_m=1e-3,
        min_valid_depth_pixels=10000,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args)
    validate_bandit_args(parser, args)
    return args


def build_experiment(args: argparse.Namespace) -> RiskBanditExperiment:
    # Hardware/model imports stay in the builder so unit tests can import this
    # executable without an installed Orbbec SDK or a loaded CUDA model.
    from ati_mde_control.full_depth_predictor import CameraErrorFullDepthPredictor
    from hardware.sensor import OrbbecColorCamera

    config = ExperimentConfig.from_args(args)
    bandit_config = RiskBanditConfig.from_args(args)
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
    safety_policy = SafetyPolicy.from_json(config.safety_path)
    policy = RiskBanditPolicy(bandit_config, safety_policy, config.default_cell)
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
    return RiskBanditExperiment(
        config,
        capture_runner,
        predictor,
        policy,
        logger,
        evaluator,
        safety_policy.brightness_guard,
    )


def main(argv: Sequence[str] | None = None) -> int:
    experiment: RiskBanditExperiment | None = None
    exit_code = 0
    try:
        args = parse_args(argv)
        experiment = build_experiment(args)
        print(
            f"[Start] {METHOD_NAME}; capture rounds={args.max_rounds}; "
            "press Ctrl-C to stop early."
        )
        while experiment.round_index < args.max_rounds:
            started = time.monotonic()
            experiment.run_round()
            remaining = args.round_interval_ms / 1000.0 - (time.monotonic() - started)
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
