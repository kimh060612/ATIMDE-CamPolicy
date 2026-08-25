#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from ati_mde_control.bidirectional_exposure_guard import (
    BidirectionalExposureGuard,
    BidirectionalExposureGuardConfig,
)
from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import (
    ExperimentConfig,
    SafetyPolicy,
    configure_parser,
    validate_args,
)
from ati_mde_control.context import build_context_provider
from ati_mde_control.logging import CaptureLogger
from ati_mde_control.risk_bandit_bidirectional_exposure_experiment import (
    METHOD_NAME,
    RiskBanditBidirectionalExposureExperiment,
)
from ati_mde_control.risk_bandit_policy import RiskBanditConfig
from ati_mde_control.saturation_guard import SaturationGuardedRiskBanditPolicy
from orbbec_ati_risk_bandit import (
    DEFAULT_SAFETY_CONFIG,
    configure_bandit_parser,
    validate_bandit_args,
)
from orbbec_ati_risk_bandit_predictive_saturation import (
    configure_predictive_saturation_parser,
    validate_predictive_saturation_args,
)
from orbbec_deterministic_probing_modelv1 import FairDepthEvaluator


def configure_bidirectional_exposure_parser(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--exposure-max-ev-step-stops", type=int, default=1)
    parser.add_argument("--shadow-pixel-threshold", type=float, default=10.0)
    parser.add_argument("--shadow-soft-ratio", type=float, default=0.30)
    parser.add_argument("--shadow-hard-ratio", type=float, default=0.50)
    parser.add_argument("--shadow-soft-mean-luminance", type=float, default=60.0)
    parser.add_argument("--shadow-hard-mean-luminance", type=float, default=40.0)
    parser.add_argument("--shadow-recovery-ratio", type=float, default=0.15)
    parser.add_argument("--shadow-recovery-mean-luminance", type=float, default=80.0)
    parser.add_argument("--shadow-recovery-frames", type=int, default=3)
    parser.add_argument("--shadow-projected-ratio-limit", type=float, default=0.30)


def validate_bidirectional_exposure_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    try:
        BidirectionalExposureGuardConfig.from_args(args)
    except ValueError as error:
        parser.error(str(error))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=METHOD_NAME)
    configure_parser(parser)
    configure_bandit_parser(parser)
    configure_predictive_saturation_parser(parser)
    configure_bidirectional_exposure_parser(parser)
    parser.set_defaults(
        safety_config=DEFAULT_SAFETY_CONFIG,
        evaluation_alignment="scale_shift_inverse",
        min_depth_m=1e-3,
        min_valid_depth_pixels=10000,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args)
    validate_bandit_args(parser, args)
    validate_predictive_saturation_args(parser, args)
    validate_bidirectional_exposure_args(parser, args)
    return args


def build_experiment(
    args: argparse.Namespace,
) -> RiskBanditBidirectionalExposureExperiment:
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
    policy = SaturationGuardedRiskBanditPolicy(
        RiskBanditConfig.from_args(args),
        SafetyPolicy.from_json(config.safety_path),
        config.default_cell,
    )
    capture_runner = CaptureRunner(
        camera, context_provider, config.max_pair_capture_gap_ms
    )
    logger = CaptureLogger(config.output_dir)
    evaluator = FairDepthEvaluator(predictor, config, args.evaluation_precision)
    guard = BidirectionalExposureGuard(
        BidirectionalExposureGuardConfig.from_args(args), policy.safe_fallback
    )
    return RiskBanditBidirectionalExposureExperiment(
        config,
        capture_runner,
        predictor,
        policy,
        logger,
        evaluator,
        guard,
    )


def main(argv: Sequence[str] | None = None) -> int:
    experiment: RiskBanditBidirectionalExposureExperiment | None = None
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
