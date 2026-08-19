#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import time

import cv2
import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import ExperimentConfig, SafetyPolicy, configure_parser, validate_args
from ati_mde_control.context import build_context_provider
from ati_mde_control.experiment import CameraControlExperiment
from ati_mde_control.logging import CaptureLogger
from ati_mde_control.pairwise_policy import PairwisePolicy


class FairDepthEvaluator:
    """Post-capture evaluator shared with ``orbbec_iqa_control.py``."""

    def __init__(self, control_predictor, config: ExperimentConfig, precision: str) -> None:
        self.control_predictor = control_predictor
        self.config = config
        self.precision = precision

    def evaluate_rows(self, rows) -> None:
        # Control is complete. Release its camera-error network before loading
        # the independent, common DA-V2 Small evaluation model.
        if not rows:
            return
        import torch
        from orbbec_iqa_control import DepthAnythingV2Small, evaluate_depth_arrays

        if hasattr(self.control_predictor, "model"):
            self.control_predictor.model = None
            gc.collect()
            torch.cuda.empty_cache()

        evaluation_args = argparse.Namespace(
            depth_device=str(self.config.device),
            depth_precision=self.precision,
            depth_alignment=self.config.evaluation_alignment,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
            min_valid_depth_pixels=self.config.min_valid_depth_pixels,
            depth_model_local_files_only=self.config.local_files_only,
        )
        predictor = DepthAnythingV2Small(evaluation_args)
        for row in rows:
            try:
                image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Failed to read {row['image_path']}")
                target = np.load(
                    str(row["depth_path"]), allow_pickle=False
                ).astype(np.float32)
                prediction, inference_ms = predictor.predict(image, target.shape)
                _, metrics = evaluate_depth_arrays(
                    prediction,
                    target,
                    alignment=self.config.evaluation_alignment,
                    min_depth_m=self.config.min_depth_m,
                    max_depth_m=self.config.max_depth_m,
                    min_valid_depth_pixels=self.config.min_valid_depth_pixels,
                )
                row.update(
                    abs_rel=metrics["abs_rel"],
                    a1=metrics["a1"],
                    valid_depth_pixels=metrics["valid_depth_pixels"],
                    evaluation_inference_ms=inference_ms,
                )
                print(
                    f"[Evaluate] round={row['round_index']} "
                    f"capture={row['capture_index']} metrics={metrics}"
                )
            except (OSError, RuntimeError, ValueError) as error:
                row["evaluation_error"] = str(error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local pairwise Orbbec camera control.")
    configure_parser(parser)
    parser.set_defaults(
        evaluation_alignment="scale_shift_inverse",
        min_depth_m=1e-3,
        min_valid_depth_pixels=10000,
    )
    parser.add_argument(
        "--evaluation-precision",
        choices=("fp16", "fp32"),
        default="fp32",
        help="Precision of the independent DA-V2 Small evaluation model.",
    )
    args = parser.parse_args()
    if args.evaluation_alignment == "metric":
        parser.error(
            "Fair DA-V2 Small comparison requires scale_shift_depth or "
            "scale_shift_inverse alignment."
        )
    validate_args(parser, args)
    return args


def build_experiment(args: argparse.Namespace) -> CameraControlExperiment:
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
    policy = PairwisePolicy(config.policy, SafetyPolicy.from_json(config.safety_path))
    capture_runner = CaptureRunner(camera, context_provider, config.max_pair_capture_gap_ms)
    logger = CaptureLogger(config.output_dir)
    evaluator = FairDepthEvaluator(predictor, config, args.evaluation_precision)
    return CameraControlExperiment(config, capture_runner, predictor, policy, logger, evaluator)


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
