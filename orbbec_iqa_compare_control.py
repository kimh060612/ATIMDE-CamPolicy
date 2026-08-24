#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence

from ati_mde_control.config import configure_parser, validate_args
from ati_mde_control.full_depth_predictor import FullDepthBatchPrediction
from hardware.utils import QScore
from iqa_control.noise_aware_iqa_control import noise_aware_iqa
import orbbec_ati_risk_bandit as risk_bandit


METHOD_NAME = f"{risk_bandit.METHOD_NAME} with Noise-Aware IQA"


class IQARiskPredictor:
    """Keep the full-depth path but replace its learned risk score with IQA."""

    def __init__(self, full_depth_predictor, resize_factor: float = 1.0) -> None:
        self.full_depth_predictor = full_depth_predictor
        self.resize_factor = resize_factor

    def predict_batch(
        self,
        images,
        contexts,
        exposure_us_values,
        gains,
    ) -> FullDepthBatchPrediction:
        prediction = self.full_depth_predictor.predict_batch(
            images,
            contexts,
            exposure_us_values,
            gains,
        )
        if len(prediction.scores) != len(images):
            raise RuntimeError("Full-depth predictor returned the wrong score batch size.")

        scores = []
        for image, base_score in zip(images, prediction.scores):
            iqa = noise_aware_iqa(image, self.resize_factor)
            risk = -iqa.score  # The GP bandit minimizes Q; IQA is maximized.
            scores.append(
                QScore(
                    q=risk,
                    uncertainty=0.0,
                    mu=risk,
                    extra={
                        **base_score.extra,
                        "iqa_score": iqa.score,
                        "iqa_gradient": iqa.gradient,
                        "iqa_entropy": iqa.entropy,
                        "iqa_noise": iqa.noise,
                        "iqa_mean_intensity": iqa.mean_intensity,
                    },
                )
            )
        return FullDepthBatchPrediction(tuple(scores), prediction.depth_maps)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=METHOD_NAME)
    configure_parser(parser)
    risk_bandit.configure_bandit_parser(parser)
    parser.add_argument("--iqa-resize-factor", type=float, default=1.0)
    parser.set_defaults(
        safety_config=risk_bandit.DEFAULT_SAFETY_CONFIG,
        evaluation_alignment="scale_shift_inverse",
        min_depth_m=1e-3,
        min_valid_depth_pixels=10000,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args)
    risk_bandit.validate_bandit_args(parser, args)
    if not math.isfinite(args.iqa_resize_factor) or args.iqa_resize_factor <= 0:
        parser.error("--iqa-resize-factor must be finite and positive.")
    return args


def build_experiment(args):
    experiment = risk_bandit.build_experiment(args)
    experiment.predictor = IQARiskPredictor(
        experiment.predictor,
        args.iqa_resize_factor,
    )
    return experiment


def main(argv: Sequence[str] | None = None) -> int:
    experiment = None
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
