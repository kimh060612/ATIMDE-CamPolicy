#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ati_mde_control.bidirectional_exposure_guard import (
    BidirectionalExposureGuard,
    BidirectionalExposureGuardConfig,
)
from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import (
    ExperimentConfig,
    SafetyPolicy,
)
from ati_mde_control.context import build_context_provider
from ati_mde_control.logging import CaptureLogger
from ati_mde_control.predictor import CameraErrorPredictor
from ati_mde_control.risk_bandit_bidirectional_exposure_sync_experiment import (
    METHOD_NAME,
    RiskBanditBidirectionalExposureSyncExperiment,
)
from ati_mde_control.risk_bandit_policy import RiskBanditConfig
from ati_mde_control.saturation_guard import SaturationGuardedRiskBanditPolicy
from orbbec_ati_risk_bandit_bidirectional_exposure_sync import (
    parse_args,
)
from orbbec_deterministic_probing_modelv1 import FairDepthEvaluator


LATENCY_CSV_FIELDS = (
    "round_index",
    "capture_index",
    "timestamp_ns",
    "captured_cell",
    "selected_next_cell",
    "camera_switched",
    "gp_update_status",
    "mde_predict_scores_total_ms",
    "mde_inference_ms",
    "mde_encoder_decoder_gpu_ms",
    "score_prediction_only_gpu_ms",
    "gp_update_ms",
    "gp_selection_ms",
    "gp_bandit_computation_ms",
    "camera_apply_latency_ms",
    "camera_switching_latency_ms",
    "pre_capture_camera_apply_ms",
    "round_total_ms",
    "error",
)


class LatencyCameraErrorPredictor(CameraErrorPredictor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reset_latency()

    def reset_latency(self) -> None:
        self.predict_scores_total_ms: float | None = None
        self.encoder_decoder_gpu_ms: float | None = None
        self.score_prediction_only_gpu_ms: float | None = None

    def _infer_scores(self, images, contexts, exposure_us_values, gains):
        pixels, vectors = self._prepare_inputs(
            images, contexts, exposure_us_values, gains
        )
        inference_started = torch.cuda.Event(enable_timing=True)
        feature_ready = torch.cuda.Event(enable_timing=True)
        inference_finished = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        stream = torch.cuda.current_stream(self.device)
        started = time.perf_counter()
        inference_started.record(stream)
        with torch.inference_mode():
            frozen_feature = self.model._extract_frozen_feature(pixels)
            feature_ready.record(stream)
            shared_feature = self.model.feature_projection(frozen_feature)
            conditioned_feature = self.model._apply_film(shared_feature, vectors)
            camera_bias, variance = self.model._scalar_heads(conditioned_feature)
            log_variance = torch.log(variance.clamp_min(1e-8))
            std = torch.sqrt(variance)
        inference_finished.record(stream)
        torch.cuda.synchronize(self.device)
        self.encoder_decoder_gpu_ms = inference_started.elapsed_time(feature_ready)
        self.score_prediction_only_gpu_ms = feature_ready.elapsed_time(
            inference_finished
        )
        return {
            "predicted_loss": camera_bias,
            "camera_bias": camera_bias,
            "log_variance": log_variance,
            "variance": variance,
            "std": std,
        }, (time.perf_counter() - started) * 1000.0

    def predict_scores(self, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return super().predict_scores(*args, **kwargs)
        finally:
            self.predict_scores_total_ms = (time.perf_counter() - started) * 1000.0


class LatencyRiskBanditPolicy(SaturationGuardedRiskBanditPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reset_latency()

    def reset_latency(self) -> None:
        self.gp_update_ms: float | None = None
        self.gp_selection_ms: float | None = None

    def add_observation(self, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return super().add_observation(*args, **kwargs)
        finally:
            self.gp_update_ms = (time.perf_counter() - started) * 1000.0

    def select_from_candidates(self, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        try:
            return super().select_from_candidates(*args, **kwargs)
        finally:
            self.gp_selection_ms = (time.perf_counter() - started) * 1000.0


class LatencyLogger:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "risk_bandit_latency.csv"
        self.rows: list[dict[str, Any]] = []

    def record(self, values: dict[str, Any]) -> None:
        self.rows.append({field: values.get(field, "") for field in LATENCY_CSV_FIELDS})

    def write(self) -> Path:
        temporary = self.path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LATENCY_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(temporary, self.path)
        return self.path


class LatencyExperiment(RiskBanditBidirectionalExposureSyncExperiment):
    predictor: LatencyCameraErrorPredictor
    policy: LatencyRiskBanditPolicy

    def __init__(
        self, *args: Any, latency_logger: LatencyLogger, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.latency_logger = latency_logger
        self._decision_apply: dict[str, Any] | None = None
        self._pre_capture_apply_ms: float | None = None

    def _apply_cell(self, cell):
        previous = self.current_cell
        is_decision_apply = self.policy.gp_selection_ms is not None
        started = time.perf_counter()
        try:
            return super()._apply_cell(cell)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if is_decision_apply:
                self._decision_apply = {
                    "latency_ms": elapsed_ms,
                    "switched": previous is not None and previous != cell,
                }
            else:
                self._pre_capture_apply_ms = elapsed_ms

    def run_round(self):
        round_index = self.round_index
        self.predictor.reset_latency()
        self.policy.reset_latency()
        self._decision_apply = None
        self._pre_capture_apply_ms = None
        result = None
        error_text = ""
        started = time.perf_counter()
        try:
            result = super().run_round()
            return result
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            raise
        finally:
            capture_row = next(
                (
                    row
                    for row in reversed(self.logger.rows)
                    if row.get("round_index") == round_index
                ),
                {},
            )
            update_ms = self.policy.gp_update_ms
            selection_ms = self.policy.gp_selection_ms
            gp_times = [
                value for value in (update_ms, selection_ms) if value is not None
            ]
            camera_apply_ms = (
                self._decision_apply["latency_ms"]
                if self._decision_apply is not None
                else None
            )
            camera_switched = bool(
                self._decision_apply is not None and self._decision_apply["switched"]
            )
            self.latency_logger.record(
                {
                    "round_index": round_index,
                    "capture_index": capture_row.get("capture_index", ""),
                    "timestamp_ns": capture_row.get("timestamp_ns", ""),
                    "captured_cell": capture_row.get("cell_id", ""),
                    "selected_next_cell": (
                        result.decision.selected_cell.cell_id
                        if result is not None
                        else capture_row.get("active_cell_after", "")
                    ),
                    "camera_switched": int(camera_switched),
                    "gp_update_status": (
                        result.gp_update_status if result is not None else ""
                    ),
                    "mde_predict_scores_total_ms": self.predictor.predict_scores_total_ms,
                    "mde_inference_ms": capture_row.get("mde_inference_ms", ""),
                    "mde_encoder_decoder_gpu_ms": self.predictor.encoder_decoder_gpu_ms,
                    "score_prediction_only_gpu_ms": self.predictor.score_prediction_only_gpu_ms,
                    "gp_update_ms": update_ms,
                    "gp_selection_ms": selection_ms,
                    "gp_bandit_computation_ms": sum(gp_times) if gp_times else None,
                    "camera_apply_latency_ms": camera_apply_ms,
                    "camera_switching_latency_ms": (
                        camera_apply_ms if camera_switched else None
                    ),
                    "pre_capture_camera_apply_ms": self._pre_capture_apply_ms,
                    "round_total_ms": (time.perf_counter() - started) * 1000.0,
                    "error": error_text,
                }
            )


def build_experiment(args: argparse.Namespace) -> LatencyExperiment:
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
    predictor = LatencyCameraErrorPredictor(
        config.checkpoint_path,
        config.model_size,
        config.device,
        config.precision,
        config.q_uncertainty_weight,
        config.local_files_only,
    )
    policy = LatencyRiskBanditPolicy(
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
    return LatencyExperiment(
        config,
        capture_runner,
        predictor,
        policy,
        logger,
        evaluator,
        guard,
        latency_logger=LatencyLogger(config.output_dir),
    )


def main(argv: Sequence[str] | None = None) -> int:
    experiment: LatencyExperiment | None = None
    exit_code = 0
    try:
        args = parse_args(argv)
        experiment = build_experiment(args)
        print(
            f"[Start] latency measurement for {METHOD_NAME}; "
            f"capture rounds={args.max_rounds}; press Ctrl-C to stop early."
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
                path = experiment.latency_logger.write()
                print(f"[Latency] wrote {path}")
            except OSError as error:
                print(f"[ERROR] latency CSV write failed: {error}")
                exit_code = 1
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
