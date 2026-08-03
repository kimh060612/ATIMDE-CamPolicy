#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from hardware.utils import (
    EXPOSURE_MS_VALUES,
    GAIN_VALUES,
    ContextKey,
    QScore,
    SensorCell,
)
from model.model_camerror import MODEL_IDS, CameraInducedErrorModel
from model.utils import CameraParameterRange, normalize_camera_parameters
from policy.basic_policy import (
    ATIMDECameraProbingController,
    ObservationUpdate,
    PolicyDecision,
    SafetyPolicy,
)


MOTION_LABEL_TO_STATE = {
    "fast": 0,
    "slow": 1,
    "stop": 2,
    "rotate": 3,
    "spin": 4,
}
LIGHT_LABEL_TO_STATE = {"normal": 0, "dim": 1, "dark": 2}
MOTION_STATE_TO_LABEL = {value: key for key, value in MOTION_LABEL_TO_STATE.items()}
LIGHT_STATE_TO_LABEL = {value: key for key, value in LIGHT_LABEL_TO_STATE.items()}

CSV_FIELDNAMES = [
    "record_type",
    "timestamp_ns",
    "round_index",
    "capture_index",
    "capture_role",
    "motion_state",
    "motion_label",
    "light_state",
    "lighting_label",
    "cell_id",
    "exposure_ms",
    "exposure_us_model",
    "gain",
    "requested_exposure_raw",
    "actual_exposure_raw",
    "actual_gain",
    "camera_parameter_ms",
    "mde_inference_ms",
    "mde_batch_size",
    "control_decision_delay_ms",
    "delay_warning",
    "delay_reasons",
    "camera_bias",
    "std",
    "q",
    "probe_step",
    "probability_good",
    "accept_threshold",
    "accept_probability",
    "cell_success_score_before",
    "cell_success_score_after",
    "consecutive_bad_frames",
    "challenger_success_score",
    "challenger_probability_good",
    "challenger_cooldown",
    "selection_source",
    "probe_stop_reason",
    "frame_action",
    "round_action",
    "decision_reason",
    "selected",
    "dropped",
    "stale_for_control",
    "control_context",
    "discard_reason",
    "offload_requested",
    "image_path",
    "depth_path",
    "abs_rel",
    "a1",
    "valid_depth_pixels",
    "evaluation_inference_ms",
    "evaluation_error",
]


class ContextProvider(Protocol):
    def get(self) -> ContextKey: ...

    def close(self) -> None: ...


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(
        checkpoint, dict
    ) else checkpoint
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint does not contain a model state dict: {path}")
    if all(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key).removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


class CameraErrorPredictor:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        model_size: str,
        device: str,
        precision: str,
        q_uncertainty_weight: float,
        local_files_only: bool,
    ) -> None:
        if model_size not in {"small", "base"}:
            raise ValueError("model_size must be 'small' or 'base'.")
        if precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be 'fp16' or 'fp32'.")
        if q_uncertainty_weight < 0.0:
            raise ValueError("q_uncertainty_weight must be non-negative.")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested but unavailable.")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "This real-time experiment and its final AbsRel/A1 evaluation "
                "require a CUDA GPU."
            )
        self.use_fp16 = precision == "fp16" and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.q_uncertainty_weight = q_uncertainty_weight

        model_id = MODEL_IDS[model_size]
        print(f"[Model] loading processor {model_id}")
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.model = CameraInducedErrorModel(
            model_id=model_id,
            context_dim=10,
            feature_channels=64,
            hidden_channels=64,
            film_hidden_dim=128,
            max_bias=0.5,
            min_log_variance=-9.21,
            max_log_variance=-1.39,
            initial_std=0.07,
            variance_head_init_std=1e-3,
        )
        load_result = self.model.load_state_dict(
            _load_checkpoint(checkpoint_path), strict=True
        )
        print(f"[Model] checkpoint loaded: {load_result}")
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.parameter_range = CameraParameterRange(
            exposure_min=4_000.0,
            exposure_max=32_000.0,
            gain_min=16.0,
            gain_max=128.0,
        )

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True

    def _context_vector(
        self,
        *,
        light_label: str,
        motion_label: str,
        exposure_us: float,
        gain: float,
    ) -> torch.Tensor:
        light = F.one_hot(
            torch.tensor(LIGHT_LABEL_TO_STATE[light_label]),
            num_classes=len(LIGHT_LABEL_TO_STATE),
        ).float()
        motion = F.one_hot(
            torch.tensor(MOTION_LABEL_TO_STATE[motion_label]),
            num_classes=len(MOTION_LABEL_TO_STATE),
        ).float()
        camera = normalize_camera_parameters(
            torch.tensor(exposure_us, dtype=torch.float32),
            torch.tensor(gain, dtype=torch.float32),
            self.parameter_range,
        )
        return torch.cat((light, motion, camera)).unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )

    def _inference_outputs(
        self,
        images_bgr: Sequence[np.ndarray],
        *,
        contexts: Sequence[ContextKey],
        exposure_us_values: Sequence[float],
        gains: Sequence[float],
        target_size: Optional[tuple[int, int]] = None,
    ) -> tuple[dict[str, torch.Tensor], float]:
        batch_size = len(images_bgr)
        if batch_size == 0 or not (
            len(contexts) == len(exposure_us_values) == len(gains) == batch_size
        ):
            raise ValueError("Inference batch inputs must have the same non-zero length.")

        rgb_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_bgr]
        pixel_values = self.processor(images=rgb_images, return_tensors="pt")[
            "pixel_values"
        ].to(device=self.device, dtype=self.dtype, non_blocking=True)
        context_vectors = torch.cat(
            [
                self._context_vector(
                    light_label=LIGHT_STATE_TO_LABEL[context.light_state],
                    motion_label=MOTION_STATE_TO_LABEL[context.motion_state],
                    exposure_us=exposure_us,
                    gain=gain,
                )
                for context, exposure_us, gain in zip(
                    contexts, exposure_us_values, gains
                )
            ],
            dim=0,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.inference(
                candidate_img=pixel_values,
                context=context_vectors,
                target_size=target_size or tuple(pixel_values.shape[-2:]),
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        return outputs, inference_ms

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        context: ContextKey,
        exposure_us: float,
        gain: float,
    ) -> QScore:
        return self.predict_batch(
            [image_bgr],
            contexts=[context],
            exposure_us_values=[exposure_us],
            gains=[gain],
        )[0]

    def predict_batch(
        self,
        images_bgr: Sequence[np.ndarray],
        *,
        contexts: Sequence[ContextKey],
        exposure_us_values: Sequence[float],
        gains: Sequence[float],
    ) -> list[QScore]:
        outputs, inference_ms = self._inference_outputs(
            images_bgr,
            contexts=contexts,
            exposure_us_values=exposure_us_values,
            gains=gains,
        )
        biases = outputs["camera_bias"].detach().float().cpu().tolist()
        standard_deviations = outputs["std"].detach().float().cpu().tolist()
        batch_size = len(images_bgr)
        if len(biases) != batch_size or len(standard_deviations) != batch_size:
            raise RuntimeError(
                "Camera error model returned a different number of batch outputs."
            )
        scores: list[QScore] = []
        for camera_bias, std in zip(biases, standard_deviations):
            q_value = camera_bias + self.q_uncertainty_weight * std
            if not all(
                math.isfinite(value) for value in (camera_bias, std, q_value)
            ):
                raise ValueError(
                    f"Non-finite prediction: bias={camera_bias}, std={std}, q={q_value}"
                )
            scores.append(
                QScore(
                    q=q_value,
                    uncertainty=std,
                    mu=camera_bias,
                    extra={
                        "camera_bias": camera_bias,
                        "std": std,
                        "mde_inference_ms": inference_ms,
                        "mde_batch_size": float(batch_size),
                    },
                )
            )
        return scores

    @staticmethod
    def _fit_scale_shift(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_values = prediction[valid].float()
        target_values = target[valid].float()
        count = prediction.new_tensor(float(predicted_values.numel())).float()
        a00 = torch.sum(predicted_values * predicted_values)
        a01 = torch.sum(predicted_values)
        b0 = torch.sum(predicted_values * target_values)
        b1 = torch.sum(target_values)
        determinant = a00 * count - a01 * a01
        if abs(float(determinant.item())) < 1e-12:
            raise ValueError("Depth scale/shift alignment is singular.")
        scale = (count * b0 - a01 * b1) / determinant
        shift = (-a01 * b0 + a00 * b1) / determinant
        return scale, shift

    def evaluate(
        self,
        image_bgr: np.ndarray,
        ground_truth_depth_m: np.ndarray,
        *,
        context: ContextKey,
        exposure_us: float,
        gain: float,
        alignment_mode: str,
        min_depth_m: float,
        max_depth_m: float,
        min_valid_pixels: int,
    ) -> dict[str, float]:
        if self.device.type != "cuda":
            raise RuntimeError("Final AbsRel/A1 evaluation requires a CUDA GPU.")

        outputs, inference_ms = self._inference_outputs(
            [image_bgr],
            contexts=[context],
            exposure_us_values=[exposure_us],
            gains=[gain],
            target_size=ground_truth_depth_m.shape,
        )
        prediction = outputs["candidate_depth"][0, 0].float()
        ground_truth = torch.from_numpy(
            np.ascontiguousarray(ground_truth_depth_m, dtype=np.float32)
        ).to(self.device, non_blocking=True)
        valid = (
            torch.isfinite(ground_truth)
            & (ground_truth >= min_depth_m)
            & (ground_truth <= max_depth_m)
            & torch.isfinite(prediction)
        )
        if int(valid.sum().item()) < min_valid_pixels:
            raise ValueError("Not enough valid depth pixels for evaluation.")

        if alignment_mode == "metric":
            aligned_depth = prediction
        elif alignment_mode == "scale_shift_depth":
            scale, shift = self._fit_scale_shift(prediction, ground_truth, valid)
            aligned_depth = scale * prediction + shift
        elif alignment_mode == "scale_shift_inverse":
            inverse_ground_truth = torch.reciprocal(ground_truth.clamp_min(1e-6))
            scale, shift = self._fit_scale_shift(
                prediction, inverse_ground_truth, valid
            )
            aligned_inverse = scale * prediction + shift
            valid = valid & torch.isfinite(aligned_inverse) & (aligned_inverse > 1e-6)
            aligned_depth = torch.reciprocal(aligned_inverse.clamp_min(1e-6))
        else:
            raise ValueError(f"Unknown evaluation alignment mode: {alignment_mode}")

        valid = valid & torch.isfinite(aligned_depth) & (aligned_depth > 1e-6)
        valid_count = int(valid.sum().item())
        if valid_count < min_valid_pixels:
            raise ValueError("Not enough valid aligned depth pixels for evaluation.")
        predicted_values = aligned_depth[valid]
        target_values = ground_truth[valid]
        abs_rel = torch.mean(torch.abs(predicted_values - target_values) / target_values)
        ratio = torch.maximum(
            predicted_values / target_values,
            target_values / predicted_values,
        )
        a1 = torch.mean((ratio < 1.25).float())
        return {
            "abs_rel": float(abs_rel.item()),
            "a1": float(a1.item()),
            "valid_depth_pixels": float(valid_count),
            "evaluation_inference_ms": inference_ms,
        }


class RosTrainingContextAdapter:
    """Map the hardware provider's stop-first indices to training label order."""

    HARDWARE_STATE_TO_LABEL = {
        0: "stop",
        1: "slow",
        2: "fast",
        3: "rotate",
        4: "spin",
    }

    def __init__(self, provider: ContextProvider) -> None:
        self.provider = provider

    def get(self) -> ContextKey:
        context = self.provider.get()
        label = self.HARDWARE_STATE_TO_LABEL[context.motion_state]
        return ContextKey(MOTION_LABEL_TO_STATE[label], context.light_state)

    def close(self) -> None:
        self.provider.close()


class DebouncedContextProvider:
    """Commit a context change only after it is both repeated and sustained."""

    def __init__(
        self,
        provider: ContextProvider,
        *,
        debounce_ms: float,
        minimum_samples: int,
    ) -> None:
        if debounce_ms < 0.0:
            raise ValueError("context debounce duration must be non-negative.")
        if minimum_samples < 1:
            raise ValueError("context debounce samples must be positive.")
        self.provider = provider
        self.debounce_sec = debounce_ms / 1000.0
        self.minimum_samples = minimum_samples
        self.committed = provider.get()
        self.committed.validate()
        self.candidate: Optional[ContextKey] = None
        self.candidate_since = 0.0
        self.candidate_samples = 0

    def get(self) -> ContextKey:
        observed = self.provider.get()
        observed.validate()
        now = time.monotonic()
        if observed == self.committed:
            self.candidate = None
            self.candidate_samples = 0
            return self.committed

        if observed != self.candidate:
            self.candidate = observed
            self.candidate_since = now
            self.candidate_samples = 1
        else:
            self.candidate_samples += 1

        if (
            self.candidate_samples >= self.minimum_samples
            and now - self.candidate_since >= self.debounce_sec
        ):
            previous = self.committed
            self.committed = observed
            self.candidate = None
            self.candidate_samples = 0
            print(
                f"[Context] confirmed {previous.table_key}->{observed.table_key} "
                f"after {self.debounce_sec * 1000.0:.0f}ms"
            )
        return self.committed

    @property
    def is_stable(self) -> bool:
        return self.candidate is None

    def transition_contexts(self) -> tuple[ContextKey, ...]:
        if self.candidate is None:
            return (self.committed,)
        return (self.committed, self.candidate)

    def close(self) -> None:
        self.provider.close()


class ModelV1Experiment:
    def __init__(
        self,
        *,
        camera: Any,
        context_provider: ContextProvider,
        predictor: CameraErrorPredictor,
        policy: ATIMDECameraProbingController,
        output_dir: Path,
        default_cell: SensorCell,
        camera_parameter_warn_ms: float,
        mde_inference_warn_ms: float,
        control_decision_warn_ms: float,
        probe_decision_budget_ms: float,
        inference_poll_ms: float,
        evaluation_alignment: str,
        min_depth_m: float,
        max_depth_m: float,
        min_valid_depth_pixels: int,
    ) -> None:
        if min(
            camera_parameter_warn_ms,
            mde_inference_warn_ms,
            control_decision_warn_ms,
            probe_decision_budget_ms,
            inference_poll_ms,
        ) <= 0.0:
            raise ValueError("Latency warning and polling intervals must be positive.")

        self.camera = camera
        self.context_provider = context_provider
        self.predictor = predictor
        self.policy = policy
        self.output_dir = output_dir
        self.default_cell = default_cell
        self.camera_parameter_warn_ms = camera_parameter_warn_ms
        self.mde_inference_warn_ms = mde_inference_warn_ms
        self.control_decision_warn_ms = control_decision_warn_ms
        self.probe_decision_budget_ms = probe_decision_budget_ms
        self.inference_poll_sec = inference_poll_ms / 1000.0
        self.evaluation_alignment = evaluation_alignment
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.min_valid_depth_pixels = min_valid_depth_pixels
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ati-mde")
        self.rows: list[dict[str, Any]] = []
        self.round_index = 0
        self.capture_index = 0
        self._executor_stopped = False

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "depth_gt").mkdir(parents=True, exist_ok=True)

    def _add_delay(self, row: dict[str, Any], reason: str) -> None:
        reasons: list[str] = row["_delay_reasons"]
        if reason not in reasons:
            reasons.append(reason)
        row["delay_warning"] = 1

    def _apply_control_cell(self, cell: SensorCell, *, reason: str) -> None:
        started = time.perf_counter()
        self.camera.apply_cell(cell)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > self.camera_parameter_warn_ms:
            print(
                f"[LATENCY WARNING] round={self.round_index} reason={reason} "
                f"camera_parameter_ms={elapsed_ms:.2f}"
            )

    def _capture(
        self,
        cell: SensorCell,
        context: ContextKey,
        role: str,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        parameter_started = time.perf_counter()
        requested_raw, actual_raw, actual_gain = self.camera.apply_cell(cell)
        camera_parameter_ms = (time.perf_counter() - parameter_started) * 1000.0
        image_bgr, depth_gt_m = self.camera.capture_rgbd()
        capture_time_ns = time.time_ns()

        # The model was trained with microseconds (4000..32000), never raw
        # Orbbec property units. Convert actual device exposure through ms first.
        exposure_us = float(cell.exposure_ms) * 1000.0
        if actual_raw is not None:
            actual_exposure_ms = actual_raw / self.camera.exposure_value_per_ms
            exposure_us = float(actual_exposure_ms) * 1000.0

        stem = (
            f"round_{self.round_index:04d}_capture_{self.capture_index:05d}_"
            f"{cell.cell_id}_{capture_time_ns}"
        )
        image_path = self.output_dir / "images" / f"{stem}.png"
        depth_path = self.output_dir / "depth_gt" / f"{stem}.npy"

        row: dict[str, Any] = {
            "record_type": "capture",
            "timestamp_ns": capture_time_ns,
            "round_index": self.round_index,
            "capture_index": self.capture_index,
            "capture_role": role,
            "motion_state": context.motion_state,
            "motion_label": MOTION_STATE_TO_LABEL[context.motion_state],
            "light_state": context.light_state,
            "lighting_label": LIGHT_STATE_TO_LABEL[context.light_state],
            "cell_id": cell.cell_id,
            "exposure_ms": cell.exposure_ms,
            "exposure_us_model": exposure_us,
            "gain": cell.gain,
            "requested_exposure_raw": requested_raw,
            "actual_exposure_raw": actual_raw,
            "actual_gain": actual_gain,
            "camera_parameter_ms": camera_parameter_ms,
            "mde_inference_ms": "",
            "mde_batch_size": "",
            "control_decision_delay_ms": "",
            "delay_warning": 0,
            "_delay_reasons": [],
            "camera_bias": "",
            "std": "",
            "q": "",
            "probe_step": 0 if role == "initial" else "",
            "probability_good": "",
            "accept_threshold": self.policy.accept_threshold,
            "accept_probability": self.policy.accept_probability,
            "cell_success_score_before": "",
            "cell_success_score_after": "",
            "consecutive_bad_frames": "",
            "challenger_success_score": "",
            "challenger_probability_good": "",
            "challenger_cooldown": "",
            "selection_source": "",
            "probe_stop_reason": "",
            "frame_action": "",
            "round_action": "",
            "decision_reason": "",
            "selected": 0,
            "dropped": 0,
            "stale_for_control": 0,
            "control_context": "",
            "discard_reason": "",
            "offload_requested": 0,
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "abs_rel": "",
            "a1": "",
            "valid_depth_pixels": "",
            "evaluation_inference_ms": "",
            "evaluation_error": "",
        }
        if camera_parameter_ms > self.camera_parameter_warn_ms:
            self._add_delay(row, "camera_parameter")
            print(
                f"[LATENCY WARNING] round={self.round_index} "
                f"capture={self.capture_index} camera_parameter_ms="
                f"{camera_parameter_ms:.2f}"
            )
        self.rows.append(row)
        self.capture_index += 1
        return row, image_bgr, depth_gt_m

    @staticmethod
    def _save_capture(
        row: dict[str, Any], image_bgr: np.ndarray, depth_gt_m: np.ndarray
    ) -> None:
        if not cv2.imwrite(str(row["image_path"]), image_bgr):
            raise OSError(f"Failed to save RGB image: {row['image_path']}")
        np.save(
            str(row["depth_path"]),
            np.ascontiguousarray(depth_gt_m, dtype=np.float32),
        )

    def _store_score(self, row: dict[str, Any], score: QScore) -> None:
        inference_ms = float(score.extra["mde_inference_ms"])
        row.update(
            {
                "camera_bias": score.mu,
                "std": score.uncertainty,
                "q": score.q,
                "mde_inference_ms": inference_ms,
                "mde_batch_size": int(score.extra.get("mde_batch_size", 1.0)),
            }
        )
        if inference_ms > self.mde_inference_warn_ms:
            self._add_delay(row, "mde_inference")
            print(
                f"[LATENCY WARNING] round={row['round_index']} "
                f"capture={row['capture_index']} mde_inference_ms={inference_ms:.2f}"
            )

    @staticmethod
    def _store_observation_update(
        row: dict[str, Any], update: ObservationUpdate
    ) -> None:
        row.update(
            {
                "probability_good": update.probability_good,
                "cell_success_score_before": update.success_score_before,
                "cell_success_score_after": update.success_score_after,
            }
        )

    def _await_probe_result(
        self,
        row: dict[str, Any],
        future: Future[QScore],
    ) -> None:
        while not future.done():
            self.context_provider.get()
            wait([future], timeout=self.inference_poll_sec)

        self._store_score(row, future.result())

    def _finish_decision(
        self,
        rows: Sequence[dict[str, Any]],
        decision: PolicyDecision,
        *,
        stale_context: Optional[ContextKey] = None,
    ) -> None:
        decision_time_ns = time.time_ns()
        for row in rows:
            row["probe_stop_reason"] = decision.reason
            if stale_context is None:
                row["round_action"] = decision.action
                row["decision_reason"] = decision.reason
                row["offload_requested"] = int(decision.action == "offload")
                row["control_context"] = (
                    f"{row['motion_state']},{row['light_state']}"
                )
            else:
                row["round_action"] = "deferred"
                row["decision_reason"] = (
                    f"{decision.reason};control_deferred_context_changed"
                )
                row["offload_requested"] = 0
                row["stale_for_control"] = 1
                row["control_context"] = stale_context.table_key
                row["discard_reason"] = (
                    f"control_only:{row['motion_state']},{row['light_state']}"
                    f"->{stale_context.table_key}"
                )
            row["selected"] = int(row["cell_id"] == decision.selected_cell.cell_id)
            delay_ms = max(
                0.0, (decision_time_ns - int(row["timestamp_ns"])) / 1_000_000.0
            )
            row["control_decision_delay_ms"] = delay_ms
            if delay_ms > self.control_decision_warn_ms:
                self._add_delay(row, "control_decision")
                print(
                    f"[LATENCY WARNING] round={row['round_index']} "
                    f"capture={row['capture_index']} "
                    f"control_decision_delay_ms={delay_ms:.2f}"
                )

    def _context_state(
        self,
    ) -> tuple[ContextKey, bool, tuple[ContextKey, ...]]:
        context = self.context_provider.get()
        stable = bool(getattr(self.context_provider, "is_stable", True))
        transition_contexts = getattr(
            self.context_provider,
            "transition_contexts",
            lambda: (context,),
        )()
        return context, stable, transition_contexts

    def _defer_for_transition(
        self,
        capture_context: ContextKey,
        rows: Sequence[dict[str, Any]],
        decision: PolicyDecision,
    ) -> bool:
        current_context, stable, transition_contexts = self._context_state()
        if stable and current_context == capture_context:
            return False

        if stable:
            control_cell = self.policy.cell_for_context(
                current_context, self.default_cell
            )
            control_context = current_context
            reason = "stable_context_active_cell"
        else:
            control_cell = self.policy.bridge_cell(
                transition_contexts, self.default_cell
            )
            control_context = transition_contexts[-1]
            reason = "transition_bridge"

        for row in rows:
            row["selection_source"] = "context_bridge"
        self._finish_decision(
            rows,
            decision,
            stale_context=control_context,
        )
        if rows[-1]["cell_id"] != control_cell.cell_id:
            self._apply_control_cell(control_cell, reason=reason)
        print(
            f"[Context control] capture={capture_context.table_key} "
            f"control={control_context.table_key} cell={control_cell.cell_id} "
            f"reason={reason}; context success statistics retained."
        )
        return True

    def _probe_fits_budget(self, initial_row: dict[str, Any]) -> bool:
        elapsed_ms = max(
            0.0,
            (time.time_ns() - int(initial_row["timestamp_ns"])) / 1_000_000.0,
        )
        next_probe_ms = (
            float(initial_row["camera_parameter_ms"])
            + float(initial_row["mde_inference_ms"])
        )
        return elapsed_ms + next_probe_ms <= self.probe_decision_budget_ms

    def run_round(self) -> None:
        context, stable, transition_contexts = self._context_state()
        context.validate()
        self.policy.start_round(context)
        current_cell = (
            self.policy.cell_for_context(context, self.default_cell)
            if stable
            else self.policy.bridge_cell(transition_contexts, self.default_cell)
        )
        print(
            f"\n[Round {self.round_index}] context={context.table_key} "
            f"cell={current_cell.cell_id} stable={stable}"
        )

        initial_row, initial_image, initial_depth = self._capture(
            current_cell, context, "initial"
        )
        initial_future = self.executor.submit(
            self.predictor.predict,
            initial_image,
            context=context,
            exposure_us=float(initial_row["exposure_us_model"]),
            gain=float(
                initial_row["actual_gain"]
                if initial_row["actual_gain"] not in (None, "")
                else initial_row["gain"]
            ),
        )
        self._save_capture(initial_row, initial_image, initial_depth)
        self._await_probe_result(initial_row, initial_future)
        initial_score = QScore(
            q=float(initial_row["q"]),
            uncertainty=float(initial_row["std"]),
            mu=float(initial_row["camera_bias"]),
        )
        initial_update = self.policy.observe(
            context, current_cell, initial_score
        )
        self._store_observation_update(initial_row, initial_update)
        current_acceptable = self.policy.is_acceptable(
            initial_update.probability_good
        )
        initial_row["frame_action"] = "use" if current_acceptable else "probe"
        print(
            f"[Decision] bias={initial_score.mu:.6f} std="
            f"{initial_score.uncertainty:.6f} q={initial_score.q:.6f} "
            f"p_good={initial_update.probability_good:.6f} "
            f"acceptable={current_acceptable}"
        )

        cached_decision = PolicyDecision(
            "use",
            "capture_context_result_cached",
            current_cell,
            initial_score,
        )
        if self._defer_for_transition(
            context, [initial_row], cached_decision
        ):
            self.round_index += 1
            return

        should_probe = self.policy.record_current_result(
            context, current_cell, initial_update.probability_good
        )
        initial_row["consecutive_bad_frames"] = (
            self.policy.consecutive_bad_frames(context)
        )
        if current_acceptable:
            initial_row["selection_source"] = "current_acceptable"
            final_decision = PolicyDecision(
                "use", "current_acceptable", current_cell, initial_score
            )
            self._finish_decision([initial_row], final_decision)
            self.round_index += 1
            return

        if not should_probe:
            initial_row["selection_source"] = "bad_frame_hysteresis"
            final_decision = PolicyDecision(
                "use", "bad_frame_hysteresis", current_cell, initial_score
            )
            self._finish_decision([initial_row], final_decision)
            self.round_index += 1
            return

        challenger = self.policy.select_challenger(context, current_cell)
        if challenger is None or not self._probe_fits_budget(initial_row):
            reason = (
                "no_available_challenger"
                if challenger is None
                else "probe_decision_budget_exhausted"
            )
            initial_row["selection_source"] = "bad_frame_hysteresis"
            final_decision = PolicyDecision(
                "offload", reason, current_cell, initial_score
            )
            self._finish_decision([initial_row], final_decision)
            self.policy.invoke_offload(context, final_decision)
            self.round_index += 1
            return

        if self._defer_for_transition(
            context, [initial_row], cached_decision
        ):
            self.round_index += 1
            return

        challenger_success_score = self.policy.success_score(context, challenger)
        print(
            f"[Probe] context={context.table_key} "
            f"challenger={challenger.cell_id} "
            f"success_score={challenger_success_score:.6f}"
        )
        probe_row, probe_image, probe_depth = self._capture(
            challenger, context, "probe"
        )
        probe_row.update(
            {
                "probe_step": 1,
                "challenger_success_score": challenger_success_score,
            }
        )
        probe_future = self.executor.submit(
            self.predictor.predict,
            probe_image,
            context=context,
            exposure_us=float(probe_row["exposure_us_model"]),
            gain=float(
                probe_row["actual_gain"]
                if probe_row["actual_gain"] not in (None, "")
                else probe_row["gain"]
            ),
        )
        self._save_capture(probe_row, probe_image, probe_depth)
        self._await_probe_result(probe_row, probe_future)

        probe_score = QScore(
            q=float(probe_row["q"]),
            uncertainty=float(probe_row["std"]),
            mu=float(probe_row["camera_bias"]),
        )
        probe_update = self.policy.observe(
            context, challenger, probe_score
        )
        self._store_observation_update(probe_row, probe_update)
        probe_row["frame_action"] = "probe_candidate"
        round_rows = [initial_row, probe_row]

        probe_cached_decision = PolicyDecision(
            "use",
            "capture_context_result_cached",
            challenger,
            probe_score,
        )
        if self._defer_for_transition(
            context, round_rows, probe_cached_decision
        ):
            self.round_index += 1
            return

        selected = self.policy.resolve_challenger(
            context,
            current_cell,
            challenger,
            probe_update.probability_good,
        )
        challenger_acceptable = selected == challenger
        selection_source = (
            "challenger_acceptable"
            if challenger_acceptable
            else "challenger_rejected"
        )
        final_decision = PolicyDecision(
            "use" if challenger_acceptable else "offload",
            selection_source,
            selected,
            probe_score if challenger_acceptable else initial_score,
        )
        challenger_cooldown = self.policy.challenger_cooldown(context, challenger)
        consecutive_bad_frames = self.policy.consecutive_bad_frames(context)
        for row in round_rows:
            row.update(
                {
                    "selection_source": selection_source,
                    "consecutive_bad_frames": consecutive_bad_frames,
                    "challenger_success_score": challenger_success_score,
                    "challenger_probability_good": probe_update.probability_good,
                    "challenger_cooldown": challenger_cooldown,
                }
            )

        self._finish_decision(round_rows, final_decision)
        print(
            f"[Probe result] selected={final_decision.selected_cell.cell_id} "
            f"p_good={probe_update.probability_good:.6f} "
            f"action={final_decision.action} reason={final_decision.reason}"
        )

        if not challenger_acceptable:
            self._apply_control_cell(
                current_cell, reason="challenger_rejected"
            )
        if final_decision.action == "offload":
            self.policy.invoke_offload(context, final_decision)
        self.round_index += 1

    def stop_inference(self) -> None:
        if not self._executor_stopped:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self._executor_stopped = True

    def finalize(self) -> Path:
        self.stop_inference()
        evaluation_failure: Optional[Exception] = None
        try:
            if self.rows and self.predictor.device.type != "cuda":
                raise RuntimeError("Final AbsRel/A1 evaluation requires a CUDA GPU.")
            print(f"[Evaluation] computing AbsRel/A1 for {len(self.rows)} captures on GPU")
            for index, row in enumerate(self.rows, start=1):
                try:
                    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
                    if image is None:
                        raise OSError(f"Failed to read {row['image_path']}")
                    depth = np.load(str(row["depth_path"]), allow_pickle=False)
                    context = ContextKey(
                        int(row["motion_state"]), int(row["light_state"])
                    )
                    metrics = self.predictor.evaluate(
                        image,
                        np.asarray(depth, dtype=np.float32),
                        context=context,
                        exposure_us=float(row["exposure_us_model"]),
                        gain=float(
                            row["actual_gain"]
                            if row["actual_gain"] not in (None, "")
                            else row["gain"]
                        ),
                        alignment_mode=self.evaluation_alignment,
                        min_depth_m=self.min_depth_m,
                        max_depth_m=self.max_depth_m,
                        min_valid_pixels=self.min_valid_depth_pixels,
                    )
                    row.update(metrics)
                except (OSError, RuntimeError, ValueError) as exc:
                    row["evaluation_error"] = str(exc)
                if index % 10 == 0 or index == len(self.rows):
                    print(f"[Evaluation] {index}/{len(self.rows)}")
            if self.rows and not any(row["abs_rel"] != "" for row in self.rows):
                raise RuntimeError(
                    "AbsRel/A1 evaluation failed for every captured frame."
                )
        except Exception as exc:
            evaluation_failure = exc
        finally:
            csv_path = self._write_csv()
        if evaluation_failure is not None:
            raise evaluation_failure
        return csv_path

    def _write_csv(self) -> Path:
        csv_path = self.output_dir / "probing_modelv1.csv"
        temporary_path = csv_path.with_suffix(".csv.tmp")
        valid_abs_rel = [
            float(row["abs_rel"])
            for row in self.rows
            if row["abs_rel"] not in (None, "")
        ]
        valid_a1 = [
            float(row["a1"])
            for row in self.rows
            if row["a1"] not in (None, "")
        ]
        summary = {field: "" for field in CSV_FIELDNAMES}
        summary.update(
            {
                "record_type": "summary",
                "capture_index": len(self.rows),
                "decision_reason": f"evaluated_frames={len(valid_abs_rel)}",
                "abs_rel": float(np.mean(valid_abs_rel)) if valid_abs_rel else "",
                "a1": float(np.mean(valid_a1)) if valid_a1 else "",
            }
        )

        with temporary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for row in self.rows:
                output = {field: row.get(field, "") for field in CSV_FIELDNAMES}
                output["delay_reasons"] = ",".join(row["_delay_reasons"])
                writer.writerow(output)
            writer.writerow(summary)
        os.replace(temporary_path, csv_path)
        print(
            f"[Saved] {csv_path} captures={len(self.rows)} "
            f"mean_abs_rel={summary['abs_rel']} mean_a1={summary['a1']}"
        )
        return csv_path


def build_context_provider(args: argparse.Namespace) -> ContextProvider:
    from hardware import robot

    light_state = LIGHT_LABEL_TO_STATE[args.lighting_state]
    fallback = ContextKey(args.motion_state, light_state)
    fallback.validate()
    if args.context_file is not None:
        provider: ContextProvider = robot.JsonContextProvider(
            args.context_file, fallback
        )
    elif args.motion_source == "ros":
        ros_provider = robot.RosMotionContextProvider(
            light_state=light_state,
            imu_topic=args.imu_topic,
            odom_topic=args.odom_topic,
            sensor_timeout_sec=args.ros_sensor_timeout_sec,
        )
        provider = RosTrainingContextAdapter(ros_provider)
    else:
        provider = robot.FixedContextProvider(fallback)
    return DebouncedContextProvider(
        provider,
        debounce_ms=args.context_debounce_ms,
        minimum_samples=args.context_debounce_samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-error-model-driven deterministic Orbbec probing."
    )
    parser.add_argument("--camera-error-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("probing_modelv1_output")
    )
    parser.add_argument("--model-size", choices=("small", "base"), default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--q-uncertainty-weight", type=float, default=1.0)

    parser.add_argument("--accept-threshold", type=float, default=0.11)
    parser.add_argument("--accept-probability", type=float, default=0.90)
    parser.add_argument("--required-bad-frames", type=int, default=2)
    parser.add_argument("--success-ema-alpha", type=float, default=0.3)
    parser.add_argument("--challenger-cooldown-rounds", type=int, default=5)
    parser.add_argument("--offload-command", type=str, default=None)
    parser.add_argument("--safety-config", type=Path, default=None)

    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--round-interval-ms", type=float, default=0.0)
    parser.add_argument(
        "--initial-exposure-ms", type=int, choices=EXPOSURE_MS_VALUES, default=32
    )
    parser.add_argument("--initial-gain", type=int, choices=GAIN_VALUES, default=64)

    parser.add_argument("--context-file", type=Path, default=None)
    parser.add_argument("--motion-source", choices=("ros", "fixed"), default="ros")
    parser.add_argument(
        "--motion-state",
        type=int,
        choices=range(5),
        default=2,
        help="Training context order: 0=fast, 1=slow, 2=stop, 3=rotate, 4=spin.",
    )
    parser.add_argument(
        "--lighting-state", choices=tuple(LIGHT_LABEL_TO_STATE), default="normal"
    )
    parser.add_argument("--imu-topic", default="/imu")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--ros-sensor-timeout-sec", type=float, default=1.0)
    parser.add_argument(
        "--context-debounce-ms",
        type=float,
        default=250.0,
        help="Require a new motion/light context to persist this long.",
    )
    parser.add_argument(
        "--context-debounce-samples",
        type=int,
        default=3,
        help="Require this many repeated observations before changing context.",
    )

    parser.add_argument("--settle-frames", type=int, default=2)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=1000)
    parser.add_argument(
        "--exposure-value-per-ms",
        type=float,
        default=1000.0,
        help="Raw Orbbec exposure-property units per ms. Model input is always "
        "converted to microseconds (for example, 2 ms becomes 2000 us).",
    )
    parser.add_argument("--disable-awb", action="store_true")
    parser.add_argument("--allow-unsupported-grid-values", action="store_true")

    parser.add_argument("--camera-parameter-warn-ms", type=float, default=50.0)
    parser.add_argument("--mde-inference-warn-ms", type=float, default=100.0)
    parser.add_argument("--control-decision-warn-ms", type=float, default=500.0)
    parser.add_argument("--probe-decision-budget-ms", type=float, default=500.0)
    parser.add_argument("--inference-poll-ms", type=float, default=10.0)

    parser.add_argument(
        "--evaluation-alignment",
        choices=("metric", "scale_shift_depth", "scale_shift_inverse"),
        default="scale_shift_inverse",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.max_rounds <= 200:
        parser.error("--max-rounds must be between 1 and 200.")
    if args.round_interval_ms < 0.0:
        parser.error("--round-interval-ms must be non-negative.")
    if not 0.0 < args.min_depth_m < args.max_depth_m:
        parser.error("Require 0 < --min-depth-m < --max-depth-m.")
    if args.min_valid_depth_pixels < 1:
        parser.error("--min-valid-depth-pixels must be positive.")
    if args.context_debounce_ms < 0.0:
        parser.error("--context-debounce-ms must be non-negative.")
    if args.context_debounce_samples < 1:
        parser.error("--context-debounce-samples must be positive.")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_provider: Optional[ContextProvider] = None
    camera: Any = None
    experiment: Optional[ModelV1Experiment] = None
    exit_code = 0

    try:
        predictor = CameraErrorPredictor(
            checkpoint_path=args.camera_error_checkpoint,
            model_size=args.model_size,
            device=args.device,
            precision=args.precision,
            q_uncertainty_weight=args.q_uncertainty_weight,
            local_files_only=args.local_files_only,
        )
        context_provider = build_context_provider(args)
        safety_policy = SafetyPolicy.from_json(args.safety_config)
        policy = ATIMDECameraProbingController(
            safety_policy,
            accept_threshold=args.accept_threshold,
            accept_probability=args.accept_probability,
            required_bad_frames=args.required_bad_frames,
            success_ema_alpha=args.success_ema_alpha,
            challenger_cooldown_rounds=args.challenger_cooldown_rounds,
            offload_command=args.offload_command,
        )

        from hardware.sensor import OrbbecColorCamera

        camera = OrbbecColorCamera(
            exposure_value_per_ms=args.exposure_value_per_ms,
            settle_frames=args.settle_frames,
            frame_timeout_ms=args.frame_timeout_ms,
            warmup_frames=args.warmup_frames,
            disable_awb=args.disable_awb,
            strict_property_grid=not args.allow_unsupported_grid_values,
        )
        experiment = ModelV1Experiment(
            camera=camera,
            context_provider=context_provider,
            predictor=predictor,
            policy=policy,
            output_dir=args.output_dir,
            default_cell=SensorCell(args.initial_exposure_ms, args.initial_gain),
            camera_parameter_warn_ms=args.camera_parameter_warn_ms,
            mde_inference_warn_ms=args.mde_inference_warn_ms,
            control_decision_warn_ms=args.control_decision_warn_ms,
            probe_decision_budget_ms=args.probe_decision_budget_ms,
            inference_poll_ms=args.inference_poll_ms,
            evaluation_alignment=args.evaluation_alignment,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            min_valid_depth_pixels=args.min_valid_depth_pixels,
        )

        print(f"[Start] capture rounds={args.max_rounds}; press Ctrl-C to stop early.")
        while experiment.round_index < args.max_rounds:
            round_started = time.monotonic()
            experiment.run_round()
            remaining = (
                args.round_interval_ms / 1000.0
                - (time.monotonic() - round_started)
            )
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[Stop] interrupted; finalizing captured frames.")
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(f"\n[ERROR] {exc}")
        exit_code = 1
    finally:
        if experiment is not None:
            try:
                experiment.stop_inference()
            except RuntimeError as exc:
                print(f"[ERROR] inference shutdown failed: {exc}")
                exit_code = 1
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                print(f"[ERROR] camera shutdown failed: {exc}")
                exit_code = 1
        if context_provider is not None:
            try:
                context_provider.close()
            except Exception as exc:
                print(f"[ERROR] context shutdown failed: {exc}")
                exit_code = 1
        if experiment is not None:
            try:
                experiment.finalize()
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"[ERROR] result finalization failed: {exc}")
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
