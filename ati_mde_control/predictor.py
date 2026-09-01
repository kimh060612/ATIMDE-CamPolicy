from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from hardware.utils import ContextKey, QScore
from model.model_camerror import MODEL_IDS, CameraInducedErrorModel
from model.utils import CameraParameterRange, normalize_camera_parameters

from .context import LIGHT_STATE_TO_LABEL, MOTION_STATE_TO_LABEL


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint does not contain a model state dict: {path}")
    if all(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


class CameraErrorPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        model_size: str,
        device: str,
        precision: str,
        q_uncertainty_weight: float,
        local_files_only: bool,
    ) -> None:
        if model_size not in MODEL_IDS or precision not in {"fp16", "fp32"}:
            raise ValueError("Unsupported model size or precision.")
        if not math.isfinite(q_uncertainty_weight) or q_uncertainty_weight < 0:
            raise ValueError("q_uncertainty_weight must be non-negative.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The real-time experiment and evaluation require CUDA.")
        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.q_uncertainty_weight = q_uncertainty_weight
        model_id = MODEL_IDS[model_size]
        self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
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
        self.model.load_state_dict(_load_checkpoint(checkpoint_path), strict=True)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.parameter_range = CameraParameterRange(4000.0, 32000.0, 16.0, 128.0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    def _context_vector(self, context: ContextKey, exposure_us: float, gain: float) -> torch.Tensor:
        light = F.one_hot(torch.tensor(context.light_state), num_classes=len(LIGHT_STATE_TO_LABEL)).float()
        motion = F.one_hot(torch.tensor(context.motion_state), num_classes=len(MOTION_STATE_TO_LABEL)).float()
        camera = normalize_camera_parameters(
            torch.tensor(exposure_us, dtype=torch.float32),
            torch.tensor(gain, dtype=torch.float32),
            self.parameter_range,
        )
        return torch.cat((light, motion, camera)).unsqueeze(0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )

    def _infer(
        self,
        images: Sequence[np.ndarray],
        contexts: Sequence[ContextKey],
        exposure_us_values: Sequence[float],
        gains: Sequence[float],
        target_size: tuple[int, int] | None = None,
    ) -> tuple[dict[str, torch.Tensor], float]:
        pixels, vectors = self._prepare_inputs(
            images, contexts, exposure_us_values, gains
        )
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.inference(
                candidate_img=pixels,
                context=vectors,
                target_size=target_size or tuple(pixels.shape[-2:]),
            )
        torch.cuda.synchronize(self.device)
        return outputs, (time.perf_counter() - started) * 1000.0

    def _prepare_inputs(
        self,
        images: Sequence[np.ndarray],
        contexts: Sequence[ContextKey],
        exposure_us_values: Sequence[float],
        gains: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = len(images)
        if not size or not (len(contexts) == len(exposure_us_values) == len(gains) == size):
            raise ValueError("Inference inputs require the same non-zero length.")
        # rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        vectors = torch.cat([
            self._context_vector(context, exposure, gain)
            for context, exposure, gain in zip(contexts, exposure_us_values, gains)
        ])
        return pixels, vectors

    def _infer_scores(
        self,
        images: Sequence[np.ndarray],
        contexts: Sequence[ContextKey],
        exposure_us_values: Sequence[float],
        gains: Sequence[float],
    ) -> tuple[dict[str, torch.Tensor], float]:
        pixels, vectors = self._prepare_inputs(
            images, contexts, exposure_us_values, gains
        )
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.inference_scores(
                candidate_img=pixels,
                context=vectors,
            )
        torch.cuda.synchronize(self.device)
        return outputs, (time.perf_counter() - started) * 1000.0

    def predict(self, image: np.ndarray, context: ContextKey, exposure_us: float, gain: float) -> QScore:
        return self.predict_batch([image], [context], [exposure_us], [gain])[0]

    def predict_scores(
        self,
        image: np.ndarray,
        context: ContextKey,
        exposure_us: float,
        gain: float,
    ) -> QScore:
        """Control-only score prediction that skips depth-head computation."""
        outputs, inference_ms = self._infer_scores(
            [image], [context], [exposure_us], [gain]
        )
        return self._scores(outputs, inference_ms, 1)[0]

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
        contexts: Sequence[ContextKey],
        exposure_us_values: Sequence[float],
        gains: Sequence[float],
    ) -> list[QScore]:
        outputs, inference_ms = self._infer(images, contexts, exposure_us_values, gains)
        return self._scores(outputs, inference_ms, len(images))

    def _scores(
        self,
        outputs: dict[str, torch.Tensor],
        inference_ms: float,
        batch_size: int,
    ) -> list[QScore]:
        biases = outputs["camera_bias"].detach().float().cpu().tolist()
        deviations = outputs["std"].detach().float().cpu().tolist()
        if len(biases) != batch_size or len(deviations) != batch_size:
            raise RuntimeError("Camera-error model returned the wrong batch size.")
        scores = []
        for mu, std in zip(biases, deviations):
            q = mu + self.q_uncertainty_weight * std
            if not all(math.isfinite(value) for value in (mu, std, q)):
                raise ValueError("Camera-error model returned a non-finite score.")
            scores.append(QScore(q, std, mu, {
                "mde_inference_ms": inference_ms,
                "mde_batch_size": float(batch_size),
            }))
        return scores

    def predict_depth(
        self,
        image: np.ndarray,
        context: ContextKey,
        exposure_us: float,
        gain: float,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, float]:
        outputs, inference_ms = self._infer(
            [image], [context], [exposure_us], [gain], target_size
        )
        return outputs["candidate_depth"][0, 0].float(), inference_ms
