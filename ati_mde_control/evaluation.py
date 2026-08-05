from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from hardware.utils import ContextKey

from .config import ExperimentConfig


def _fit_scale_shift(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = prediction[valid].float()
    expected = target[valid].float()
    count = prediction.new_tensor(float(predicted.numel())).float()
    a00, a01 = torch.sum(predicted * predicted), torch.sum(predicted)
    determinant = a00 * count - a01 * a01
    if abs(float(determinant.item())) < 1e-12:
        raise ValueError("Depth scale/shift alignment is singular.")
    b0, b1 = torch.sum(predicted * expected), torch.sum(expected)
    return (count * b0 - a01 * b1) / determinant, (-a01 * b0 + a00 * b1) / determinant


class DepthEvaluator:
    def __init__(self, predictor: Any, config: ExperimentConfig) -> None:
        self.predictor = predictor
        self.config = config

    def evaluate_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            try:
                image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise OSError(f"Failed to read {row['image_path']}")
                depth = np.load(str(row["depth_path"]), allow_pickle=False).astype(np.float32)
                context = ContextKey(int(row["motion_state"]), int(row["light_state"]))
                prediction, inference_ms = self.predictor.predict_depth(
                    image,
                    context,
                    float(row["exposure_us_model"]),
                    float(row["actual_gain"] or row["gain"]),
                    depth.shape,
                )
                metrics = self._metrics(prediction, depth)
                row.update(metrics, evaluation_inference_ms=inference_ms)
                print(f"[Evaluate] round={row['round_index']} capture={row['capture_index']} metrics={metrics}")
            except (OSError, RuntimeError, ValueError) as error:
                row["evaluation_error"] = str(error)

    def _metrics(self, prediction: torch.Tensor, depth: np.ndarray) -> dict[str, float]:
        target = torch.from_numpy(np.ascontiguousarray(depth)).to(self.predictor.device)
        valid = (
            torch.isfinite(target)
            & (target >= self.config.min_depth_m)
            & (target <= self.config.max_depth_m)
            & torch.isfinite(prediction)
        )
        if int(valid.sum()) < self.config.min_valid_depth_pixels:
            raise ValueError("Not enough valid depth pixels for evaluation.")
        mode = self.config.evaluation_alignment
        if mode == "metric":
            aligned = prediction
        elif mode == "scale_shift_depth":
            scale, shift = _fit_scale_shift(prediction, target, valid)
            aligned = scale * prediction + shift
        else:
            inverse = torch.reciprocal(target.clamp_min(1e-6))
            scale, shift = _fit_scale_shift(prediction, inverse, valid)
            aligned_inverse = scale * prediction + shift
            valid &= torch.isfinite(aligned_inverse) & (aligned_inverse > 1e-6)
            aligned = torch.reciprocal(aligned_inverse.clamp_min(1e-6))
        valid &= torch.isfinite(aligned) & (aligned > 1e-6)
        count = int(valid.sum())
        if count < self.config.min_valid_depth_pixels:
            raise ValueError("Not enough valid aligned depth pixels for evaluation.")
        predicted, expected = aligned[valid], target[valid]
        ratio = torch.maximum(predicted / expected, expected / predicted)
        return {
            "abs_rel": float(torch.mean(torch.abs(predicted - expected) / expected)),
            "a1": float(torch.mean((ratio < 1.25).float())),
            "valid_depth_pixels": count,
        }
