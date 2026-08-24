from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hardware.utils import ContextKey, QScore

from .predictor import CameraErrorPredictor
from .types import CapturedFrame


@dataclass(frozen=True)
class FullDepthBatchPrediction:
    """Scores and raw depth predictions produced by one model call."""

    scores: tuple[QScore, ...]
    depth_maps: tuple[np.ndarray, ...]


class CameraErrorFullDepthPredictor(CameraErrorPredictor):
    """Camera-error predictor variant used only by full-depth control paths."""

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
        contexts: Sequence[ContextKey],
        exposure_us_values: Sequence[float],
        gains: Sequence[float],
    ) -> FullDepthBatchPrediction:
        if not images:
            raise ValueError("Full-depth prediction requires at least one image.")
        target_size = tuple(images[0].shape[:2])
        if any(tuple(image.shape[:2]) != target_size for image in images):
            raise ValueError("All full-depth batch images must have the same size.")
        outputs, inference_ms = self._infer(
            images,
            contexts,
            exposure_us_values,
            gains,
            target_size=target_size,
        )
        scores = tuple(self._scores(outputs, inference_ms, len(images)))
        depths = outputs["candidate_depth"]
        if depths.ndim != 4 or depths.shape[0] != len(images) or depths.shape[1] != 1:
            raise RuntimeError("Camera-error model returned the wrong depth batch shape.")
        depth_maps = tuple(
            np.ascontiguousarray(depths[index, 0].detach().float().cpu().numpy())
            for index in range(len(images))
        )
        return FullDepthBatchPrediction(scores, depth_maps)

    def predict(
        self,
        image: np.ndarray,
        context: ContextKey,
        exposure_us: float,
        gain: float,
    ) -> QScore:
        return self.predict_batch(
            [image], [context], [exposure_us], [gain]
        ).scores[0]


def save_raw_depth_prediction(
    output_dir: Path,
    frame: CapturedFrame,
    depth_map: np.ndarray,
) -> Path:
    """Save the raw control-model depth using the capture logger's stem."""

    prediction_dir = output_dir / "depth_pred_raw"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"round_{frame.round_index:04d}_capture_{frame.capture_index:05d}_"
        f"{frame.cell.cell_id}_{frame.timestamp_ns}"
    )
    path = prediction_dir / f"{stem}.npy"
    np.save(path, np.ascontiguousarray(depth_map, dtype=np.float32))
    return path
