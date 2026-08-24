import unittest

import numpy as np
import torch

from ati_mde_control.full_depth_predictor import CameraErrorFullDepthPredictor
from hardware.utils import ContextKey, QScore


class FakeFullDepthPredictor(CameraErrorFullDepthPredictor):
    def __init__(self):
        self.infer_calls = []

    def _infer(
        self,
        images,
        contexts,
        exposure_us_values,
        gains,
        target_size=None,
    ):
        self.infer_calls.append(
            (images, contexts, exposure_us_values, gains, target_size)
        )
        height, width = target_size
        return {
            "camera_bias": torch.tensor([0.2]),
            "std": torch.tensor([0.1]),
            "candidate_depth": torch.full((1, 1, height, width), 3.5),
        }, 12.0

    def _scores(self, outputs, inference_ms, batch_size):
        return [QScore(0.3, 0.1, 0.2, {"mde_inference_ms": inference_ms})]


class CameraErrorFullDepthPredictorTest(unittest.TestCase):
    def test_predict_batch_returns_score_and_full_resolution_depth_from_one_call(self):
        predictor = FakeFullDepthPredictor()
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        result = predictor.predict_batch(
            [image], [ContextKey(0, 0)], [16_000.0], [64.0]
        )
        self.assertEqual(len(predictor.infer_calls), 1)
        self.assertEqual(predictor.infer_calls[0][-1], (3, 5))
        self.assertEqual(result.scores[0].q, 0.3)
        self.assertEqual(result.depth_maps[0].shape, (3, 5))
        self.assertEqual(result.depth_maps[0].dtype, np.float32)
        self.assertTrue(np.all(result.depth_maps[0] == 3.5))


if __name__ == "__main__":
    unittest.main()
