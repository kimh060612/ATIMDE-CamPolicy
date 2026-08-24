import unittest

import numpy as np

from ati_mde_control.full_depth_predictor import FullDepthBatchPrediction
from hardware.utils import QScore
from iqa_control.noise_aware_iqa_control import noise_aware_iqa
from orbbec_iqa_compare_control import IQARiskPredictor


class IQARiskPredictorTest(unittest.TestCase):
    def test_replaces_only_the_q_score(self) -> None:
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        depth = np.ones((12, 16), dtype=np.float32)

        class FullDepthPredictor:
            def predict_batch(self, *args):
                score = QScore(9.0, 2.0, 7.0, {"mde_inference_ms": 3.0})
                return FullDepthBatchPrediction((score,), (depth,))

        result = IQARiskPredictor(FullDepthPredictor()).predict_batch(
            [image], [object()], [32000.0], [64.0]
        )
        expected_q = -noise_aware_iqa(image).score

        self.assertAlmostEqual(result.scores[0].q, expected_q)
        self.assertAlmostEqual(result.scores[0].mu, expected_q)
        self.assertEqual(result.scores[0].uncertainty, 0.0)
        self.assertEqual(result.scores[0].extra["mde_inference_ms"], 3.0)
        self.assertIs(result.depth_maps[0], depth)


if __name__ == "__main__":
    unittest.main()
