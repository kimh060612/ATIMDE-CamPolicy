import unittest

from drl_policy.progressive_allocator import allocate_progressive_exposure


class ProgressiveExposureAllocatorTest(unittest.TestCase):
    def allocate(self, target: float, **overrides):
        arguments = dict(
            t_min_us=100.0,
            t_max_us=20_000.0,
            gain_min_db=0.0,
            gain_max_db=30.0,
            gain_step_db=1.0,
            t_step_us=1.0,
        )
        arguments.update(overrides)
        return allocate_progressive_exposure(target, **arguments)

    def test_paper_example_10_ms_uses_5_db(self):
        allocation = self.allocate(10_000.0)
        self.assertAlmostEqual(allocation.exposure_time_us, 5_623.0, delta=1.0)
        self.assertEqual(allocation.gain_db, 5.0)
        self.assertAlmostEqual(allocation.realized_exposure, 10_000.0, delta=1.0)
        self.assertEqual(allocation.stage, "interval_0")

    def test_minimum_and_maximum_clipping(self):
        low = self.allocate(1.0, gain_max_db=20.0)
        high = self.allocate(1_000_000.0, gain_max_db=20.0)
        self.assertTrue(low.was_clipped)
        self.assertEqual((low.exposure_time_us, low.gain_db), (100.0, 0.0))
        self.assertTrue(high.was_clipped)
        self.assertEqual((high.exposure_time_us, high.gain_db), (20_000.0, 20.0))

    def test_gain_step_quantization(self):
        allocation = self.allocate(10_000.0, gain_step_db=2.0)
        self.assertEqual(allocation.gain_db, 6.0)

    def test_realized_exposure_matches_quantized_time_and_gain(self):
        allocation = self.allocate(300_000.0, gain_step_db=2.0, t_step_us=100.0)
        expected = allocation.exposure_time_us * 10.0 ** (allocation.gain_db / 20.0)
        self.assertAlmostEqual(allocation.realized_exposure, expected)


if __name__ == "__main__":
    unittest.main()
