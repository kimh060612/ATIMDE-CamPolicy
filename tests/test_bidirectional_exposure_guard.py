import unittest

import numpy as np

from ati_mde_control.bidirectional_exposure_guard import (
    BidirectionalExposureGuard,
    BidirectionalExposureGuardConfig,
)
from ati_mde_control.predictive_saturation_guard import (
    PredictiveSaturationGuardConfig,
)
from ati_mde_control.saturation_guard import ev_index
from hardware.utils import ALL_CELLS, ContextKey, SensorCell


def grayscale(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8).reshape(1, -1, 1)
    return np.repeat(values, 3, axis=-1)


class BidirectionalExposureGuardTest(unittest.TestCase):
    context = ContextKey(0, 0)
    current = SensorCell(8, 64)  # EV 3

    def setUp(self) -> None:
        self.guard = BidirectionalExposureGuard(BidirectionalExposureGuardConfig())

    def observe(self, image, *, effective=True, round_index=0, context=None):
        return self.guard.observe(
            context or self.context,
            self.current,
            round_index,
            image,
            setting_effective=effective,
        )

    def filtered(self, observation, *, base=ALL_CELLS, context=None, round_index=0):
        return self.guard.filter_candidates(
            context or self.context,
            self.current,
            round_index,
            observation,
            base,
        )

    def test_black_and_white_shadow_ratios(self) -> None:
        black = self.guard.measure(np.zeros((2, 2, 3), np.uint8))
        white = self.guard.measure(np.full((2, 2, 3), 255, np.uint8))
        self.assertEqual(black.shadow_clip_ratio, 1.0)
        self.assertEqual(white.shadow_clip_ratio, 0.0)

    def test_partial_shadow_ratio_and_input_is_unchanged(self) -> None:
        image = grayscale([0, 10, 11, 100])
        original = image.copy()
        metrics = self.guard.measure(image)
        self.assertEqual(metrics.shadow_clip_ratio, 0.5)
        np.testing.assert_array_equal(image, original)

    def test_bgr_luminance_order(self) -> None:
        image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
        metrics = self.guard.measure(image)
        bgr = image.astype(np.float32)
        luminance = (
            0.2126 * bgr[..., 2]
            + 0.7152 * bgr[..., 1]
            + 0.0722 * bgr[..., 0]
        )
        np.testing.assert_allclose(
            luminance[0], [0.0722 * 255, 0.7152 * 255, 0.2126 * 255]
        )
        self.assertAlmostEqual(metrics.mean_luminance, float(np.mean(luminance)))

    def test_invalid_and_non_finite_images_are_rejected(self) -> None:
        invalid = (
            np.zeros((2, 2), np.uint8),
            np.zeros((0, 2, 3), np.uint8),
            np.full((1, 1, 3), np.nan, np.float32),
        )
        for image in invalid:
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.guard.measure(image)

    def test_normal_symmetric_step_allows_minus_one_zero_plus_one(self) -> None:
        result = self.filtered(self.observe(grayscale([100] * 10)))
        deltas = {ev_index(cell) - ev_index(self.current) for cell in result.candidates}
        self.assertEqual(deltas, {-1, 0, 1})
        self.assertTrue(
            all(
                abs(ev_index(cell) - ev_index(self.current)) >= 2
                for cell in result.ev_step_rejected_cells
            )
        )

    def test_same_ev_exposure_gain_pair_is_allowed(self) -> None:
        same_ev = SensorCell(16, 32)
        result = self.filtered(self.observe(grayscale([100] * 10)))
        self.assertEqual(ev_index(same_ev), ev_index(self.current))
        self.assertIn(same_ev, result.candidates)

    def test_soft_over_allows_only_minus_one_and_zero(self) -> None:
        observation = self.observe(grayscale([255] * 6 + [100] * 4))
        result = self.filtered(observation)
        self.assertEqual(observation.guard_state, "soft_overexposed")
        self.assertEqual(
            {ev_index(cell) - ev_index(self.current) for cell in result.candidates},
            {-1, 0},
        )

    def test_hard_over_allows_exactly_minus_one(self) -> None:
        observation = self.observe(grayscale([255] * 9 + [100]))
        result = self.filtered(observation)
        self.assertEqual(observation.guard_state, "hard_overexposed")
        self.assertEqual(
            {ev_index(cell) - ev_index(self.current) for cell in result.candidates},
            {-1},
        )

    def test_soft_under_allows_only_zero_and_plus_one(self) -> None:
        observation = self.observe(grayscale([0] * 3 + [50] * 7))
        result = self.filtered(observation)
        self.assertEqual(observation.guard_state, "soft_underexposed")
        self.assertEqual(
            {ev_index(cell) - ev_index(self.current) for cell in result.candidates},
            {0, 1},
        )

    def test_hard_under_allows_exactly_plus_one(self) -> None:
        observation = self.observe(grayscale([0] * 6 + [40] * 4))
        result = self.filtered(observation)
        self.assertEqual(observation.guard_state, "hard_underexposed")
        self.assertEqual(
            {ev_index(cell) - ev_index(self.current) for cell in result.candidates},
            {1},
        )

    def test_underexposure_thresholds_require_ratio_and_mean(self) -> None:
        soft = self.observe(grayscale([0] * 3 + [50] * 7))
        hard = self.observe(grayscale([0] * 5 + [30] * 5))
        bright_shadows = self.observe(grayscale([0] * 5 + [255] * 5))
        self.assertTrue(soft.metrics.soft_underexposed)
        self.assertTrue(hard.metrics.hard_underexposed)
        self.assertFalse(bright_shadows.metrics.hard_underexposed)
        self.assertFalse(bright_shadows.metrics.soft_underexposed)

    def test_under_recovery_requires_three_consecutive_frames(self) -> None:
        self.observe(grayscale([0] * 6 + [40] * 4))
        first = self.observe(grayscale([100] * 10), round_index=1)
        second = self.observe(grayscale([100] * 10), round_index=2)
        third = self.observe(grayscale([100] * 10), round_index=3)
        self.assertEqual(first.guard_state, "hard_underexposed")
        self.assertEqual(first.under_recovery_count, 1)
        self.assertEqual(second.under_recovery_count, 2)
        self.assertEqual(third.guard_state, "normal")
        self.assertEqual(third.under_recovery_count, 0)

    def test_under_recovery_count_resets_on_failure(self) -> None:
        self.observe(grayscale([0] * 6 + [40] * 4))
        self.observe(grayscale([100] * 10), round_index=1)
        failed = self.observe(grayscale([50] * 10), round_index=2)
        self.assertEqual(failed.guard_state, "hard_underexposed")
        self.assertEqual(failed.under_recovery_count, 0)

    def test_ineffective_frame_does_not_update_guard_state(self) -> None:
        ineffective = self.observe(grayscale([0] * 10), effective=False)
        self.assertTrue(ineffective.metrics.hard_underexposed)
        self.assertEqual(ineffective.guard_state, "normal")
        effective = self.observe(grayscale([0] * 10), round_index=1)
        ignored_recovery = self.observe(
            grayscale([100] * 10), effective=False, round_index=2
        )
        self.assertEqual(effective.guard_state, "hard_underexposed")
        self.assertEqual(ignored_recovery.guard_state, "hard_underexposed")
        self.assertEqual(ignored_recovery.under_recovery_count, 0)

    def test_under_over_conflict_holds_current_safe_cell(self) -> None:
        self.observe(grayscale([0] * 10))
        conflict = self.observe(grayscale([255] * 10), round_index=1)
        result = self.filtered(conflict, round_index=1)
        self.assertTrue(conflict.guard_conflict)
        self.assertEqual(conflict.guard_state, "guard_conflict")
        self.assertEqual(result.candidates, (self.current,))
        self.assertEqual(result.fallback_reason, "guard_conflict")

    def test_projected_shadow_uses_two_to_delta_and_matches_formula(self) -> None:
        image = grayscale([20, 40])
        lower = SensorCell(8, 32)
        ratio = self.guard.projected_shadow_clip_ratio(image, self.current, lower)
        luminance = image[..., 0].astype(np.float32)
        expected = float(np.mean(luminance * (2.0**-1) <= 10.0))
        self.assertEqual(ratio, expected)
        self.assertEqual(ratio, 0.5)

    def test_projected_shadow_rejects_at_limit_only_for_darker_cells(self) -> None:
        image = grayscale([20] * 3 + [100] * 7)
        lower = SensorCell(8, 32)
        same = SensorCell(16, 32)
        higher = SensorCell(16, 64)
        observation = self.observe(image)
        result = self.filtered(observation, base=(lower, same, self.current, higher))
        self.assertIn(lower, result.shadow_rejected_cells)
        self.assertNotIn(lower, result.candidates)
        self.assertIsNone(
            self.guard.projected_shadow_clip_ratio(image, self.current, same)
        )
        self.assertIsNone(
            self.guard.projected_shadow_clip_ratio(image, self.current, higher)
        )

    def test_projected_saturation_and_shadow_both_filter(self) -> None:
        config = BidirectionalExposureGuardConfig(
            predictive=PredictiveSaturationGuardConfig(projected_hard_clip_ratio=0.70)
        )
        guard = BidirectionalExposureGuard(config)
        image = grayscale([20] * 3 + [125] * 7)
        observation = guard.observe(
            self.context, self.current, 0, image, setting_effective=True
        )
        lower = SensorCell(8, 32)
        higher = SensorCell(16, 64)
        result = guard.filter_candidates(
            self.context, self.current, 0, observation, (lower, self.current, higher)
        )
        self.assertIn(lower, result.shadow_rejected_cells)
        self.assertIn(higher, result.saturation_rejected_cells)
        self.assertEqual(result.candidates, (self.current,))

    def test_hard_under_fallback_and_unrecoverable_reason(self) -> None:
        observation = self.observe(grayscale([0] * 10))
        result = self.filtered(observation, base=(self.current,))
        self.assertEqual(result.candidates, (self.current,))
        self.assertEqual(
            result.fallback_reason,
            "underexposure_unrecoverable_within_safety_policy",
        )

    def test_hard_fallbacks_choose_nearest_safe_direction(self) -> None:
        lower_far = SensorCell(4, 16)
        lower_near = SensorCell(8, 16)
        over = self.observe(grayscale([255] * 10))
        over_result = self.filtered(over, base=(lower_far, lower_near, self.current))
        self.assertEqual(over_result.candidates, (lower_near,))
        self.assertTrue(over_result.fallback_used)

        guard = BidirectionalExposureGuard(BidirectionalExposureGuardConfig())
        under = guard.observe(
            self.context,
            self.current,
            0,
            grayscale([0] * 10),
            setting_effective=True,
        )
        higher_near = SensorCell(16, 128)
        higher_far = SensorCell(32, 128)
        under_result = guard.filter_candidates(
            self.context,
            self.current,
            0,
            under,
            (self.current, higher_far, higher_near),
        )
        self.assertEqual(under_result.candidates, (higher_near,))
        self.assertTrue(under_result.fallback_used)

    def test_only_hard_over_creates_long_term_quarantine(self) -> None:
        self.observe(grayscale([0] * 10))
        self.assertIsNone(self.guard.quarantine(self.context, 0))

        guard = BidirectionalExposureGuard(BidirectionalExposureGuardConfig())
        guard.observe(
            self.context,
            self.current,
            0,
            grayscale([255] * 10),
            setting_effective=True,
        )
        self.assertIsNotNone(guard.quarantine(self.context, 0))

    def test_unsafe_current_uses_injected_policy_fallback(self) -> None:
        fallback = SensorCell(4, 16)
        guard = BidirectionalExposureGuard(
            BidirectionalExposureGuardConfig(), lambda context: fallback
        )
        observation = guard.observe(
            self.context,
            self.current,
            0,
            grayscale([100] * 10),
            setting_effective=True,
        )
        result = guard.filter_candidates(
            self.context, self.current, 0, observation, (fallback,)
        )
        self.assertEqual(result.candidates, (fallback,))
        self.assertEqual(result.fallback_reason, "current_cell_unsafe")


class BidirectionalExposureConfigTest(unittest.TestCase):
    def test_defaults_and_validation(self) -> None:
        config = BidirectionalExposureGuardConfig()
        self.assertEqual(config.max_ev_step_stops, 1)
        self.assertEqual(config.shadow_pixel_threshold, 10.0)
        self.assertEqual(config.soft_shadow_ratio, 0.30)
        self.assertEqual(config.hard_shadow_ratio, 0.50)
        self.assertEqual(config.projected_shadow_ratio_limit, 0.30)
        self.assertEqual(config.predictive.soft_clip_ratio, 0.60)
        self.assertEqual(config.predictive.hard_clip_ratio, 0.89)

        invalid = (
            {"max_ev_step_stops": 0},
            {"shadow_pixel_threshold": -1},
            {"shadow_pixel_threshold": float("nan")},
            {"soft_shadow_ratio": 0.50},
            {"hard_under_mean_luminance": 60},
            {"projected_shadow_ratio_limit": 1.1},
            {"under_recovery_consecutive_frames": 0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                BidirectionalExposureGuardConfig(**values)


if __name__ == "__main__":
    unittest.main()
