import unittest

import numpy as np

from ati_mde_control.config import SafetyPolicy
from ati_mde_control.risk_bandit_policy import RiskBanditConfig
from ati_mde_control.saturation_guard import (
    SaturationGuard,
    SaturationGuardConfig,
    SaturationGuardedRiskBanditPolicy,
    ev_index,
)
from hardware.utils import ALL_CELLS, ContextKey, QScore, SensorCell


def clipped_image(count: int, total: int = 10) -> np.ndarray:
    image = np.zeros((1, total, 3), dtype=np.uint8)
    image[0, :count] = 255
    return image


class SaturationMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = SaturationGuard(SaturationGuardConfig())

    def test_black_image_has_zero_luminance_and_clipping(self) -> None:
        metrics = self.guard.measure(np.zeros((2, 3, 3), dtype=np.uint8))
        self.assertEqual(metrics.mean_luminance, 0.0)
        self.assertEqual(metrics.channel_clip_ratio, 0.0)
        self.assertEqual(metrics.luminance_clip_ratio, 0.0)

    def test_white_image_has_unit_clipping(self) -> None:
        metrics = self.guard.measure(np.full((2, 3, 3), 255, dtype=np.uint8))
        self.assertAlmostEqual(metrics.mean_luminance, 255.0, places=4)
        self.assertEqual(metrics.channel_clip_ratio, 1.0)
        self.assertEqual(metrics.luminance_clip_ratio, 1.0)

    def test_bgr_channel_order_is_explicit(self) -> None:
        blue = self.guard.measure(np.array([[[255, 0, 0]]], dtype=np.uint8))
        red = self.guard.measure(np.array([[[0, 0, 255]]], dtype=np.uint8))
        self.assertAlmostEqual(blue.mean_luminance, 0.0722 * 255, places=4)
        self.assertAlmostEqual(red.mean_luminance, 0.2126 * 255, places=4)

    def test_partial_pixel_clipping_ratio(self) -> None:
        metrics = self.guard.measure(clipped_image(3))
        self.assertAlmostEqual(metrics.channel_clip_ratio, 0.3)
        self.assertAlmostEqual(metrics.luminance_clip_ratio, 0.3)

    def test_invalid_shape_and_empty_image_are_rejected(self) -> None:
        for image in (
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2, 4), dtype=np.uint8),
            np.zeros((0, 2, 3), dtype=np.uint8),
        ):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.guard.measure(image)

    def test_non_finite_image_is_rejected(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.float32)
        image[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.guard.measure(image)


class SaturationGuardLogicTest(unittest.TestCase):
    context = ContextKey(0, 0)
    other_context = ContextKey(1, 0)
    current = SensorCell(16, 64)

    def setUp(self) -> None:
        self.guard = SaturationGuard(SaturationGuardConfig(quarantine_rounds=3))
        self.base = tuple(ALL_CELLS)

    def observe(self, image, round_index=0, context=None, effective=True, cell=None):
        return self.guard.observe(
            context or self.context,
            cell or self.current,
            round_index,
            image,
            setting_effective=effective,
        )

    def filtered(self, observation, round_index=0, context=None, base=None, current=None):
        return self.guard.filter_candidates(
            context or self.context,
            current or self.current,
            round_index,
            observation,
            self.base if base is None else base,
        )

    def test_normal_frame_preserves_base_safe_candidates(self) -> None:
        result = self.filtered(self.observe(clipped_image(0)))
        self.assertEqual(result.candidates, self.base)
        self.assertFalse(result.fallback_used)

    def test_soft_frame_removes_only_higher_ev_candidates(self) -> None:
        observation = self.observe(clipped_image(8))
        self.assertTrue(observation.metrics.soft_overexposed)
        result = self.filtered(observation)
        self.assertTrue(result.candidates)
        self.assertTrue(
            all(ev_index(cell) <= ev_index(self.current) for cell in result.candidates)
        )

    def test_hard_frame_requires_minimum_ev_drop(self) -> None:
        observation = self.observe(clipped_image(9))
        result = self.filtered(observation)
        self.assertTrue(observation.metrics.hard_overexposed)
        self.assertTrue(
            all(ev_index(cell) <= ev_index(self.current) - 1 for cell in result.candidates)
        )

    def test_hard_frame_creates_context_quarantine_ceiling(self) -> None:
        self.observe(clipped_image(9), round_index=2)
        entry = self.guard.quarantine(self.context, 2)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.blocked_ev_min, ev_index(self.current))
        self.assertEqual(entry.expiry_round, 5)

    def test_quarantine_is_context_local(self) -> None:
        hard = self.observe(clipped_image(9))
        normal_other = self.observe(clipped_image(0), context=self.other_context)
        local = self.filtered(hard)
        other = self.filtered(normal_other, context=self.other_context)
        self.assertLess(len(local.candidates), len(self.base))
        self.assertEqual(other.candidates, self.base)

    def test_quarantine_expires_at_configured_round(self) -> None:
        self.observe(clipped_image(9), round_index=2)
        normal = self.observe(clipped_image(0), round_index=3)
        self.assertTrue(self.filtered(normal, round_index=4).quarantine_active)
        expired = self.filtered(normal, round_index=5)
        self.assertFalse(expired.quarantine_active)
        self.assertEqual(expired.candidates, self.base)

    def test_recovery_requires_consecutive_frames(self) -> None:
        config = SaturationGuardConfig(recovery_consecutive_frames=3)
        guard = SaturationGuard(config)
        guard.observe(
            self.context, self.current, 0, clipped_image(9), setting_effective=True
        )
        states = [
            guard.observe(
                self.context,
                self.current,
                index,
                clipped_image(0),
                setting_effective=True,
            ).guard_state
            for index in (1, 2, 3)
        ]
        self.assertEqual(states, ["hard", "hard", "normal"])

    def test_same_ev_exposure_gain_combinations_survive_soft_filter(self) -> None:
        current = SensorCell(16, 32)
        same_ev = SensorCell(8, 64)
        observation = self.observe(clipped_image(8), cell=current)
        result = self.filtered(observation, current=current)
        self.assertEqual(ev_index(current), ev_index(same_ev))
        self.assertIn(current, result.candidates)
        self.assertIn(same_ev, result.candidates)

    def test_empty_guard_result_uses_deterministic_safe_fallback(self) -> None:
        current = SensorCell(4, 16)
        base = (SensorCell(8, 16), SensorCell(4, 32), SensorCell(32, 128))
        observation = self.observe(clipped_image(9), cell=current)
        result = self.filtered(observation, base=base, current=current)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.candidates, (SensorCell(4, 32),))

    def test_adapter_never_selects_outside_base_safety_or_guard_candidates(self) -> None:
        safety = SafetyPolicy(max_exposure_ms_by_motion=(8, 32, 32, 32, 32))
        policy = SaturationGuardedRiskBanditPolicy(
            RiskBanditConfig(), safety, self.current
        )
        allowed = (SensorCell(4, 16), SensorCell(8, 16))
        policy.add_observation(
            self.context,
            SensorCell(4, 16),
            0,
            QScore(0.2, 0.1, 0.1),
        )
        selected = policy.select_from_candidates(
            self.context, self.current, 1, allowed
        ).selected_cell
        self.assertIn(selected, allowed)
        self.assertIn(selected, safety.safe_cells(self.context))

    def test_ineffective_hard_frame_does_not_create_quarantine(self) -> None:
        self.observe(clipped_image(9), effective=False)
        self.assertIsNone(self.guard.quarantine(self.context, 0))


class SaturationConfigTest(unittest.TestCase):
    def test_invalid_thresholds_are_rejected(self) -> None:
        invalid = (
            {"pixel_clip_threshold": 0},
            {"pixel_clip_threshold": float("nan")},
            {"recovery_clip_ratio": 0.8},
            {"soft_clip_ratio": 0.9, "secondary_clip_ratio": 0.85},
            {"hard_mean_luminance": 256},
            {"recovery_consecutive_frames": 0},
            {"quarantine_rounds": 0},
            {"minimum_ev_drop_stops": 0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SaturationGuardConfig(**values)


if __name__ == "__main__":
    unittest.main()
