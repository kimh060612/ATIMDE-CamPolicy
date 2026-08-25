import unittest

import numpy as np

from ati_mde_control.predictive_saturation_guard import (
    PredictiveSaturationGuard,
    PredictiveSaturationGuardConfig,
)
from ati_mde_control.saturation_guard import ev_index
from hardware.utils import ALL_CELLS, ContextKey, SensorCell


def clipped_image(count: int, total: int = 10) -> np.ndarray:
    image = np.zeros((1, total, 3), dtype=np.uint8)
    image[0, :count] = 255
    return image


class PredictiveSaturationGuardTest(unittest.TestCase):
    context = ContextKey(0, 0)

    def setUp(self) -> None:
        self.guard = PredictiveSaturationGuard(PredictiveSaturationGuardConfig())

    def observe(
        self,
        image=None,
        *,
        context=None,
        cell=SensorCell(8, 32),
        round_index=0,
        effective=True,
    ):
        return self.guard.observe(
            context or self.context,
            cell,
            round_index,
            clipped_image(0) if image is None else image,
            setting_effective=effective,
        )

    def filtered(
        self,
        observation,
        *,
        context=None,
        current=SensorCell(8, 32),
        round_index=0,
        base=ALL_CELLS,
    ):
        return self.guard.filter_candidates(
            context or self.context,
            current,
            round_index,
            observation,
            base,
        )

    def test_ev_index_uses_existing_geometric_grid(self) -> None:
        expected = {
            SensorCell(4, 16): 0,
            SensorCell(8, 16): 1,
            SensorCell(8, 32): 2,
            SensorCell(8, 64): 3,
            SensorCell(16, 32): 3,
            SensorCell(16, 64): 4,
        }
        self.assertEqual({cell: ev_index(cell) for cell in expected}, expected)

    def test_large_jump_and_same_ev_delta(self) -> None:
        self.assertEqual(ev_index(SensorCell(16, 64)) - ev_index(SensorCell(4, 16)), 4)
        self.assertEqual(ev_index(SensorCell(16, 32)) - ev_index(SensorCell(8, 64)), 0)

    def test_projected_ratio_matches_direct_scaling_formula(self) -> None:
        image = np.array([[[124, 10, 20], [125, 1, 2], [200, 3, 4]]], np.uint8)
        ratio = self.guard.projected_channel_clip_ratio(
            image, SensorCell(8, 32), SensorCell(16, 32)
        )
        expected = float(np.mean(np.max(image.astype(np.float32), axis=-1) * 2 >= 250))
        self.assertEqual(ratio, expected)
        self.assertEqual(ratio, 2 / 3)

    def test_non_brighter_candidates_are_not_projected_or_removed(self) -> None:
        current = SensorCell(16, 32)
        lower = SensorCell(8, 32)
        same = SensorCell(8, 64)
        observation = self.observe(image=clipped_image(0), cell=current)
        result = self.filtered(
            observation, current=current, base=(lower, same, current)
        )
        self.assertIsNone(
            self.guard.projected_channel_clip_ratio(
                clipped_image(10), current, lower
            )
        )
        self.assertIn(lower, result.candidates)
        self.assertIn(same, result.candidates)
        self.assertEqual(result.projected_clip_ratios, ())

    def test_projection_uses_two_to_delta_ev_and_rejects_at_threshold(self) -> None:
        current = SensorCell(4, 16)
        candidate = SensorCell(8, 16)
        image = np.zeros((1, 10, 3), np.uint8)
        image[0, :9] = 125
        observation = self.observe(image=image, cell=current)
        result = self.filtered(
            observation, current=current, base=(current, candidate)
        )
        ratios = dict(result.projected_clip_ratios)
        self.assertEqual(ratios[candidate], 0.9)
        self.assertIn(candidate, result.projected_rejected_cells)
        self.assertNotIn(candidate, result.candidates)

    def test_four_stop_projection_uses_scale_sixteen(self) -> None:
        image = np.full((1, 1, 3), 16, np.uint8)
        ratio = self.guard.projected_channel_clip_ratio(
            image, SensorCell(4, 16), SensorCell(16, 64)
        )
        self.assertEqual(ratio, 1.0)

    def test_projection_rejects_invalid_or_non_finite_image(self) -> None:
        invalid = (
            np.zeros((2, 2), np.uint8),
            np.zeros((0, 2, 3), np.uint8),
            np.full((1, 1, 3), np.nan, np.float32),
        )
        for image in invalid:
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.guard.projected_channel_clip_ratio(
                    image, SensorCell(4, 16), SensorCell(8, 16)
                )

    def test_normal_state_limits_upward_step_to_one(self) -> None:
        current = SensorCell(8, 32)
        observation = self.observe(cell=current)
        result = self.filtered(observation, current=current)
        self.assertTrue(
            all(ev_index(cell) <= ev_index(current) + 1 for cell in result.candidates)
        )
        self.assertTrue(
            all(ev_index(cell) - ev_index(current) >= 2 for cell in result.ev_step_rejected_cells)
        )

    def test_downward_moves_have_no_step_limit(self) -> None:
        current = SensorCell(16, 64)
        lowest = SensorCell(4, 16)
        result = self.filtered(self.observe(cell=current), current=current)
        self.assertIn(lowest, result.candidates)

    def test_same_ev_change_is_allowed(self) -> None:
        current = SensorCell(8, 64)
        same = SensorCell(16, 32)
        result = self.filtered(self.observe(cell=current), current=current)
        self.assertEqual(ev_index(current), ev_index(same))
        self.assertIn(same, result.candidates)

    def test_soft_state_removes_all_upward_ev(self) -> None:
        current = SensorCell(8, 32)
        observation = self.observe(image=clipped_image(6), cell=current)
        result = self.filtered(observation, current=current)
        self.assertTrue(observation.metrics.soft_overexposed)
        self.assertTrue(
            all(ev_index(cell) <= ev_index(current) for cell in result.candidates)
        )

    def test_hard_state_requires_one_stop_drop(self) -> None:
        current = SensorCell(16, 64)
        observation = self.observe(image=clipped_image(9), cell=current)
        result = self.filtered(observation, current=current)
        self.assertTrue(observation.metrics.hard_overexposed)
        self.assertTrue(
            all(ev_index(cell) <= ev_index(current) - 1 for cell in result.candidates)
        )

    def test_quarantine_is_shared_across_motion_for_same_light(self) -> None:
        current = SensorCell(16, 64)
        source = ContextKey(0, 1)
        other_motion = ContextKey(4, 1)
        self.observe(
            image=clipped_image(9), context=source, cell=current, round_index=2
        )
        entry = self.guard.quarantine(other_motion, 2)
        self.assertEqual(entry.blocked_ev_min, ev_index(current))
        self.assertEqual(entry.expiry_round, 62)
        normal = self.observe(context=other_motion, cell=SensorCell(8, 32), round_index=3)
        result = self.filtered(
            normal,
            context=other_motion,
            current=SensorCell(8, 32),
            round_index=3,
        )
        self.assertTrue(result.quarantine_active)
        self.assertTrue(
            all(ev_index(cell) < ev_index(current) for cell in result.candidates)
        )

    def test_quarantine_does_not_cross_light_state(self) -> None:
        self.observe(
            image=clipped_image(9),
            context=ContextKey(0, 0),
            cell=SensorCell(16, 64),
        )
        self.assertIsNone(self.guard.quarantine(ContextKey(0, 1), 0))

    def test_quarantine_expires_after_sixty_rounds(self) -> None:
        self.observe(
            image=clipped_image(9), cell=SensorCell(16, 64), round_index=0
        )
        self.assertIsNotNone(self.guard.quarantine(self.context, 59))
        self.assertIsNone(self.guard.quarantine(self.context, 60))

    def test_ineffective_hard_frame_does_not_create_quarantine(self) -> None:
        observation = self.observe(
            image=clipped_image(9), cell=SensorCell(16, 64), effective=False
        )
        self.assertTrue(observation.metrics.hard_overexposed)
        self.assertIsNone(self.guard.quarantine(self.context, 0))

    def test_empty_hard_fallback_is_base_safe_and_never_higher(self) -> None:
        current = SensorCell(4, 16)
        base = (current, SensorCell(8, 16))
        observation = self.observe(
            image=clipped_image(9), cell=current, effective=False
        )
        result = self.filtered(observation, current=current, base=base)
        self.assertTrue(result.fallback_used)
        self.assertIn(result.candidates[0], base)
        self.assertLessEqual(ev_index(result.candidates[0]), ev_index(current))
        self.assertEqual(result.fallback_reason, "observed_guard_empty")


class PredictiveSaturationConfigTest(unittest.TestCase):
    def test_defaults_match_predictive_path(self) -> None:
        config = PredictiveSaturationGuardConfig()
        self.assertEqual(config.soft_clip_ratio, 0.60)
        self.assertEqual(config.hard_clip_ratio, 0.89)
        self.assertEqual(config.hard_mean_luminance, 243.0)
        self.assertEqual(config.recovery_clip_ratio, 0.50)
        self.assertEqual(config.quarantine_rounds, 60)
        self.assertEqual(config.max_upward_ev_stops, 1)
        self.assertEqual(config.projected_hard_clip_ratio, 0.90)

    def test_invalid_predictive_config_is_rejected(self) -> None:
        invalid = (
            {"soft_clip_ratio": 0.5},
            {"projected_pixel_clip_threshold": 0},
            {"projected_pixel_clip_threshold": float("nan")},
            {"projected_hard_clip_ratio": 0},
            {"projected_hard_clip_ratio": float("inf")},
            {"max_upward_ev_stops": -1},
            {"quarantine_rounds": 0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PredictiveSaturationGuardConfig(**values)


if __name__ == "__main__":
    unittest.main()
