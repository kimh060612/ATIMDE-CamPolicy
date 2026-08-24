import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ati_mde_control.brightness_safety import (
    BrightnessGuard,
    BrightnessGuardConfig,
    BrightnessGuardMode,
    BrightnessState,
    ev_step,
)
from ati_mde_control.config import SafetyPolicy
from hardware.utils import ContextKey, SensorCell


class BrightnessGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.current = SensorCell(16, 64)
        self.hard_safe = tuple(SafetyPolicy().safe_cells(ContextKey(0, 0)))

    def test_white_is_immediately_severe_over(self) -> None:
        decision = BrightnessGuard().observe(
            np.full((4, 4, 3), 255, np.uint8), self.current, self.hard_safe
        )
        self.assertEqual(decision.state, BrightnessState.SEVERE_OVER)

    def test_black_is_severe_under(self) -> None:
        decision = BrightnessGuard().observe(
            np.zeros((32, 32, 3), np.uint8), self.current, self.hard_safe
        )
        self.assertEqual(decision.state, BrightnessState.SEVERE_UNDER)

    def test_small_highlight_does_not_make_the_frame_overexposed(self) -> None:
        image = np.full((100, 100, 3), 100, np.uint8)
        image[:5, :5] = 255
        decision = BrightnessGuard().observe(image, self.current, self.hard_safe)
        self.assertEqual(decision.state, BrightnessState.GOOD)

    def test_high_p99_without_clipping_evidence_is_not_overexposed(self) -> None:
        image = np.full((100, 100, 3), 100, np.uint8)
        image[:11, :10] = 249
        decision = BrightnessGuard().observe(image, self.current, self.hard_safe)
        self.assertGreaterEqual(decision.raw_stats.p99, 248)
        self.assertEqual(decision.state, BrightnessState.GOOD)

    def test_ema_and_debounce_and_immediate_severe_over(self) -> None:
        guard = BrightnessGuard(
            BrightnessGuardConfig(ema_alpha=1.0, state_change_frames=2)
        )
        good = np.full((32, 32, 3), 100, np.uint8)
        black = np.zeros_like(good)
        white = np.full_like(good, 255)

        self.assertEqual(
            guard.observe(good, self.current, self.hard_safe).state,
            BrightnessState.GOOD,
        )
        first_under = guard.observe(black, self.current, self.hard_safe)
        self.assertEqual(first_under.state, BrightnessState.GOOD)
        self.assertFalse(first_under.state_changed)
        second_under = guard.observe(black, self.current, self.hard_safe)
        self.assertEqual(second_under.state, BrightnessState.SEVERE_UNDER)
        self.assertTrue(second_under.state_changed)
        severe_over = guard.observe(white, self.current, self.hard_safe)
        self.assertEqual(severe_over.state, BrightnessState.SEVERE_OVER)
        self.assertTrue(severe_over.state_changed)

    def test_raw_severe_over_bypasses_ema_damping(self) -> None:
        guard = BrightnessGuard(
            BrightnessGuardConfig(ema_alpha=0.1, state_change_frames=3)
        )
        good = np.full((32, 32, 3), 100, np.uint8)
        clipped = good.copy()
        clipped[:2] = 255
        guard.observe(good, self.current, self.hard_safe)

        decision = guard.observe(clipped, self.current, self.hard_safe)

        self.assertGreaterEqual(
            decision.raw_stats.luma_clip_ratio,
            guard.config.severe_over_luma_clip_ratio,
        )
        self.assertEqual(decision.state, BrightnessState.SEVERE_OVER)
        self.assertTrue(decision.state_changed)

    def test_each_state_has_the_exact_ev_envelope(self) -> None:
        guard = BrightnessGuard()
        current_ev = ev_step(self.current)
        predicates = {
            BrightnessState.SEVERE_OVER: lambda value: value <= current_ev - 1,
            BrightnessState.OVER: lambda value: value <= current_ev,
            BrightnessState.GOOD: lambda value: abs(value - current_ev) <= 1,
            BrightnessState.UNDER: lambda value: current_ev <= value <= current_ev + 1,
            BrightnessState.SEVERE_UNDER: lambda value: current_ev <= value <= current_ev + 2,
        }
        for state, predicate in predicates.items():
            with self.subTest(state=state):
                actual = set(guard._admissible(state, self.current, self.hard_safe))
                expected = {cell for cell in self.hard_safe if predicate(ev_step(cell))}
                expected.add(self.current)
                self.assertEqual(actual, expected)

    def test_recovery_prefers_exact_minus_one_ev_and_exposure_reduction(self) -> None:
        recovery = BrightnessGuard._recovery(self.current, self.hard_safe)
        self.assertEqual(recovery, SensorCell(8, 64))
        self.assertEqual(ev_step(recovery), ev_step(self.current) - 1)

    def test_recovery_unavailable_keeps_the_current_cell_admissible(self) -> None:
        current = SensorCell(4, 16)
        decision = BrightnessGuard(
            BrightnessGuardConfig(mode=BrightnessGuardMode.ENFORCE)
        ).observe(np.full((8, 8, 3), 255, np.uint8), current, (current,))
        self.assertIsNone(decision.recovery_cell)
        self.assertFalse(decision.force_recovery)
        self.assertIn(current, decision.admissible_cells)


class SafetyConfigTest(unittest.TestCase):
    def _load(self, payload) -> SafetyPolicy:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "safety.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return SafetyPolicy.from_json(path)

    def test_global_allowed_gains_apply_to_every_lighting_state(self) -> None:
        safety = self._load({"allowed_gains": [16, 64]})
        for light_state in range(3):
            self.assertEqual(
                {cell.gain for cell in safety.safe_cells(ContextKey(0, light_state))},
                {16, 64},
            )

    def test_legacy_allowed_gains_by_light_is_unchanged(self) -> None:
        expected = ((16,), (32, 64), (128,))
        safety = self._load({"allowed_gains_by_light": expected})
        for light_state, gains in enumerate(expected):
            self.assertEqual(
                {cell.gain for cell in safety.safe_cells(ContextKey(0, light_state))},
                set(gains),
            )

    def test_missing_brightness_section_defaults_to_off(self) -> None:
        self.assertEqual(
            self._load({}).brightness_guard.mode,
            BrightnessGuardMode.OFF,
        )

    def test_invalid_threshold_and_unknown_gain_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._load({"allowed_gains": [17]})
        with self.assertRaises(ValueError):
            self._load({"brightness_guard": {"ema_alpha": 0}})


if __name__ == "__main__":
    unittest.main()
