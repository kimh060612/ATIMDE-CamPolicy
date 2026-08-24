import math
import unittest
from unittest.mock import patch

from ati_mde_control.config import SafetyPolicy
from ati_mde_control.risk_bandit_policy import (
    GPNumericalError,
    RiskBanditConfig,
    RiskBanditPolicy,
)
from hardware.utils import ALL_CELLS, ContextKey, QScore, SensorCell


def q_score(q: float, *, mu: float | None = None, uncertainty: float = 0.2) -> QScore:
    return QScore(q=q, mu=q - uncertainty if mu is None else mu, uncertainty=uncertainty)


class RiskBanditPolicyTest(unittest.TestCase):
    def make_policy(self, **changes) -> RiskBanditPolicy:
        config = RiskBanditConfig(**changes)
        return RiskBanditPolicy(config, SafetyPolicy(), SensorCell(16, 64))

    def test_history_target_is_exactly_q_and_is_bounded(self) -> None:
        policy = self.make_policy(window_size=3)
        for index in range(5):
            policy.add_observation(
                ContextKey(0, 0), ALL_CELLS[index], index, q_score(10.0 + index)
            )
        self.assertEqual([item.q for item in policy.history], [12.0, 13.0, 14.0])

    def test_aleatoric_uncertainty_is_not_averaged_or_used_as_gp_noise(self) -> None:
        low = self.make_policy()
        high = self.make_policy()
        cell = SensorCell(16, 64)
        low.add_observation(ContextKey(0, 0), cell, 0, q_score(0.5, mu=0.49, uncertainty=0.01))
        high.add_observation(ContextKey(0, 0), cell, 0, q_score(0.5, mu=-9.5, uncertainty=10.0))
        low_prediction = low.posterior_predictions(ContextKey(0, 0), [cell], 0)[0]
        high_prediction = high.posterior_predictions(ContextKey(0, 0), [cell], 0)[0]
        self.assertEqual(low.history[0].q, 0.5)
        self.assertEqual(high.history[0].q, 0.5)
        self.assertAlmostEqual(low_prediction.mean_risk, high_prediction.mean_risk)
        self.assertAlmostEqual(low_prediction.epistemic_std, high_prediction.epistemic_std)

    def test_temporal_kernel_makes_old_observation_less_certain(self) -> None:
        policy = self.make_policy(temporal_scale_sec=1.0)
        cell = SensorCell(16, 64)
        policy.add_observation(ContextKey(0, 0), cell, 0, q_score(0.5))
        recent = policy.posterior_predictions(ContextKey(0, 0), [cell], 0)[0]
        old = policy.posterior_predictions(ContextKey(0, 0), [cell], 10_000_000_000)[0]
        self.assertGreater(old.epistemic_std, recent.epistemic_std)

    def test_selection_is_always_in_safe_cells(self) -> None:
        safety = SafetyPolicy(
            max_exposure_ms_by_motion=(8, 32, 32, 32, 32),
            allowed_gains_by_light=((16, 32), (16, 32, 64, 128), (16, 32, 64, 128)),
        )
        policy = RiskBanditPolicy(RiskBanditConfig(), safety, SensorCell(16, 64))
        context = ContextKey(0, 0)
        policy.add_observation(context, SensorCell(4, 16), 0, q_score(0.5))
        selected = policy.select_action(context, SensorCell(32, 128), 1).selected_cell
        self.assertIn(selected, safety.safe_cells(context))

    def test_no_history_keeps_safe_current_or_uses_default_nearest_fallback(self) -> None:
        safety = SafetyPolicy(max_exposure_ms_by_motion=(8, 32, 32, 32, 32))
        policy = RiskBanditPolicy(RiskBanditConfig(), safety, SensorCell(16, 64))
        context = ContextKey(0, 0)
        self.assertEqual(
            policy.select_action(context, SensorCell(8, 32), 0).selected_cell,
            SensorCell(8, 32),
        )
        self.assertEqual(
            policy.select_action(context, SensorCell(32, 128), 0).selected_cell,
            SensorCell(8, 64),
        )

    def test_equal_acquisition_prefers_current_then_cell_id(self) -> None:
        policy = self.make_policy(exploration_beta=0.0, switch_penalty=0.0)
        current = SensorCell(16, 64)
        policy.add_observation(ContextKey(0, 0), current, 0, q_score(0.5))
        decision = policy.select_action(ContextKey(0, 0), current, 0)
        self.assertEqual(decision.selected_cell, current)

        unsafe_current = SensorCell(1, 1)
        decision = policy.select_action(ContextKey(0, 0), unsafe_current, 0)
        self.assertEqual(decision.selected_cell, min(ALL_CELLS, key=lambda cell: cell.cell_id))

    def test_switch_penalty_is_part_of_acquisition(self) -> None:
        policy = self.make_policy(exploration_beta=0.0, switch_penalty=0.125)
        current = SensorCell(16, 64)
        policy.add_observation(ContextKey(0, 0), current, 0, q_score(0.5))
        decision = policy.select_action(ContextKey(0, 0), current, 0)
        by_cell = {item.cell: item for item in decision.candidates}
        other = SensorCell(16, 32)
        self.assertAlmostEqual(
            by_cell[other].acquisition - by_cell[other].mean_risk,
            0.125,
        )
        self.assertAlmostEqual(
            by_cell[current].acquisition - by_cell[current].mean_risk,
            0.0,
        )

    def test_non_finite_score_is_rejected(self) -> None:
        for score in (
            QScore(math.nan, 0.1, 0.1),
            QScore(0.1, math.inf, 0.1),
            QScore(0.1, 0.1, None),
        ):
            policy = self.make_policy()
            self.assertFalse(policy.add_observation(ContextKey(0, 0), ALL_CELLS[0], 0, score))
            self.assertEqual(policy.history, ())
            self.assertEqual(policy.last_update_status, "non_finite_score")

    def test_numerical_failure_returns_safe_fallback_without_observation(self) -> None:
        policy = self.make_policy()
        context = ContextKey(0, 0)
        current = SensorCell(16, 64)
        policy.add_observation(context, current, 0, q_score(0.5))
        with patch.object(policy, "posterior_predictions", side_effect=GPNumericalError("synthetic")):
            decision = policy.select_action(context, current, 1)
        self.assertEqual(decision.selected_cell, current)
        self.assertEqual(decision.status, "gp_numerical_fallback")

    def test_brightness_filter_and_recovery_stay_inside_hard_safety(self) -> None:
        policy = self.make_policy()
        context = ContextKey(0, 0)
        current = SensorCell(16, 64)
        admissible = (current, SensorCell(8, 64))
        policy.add_observation(context, SensorCell(32, 128), 0, q_score(-10.0))
        filtered = policy.select_action(
            context,
            current,
            1,
            admissible_cells=admissible,
            prefer_brighter=False,
        )
        self.assertIn(filtered.selected_cell, admissible)
        recovered = policy.select_action(
            context,
            current,
            1,
            admissible_cells=admissible,
            forced_cell=SensorCell(8, 64),
        )
        self.assertEqual(recovered.selected_cell, SensorCell(8, 64))
        self.assertEqual(recovered.status, "brightness_recovery")
        with self.assertRaises(ValueError):
            policy.select_action(
                context,
                current,
                1,
                admissible_cells=admissible,
                forced_cell=SensorCell(32, 128),
            )

    def test_brightness_change_discards_only_matching_context_history(self) -> None:
        policy = self.make_policy()
        changed = ContextKey(0, 0)
        retained = ContextKey(1, 0)
        policy.add_observation(changed, SensorCell(16, 64), 0, q_score(0.5))
        policy.add_observation(retained, SensorCell(16, 64), 1, q_score(0.4))
        policy.reset_for_brightness_change(changed)
        self.assertEqual(tuple(item.context for item in policy.history), (retained,))


if __name__ == "__main__":
    unittest.main()
