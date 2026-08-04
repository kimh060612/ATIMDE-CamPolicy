import unittest

from ati_mde_control.config import PolicyConfig, SafetyPolicy
from ati_mde_control.pairwise_policy import PairwisePolicy
from ati_mde_control.types import PairStatus
from hardware.utils import ContextKey, QScore, SensorCell


def score(mu: float, std: float = 0.01) -> QScore:
    return QScore(mu + std, std, mu)


class PairwisePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)
        self.policy = PairwisePolicy(PolicyConfig(), SafetyPolicy())
        self.current = SensorCell(16, 64)
        self.policy.committed_cell(self.context, self.current)

    def challenger(self):
        return self.policy.select_challenger(self.context, self.current)

    def test_pairwise_margin_uses_mu_for_direction_and_std_for_confidence(self) -> None:
        challenger = self.challenger()
        decision = self.policy.resolve(
            self.context, self.current, score(.20, .10), challenger, score(.17, .10), 0
        )
        self.assertEqual(decision.status, PairStatus.AMBIGUOUS)
        self.assertAlmostEqual(decision.effective_margin, .01 + .25 * (2 * .1**2) ** .5)
        self.assertEqual(decision.selected_cell, self.current)

    def test_ambiguous_does_not_mark_challenger_inferior(self) -> None:
        challenger = self.challenger()
        self.policy.resolve(self.context, self.current, score(.10), challenger, score(.095), 0)
        state = self.policy.state(self.context)
        edge = state.local_edges[(self.current.cell_id, challenger.cell_id)]
        self.assertFalse(state.exposure_negative_tested)
        self.assertEqual(edge.ambiguous_count, 1)
        self.assertEqual(state.active_cell_id, self.current.cell_id)

    def test_invalid_updates_no_quality_superiority(self) -> None:
        challenger = self.challenger()
        edge = self.policy.record_invalid(self.context, self.current, challenger, 0)
        self.assertEqual(edge.valid_count, 0)
        self.assertIsNone(edge.last_delta_mu)
        self.assertEqual(self.policy.state(self.context).active_cell_id, self.current.cell_id)

    def test_invalid_does_not_close_exposure_direction(self) -> None:
        challenger = self.challenger()
        self.policy.record_invalid(self.context, self.current, challenger, 0)
        state = self.policy.state(self.context)
        self.assertFalse(state.exposure_negative_tested)
        self.assertFalse(state.exposure_positive_tested)

    def test_three_invalid_pairs_trigger_bounded_recovery(self) -> None:
        policy = PairwisePolicy(PolicyConfig(max_consecutive_invalid_pairs=3, recovery_current_only_rounds=2), SafetyPolicy())
        policy.committed_cell(self.context, self.current)
        challengers = (SensorCell(8, 64), SensorCell(32, 64), SensorCell(16, 32))
        for index, challenger in enumerate(challengers):
            policy.record_invalid(self.context, self.current, challenger, index)
        state = policy.state(self.context)
        self.assertEqual(state.force_current_only_rounds, 2)
        self.assertEqual(state.consecutive_invalid_pairs, 0)


if __name__ == "__main__":
    unittest.main()
