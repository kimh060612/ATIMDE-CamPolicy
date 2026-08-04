import unittest

from ati_mde_control.config import PolicyConfig, SafetyPolicy
from ati_mde_control.pairwise_policy import PairwisePolicy
from ati_mde_control.types import PairMode, PairStatus, SwitchEvent
from hardware.utils import ContextKey, QScore, SensorCell


def score(mu: float, std: float = 0.01) -> QScore:
    return QScore(mu + std, std, mu)


class PairwisePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)
        self.current = SensorCell(16, 64)

    def policy(self, **values) -> PairwisePolicy:
        policy = PairwisePolicy(PolicyConfig(**values), SafetyPolicy())
        policy.committed_cell(self.context, self.current)
        return policy

    def first_win(self, policy: PairwisePolicy):
        challenger = policy.select_challenger(self.context, self.current)
        decision = policy.resolve(
            self.context, self.current, score(.20), challenger, score(.10), 0
        )
        return challenger, decision

    def commit(self, policy: PairwisePolicy):
        challenger, _ = self.first_win(policy)
        self.assertEqual(policy.select_challenger(self.context, self.current), challenger)
        decision = policy.resolve(
            self.context, self.current, score(.20), challenger, score(.10), 1
        )
        return challenger, decision

    def test_one_win_does_not_commit_or_select_challenger(self) -> None:
        policy = self.policy()
        challenger, decision = self.first_win(policy)
        state = policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CHALLENGER_WON)
        self.assertEqual(decision.switch_event, SwitchEvent.CONFIRMATION_PENDING)
        self.assertEqual(decision.selected_cell, self.current)
        self.assertEqual(state.active_cell_id, self.current.cell_id)
        self.assertEqual(state.pending_switch_to_id, challenger.cell_id)

    def test_pending_challenger_suspends_normal_search(self) -> None:
        policy = self.policy()
        challenger, _ = self.first_win(policy)
        self.assertEqual(policy.pair_mode(self.context), PairMode.SWITCH_CONFIRMATION)
        self.assertEqual(policy.select_challenger(self.context, self.current), challenger)

    def test_two_wins_commit_and_store_rollback_cell(self) -> None:
        policy = self.policy()
        challenger, decision = self.commit(policy)
        state = policy.state(self.context)
        self.assertEqual(decision.switch_event, SwitchEvent.COMMITTED)
        self.assertEqual(decision.selected_cell, challenger)
        self.assertEqual(state.active_cell_id, challenger.cell_id)
        self.assertEqual(state.rollback_cell_id, self.current.cell_id)
        self.assertTrue(state.rollback_verification_pending)

    def test_current_win_cancels_pending_switch(self) -> None:
        policy = self.policy()
        challenger, _ = self.first_win(policy)
        decision = policy.resolve(
            self.context, self.current, score(.10), challenger, score(.20), 1
        )
        state = policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CURRENT_WON)
        self.assertIsNone(state.pending_switch_to_id)
        self.assertEqual(state.pending_switch_wins, 0)
        self.assertTrue(state.exposure_negative_tested)

    def test_ambiguous_preserves_pending_win(self) -> None:
        policy = self.policy()
        challenger, _ = self.first_win(policy)
        policy.resolve(self.context, self.current, score(.10), challenger, score(.10), 1)
        state = policy.state(self.context)
        self.assertEqual(state.pending_switch_wins, 1)
        self.assertEqual(state.pending_switch_to_id, challenger.cell_id)
        self.assertFalse(state.exposure_negative_tested)

    def test_invalid_preserves_pending_win(self) -> None:
        policy = self.policy()
        challenger, _ = self.first_win(policy)
        edge = policy.record_invalid(self.context, self.current, challenger, 1)
        state = policy.state(self.context)
        self.assertEqual(state.pending_switch_wins, 1)
        self.assertEqual(state.pending_switch_to_id, challenger.cell_id)
        self.assertEqual(edge.valid_count, 1)
        self.assertEqual(edge.invalid_count, 1)

    def test_confirmation_timeout_clears_pending_without_rejection(self) -> None:
        policy = self.policy(switch_confirmation_timeout_rounds=8)
        challenger, _ = self.first_win(policy)
        event = policy.begin_round(self.context, 8)
        state = policy.state(self.context)
        self.assertEqual(event, SwitchEvent.CONFIRMATION_TIMEOUT)
        self.assertEqual(state.active_cell_id, self.current.cell_id)
        self.assertIsNone(state.pending_switch_to_id)
        self.assertFalse(state.exposure_negative_tested)
        self.assertGreater(state.local_edges[(self.current.cell_id, challenger.cell_id)].ambiguous_cooldown, 0)

    def test_pending_switch_is_context_specific(self) -> None:
        policy = self.policy()
        challenger, _ = self.first_win(policy)
        other = ContextKey(1, 0)
        policy.committed_cell(other, self.current)
        self.assertEqual(policy.state(self.context).pending_switch_to_id, challenger.cell_id)
        self.assertIsNone(policy.state(other).pending_switch_to_id)

    def test_rollback_old_cell_win_restores_previous_cell(self) -> None:
        policy = self.policy(post_switch_dwell_rounds=0)
        new_cell, _ = self.commit(policy)
        old_cell = policy.select_challenger(self.context, new_cell)
        decision = policy.resolve(
            self.context, new_cell, score(.20), old_cell, score(.10), 2
        )
        self.assertEqual(decision.pair_mode, PairMode.ROLLBACK_VERIFICATION)
        self.assertEqual(decision.switch_event, SwitchEvent.ROLLED_BACK)
        self.assertEqual(policy.state(self.context).active_cell_id, self.current.cell_id)

    def test_rollback_new_cell_win_keeps_commit(self) -> None:
        policy = self.policy(post_switch_dwell_rounds=0)
        new_cell, _ = self.commit(policy)
        old_cell = policy.select_challenger(self.context, new_cell)
        decision = policy.resolve(
            self.context, new_cell, score(.10), old_cell, score(.20), 2
        )
        self.assertEqual(decision.switch_event, SwitchEvent.ROLLBACK_VERIFIED)
        self.assertEqual(policy.state(self.context).active_cell_id, new_cell.cell_id)
        self.assertFalse(policy.state(self.context).rollback_verification_pending)

    def test_ambiguous_rollback_keeps_new_cell(self) -> None:
        policy = self.policy(post_switch_dwell_rounds=0)
        new_cell, _ = self.commit(policy)
        old_cell = policy.select_challenger(self.context, new_cell)
        decision = policy.resolve(
            self.context, new_cell, score(.10), old_cell, score(.10), 2
        )
        self.assertEqual(decision.switch_event, SwitchEvent.ROLLBACK_INCONCLUSIVE)
        self.assertEqual(policy.state(self.context).active_cell_id, new_cell.cell_id)

    def test_invalid_rollback_stays_pending(self) -> None:
        policy = self.policy(post_switch_dwell_rounds=0)
        new_cell, _ = self.commit(policy)
        old_cell = policy.select_challenger(self.context, new_cell)
        policy.record_invalid(self.context, new_cell, old_cell, 2)
        state = policy.state(self.context)
        self.assertEqual(state.active_cell_id, new_cell.cell_id)
        self.assertTrue(state.rollback_verification_pending)

    def test_rollback_timeout_keeps_new_cell(self) -> None:
        policy = self.policy(post_switch_dwell_rounds=0, rollback_verification_timeout_rounds=8)
        new_cell, _ = self.commit(policy)
        policy.begin_round(self.context, 2)
        event = policy.begin_round(self.context, 10)
        state = policy.state(self.context)
        self.assertEqual(event, SwitchEvent.ROLLBACK_TIMEOUT)
        self.assertEqual(state.active_cell_id, new_cell.cell_id)
        self.assertFalse(state.rollback_verification_pending)

    def test_rolled_back_direction_is_not_immediately_recommitted(self) -> None:
        policy = self.policy(post_switch_dwell_rounds=0)
        new_cell, _ = self.commit(policy)
        old_cell = policy.select_challenger(self.context, new_cell)
        policy.resolve(self.context, new_cell, score(.20), old_cell, score(.10), 2)
        state = policy.state(self.context)
        self.assertTrue(state.exposure_negative_tested)
        self.assertNotEqual(policy.select_challenger(self.context, self.current), new_cell)

    def test_pairwise_margin_still_uses_mu_and_std(self) -> None:
        policy = self.policy()
        challenger = policy.select_challenger(self.context, self.current)
        decision = policy.resolve(
            self.context, self.current, score(.20, .10), challenger, score(.17, .10), 0
        )
        self.assertEqual(decision.status, PairStatus.AMBIGUOUS)
        self.assertAlmostEqual(decision.effective_margin, .01 + .25 * (2 * .1**2) ** .5)


if __name__ == "__main__":
    unittest.main()
