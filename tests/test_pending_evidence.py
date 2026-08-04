import unittest

from ati_mde_control.config import PolicyConfig, SafetyPolicy
from ati_mde_control.pairwise_policy import PairwisePolicy
from ati_mde_control.types import PairMode, PairStatus, SwitchEvent
from hardware.utils import ContextKey, QScore, SensorCell


def score(mu: float, std: float) -> QScore:
    return QScore(mu, std, mu)


class PendingEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)
        self.current = SensorCell(16, 64)
        self.challenger = SensorCell(8, 64)

    def policy(self, **values) -> PairwisePolicy:
        policy = PairwisePolicy(PolicyConfig(**values), SafetyPolicy())
        policy.committed_cell(self.context, self.current)
        self.assertEqual(policy.select_challenger(self.context, self.current), self.challenger)
        return policy

    def resolve(self, policy, delta, std, round_index=0):
        return policy.resolve(
            self.context,
            self.current,
            score(.1 + delta, std),
            self.challenger,
            score(.1, std),
            round_index,
        )

    def test_high_confidence_win_commits_immediately_without_pending(self) -> None:
        policy = self.policy()
        decision = self.resolve(policy, .1, .01)
        state = policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CHALLENGER_WON)
        self.assertEqual(decision.switch_event, SwitchEvent.IMMEDIATE_COMMIT)
        self.assertEqual(state.active_cell_id, self.challenger.cell_id)
        self.assertIsNone(state.pending_edge_from_id)

    def test_positive_uncertain_result_starts_pending_and_keeps_current(self) -> None:
        policy = self.policy()
        decision = self.resolve(policy, .006, .01)
        state = policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.AMBIGUOUS)
        self.assertEqual(decision.switch_event, SwitchEvent.PENDING_STARTED)
        self.assertGreaterEqual(decision.pair_z, policy.config.pending_min_z)
        self.assertTrue(decision.pending_admitted)
        self.assertEqual(decision.pending_observation_count, 1)
        self.assertAlmostEqual(decision.aggregated_delta_mu, .006)
        self.assertEqual(decision.selected_cell, self.current)
        self.assertEqual(state.active_cell_id, self.current.cell_id)
        self.assertEqual(state.pending_edge_to_id, self.challenger.cell_id)

    def test_weak_positive_ambiguity_does_not_start_pending(self) -> None:
        policy = self.policy()
        decision = self.resolve(policy, .005, .03)
        edge = policy.state(self.context).local_edges[(self.current.cell_id, self.challenger.cell_id)]
        self.assertEqual(decision.status, PairStatus.AMBIGUOUS)
        self.assertLess(decision.pair_z, policy.config.pending_min_z)
        self.assertFalse(decision.pending_admitted)
        self.assertFalse(edge.pending)
        self.assertGreater(edge.ambiguous_cooldown, 0)

    def test_weak_nonpositive_ambiguity_does_not_start_pending(self) -> None:
        policy = self.policy()
        decision = self.resolve(policy, -.005, .01)
        edge = policy.state(self.context).local_edges[(self.current.cell_id, self.challenger.cell_id)]
        self.assertEqual(decision.switch_event, SwitchEvent.NONE)
        self.assertFalse(edge.pending)
        self.assertGreater(edge.ambiguous_cooldown, 0)

    def test_pending_edge_has_priority(self) -> None:
        policy = self.policy()
        self.resolve(policy, .006, .01)
        self.assertEqual(policy.select_challenger(self.context, self.current), self.challenger)

    def test_stale_pending_edge_is_cleared(self) -> None:
        policy = self.policy()
        self.resolve(policy, .006, .01)
        policy.select_challenger(self.context, SensorCell(32, 64))
        state = policy.state(self.context)
        edge = state.local_edges[(self.current.cell_id, self.challenger.cell_id)]
        self.assertIsNone(state.pending_edge_from_id)
        self.assertFalse(edge.pending)
        self.assertEqual(edge.pending_observation_count, 0)

    def test_lower_sigma_contributes_more_precision(self) -> None:
        policy = self.policy(max_pending_observations=3, pending_min_z=.1)
        high_sigma = self.resolve(policy, .005, .03)
        low_sigma = self.resolve(policy, .005, .005, 1)
        low_sigma_weight = low_sigma.pending_precision_sum - high_sigma.pending_precision_sum
        self.assertGreater(low_sigma_weight, high_sigma.pending_precision_sum)

    def test_two_consistent_uncertain_observations_commit(self) -> None:
        policy = self.policy()
        first = self.resolve(policy, .006, .01)
        decision = self.resolve(policy, .006, .01, 1)
        self.assertEqual(decision.pair_mode, PairMode.PENDING_RECHECK)
        self.assertEqual(decision.status, PairStatus.AMBIGUOUS)
        self.assertEqual(decision.switch_event, SwitchEvent.PENDING_COMMITTED)
        self.assertEqual(decision.pending_observation_count, 2)
        self.assertLess(first.delta_mu, first.effective_margin)
        self.assertLess(decision.delta_mu, decision.effective_margin)
        self.assertGreater(decision.aggregated_delta_mu, policy.config.pending_margin)
        self.assertGreaterEqual(decision.aggregated_z, policy.config.pending_commit_z)
        self.assertEqual(policy.state(self.context).active_cell_id, self.challenger.cell_id)

    def test_raw_challenger_win_is_aggregated_before_pending_commit(self) -> None:
        policy = self.policy()
        first = self.resolve(policy, .012, .03)
        decision = self.resolve(policy, .03, .005, 1)
        first_precision = 1.0 / (2 * .03**2)
        second_precision = 1.0 / (2 * .005**2)
        self.assertEqual(first.switch_event, SwitchEvent.PENDING_STARTED)
        self.assertEqual(decision.status, PairStatus.CHALLENGER_WON)
        self.assertEqual(decision.switch_event, SwitchEvent.PENDING_COMMITTED)
        self.assertEqual(decision.pending_observation_count, 2)
        self.assertAlmostEqual(decision.pending_precision_sum, first_precision + second_precision)
        self.assertAlmostEqual(
            decision.pending_weighted_delta_sum,
            .012 * first_precision + .03 * second_precision,
        )

    def test_raw_current_win_is_aggregated_before_pending_rejection(self) -> None:
        policy = self.policy()
        self.resolve(policy, .012, .03)
        decision = self.resolve(policy, -.03, .005, 1)
        state = policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CURRENT_WON)
        self.assertEqual(decision.switch_event, SwitchEvent.PENDING_REJECTED)
        self.assertEqual(decision.pending_observation_count, 2)
        self.assertLess(decision.aggregated_delta_mu, -policy.config.pending_margin)
        self.assertLessEqual(decision.aggregated_z, -policy.config.pending_commit_z)
        self.assertEqual(state.active_cell_id, self.current.cell_id)
        self.assertTrue(state.exposure_negative_tested)
        self.assertIsNone(state.pending_edge_from_id)

    def test_pending_exhaustion_keeps_current_and_closes_direction(self) -> None:
        policy = self.policy()
        self.resolve(policy, .0045, .01)
        decision = self.resolve(policy, .0045, .01, 1)
        state = policy.state(self.context)
        self.assertEqual(decision.switch_event, SwitchEvent.PENDING_EXHAUSTED)
        self.assertEqual(state.active_cell_id, self.current.cell_id)
        self.assertTrue(state.exposure_negative_tested)
        self.assertIsNone(state.pending_edge_from_id)

    def test_pending_timeout_keeps_current_and_resumes_search(self) -> None:
        policy = self.policy()
        self.resolve(policy, .006, .01)
        event = policy.begin_round(self.context, 3)
        state = policy.state(self.context)
        self.assertEqual(event, SwitchEvent.PENDING_TIMEOUT)
        self.assertEqual(state.active_cell_id, self.current.cell_id)
        self.assertTrue(state.exposure_negative_tested)
        self.assertIsNone(state.pending_edge_from_id)
        self.assertTrue(state.probe_pending)

    def test_invalid_pending_pair_preserves_evidence_and_pending_state(self) -> None:
        policy = self.policy(invalid_edge_cooldown_rounds=2, pending_timeout_rounds=4)
        decision = self.resolve(policy, .006, .01)
        edge = policy.state(self.context).local_edges[(self.current.cell_id, self.challenger.cell_id)]
        before = (
            edge.pending_weighted_delta_sum,
            edge.pending_precision_sum,
            edge.pending_observation_count,
        )
        policy.record_invalid(self.context, self.current, self.challenger, 1)
        self.assertEqual(before, (
            edge.pending_weighted_delta_sum,
            edge.pending_precision_sum,
            edge.pending_observation_count,
        ))
        self.assertTrue(edge.pending)
        self.assertEqual(policy.state(self.context).pending_edge_to_id, self.challenger.cell_id)
        self.assertEqual(decision.pending_observation_count, 1)
        policy.begin_round(self.context, 2)
        self.assertIsNone(policy.select_challenger(self.context, self.current))
        policy.begin_round(self.context, 3)
        self.assertEqual(policy.select_challenger(self.context, self.current), self.challenger)

    def test_pending_state_is_context_specific(self) -> None:
        policy = self.policy()
        self.resolve(policy, .006, .01)
        other = ContextKey(1, 0)
        policy.committed_cell(other, self.current)
        other_challenger = policy.select_challenger(other, self.current)
        decision = policy.resolve(
            other, self.current, score(.2, .01), other_challenger, score(.1, .01), 1
        )
        self.assertEqual(decision.switch_event, SwitchEvent.IMMEDIATE_COMMIT)
        self.assertEqual(policy.state(other).active_cell_id, other_challenger.cell_id)
        self.assertEqual(policy.state(self.context).pending_edge_to_id, self.challenger.cell_id)

    def test_commit_continues_forward_without_dwell_or_rollback(self) -> None:
        policy = self.policy()
        self.resolve(policy, .1, .01)
        self.assertEqual(
            policy.select_challenger(self.context, self.challenger), SensorCell(4, 64)
        )
        self.assertEqual(
            set(PairMode), {PairMode.NORMAL_SEARCH, PairMode.PENDING_RECHECK}
        )

    def test_pending_configuration_is_bounded(self) -> None:
        config = PolicyConfig()
        self.assertEqual((config.max_pending_observations, config.pending_timeout_rounds), (2, 3))
        self.assertEqual(config.pending_std_floor, 1e-4)
        self.assertEqual(
            (config.pending_min_z, config.pending_margin, config.pending_commit_z),
            (.25, .005, .5),
        )
        with self.assertRaises(ValueError):
            PolicyConfig(max_pending_observations=1)
        with self.assertRaises(ValueError):
            PolicyConfig(pending_std_floor=0)
        with self.assertRaises(ValueError):
            PolicyConfig(pending_min_z=-1)
        with self.assertRaises(ValueError):
            PolicyConfig(pending_margin=-1)
        with self.assertRaises(ValueError):
            PolicyConfig(pending_commit_z=0)


if __name__ == "__main__":
    unittest.main()
