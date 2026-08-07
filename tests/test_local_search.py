import unittest

from ati_mde_control import local_search
from ati_mde_control.config import PolicyConfig, SafetyPolicy
from ati_mde_control.pairwise_policy import PairwisePolicy
from ati_mde_control.state import ContextState
from ati_mde_control.types import PairStatus, SearchAxis, SearchDirection
from hardware.utils import ContextKey, QScore, SensorCell


def score(mu: float, std: float = 0.01) -> QScore:
    return QScore(mu + std, std, mu)


class LocalSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)
        self.policy = PairwisePolicy(PolicyConfig(), SafetyPolicy())

    def test_neighbor_changes_exactly_one_axis_by_one_step(self) -> None:
        cell = SensorCell(16, 64)
        neighbors = {
            local_search.neighbor(cell, axis, direction)
            for axis in SearchAxis
            for direction in SearchDirection
        }
        self.assertEqual(
            neighbors,
            {SensorCell(8, 64), SensorCell(32, 64), SensorCell(16, 32), SensorCell(16, 128)},
        )

    def test_middle_cell_tests_both_directions_before_axis_closes(self) -> None:
        current = SensorCell(16, 64)
        self.policy.committed_cell(self.context, current)
        first = self.policy.select_challenger(self.context, current)
        self.assertEqual(first, SensorCell(8, 64))
        self.policy.resolve(self.context, current, score(.1), first, score(.2), 0)
        second = self.policy.select_challenger(self.context, current)
        self.assertEqual(second, SensorCell(32, 64))

    def test_current_win_closes_only_tested_direction(self) -> None:
        current = SensorCell(16, 64)
        self.policy.committed_cell(self.context, current)
        challenger = self.policy.select_challenger(self.context, current)
        decision = self.policy.resolve(self.context, current, score(.1), challenger, score(.2), 0)
        state = self.policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CURRENT_WON)
        self.assertTrue(state.exposure_negative_tested)
        self.assertFalse(state.exposure_positive_tested)
        self.assertEqual(
            self.policy.select_challenger(self.context, current), SensorCell(32, 64)
        )

    def test_exposure_win_continues_in_same_direction(self) -> None:
        current = SensorCell(16, 64)
        self.policy.committed_cell(self.context, current)
        challenger = self.policy.select_challenger(self.context, current)
        decision = self.policy.resolve(self.context, current, score(.2), challenger, score(.1), 0)
        state = self.policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CHALLENGER_WON)
        self.assertEqual(state.search_axis, SearchAxis.EXPOSURE)
        self.assertEqual(state.search_direction, SearchDirection.NEGATIVE)
        self.assertFalse(state.exposure_positive_tested)
        self.assertEqual(self.policy.select_challenger(self.context, challenger), SensorCell(4, 64))

    def test_commit_at_boundary_reverses_on_same_axis(self) -> None:
        current = SensorCell(8, 64)
        self.policy.committed_cell(self.context, current)
        challenger = self.policy.select_challenger(self.context, current)
        self.assertEqual(challenger, SensorCell(4, 64))

        self.policy.resolve(self.context, current, score(.2), challenger, score(.1), 0)
        state = self.policy.state(self.context)
        self.assertFalse(state.exposure_positive_tested)
        self.assertEqual(
            self.policy.select_challenger(self.context, challenger), SensorCell(8, 64)
        )
        self.assertEqual(state.search_axis, SearchAxis.EXPOSURE)
        self.assertEqual(state.search_direction, SearchDirection.POSITIVE)

    def test_axis_advances_only_after_both_directions_are_rejected(self) -> None:
        current = SensorCell(16, 64)
        self.policy.committed_cell(self.context, current)
        negative = self.policy.select_challenger(self.context, current)
        self.policy.resolve(self.context, current, score(.1), negative, score(.2), 0)

        positive = self.policy.select_challenger(self.context, current)
        state = self.policy.state(self.context)
        self.assertEqual(positive, SensorCell(32, 64))
        self.assertEqual(state.search_axis, SearchAxis.EXPOSURE)
        self.policy.resolve(self.context, current, score(.1), positive, score(.2), 1)

        self.assertEqual(
            self.policy.select_challenger(self.context, current), SensorCell(16, 32)
        )
        self.assertEqual(state.search_axis, SearchAxis.GAIN)

    def test_exposure_and_gain_phases_do_not_starve(self) -> None:
        state = ContextState(
            active_cell_id="E16_G064",
            exposure_negative_tested=True,
            exposure_positive_tested=True,
        )
        candidate = local_search.select_challenger(state, SensorCell(16, 64), SafetyPolicy().safe_cells(self.context))
        self.assertEqual(state.search_axis, SearchAxis.GAIN)
        self.assertEqual(candidate, SensorCell(16, 32))
        state.gain_negative_tested = state.gain_positive_tested = True
        state.search_direction = None
        candidate = local_search.select_challenger(state, SensorCell(16, 64), SafetyPolicy().safe_cells(self.context))
        self.assertEqual(state.search_axis, SearchAxis.EXPOSURE)
        self.assertTrue(state.exposure_recheck)
        self.assertEqual(candidate, SensorCell(8, 64))

    def test_full_cycle_without_switch_enters_exploitation(self) -> None:
        state = ContextState(
            exposure_recheck=True,
            exposure_negative_tested=True,
            exposure_positive_tested=True,
        )
        self.assertIsNone(local_search.select_challenger(
            state, SensorCell(16, 64), SafetyPolicy().safe_cells(self.context)
        ))
        self.assertTrue(state.search_cycle_complete)
        self.assertFalse(state.probe_pending)

    def test_cycle_with_switch_starts_another_cycle(self) -> None:
        state = ContextState(
            exposure_recheck=True,
            exposure_negative_tested=True,
            exposure_positive_tested=True,
            cycle_had_switch=True,
        )
        challenger = local_search.select_challenger(
            state, SensorCell(16, 64), SafetyPolicy().safe_cells(self.context)
        )
        self.assertEqual(challenger, SensorCell(8, 64))
        self.assertFalse(state.search_cycle_complete)
        self.assertFalse(state.cycle_had_switch)

    def test_periodic_reprobe_restarts_converged_search(self) -> None:
        state = self.policy.state(self.context)
        state.search_cycle_complete = True
        state.probe_pending = False
        state.rounds_since_valid_probe = 29
        self.policy.schedule_after_current(self.context, .01)
        self.assertTrue(state.probe_pending)
        self.assertFalse(state.search_cycle_complete)

    def test_high_mu_restarts_converged_search(self) -> None:
        state = self.policy.state(self.context)
        state.search_cycle_complete = True
        state.probe_pending = False
        self.policy.schedule_after_current(self.context, .20)
        self.assertTrue(state.probe_pending)
        self.assertFalse(state.search_cycle_complete)

    def test_low_mu_cannot_cancel_unfinished_cycle(self) -> None:
        state = self.policy.state(self.context)
        state.search_cycle_complete = False
        self.policy.schedule_after_current(self.context, .01)
        self.assertTrue(state.probe_pending)


if __name__ == "__main__":
    unittest.main()
