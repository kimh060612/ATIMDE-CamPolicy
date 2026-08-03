import math
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hardware.utils import ALL_CELLS, CELL_BY_ID, ContextKey, QScore
from orbbec_deterministic_probing_modelv1 import ModelV1Experiment
from policy.basic_policy import (
    ATIMDECameraProbingController,
    SafetyPolicy,
    expected_improvement,
    normal_cdf,
    normal_pdf,
    probability_of_improvement,
)


def score(mu: float, std: float) -> QScore:
    return QScore(q=mu + std, mu=mu, uncertainty=std)


class SequentialEIPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)

    def controller(self, **overrides) -> ATIMDECameraProbingController:
        arguments = {
            "unobserved_prior_mean": 0.2,
            "unobserved_prior_variance": 0.04,
            "belief_variance_floor": 1e-6,
            "belief_process_variance": 0.0,
            "belief_age_variance_per_cycle": 0.0,
        }
        arguments.update(overrides)
        return ATIMDECameraProbingController(SafetyPolicy(), **arguments)

    def test_normal_helpers_and_acquisition_match_closed_form(self) -> None:
        self.assertAlmostEqual(normal_cdf(0.0), 0.5)
        self.assertAlmostEqual(normal_pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi))

        candidate_mean = 0.1
        candidate_variance = 0.04
        best_mean = 0.2
        best_variance = 0.01
        margin = 0.02
        difference = best_mean - margin - candidate_mean
        scale = math.sqrt(candidate_variance + best_variance)
        z_value = difference / scale
        self.assertAlmostEqual(
            probability_of_improvement(
                candidate_mean,
                candidate_variance,
                best_mean,
                best_variance,
                margin,
            ),
            normal_cdf(z_value),
        )
        self.assertAlmostEqual(
            expected_improvement(
                candidate_mean,
                candidate_variance,
                best_mean,
                best_variance,
                margin,
            ),
            difference * normal_cdf(z_value) + scale * normal_pdf(z_value),
        )

    def test_precision_fusion_preserves_ema_and_updates_belief(self) -> None:
        controller = self.controller()
        cell = ALL_CELLS[0]
        controller.observe(self.context, cell, score(0.2, 0.1))
        update = controller.observe(self.context, cell, score(0.1, 0.2))

        self.assertAlmostEqual(update.belief_variance, 0.008)
        self.assertAlmostEqual(update.belief_mean, 0.18)
        stats = controller.context_states[self.context.table_key].cells[cell.cell_id]
        self.assertEqual(stats.count, 2)
        self.assertAlmostEqual(stats.ema_score, 0.3)
        self.assertEqual(stats.last_scene_epoch, 0)

    def test_unobserved_prior_and_all_cells_tie_breaking(self) -> None:
        controller = self.controller()
        self.assertEqual(controller.posterior_best(self.context), ALL_CELLS[0])

        # A high first observation leaves the broad unobserved prior as best.
        controller.observe(self.context, ALL_CELLS[-1], score(0.3, 0.02))
        best = controller.posterior_best(self.context)
        self.assertEqual(best, ALL_CELLS[0])
        next_cell, probability, improvement = controller.select_next_probe_cell(
            self.context, best, set()
        )
        self.assertEqual(next_cell, best)
        self.assertEqual(probability, 0.0)
        self.assertEqual(improvement, 0.0)

    def test_equal_prior_cells_use_grid_diverse_order(self) -> None:
        controller = self.controller()
        best = ALL_CELLS[0]
        controller.observe(self.context, best, score(0.1, 0.02))
        probed: set[str] = set()
        expected = [
            "E04_G128",
            "E32_G016",
            "E32_G128",
            "E08_G032",
            "E08_G064",
            "E16_G032",
            "E16_G064",
        ]
        for expected_cell_id in expected:
            next_cell, probability, improvement = (
                controller.select_next_probe_cell(self.context, best, probed)
            )
            self.assertIsNotNone(next_cell)
            assert next_cell is not None
            self.assertEqual(next_cell.cell_id, expected_cell_id)
            self.assertGreater(probability, 0.0)
            self.assertGreater(improvement, 0.0)
            probed.add(next_cell.cell_id)

    def test_raw_scores_choose_final_cell_not_belief_mean(self) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.observe(self.context, current, score(0.02, 0.01))
        controller.observe(self.context, challenger, score(0.40, 0.01))

        selected = controller.select_raw_cell(
            self.context,
            current,
            QScore(q=0.30, mu=0.30, uncertainty=0.01),
            challenger,
            QScore(q=0.10, mu=0.10, uncertainty=0.01),
        )

        self.assertEqual(selected, challenger)

    def test_stale_low_belief_cannot_override_worse_current_raw_q(self) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.observe(self.context, current, score(0.30, 0.01))
        controller.observe(self.context, challenger, score(0.01, 0.01))

        selected = controller.select_raw_cell(
            self.context,
            current,
            QScore(q=0.10, mu=0.10, uncertainty=0.01),
            challenger,
            QScore(q=0.20, mu=0.20, uncertainty=0.01),
        )

        self.assertEqual(selected, current)

    def test_process_variance_prevents_permanent_variance_collapse(self) -> None:
        controller = self.controller(
            belief_process_variance=0.001,
            maximum_belief_variance=0.04,
        )
        cell = ALL_CELLS[0]
        for _ in range(100):
            update = controller.observe(self.context, cell, score(0.1, 0.01))

        self.assertGreater(update.belief_variance, 1e-5)
        self.assertLessEqual(update.belief_variance, 0.04)

    def test_failed_challenger_is_temporarily_excluded(self) -> None:
        controller = self.controller(challenger_cooldown_rounds=2)
        current = CELL_BY_ID["E04_G016"]
        failed = CELL_BY_ID["E04_G128"]
        controller.observe(self.context, current, score(0.1, 0.01))
        controller.select_raw_cell(
            self.context,
            current,
            QScore(q=0.1, mu=0.1, uncertainty=0.01),
            failed,
            QScore(q=0.2, mu=0.2, uncertainty=0.01),
        )

        selected, _, _ = controller.select_next_probe_cell(
            self.context, current, set()
        )
        self.assertNotEqual(selected, failed)
        controller.start_round(self.context)
        selected, _, _ = controller.select_next_probe_cell(
            self.context, current, set()
        )
        self.assertNotEqual(selected, failed)
        controller.start_round(self.context)
        selected, _, _ = controller.select_next_probe_cell(
            self.context, current, set()
        )
        self.assertEqual(selected, failed)

    def test_tentative_switch_is_confirmed_by_next_frame(self) -> None:
        controller = self.controller(switch_confirmation_margin=0.02)
        current, challenger = ALL_CELLS[:2]
        selected = controller.select_raw_cell(
            self.context,
            current,
            QScore(q=0.2, mu=0.2, uncertainty=0.01),
            challenger,
            QScore(q=0.1, mu=0.1, uncertainty=0.01),
        )

        result, confirmed = controller.resolve_tentative_switch(
            self.context,
            selected,
            QScore(q=0.21, mu=0.21, uncertainty=0.01),
        )

        self.assertEqual((result, confirmed), ("confirmed", challenger))
        state = controller.context_states[self.context.table_key]
        self.assertIsNone(state.tentative_cell_id)
        self.assertEqual(state.best_cell_id, challenger.cell_id)

    def test_bad_tentative_switch_rolls_back_without_probe(self) -> None:
        controller = self.controller(
            switch_confirmation_margin=0.01,
            challenger_cooldown_rounds=3,
        )
        previous, challenger = ALL_CELLS[:2]
        controller.select_raw_cell(
            self.context,
            previous,
            QScore(q=0.2, mu=0.2, uncertainty=0.01),
            challenger,
            QScore(q=0.1, mu=0.1, uncertainty=0.01),
        )

        result, selected = controller.resolve_tentative_switch(
            self.context,
            challenger,
            QScore(q=0.22, mu=0.22, uncertainty=0.01),
        )

        self.assertEqual((result, selected), ("rolled_back", previous))
        self.assertEqual(controller.candidate_cooldown(self.context, challenger), 3)

    def test_belief_aging_is_effective_only_and_does_not_mutate_storage(self) -> None:
        controller = self.controller(
            belief_age_variance_per_cycle=0.005,
            maximum_belief_variance=0.04,
        )
        cell = ALL_CELLS[0]
        controller.observe(self.context, cell, score(0.1, 0.01))
        state = controller.context_states[self.context.table_key]
        stored = state.cells[cell.cell_id].belief_variance
        controller.remember_used_cell(self.context, cell)

        _, aged = controller.belief(self.context, cell)

        self.assertAlmostEqual(aged, float(stored) + 0.005)
        self.assertEqual(state.cells[cell.cell_id].belief_variance, stored)

    def test_runtime_captures_at_most_one_challenger(self) -> None:
        controller = self.controller()
        experiment = ModelV1Experiment.__new__(ModelV1Experiment)
        experiment.policy = controller
        experiment.context_provider = SimpleNamespace(
            get=lambda: self.context,
            is_stable=True,
            transition_contexts=lambda: (self.context,),
        )
        experiment.default_cell = ALL_CELLS[0]
        experiment.predictor = SimpleNamespace(predict=mock.Mock())
        experiment.executor = SimpleNamespace(submit=lambda *args, **kwargs: object())
        experiment.round_index = 0
        captures: list[str] = []

        def capture(cell, context, role):
            captures.append(role)
            return (
                {
                    "timestamp_ns": 0,
                    "cell_id": cell.cell_id,
                    "exposure_us_model": cell.exposure_ms * 1000.0,
                    "actual_gain": "",
                    "gain": cell.gain,
                    "camera_parameter_ms": 1.0,
                    "mde_inference_ms": 1.0,
                },
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1), dtype=np.float32),
            )

        def await_initial(jobs):
            jobs[0][0].update(q=0.31, std=0.01, camera_bias=0.30)

        def await_probe(row, future):
            row.update(q=0.41, std=0.01, camera_bias=0.40)

        experiment._capture = capture
        experiment._save_capture = mock.Mock()
        experiment._await_jobs = await_initial
        experiment._await_probe_result = await_probe
        experiment._defer_for_transition = mock.Mock(return_value=False)
        experiment._probe_fits_budget = mock.Mock(return_value=True)
        experiment._finish_decision = mock.Mock()
        experiment._apply_control_cell = mock.Mock()
        controller.invoke_offload = mock.Mock()

        with redirect_stdout(StringIO()):
            experiment.run_round()

        self.assertEqual(captures.count("initial"), 1)
        self.assertEqual(captures.count("probe"), 1)

    def test_minimum_initial_probes_counts_distinct_current_scene_cells(self) -> None:
        unchanged_default = self.controller()
        unchanged_default.observe(
            self.context, ALL_CELLS[0], score(0.1, 0.02)
        )
        self.assertEqual(
            unchanged_default.minimum_initial_probes_remaining(self.context),
            0,
        )

        controller = self.controller(min_initial_probes=4)
        controller.observe(self.context, ALL_CELLS[0], score(0.1, 0.02))
        self.assertEqual(
            controller.minimum_initial_probes_remaining(self.context),
            4,
        )

        for cell in ALL_CELLS[1:4]:
            controller.observe(self.context, cell, score(0.1, 0.02))
        self.assertEqual(
            controller.minimum_initial_probes_remaining(self.context),
            1,
        )

        # Re-observing a cell does not satisfy distinct-cell exploration.
        controller.observe(self.context, ALL_CELLS[1], score(0.1, 0.02))
        self.assertEqual(
            controller.minimum_initial_probes_remaining(self.context),
            1,
        )
        controller.observe(self.context, ALL_CELLS[4], score(0.1, 0.02))
        self.assertEqual(
            controller.minimum_initial_probes_remaining(self.context),
            0,
        )

    def test_scene_change_inflates_variance_without_reordering_means(self) -> None:
        controller = self.controller(
            min_initial_probes=4,
            scene_change_z_threshold=1.0,
            scene_change_confirmations=2,
            scene_variance_inflation=4.0,
            scene_reset_variance=0.01,
        )
        first, second, untouched = ALL_CELLS[:3]
        controller.observe(self.context, first, score(0.10, 0.01))
        controller.observe(self.context, second, score(0.20, 0.01))
        controller.observe(self.context, untouched, score(0.15, 0.01))

        controller.observe(self.context, first, score(0.40, 0.01))
        state = controller.context_states[self.context.table_key]
        untouched_mean_before = state.cells[untouched.cell_id].belief_mean
        update = controller.observe(self.context, second, score(0.50, 0.01))

        self.assertTrue(update.scene_change_detected)
        self.assertEqual(state.scene_epoch, 1)
        self.assertEqual(state.scene_change_streak, 0)
        self.assertEqual(
            state.cells[untouched.cell_id].belief_mean,
            untouched_mean_before,
        )
        self.assertAlmostEqual(
            state.cells[untouched.cell_id].belief_variance,
            0.01,
        )
        self.assertEqual(state.cells[untouched.cell_id].last_scene_epoch, 0)
        self.assertEqual(state.cells[second.cell_id].last_scene_epoch, 1)
        self.assertEqual(
            controller.minimum_initial_probes_remaining(self.context),
            4,
        )
        self.assertEqual(controller.posterior_best(self.context), untouched)
        next_cell, probability, improvement = controller.select_next_probe_cell(
            self.context,
            untouched,
            {first.cell_id, second.cell_id},
        )
        self.assertEqual(next_cell, untouched)
        self.assertEqual(probability, 0.0)
        self.assertEqual(improvement, 0.0)

    def test_use_rejects_a_best_not_observed_in_current_scene(self) -> None:
        controller = self.controller()
        controller.observe(self.context, ALL_CELLS[-1], score(0.3, 0.01))
        with self.assertRaises(RuntimeError):
            controller.complete_probe(
                self.context,
                action="use",
                reason="invalid_unobserved_use",
            )

    def test_local_shortlist_uses_one_acceptable_challenger(self) -> None:
        controller = self.controller()
        current, challenger = ALL_CELLS[:2]
        controller.observe(self.context, current, score(0.18, 0.01))
        controller.observe(self.context, challenger, score(0.05, 0.01))

        selected, probability = controller.select_local_challenger(
            self.context, current
        )
        self.assertEqual(selected, challenger)
        self.assertGreaterEqual(probability, controller.good_enough_probability)
        self.assertEqual(
            controller.acceptable_cell(
                self.context, [current, challenger]
            ),
            challenger,
        )

    def test_bridge_cell_reuses_best_cell_shared_by_contexts(self) -> None:
        controller = self.controller()
        other_context = ContextKey(1, 0)
        bridge = ALL_CELLS[3]
        controller.observe(self.context, bridge, score(0.04, 0.01))
        controller.observe(other_context, bridge, score(0.06, 0.01))

        self.assertEqual(
            controller.bridge_cell(
                (self.context, other_context), ALL_CELLS[-1]
            ),
            bridge,
        )

    def test_late_observation_becomes_cached_context_cell(self) -> None:
        controller = self.controller()
        observed = ALL_CELLS[3]
        controller.observe(self.context, observed, score(0.05, 0.01))
        self.assertEqual(
            controller.cell_for_context(self.context, ALL_CELLS[-1]),
            observed,
        )

    def test_unacceptable_frame_probes_instead_of_direct_offload(self) -> None:
        controller = self.controller()
        decision = controller.decide(ALL_CELLS[0], score(0.18, 0.02))
        self.assertEqual(decision.action, "probe")


if __name__ == "__main__":
    unittest.main()
