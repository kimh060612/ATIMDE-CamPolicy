import math
import unittest

from hardware.utils import ALL_CELLS, ContextKey, QScore
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

    def test_ei_tie_breaks_by_all_cells_order(self) -> None:
        controller = self.controller()
        best = ALL_CELLS[0]
        controller.observe(self.context, best, score(0.1, 0.02))
        next_cell, probability, improvement = controller.select_next_probe_cell(
            self.context, best, set()
        )
        self.assertEqual(next_cell, ALL_CELLS[1])
        self.assertGreater(probability, 0.0)
        self.assertGreater(improvement, 0.0)

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


if __name__ == "__main__":
    unittest.main()
