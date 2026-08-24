import math
import unittest

from ati_mde_control.config import SafetyPolicy
from ati_mde_control.nelder_mead_policy import (
    ContextualRiskNelderMeadPolicy,
    RiskNelderMeadConfig,
)
from hardware.utils import ContextKey, QScore, SensorCell


def score(q: float, mu: float = 0.1, uncertainty: float = 0.2) -> QScore:
    return QScore(q=q, mu=mu, uncertainty=uncertainty)


class ContextualRiskNelderMeadPolicyTest(unittest.TestCase):
    def make_policy(self, safety=None, default=SensorCell(16, 64)):
        return ContextualRiskNelderMeadPolicy(
            RiskNelderMeadConfig(restart_frames=30, simplex_tolerance=0.0),
            safety or SafetyPolicy(),
            default,
        )

    def test_every_nelder_mead_proposal_is_in_context_safety_envelope(self) -> None:
        safety = SafetyPolicy(
            max_exposure_ms_by_motion=(8, 32, 32, 32, 32),
            allowed_gains_by_light=(
                (16, 32),
                (16, 32, 64, 128),
                (16, 32, 64, 128),
            ),
            disabled_cells_by_context={
                "0,0": frozenset({SensorCell(8, 32).cell_id})
            },
        )
        policy = self.make_policy(safety, SensorCell(32, 128))
        context = ContextKey(0, 0)
        safe = safety.safe_cells(context)
        for _ in range(25):
            cell = policy.next_cell(context)
            self.assertIn(cell, safe)
            q = abs(cell.exposure_ms - 4) / 32 + abs(cell.gain - 16) / 128
            self.assertTrue(policy.observe(context, cell, score(q)))

    def test_objective_is_exact_q_and_lower_q_becomes_best(self) -> None:
        policy = self.make_policy()
        context = ContextKey(0, 0)
        observed = []
        for q in (0.8, 0.2, 0.6):
            cell = policy.next_cell(context)
            observed.append((cell, q))
            policy.observe(context, cell, score(q, mu=99.0, uncertainty=50.0))
        expected_cell, expected_q = min(observed, key=lambda item: item[1])
        self.assertEqual(policy.best_cell(context), expected_cell)
        self.assertAlmostEqual(policy.best_risk(context), expected_q)

    def test_mu_and_aleatoric_uncertainty_do_not_replace_q_objective(self) -> None:
        policy = self.make_policy()
        context = ContextKey(0, 0)
        first = policy.next_cell(context)
        policy.observe(context, first, score(0.1, mu=100.0, uncertainty=100.0))
        second = policy.next_cell(context)
        policy.observe(context, second, score(0.9, mu=-100.0, uncertainty=0.0))
        self.assertEqual(policy.best_cell(context), first)
        self.assertEqual(policy.best_risk(context), 0.1)

    def test_contexts_keep_independent_simplexes(self) -> None:
        policy = self.make_policy()
        first_context = ContextKey(0, 0)
        second_context = ContextKey(1, 2)
        first_cell = policy.next_cell(first_context)
        policy.observe(first_context, first_cell, score(0.3))
        second_cell = policy.next_cell(second_context)
        self.assertEqual(second_cell, SensorCell(16, 64))
        self.assertIsNone(policy.best_risk(second_context))
        self.assertEqual(policy.best_risk(first_context), 0.3)

    def test_non_finite_scores_are_rejected_without_advancing(self) -> None:
        for invalid in (
            QScore(math.nan, 0.1, 0.1),
            QScore(0.1, math.inf, 0.1),
            QScore(0.1, 0.1, None),
        ):
            policy = self.make_policy()
            context = ContextKey(0, 0)
            pending = policy.next_cell(context)
            self.assertFalse(policy.observe(context, pending, invalid))
            self.assertEqual(policy.next_cell(context), pending)
            self.assertEqual(policy.last_update_status, "non_finite_score")

    def test_requires_three_safe_cells_for_two_dimensional_simplex(self) -> None:
        disabled = frozenset(
            cell.cell_id
            for cell in SafetyPolicy().safe_cells(ContextKey(0, 0))
            if cell not in (SensorCell(4, 16), SensorCell(8, 16))
        )
        policy = self.make_policy(
            SafetyPolicy(disabled_cells_by_context={"0,0": disabled})
        )
        with self.assertRaisesRegex(RuntimeError, "at least three safe cells"):
            policy.next_cell(ContextKey(0, 0))

    def test_brightness_filter_projects_pending_cell_and_restarts_at_capture(self) -> None:
        policy = self.make_policy()
        context = ContextKey(0, 0)
        current = policy.next_cell(context)
        policy.observe(context, current, score(0.5))
        old_controller = policy._controllers[context]
        policy.configure_brightness(
            context,
            current,
            (current,),
            prefer_brighter=False,
            reset=True,
        )
        self.assertEqual(policy.next_cell(context), current)
        self.assertTrue(policy.observe(context, current, score(0.4)))
        self.assertIsNot(policy._controllers[context], old_controller)
        self.assertEqual(policy.best_risk(context), 0.4)

    def test_forced_brightness_recovery_overrides_pending_simplex_cell(self) -> None:
        policy = self.make_policy()
        context = ContextKey(0, 0)
        current = policy.next_cell(context)
        policy.observe(context, current, score(0.1))
        recovery = SensorCell(8, 64)
        policy.configure_brightness(
            context,
            current,
            (current, recovery),
            prefer_brighter=False,
            forced_cell=recovery,
        )
        self.assertEqual(policy.next_cell(context), recovery)
        self.assertEqual(policy.operation(context), "brightness_recovery")


if __name__ == "__main__":
    unittest.main()
