import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hardware.utils import ALL_CELLS, CELL_BY_ID, ContextKey, QScore
from orbbec_deterministic_probing_modelv1 import ModelV1Experiment
from policy.basic_policy import ATIMDECameraProbingController, SafetyPolicy


def score(mu: float, std: float = 0.01, q: float | None = None) -> QScore:
    return QScore(q=mu + std if q is None else q, mu=mu, uncertainty=std)


class SatisficingPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)

    def controller(self, **overrides) -> ATIMDECameraProbingController:
        arguments = {
            "accept_threshold": 0.11,
            "accept_probability": 0.90,
            "required_bad_frames": 2,
            "success_ema_alpha": 0.3,
            "challenger_cooldown_rounds": 5,
        }
        arguments.update(overrides)
        return ATIMDECameraProbingController(SafetyPolicy(), **arguments)

    def runtime(
        self,
        controller: ATIMDECameraProbingController,
        *,
        initial_scores: list[QScore],
        challenger_scores: list[QScore] | None = None,
    ) -> tuple[ModelV1Experiment, list[str]]:
        experiment = ModelV1Experiment.__new__(ModelV1Experiment)
        experiment.policy = controller
        experiment.context_provider = SimpleNamespace(
            get=lambda: self.context,
            is_stable=True,
            transition_contexts=lambda: (self.context,),
        )
        experiment.default_cell = CELL_BY_ID["E04_G016"]
        experiment.predictor = SimpleNamespace(predict=mock.Mock())
        experiment.executor = SimpleNamespace(submit=lambda *args, **kwargs: object())
        experiment.round_index = 0
        captures: list[str] = []
        pending_initial = list(initial_scores)
        pending_challengers = list(challenger_scores or [])

        def capture(cell, context, role):
            captures.append(role)
            return (
                {
                    "timestamp_ns": 0,
                    "capture_role": role,
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

        def store(row, observation):
            row.update(
                q=observation.q,
                std=observation.uncertainty,
                camera_bias=observation.mu,
            )

        def await_result(row, future):
            scores = (
                pending_initial
                if row["capture_role"] == "initial"
                else pending_challengers
            )
            store(row, scores.pop(0))

        experiment._capture = capture
        experiment._save_capture = mock.Mock()
        experiment._await_probe_result = await_result
        experiment._defer_for_transition = mock.Mock(return_value=False)
        experiment._probe_fits_budget = mock.Mock(return_value=True)
        experiment._finish_decision = mock.Mock()
        experiment._apply_control_cell = mock.Mock()
        controller.invoke_offload = mock.Mock()
        return experiment, captures

    @staticmethod
    def run_runtime(experiment: ModelV1Experiment, rounds: int = 1) -> None:
        with redirect_stdout(StringIO()):
            for _ in range(rounds):
                experiment.run_round()

    def test_success_statistics_are_independent_across_contexts(self) -> None:
        controller = self.controller(success_ema_alpha=1.0)
        cell = ALL_CELLS[0]
        other_context = ContextKey(1, 0)

        controller.observe(self.context, cell, score(0.01))
        controller.success_score(other_context, cell)

        first = controller.context_states[self.context.table_key].cells[cell.cell_id]
        second = controller.context_states[other_context.table_key].cells[cell.cell_id]
        self.assertGreater(first.success_score, 0.99)
        self.assertEqual(first.observation_count, 1)
        self.assertEqual(second.success_score, 0.5)
        self.assertEqual(second.observation_count, 0)

    def test_probability_good_uses_current_mu_and_std(self) -> None:
        controller = self.controller()

        self.assertAlmostEqual(controller.probability_good(score(0.11, 0.01)), 0.5)
        self.assertAlmostEqual(
            controller.probability_good(score(0.10, 0.01)),
            0.8413447460685429,
        )

    def test_acceptable_current_frame_does_not_probe(self) -> None:
        controller = self.controller()
        experiment, captures = self.runtime(
            controller, initial_scores=[score(0.01)]
        )

        self.run_runtime(experiment)

        self.assertEqual(captures, ["initial"])
        self.assertEqual(controller.consecutive_bad_frames(self.context), 0)

    def test_one_bad_frame_does_not_probe_with_two_frame_hysteresis(self) -> None:
        controller = self.controller(required_bad_frames=2)
        experiment, captures = self.runtime(
            controller, initial_scores=[score(0.30)]
        )

        self.run_runtime(experiment)

        self.assertEqual(captures, ["initial"])
        self.assertEqual(controller.consecutive_bad_frames(self.context), 1)

    def test_consecutive_bad_frames_capture_exactly_one_challenger(self) -> None:
        controller = self.controller(required_bad_frames=2)
        experiment, captures = self.runtime(
            controller,
            initial_scores=[score(0.30), score(0.30)],
            challenger_scores=[score(0.30)],
        )

        self.run_runtime(experiment, rounds=2)

        self.assertEqual(captures, ["initial", "initial", "probe"])

    def test_challenger_switches_when_its_probability_is_acceptable(self) -> None:
        controller = self.controller(required_bad_frames=1)
        experiment, captures = self.runtime(
            controller,
            initial_scores=[score(0.30)],
            challenger_scores=[score(0.01)],
        )

        self.run_runtime(experiment)

        state = controller.context_states[self.context.table_key]
        self.assertEqual(captures.count("probe"), 1)
        self.assertEqual(state.active_cell_id, CELL_BY_ID["E04_G128"].cell_id)
        self.assertEqual(state.consecutive_bad_frames, 0)

    def test_lower_challenger_q_does_not_switch_when_unacceptable(self) -> None:
        controller = self.controller(required_bad_frames=1)
        experiment, _ = self.runtime(
            controller,
            initial_scores=[score(0.30, q=0.50)],
            challenger_scores=[score(0.30, q=0.01)],
        )

        self.run_runtime(experiment)

        state = controller.context_states[self.context.table_key]
        self.assertEqual(state.active_cell_id, CELL_BY_ID["E04_G016"].cell_id)

    def test_rejected_challenger_enters_cooldown(self) -> None:
        controller = self.controller(
            required_bad_frames=1, challenger_cooldown_rounds=5
        )
        experiment, _ = self.runtime(
            controller,
            initial_scores=[score(0.30)],
            challenger_scores=[score(0.30)],
        )

        self.run_runtime(experiment)

        failed = CELL_BY_ID["E04_G128"]
        self.assertEqual(controller.challenger_cooldown(self.context, failed), 5)
        self.assertNotEqual(
            controller.select_challenger(self.context, ALL_CELLS[0]), failed
        )

    def test_challenger_uses_highest_context_specific_success_score(self) -> None:
        controller = self.controller(success_ema_alpha=1.0)
        current, low, high = ALL_CELLS[:3]
        other_context = ContextKey(1, 0)
        controller.observe(self.context, low, score(0.30))
        controller.observe(self.context, high, score(0.01))
        controller.observe(other_context, low, score(0.01))
        controller.observe(other_context, high, score(0.30))

        self.assertEqual(controller.select_challenger(self.context, current), high)
        self.assertEqual(controller.select_challenger(other_context, current), low)

    def test_equal_success_scores_use_grid_diverse_order(self) -> None:
        controller = self.controller()
        current = CELL_BY_ID["E04_G016"]

        first = controller.select_challenger(self.context, current)
        assert first is not None
        controller.context_states[self.context.table_key].cells[
            first.cell_id
        ].cooldown = 1
        second = controller.select_challenger(self.context, current)

        self.assertEqual(first.cell_id, "E04_G128")
        self.assertEqual(second, CELL_BY_ID["E32_G016"])

    def test_runtime_never_captures_more_than_one_challenger(self) -> None:
        controller = self.controller(required_bad_frames=1)
        experiment, captures = self.runtime(
            controller,
            initial_scores=[score(0.30)],
            challenger_scores=[score(0.30)],
        )

        self.run_runtime(experiment)

        self.assertEqual(captures.count("initial"), 1)
        self.assertEqual(captures.count("probe"), 1)

    def test_context_changes_do_not_transfer_active_cells_or_statistics(self) -> None:
        controller = self.controller(success_ema_alpha=1.0)
        first_context = ContextKey(0, 0)
        second_context = ContextKey(1, 0)
        first_cell = CELL_BY_ID["E04_G016"]
        second_cell = CELL_BY_ID["E32_G128"]

        controller.cell_for_context(first_context, first_cell)
        controller.observe(first_context, first_cell, score(0.01))
        controller.cell_for_context(second_context, second_cell)

        first_state = controller.context_states[first_context.table_key]
        second_state = controller.context_states[second_context.table_key]
        self.assertEqual(first_state.active_cell_id, first_cell.cell_id)
        self.assertEqual(second_state.active_cell_id, second_cell.cell_id)
        self.assertEqual(
            second_state.cells[first_cell.cell_id].observation_count, 0
        )
        self.assertEqual(
            controller.cell_for_context(first_context, second_cell), first_cell
        )


if __name__ == "__main__":
    unittest.main()
