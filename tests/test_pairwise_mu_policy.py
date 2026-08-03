import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hardware.utils import ALL_CELLS, CELL_BY_ID, ContextKey, QScore
from orbbec_deterministic_probing_modelv1 import ModelV1Experiment
from policy.basic_policy import ATIMDECameraProbingController, SafetyPolicy


def score(mu: float, *, std: float = 0.01, q: float | None = None) -> QScore:
    return QScore(q=mu + std if q is None else q, mu=mu, uncertainty=std)


class ScriptedContextProvider:
    def __init__(self, states: list[tuple[ContextKey, bool]]) -> None:
        self.states = states
        self.index = 0
        self.current, self.stable = states[0]

    def get(self) -> ContextKey:
        if self.index < len(self.states):
            self.current, self.stable = self.states[self.index]
            self.index += 1
        return self.current

    @property
    def is_stable(self) -> bool:
        return self.stable


class PairwiseMuPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)

    def controller(self, **overrides) -> ATIMDECameraProbingController:
        arguments = {
            "probe_trigger_threshold": 0.11,
            "switch_margin": 0.01,
            "mu_ema_alpha": 0.3,
            "challenger_cooldown_rounds": 5,
        }
        arguments.update(overrides)
        return ATIMDECameraProbingController(SafetyPolicy(), **arguments)

    def runtime(
        self,
        controller: ATIMDECameraProbingController,
        *,
        contexts: list[tuple[ContextKey, bool]] | None = None,
        initial_scores: list[QScore],
        challenger_scores: list[QScore] | None = None,
        last_applied_cell=None,
    ):
        experiment = ModelV1Experiment.__new__(ModelV1Experiment)
        experiment.policy = controller
        experiment.context_provider = ScriptedContextProvider(
            contexts or [(self.context, True)]
        )
        experiment.default_cell = CELL_BY_ID["E04_G016"]
        experiment.last_applied_cell = last_applied_cell or experiment.default_cell
        experiment.camera = SimpleNamespace(exposure_value_per_ms=1000.0)
        experiment.predictor = SimpleNamespace(predict=mock.Mock())
        experiment.executor = SimpleNamespace(submit=lambda *args, **kwargs: object())
        experiment.round_index = 0
        captures: list[tuple[str, str, bool]] = []
        rows: list[dict] = []
        applied: list[str] = []
        pending_initial = list(initial_scores)
        pending_challengers = list(challenger_scores or [])

        def capture(cell, context, role, *, apply_cell=True):
            captures.append((role, cell.cell_id, apply_cell))
            if apply_cell:
                experiment.last_applied_cell = cell
                applied.append(cell.cell_id)
            row = {
                "timestamp_ns": 0,
                "capture_role": role,
                "cell_id": cell.cell_id,
                "exposure_us_model": cell.exposure_ms * 1000.0,
                "actual_gain": "",
                "gain": cell.gain,
                "camera_parameter_ms": 1.0,
                "mde_inference_ms": 1.0,
                "selection_source": "",
            }
            rows.append(row)
            return (
                row,
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1), dtype=np.float32),
            )

        def await_result(row, future):
            scores = (
                pending_initial
                if row["capture_role"] == "initial"
                else pending_challengers
            )
            observation = scores.pop(0)
            row.update(
                q=observation.q,
                std=observation.uncertainty,
                camera_bias=observation.mu,
            )

        def apply_control(cell, *, reason):
            experiment.last_applied_cell = cell
            applied.append(cell.cell_id)

        experiment._capture = capture
        experiment._save_capture = mock.Mock()
        experiment._await_probe_result = await_result
        experiment._probe_fits_budget = mock.Mock(return_value=True)
        experiment._finish_decision = mock.Mock()
        experiment._apply_control_cell = apply_control
        controller.invoke_offload = mock.Mock()
        return experiment, captures, rows, applied

    @staticmethod
    def run_runtime(experiment: ModelV1Experiment, rounds: int = 1) -> None:
        with redirect_stdout(StringIO()):
            for _ in range(rounds):
                experiment.run_round()

    def test_transition_frame_never_updates_active_cell_or_ema(self) -> None:
        controller = self.controller()
        active = CELL_BY_ID["E04_G016"]
        held = CELL_BY_ID["E16_G064"]
        controller.cell_for_context(self.context, active)
        controller.context_states[self.context.table_key].cells[
            ALL_CELLS[-1].cell_id
        ].cooldown = 3
        experiment, _, rows, _ = self.runtime(
            controller,
            contexts=[(self.context, False)],
            initial_scores=[score(0.30)],
            last_applied_cell=held,
        )

        self.run_runtime(experiment)

        state = controller.context_states[self.context.table_key]
        self.assertEqual(state.active_cell_id, active.cell_id)
        self.assertTrue(
            all(cell.observation_count == 0 for cell in state.cells.values())
        )
        self.assertEqual(state.cells[ALL_CELLS[-1].cell_id].cooldown, 3)
        self.assertEqual(rows[0]["selection_source"], "transition_hold")
        self.assertEqual(rows[0]["transition_only"], 1)

    def test_unstable_context_uses_last_applied_cell_without_default_apply(
        self,
    ) -> None:
        controller = self.controller()
        held = CELL_BY_ID["E16_G064"]
        experiment, captures, _, applied = self.runtime(
            controller,
            contexts=[(self.context, False)],
            initial_scores=[score(0.30)],
            last_applied_cell=held,
        )

        self.run_runtime(experiment)

        self.assertEqual(captures, [("initial", held.cell_id, False)])
        self.assertEqual(applied, [])
        self.assertEqual(experiment.last_applied_cell, held)

    def test_record_current_result_cannot_change_active_cell(self) -> None:
        controller = self.controller()
        active, other = ALL_CELLS[:2]
        controller.cell_for_context(self.context, active)

        with self.assertRaises(RuntimeError):
            controller.record_current_result(self.context, other, 0.30)

        self.assertEqual(controller.active_cell_id(self.context), active.cell_id)

    def test_rejected_challenger_cannot_become_active(self) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.cell_for_context(self.context, current)

        selected = controller.resolve_challenger(
            self.context, current, 0.30, challenger, 0.295
        )

        self.assertEqual(selected, current)
        self.assertEqual(controller.active_cell_id(self.context), current.cell_id)
        self.assertEqual(controller.challenger_cooldown(self.context, challenger), 5)

    def test_challenger_switches_only_when_raw_mu_wins_by_margin(self) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.cell_for_context(self.context, current)

        selected = controller.resolve_challenger(
            self.context, current, 0.30, challenger, 0.28
        )

        self.assertEqual(selected, challenger)
        self.assertEqual(controller.active_cell_id(self.context), challenger.cell_id)

    def test_std_q_and_absolute_probability_do_not_affect_switching(self) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.cell_for_context(self.context, current)
        experiment, _, _, _ = self.runtime(
            controller,
            initial_scores=[score(0.30, std=0.50, q=99.0)],
            challenger_scores=[score(0.295, std=0.0001, q=-99.0)],
        )

        self.run_runtime(experiment)

        self.assertEqual(controller.active_cell_id(self.context), current.cell_id)
        self.assertFalse(hasattr(controller, "probability_good"))

    def test_contexts_keep_independent_active_cells_and_ema_tables(self) -> None:
        controller = self.controller(mu_ema_alpha=1.0)
        other_context = ContextKey(1, 0)
        first, second = ALL_CELLS[:2]
        controller.cell_for_context(self.context, first)
        controller.cell_for_context(other_context, second)
        controller.observe(self.context, first, 0.10, round_index=1)
        controller.observe(other_context, first, 0.30, round_index=2)

        self.assertEqual(controller.active_cell_id(self.context), first.cell_id)
        self.assertEqual(controller.active_cell_id(other_context), second.cell_id)
        self.assertEqual(controller.ema_mu(self.context, first), 0.10)
        self.assertEqual(controller.ema_mu(other_context, first), 0.30)

    def test_stale_initial_and_probe_results_do_not_modify_policy_state(self) -> None:
        other_context = ContextKey(1, 0)

        initial_controller = self.controller()
        current = ALL_CELLS[0]
        initial_controller.cell_for_context(self.context, current)
        initial_controller.context_states[self.context.table_key].cells[
            ALL_CELLS[-1].cell_id
        ].cooldown = 3
        initial_experiment, _, _, _ = self.runtime(
            initial_controller,
            contexts=[(self.context, True), (other_context, True)],
            initial_scores=[score(0.30)],
        )
        self.run_runtime(initial_experiment)
        initial_state = initial_controller.context_states[self.context.table_key]
        self.assertTrue(
            all(cell.observation_count == 0 for cell in initial_state.cells.values())
        )
        self.assertEqual(initial_state.cells[ALL_CELLS[-1].cell_id].cooldown, 3)

        probe_controller = self.controller()
        probe_controller.cell_for_context(self.context, current)
        probe_controller.context_states[self.context.table_key].cells[
            ALL_CELLS[-1].cell_id
        ].cooldown = 3
        probe_experiment, _, rows, _ = self.runtime(
            probe_controller,
            contexts=[
                (self.context, True),
                (self.context, True),
                (self.context, True),
                (other_context, True),
            ],
            initial_scores=[score(0.30)],
            challenger_scores=[score(0.10)],
        )
        self.run_runtime(probe_experiment)
        probe_state = probe_controller.context_states[self.context.table_key]
        self.assertTrue(
            all(cell.observation_count == 0 for cell in probe_state.cells.values())
        )
        self.assertEqual(probe_state.cells[ALL_CELLS[-1].cell_id].cooldown, 3)
        self.assertEqual(probe_state.cells[ALL_CELLS[1].cell_id].cooldown, 0)
        self.assertEqual(probe_state.active_cell_id, current.cell_id)
        self.assertTrue(
            all(row["selection_source"] == "stale_pair_discarded" for row in rows)
        )

    def test_unobserved_challengers_follow_grid_diverse_order(self) -> None:
        controller = self.controller()
        current = CELL_BY_ID["E04_G016"]

        first = controller.select_challenger(self.context, current)
        assert first is not None
        cells = controller.context_states[self.context.table_key].cells
        cells[first.cell_id].cooldown = 1
        second = controller.select_challenger(self.context, current)

        self.assertEqual(first, CELL_BY_ID["E04_G128"])
        self.assertEqual(second, CELL_BY_ID["E32_G016"])

    def test_observed_challenger_with_lowest_ema_mu_is_selected(self) -> None:
        controller = self.controller(mu_ema_alpha=1.0)
        current = ALL_CELLS[0]
        target = ALL_CELLS[-1]
        for cell in ALL_CELLS[1:]:
            controller.observe(
                self.context,
                cell,
                0.10 if cell == target else 0.30,
                round_index=0,
            )

        self.assertEqual(controller.select_challenger(self.context, current), target)

    def test_runtime_captures_at_most_one_challenger_per_round(self) -> None:
        controller = self.controller()
        experiment, captures, _, _ = self.runtime(
            controller,
            initial_scores=[score(0.30)],
            challenger_scores=[score(0.30)],
        )

        self.run_runtime(experiment)

        roles = [role for role, _, _ in captures]
        self.assertEqual(roles.count("initial"), 1)
        self.assertEqual(roles.count("probe"), 1)


if __name__ == "__main__":
    unittest.main()
