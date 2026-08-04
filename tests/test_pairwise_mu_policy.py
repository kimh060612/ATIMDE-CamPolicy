import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hardware.utils import ALL_CELLS, CELL_BY_ID, ContextKey, EdgeStats, QScore
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
            "challenger_cooldown_rounds": 5,
            "pair_uncertainty_weight": 0.25,
            "reference_pair_std": 0.03,
            "edge_ema_alpha": 0.3,
            "offload_uncertainty_weight": 1.0,
            "offload_threshold": 0.15,
            "ambiguous_offload_threshold": 0.11,
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
        experiment.predictor = SimpleNamespace(
            predict=mock.Mock(),
            predict_batch=mock.Mock(),
        )
        experiment.executor = SimpleNamespace(submit=lambda *args, **kwargs: object())
        experiment.round_index = 0
        captures: list[tuple[str, str, bool]] = []
        rows: list[dict] = []
        applied: list[str] = []
        pending_initial = list(initial_scores)
        pending_challengers = list(challenger_scores or [])
        last_initial: list[QScore] = []

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
            if row["capture_role"] == "initial":
                last_initial[:] = [observation]
            row.update(
                q=observation.q,
                std=observation.uncertainty,
                camera_bias=observation.mu,
            )

        def await_pair_result(pair_rows, future):
            observations = [last_initial[0], pending_challengers.pop(0)]
            for row, observation in zip(pair_rows, observations):
                row.update(
                    q=observation.q,
                    std=observation.uncertainty,
                    camera_bias=observation.mu,
                    mde_batch_size=2,
                )

        def apply_control(cell, *, reason):
            experiment.last_applied_cell = cell
            applied.append(cell.cell_id)

        experiment._capture = capture
        experiment._save_capture = mock.Mock()
        experiment._await_probe_result = await_result
        experiment._await_pair_result = await_pair_result
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

    def test_transition_frame_never_updates_active_cell_or_edges(self) -> None:
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
        self.assertEqual(state.edges, {})
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

    def test_clearly_losing_challenger_stays_inactive_and_enters_cooldown(
        self,
    ) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.cell_for_context(self.context, current)

        pair = controller.resolve_challenger(
            self.context, current, 0.30, 0.01, challenger, 0.34, 0.01
        )

        self.assertEqual(pair.status, "current_won")
        self.assertEqual(pair.selected_cell, current)
        self.assertEqual(controller.active_cell_id(self.context), current.cell_id)
        self.assertEqual(controller.challenger_cooldown(self.context, challenger), 5)

    def test_challenger_switches_only_above_effective_margin(self) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.cell_for_context(self.context, current)

        pair = controller.resolve_challenger(
            self.context, current, 0.30, 0.01, challenger, 0.28, 0.01
        )

        self.assertGreater(pair.delta_mu, pair.effective_margin)
        self.assertEqual(pair.status, "challenger_won")
        self.assertEqual(pair.selected_cell, challenger)
        self.assertEqual(controller.active_cell_id(self.context), challenger.cell_id)
        edge = controller.context_states[self.context.table_key].edges[
            (current.cell_id, challenger.cell_id)
        ]
        self.assertEqual(edge.challenger_win_count, 1)

    def test_large_std_increases_margin_and_prevents_switch(self) -> None:
        controller = self.controller(switch_margin=0.01)
        current, challenger = ALL_CELLS[:2]
        controller.cell_for_context(self.context, current)

        pair = controller.resolve_challenger(
            self.context, current, 0.30, 0.20, challenger, 0.27, 0.20
        )

        self.assertEqual(pair.status, "ambiguous")
        self.assertGreater(pair.effective_margin, pair.delta_mu)
        self.assertEqual(controller.active_cell_id(self.context), current.cell_id)

    def test_q_is_not_used_for_cell_ranking_or_pair_resolution(self) -> None:
        controller = self.controller()
        current = ALL_CELLS[0]
        controller.cell_for_context(self.context, current)
        experiment, _, rows, _ = self.runtime(
            controller,
            initial_scores=[score(0.30, std=0.01, q=-999.0)],
            challenger_scores=[score(0.20, std=0.01, q=999.0)],
        )

        self.run_runtime(experiment)

        self.assertEqual(controller.active_cell_id(self.context), rows[1]["cell_id"])

    def test_ambiguous_pair_keeps_current_without_cooldown(self) -> None:
        controller = self.controller()
        current, challenger = ALL_CELLS[:2]
        controller.cell_for_context(self.context, current)

        pair = controller.resolve_challenger(
            self.context, current, 0.30, 0.01, challenger, 0.295, 0.01
        )

        self.assertEqual(pair.status, "ambiguous")
        self.assertEqual(pair.selected_cell, current)
        self.assertEqual(controller.challenger_cooldown(self.context, challenger), 0)
        edge = controller.context_states[self.context.table_key].edges[
            (current.cell_id, challenger.cell_id)
        ]
        self.assertEqual(edge.comparison_count, 1)
        self.assertEqual(edge.ambiguous_count, 1)

    def test_low_confidence_pair_has_smaller_edge_ema_update(self) -> None:
        current, challenger = ALL_CELLS[:2]
        high_confidence = self.controller()
        low_confidence = self.controller()
        high_confidence.cell_for_context(self.context, current)
        low_confidence.cell_for_context(self.context, current)

        high = high_confidence.resolve_challenger(
            self.context, current, 0.30, 0.001, challenger, 0.20, 0.001
        )
        low = low_confidence.resolve_challenger(
            self.context, current, 0.30, 0.10, challenger, 0.20, 0.10
        )

        self.assertGreater(high.confidence, low.confidence)
        self.assertGreater(high.edge_ema_after, low.edge_ema_after)

    def test_offload_uses_selected_frame_mu_and_std(self) -> None:
        controller = self.controller(
            pair_uncertainty_weight=0.0,
            offload_threshold=0.15,
        )
        experiment, _, rows, _ = self.runtime(
            controller,
            initial_scores=[score(0.30, std=0.50)],
            challenger_scores=[score(0.10, std=0.01)],
        )

        self.run_runtime(experiment)

        self.assertEqual(rows[0]["pair_status"], "challenger_won")
        self.assertAlmostEqual(rows[0]["selected_mu"], 0.10)
        self.assertAlmostEqual(rows[0]["selected_std"], 0.01)
        self.assertAlmostEqual(rows[0]["offload_risk"], 0.11)
        controller.invoke_offload.assert_not_called()

    def test_offload_does_not_change_committed_cell(self) -> None:
        controller = self.controller(offload_threshold=0.20)
        current = ALL_CELLS[0]
        controller.cell_for_context(self.context, current)
        experiment, _, rows, _ = self.runtime(
            controller,
            initial_scores=[score(0.30, std=0.05)],
            challenger_scores=[score(0.295, std=0.05)],
        )

        self.run_runtime(experiment)

        self.assertEqual(rows[0]["pair_status"], "ambiguous")
        self.assertEqual(controller.active_cell_id(self.context), current.cell_id)
        controller.invoke_offload.assert_called_once()

    def test_contexts_keep_independent_active_cells_and_edge_tables(self) -> None:
        controller = self.controller()
        other_context = ContextKey(1, 0)
        first, second, challenger = ALL_CELLS[:3]
        controller.cell_for_context(self.context, first)
        controller.cell_for_context(other_context, second)
        controller.resolve_challenger(
            self.context, first, 0.30, 0.01, challenger, 0.295, 0.01
        )
        controller.resolve_challenger(
            other_context, second, 0.30, 0.01, challenger, 0.295, 0.01
        )

        self.assertEqual(controller.active_cell_id(self.context), first.cell_id)
        self.assertEqual(controller.active_cell_id(other_context), second.cell_id)
        first_edges = controller.context_states[self.context.table_key].edges
        second_edges = controller.context_states[other_context.table_key].edges
        self.assertIn((first.cell_id, challenger.cell_id), first_edges)
        self.assertNotIn((first.cell_id, challenger.cell_id), second_edges)

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
        self.assertEqual(initial_state.edges, {})
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
        self.assertEqual(probe_state.cells[ALL_CELLS[-1].cell_id].cooldown, 3)
        self.assertEqual(probe_state.cells[ALL_CELLS[1].cell_id].cooldown, 0)
        self.assertEqual(probe_state.active_cell_id, current.cell_id)
        self.assertEqual(probe_state.edges, {})
        self.assertTrue(all(row["pair_status"] == "invalid_pair" for row in rows))
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

    def test_challenger_with_best_edge_improvement_is_selected(self) -> None:
        controller = self.controller()
        current = ALL_CELLS[0]
        target = ALL_CELLS[-1]
        controller.cell_for_context(self.context, current)
        state = controller.context_states[self.context.table_key]
        for cell in ALL_CELLS[1:]:
            state.edges[(current.cell_id, cell.cell_id)] = EdgeStats(
                ema_improvement=0.10 if cell == target else -0.30,
                comparison_count=1,
            )

        self.assertEqual(controller.select_challenger(self.context, current), target)

    def test_runtime_captures_at_most_one_challenger_per_round(self) -> None:
        controller = self.controller()
        experiment, captures, rows, _ = self.runtime(
            controller,
            initial_scores=[score(0.30)],
            challenger_scores=[score(0.30)],
        )

        self.run_runtime(experiment)

        roles = [role for role, _, _ in captures]
        self.assertEqual(roles.count("initial"), 1)
        self.assertEqual(roles.count("probe"), 1)
        self.assertTrue(all(row["mde_batch_size"] == 2 for row in rows))


if __name__ == "__main__":
    unittest.main()
