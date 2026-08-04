import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import ExperimentConfig, PolicyConfig, SafetyPolicy
from ati_mde_control.experiment import CameraControlExperiment
from ati_mde_control.logging import CaptureLogger
from ati_mde_control.pairwise_policy import PairwisePolicy
from hardware.utils import ContextKey, QScore, SensorCell


def score(mu: float, batch: int = 1) -> QScore:
    return QScore(mu + .01, .01, mu, {"mde_batch_size": batch, "mde_inference_ms": 1.0})


class Provider:
    def __init__(self, context=ContextKey(0, 0)) -> None:
        self.current = context
        self.is_stable = True

    def get(self):
        return self.current

    def close(self):
        pass


class Camera:
    exposure_value_per_ms = 1000.0

    def __init__(self, events):
        self.events = events
        self.frame_sequence = 0

    def apply_cell(self, cell):
        self.events.append(f"apply:{cell.cell_id}")
        return cell.exposure_ms * 1000, cell.exposure_ms * 1000, cell.gain

    def capture_rgbd(self):
        self.events.append("capture")
        self.frame_sequence += 1
        return np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32)

    def close(self):
        pass


class Predictor:
    device = "cpu"

    def __init__(
        self,
        events,
        provider=None,
        batch_results=None,
        single_mu=.05,
        change_on_batch=1,
    ):
        self.events = events
        self.provider = provider
        self.single_calls = 0
        self.batch_sizes = []
        self.batch_results = list(batch_results or [(.20, .10)])
        self.single_mu = single_mu
        self.change_on_batch = change_on_batch

    def predict(self, image, context, exposure_us, gain):
        self.events.append("predict_single")
        self.single_calls += 1
        return score(self.single_mu)

    def predict_batch(self, images, contexts, exposure_us_values, gains):
        self.events.append("predict_batch")
        self.batch_sizes.append(len(images))
        if self.provider is not None and len(self.batch_sizes) == self.change_on_batch:
            self.provider.current = ContextKey(1, 0)
        current_mu, challenger_mu = (
            self.batch_results.pop(0) if len(self.batch_results) > 1
            else self.batch_results[0]
        )
        return [score(current_mu, 2), score(challenger_mu, 2)]


class Logger:
    def __init__(self):
        self.rows = []

    def record(self, frame, score, values):
        row = {
            "round_index": frame.round_index,
            "capture_role": frame.role,
            "cell_id": frame.cell.cell_id,
            "output_delivered": int(frame.role == "initial"),
            **values,
        }
        self.rows.append(row)
        return row


class Evaluator:
    def evaluate_rows(self, rows):
        pass


def config(policy=None):
    return ExperimentConfig(
        checkpoint_path=Path("unused"),
        output_dir=Path("unused"),
        model_size="small",
        device="cpu",
        precision="fp32",
        local_files_only=True,
        q_uncertainty_weight=1.0,
        policy=policy or PolicyConfig(),
        safety_path=None,
        default_cell=SensorCell(16, 64),
        max_rounds=10,
        round_interval_ms=0,
        max_pair_capture_gap_ms=10_000,
        camera_parameter_warn_ms=1000,
        mde_inference_warn_ms=1000,
        control_decision_warn_ms=1000,
        evaluation_alignment="metric",
        min_depth_m=.2,
        max_depth_m=10,
        min_valid_depth_pixels=1,
    )


def experiment(
    change_context_during_batch=False,
    policy_config=None,
    batch_results=None,
    change_on_batch=1,
):
    events = []
    provider = Provider()
    cfg = config(policy_config)
    policy = PairwisePolicy(cfg.policy, SafetyPolicy())
    predictor = Predictor(
        events,
        provider if change_context_during_batch else None,
        batch_results,
        change_on_batch=change_on_batch,
    )
    logger = Logger()
    runtime = CameraControlExperiment(
        cfg,
        CaptureRunner(Camera(events), provider, cfg.max_pair_capture_gap_ms),
        predictor,
        policy,
        logger,
        Evaluator(),
    )
    return runtime, events, provider, predictor, policy, logger


class ExperimentTest(unittest.TestCase):
    def test_terminal_log_contains_control_state_not_performance(self) -> None:
        runtime, _, _, _, _, _ = experiment()
        output = StringIO()
        with redirect_stdout(output):
            runtime.run_round()
        line = output.getvalue()
        self.assertIn("active=E16_G064 E=16ms G=64", line)
        self.assertIn("search=exposure/negative", line)
        self.assertIn("decision=challenger_won/confirmation_pending", line)
        self.assertNotIn("abs_rel", line.lower())
        self.assertNotIn("absrel", line.lower())
        self.assertNotIn("a1=", line.lower())

    def test_no_inference_occurs_between_pair_captures(self) -> None:
        runtime, events, _, _, _, _ = experiment()
        runtime.run_round()
        captures = [index for index, event in enumerate(events) if event == "capture"]
        inference = events.index("predict_batch")
        self.assertEqual(len(captures), 2)
        self.assertGreater(inference, captures[1])
        self.assertNotIn("predict_single", events[captures[0] + 1 : captures[1]])
        self.assertNotIn("predict_batch", events[captures[0] + 1 : captures[1]])

    def test_pair_inference_is_one_batch_of_two(self) -> None:
        runtime, _, _, predictor, _, _ = experiment()
        runtime.run_round()
        self.assertEqual(predictor.batch_sizes, [2])
        self.assertEqual(predictor.single_calls, 0)

    def test_every_round_delivers_initial_frame(self) -> None:
        runtime, _, _, _, _, logger = experiment()
        result = runtime.run_round()
        self.assertTrue(result.output_delivered)
        self.assertEqual(sum(row["output_delivered"] for row in logger.rows), 1)
        self.assertEqual(logger.rows[0]["capture_role"], "initial")

    def test_recovery_round_is_current_only(self) -> None:
        runtime, events, _, predictor, policy, _ = experiment()
        state = policy.state(ContextKey(0, 0))
        state.force_current_only_rounds = 2
        runtime.run_round()
        self.assertEqual(events.count("capture"), 1)
        self.assertEqual(predictor.batch_sizes, [])
        self.assertEqual(predictor.single_calls, 1)
        self.assertEqual(state.force_current_only_rounds, 1)

    def test_valid_old_context_pair_updates_old_state(self) -> None:
        runtime, _, _, _, policy, _ = experiment(change_context_during_batch=True)
        result = runtime.run_round()
        old_state = policy.state(ContextKey(0, 0))
        self.assertEqual(result.decision.status.value, "challenger_won")
        self.assertEqual(old_state.active_cell_id, SensorCell(16, 64).cell_id)
        self.assertEqual(old_state.pending_switch_to_id, SensorCell(8, 64).cell_id)
        self.assertEqual(len(old_state.local_edges), 1)

    def test_old_context_winner_is_not_left_applied_in_new_context(self) -> None:
        runtime, events, _, _, policy, _ = experiment(change_context_during_batch=True)
        runtime.run_round()
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))
        self.assertEqual(policy.state(ContextKey(1, 0)).active_cell_id, SensorCell(16, 64).cell_id)
        self.assertEqual(events[-1], "apply:E16_G064")

    def test_confirmation_pending_restores_current_cell_physically(self) -> None:
        runtime, events, _, _, policy, logger = experiment()
        runtime.run_round()
        self.assertEqual(policy.state(ContextKey(0, 0)).active_cell_id, "E16_G064")
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))
        self.assertEqual(events[-1], "apply:E16_G064")
        self.assertEqual(logger.rows[0]["pair_status"], "challenger_won")
        self.assertEqual(logger.rows[0]["switch_event"], "confirmation_pending")
        self.assertEqual(logger.rows[0]["active_cell_after"], "E16_G064")
        self.assertEqual(logger.rows[0]["committed_switch"], 0)

    def test_old_context_confirmation_commits_only_old_state(self) -> None:
        runtime, events, _, _, policy, _ = experiment(
            change_context_during_batch=True, change_on_batch=2
        )
        runtime.run_round()
        runtime.run_round()
        self.assertEqual(policy.state(ContextKey(0, 0)).active_cell_id, "E08_G064")
        self.assertEqual(policy.state(ContextKey(1, 0)).active_cell_id, "E16_G064")
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))
        self.assertEqual(events[-1], "apply:E16_G064")

    def test_old_context_rollback_updates_old_state_but_applies_new_context(self) -> None:
        runtime, events, _, _, policy, _ = experiment(
            change_context_during_batch=True, change_on_batch=3
        )
        policy.committed_cell(ContextKey(1, 0), SensorCell(32, 64))
        for _ in range(5):
            runtime.run_round()
        self.assertEqual(policy.state(ContextKey(0, 0)).active_cell_id, "E16_G064")
        self.assertEqual(runtime.last_applied_cell, SensorCell(32, 64))
        self.assertEqual(events[-1], "apply:E32_G064")

    def test_challenger_is_excluded_from_primary_metric(self) -> None:
        logger = CaptureLogger.__new__(CaptureLogger)
        logger.rows = [
            {
                "round_index": 0, "capture_role": "initial", "output_delivered": 1,
                "pair_status": "challenger_won", "capture_valid_pair": 1,
                "switch_event": "committed",
                "abs_rel": .20, "a1": .7, "cell_id": "E16_G064",
                "control_decision_delay_ms": 1, "pair_capture_gap_ms": 5,
                "timestamp_ns": 1,
            },
            {
                "round_index": 0, "capture_role": "challenger", "output_delivered": 0,
                "pair_status": "challenger_won", "capture_valid_pair": 1,
                "switch_event": "committed",
                "abs_rel": .01, "a1": .99, "cell_id": "E08_G064",
                "control_decision_delay_ms": 1, "pair_capture_gap_ms": 5,
                "timestamp_ns": 2,
            },
        ]
        summary = logger._summary()
        self.assertEqual(summary["abs_rel"], .20)
        self.assertEqual(summary["a1"], .7)
        self.assertEqual(summary["output_coverage"], 1.0)

    def committed_runtime(self):
        runtime, events, provider, predictor, policy, logger = experiment()
        runtime.run_round()
        runtime.run_round()
        self.assertEqual(policy.state(ContextKey(0, 0)).active_cell_id, "E08_G064")
        return runtime, events, provider, predictor, policy, logger

    def test_confirmed_switch_is_logged_as_one_commit(self) -> None:
        _, _, _, _, _, logger = self.committed_runtime()
        committed = [
            row for row in logger.rows
            if row["capture_role"] == "initial" and row["switch_event"] == "committed"
        ]
        self.assertEqual(len(committed), 1)
        self.assertEqual(committed[0]["active_cell_after"], "E08_G064")
        self.assertEqual(committed[0]["committed_switch"], 1)

    def test_exact_post_switch_dwell_rounds_are_current_only(self) -> None:
        runtime, _, _, predictor, _, _ = self.committed_runtime()
        runtime.run_round()
        runtime.run_round()
        self.assertEqual(predictor.single_calls, 2)
        self.assertEqual(predictor.batch_sizes, [2, 2])
        runtime.run_round()
        self.assertEqual(predictor.batch_sizes, [2, 2, 2])

    def test_no_challenger_capture_during_dwell(self) -> None:
        runtime, events, _, _, _, _ = self.committed_runtime()
        events.clear()
        runtime.run_round()
        runtime.run_round()
        self.assertEqual(events.count("capture"), 2)
        self.assertNotIn("predict_batch", events)

    def test_reverse_pair_uses_new_current_and_old_challenger(self) -> None:
        runtime, _, _, _, _, logger = self.committed_runtime()
        runtime.run_round()
        runtime.run_round()
        runtime.run_round()
        self.assertEqual(
            [row["cell_id"] for row in logger.rows[-2:]],
            ["E08_G064", "E16_G064"],
        )
        self.assertEqual(logger.rows[-2]["pair_mode"], "rollback_verification")
        self.assertEqual(logger.rows[-2]["switch_event"], "rolled_back")
        self.assertEqual(logger.rows[-2]["rollback_applied"], 1)

    def test_current_only_mu_never_rolls_back(self) -> None:
        runtime, _, _, predictor, policy, _ = self.committed_runtime()
        predictor.single_mu = 1.0
        runtime.run_round()
        state = policy.state(ContextKey(0, 0))
        self.assertEqual(state.active_cell_id, "E08_G064")
        self.assertTrue(state.rollback_verification_pending)

    def test_summary_counts_only_physical_switch_events(self) -> None:
        logger = CaptureLogger.__new__(CaptureLogger)
        logger.rows = []
        for round_index, event, initial_error, probe_error in (
            (0, "confirmation_pending", .20, .01),
            (1, "committed", .20, .30),
            (4, "rolled_back", .20, .10),
        ):
            for role, delivered, error in (
                ("initial", 1, initial_error),
                ("challenger", 0, probe_error),
            ):
                logger.rows.append({
                    "round_index": round_index,
                    "capture_role": role,
                    "output_delivered": delivered,
                    "pair_status": "challenger_won",
                    "capture_valid_pair": 1,
                    "switch_event": event,
                    "abs_rel": error,
                    "a1": .8,
                    "cell_id": "E16_G064",
                    "control_decision_delay_ms": 1,
                    "pair_capture_gap_ms": 5,
                    "timestamp_ns": round_index + delivered,
                })
        summary = logger._summary()
        self.assertEqual(summary["confirmation_pending_count"], 1)
        self.assertEqual(summary["switch_count"], 1)
        self.assertEqual(summary["rollback_count"], 1)
        self.assertEqual(summary["switch_precision"], 0.0)
        self.assertEqual(summary["harmful_switch_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
