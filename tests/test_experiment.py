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

    def __init__(self, events, provider=None):
        self.events = events
        self.provider = provider
        self.single_calls = 0
        self.batch_sizes = []

    def predict(self, image, context, exposure_us, gain):
        self.events.append("predict_single")
        self.single_calls += 1
        return score(.05)

    def predict_batch(self, images, contexts, exposure_us_values, gains):
        self.events.append("predict_batch")
        self.batch_sizes.append(len(images))
        if self.provider is not None:
            self.provider.current = ContextKey(1, 0)
        return [score(.20, 2), score(.10, 2)]


class Logger:
    def __init__(self):
        self.rows = []

    def record(self, frame, score, values):
        row = {
            "round_index": frame.round_index,
            "capture_role": frame.role,
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


def experiment(change_context_during_batch=False):
    events = []
    provider = Provider()
    cfg = config()
    policy = PairwisePolicy(cfg.policy, SafetyPolicy())
    predictor = Predictor(events, provider if change_context_during_batch else None)
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
        self.assertIn("active=E08_G064 E=8ms G=64", line)
        self.assertIn("search=exposure/negative", line)
        self.assertIn("decision=challenger_won/immediate_commit flow=normal_search", line)
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

    def test_immediate_commit_changes_next_initial_frame_without_dwell(self) -> None:
        runtime, _, _, predictor, _, _ = experiment()
        runtime.run_round()
        result = runtime.run_round()
        self.assertEqual(result.initial.cell, SensorCell(8, 64))
        self.assertEqual(result.challenger.cell, SensorCell(4, 64))
        self.assertEqual(predictor.batch_sizes, [2, 2])
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
        self.assertEqual(old_state.active_cell_id, SensorCell(8, 64).cell_id)
        self.assertEqual(len(old_state.local_edges), 1)

    def test_old_context_winner_is_not_left_applied_in_new_context(self) -> None:
        runtime, events, _, _, policy, _ = experiment(change_context_during_batch=True)
        runtime.run_round()
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))
        self.assertEqual(policy.state(ContextKey(1, 0)).active_cell_id, SensorCell(16, 64).cell_id)
        self.assertEqual(events[-1], "apply:E16_G064")

    def test_challenger_is_excluded_from_primary_metric(self) -> None:
        logger = CaptureLogger.__new__(CaptureLogger)
        logger.rows = [
            {
                "round_index": 0, "capture_role": "initial", "output_delivered": 1,
                "pair_status": "challenger_won", "capture_valid_pair": 1,
                "abs_rel": .20, "a1": .7, "cell_id": "E16_G064",
                "control_decision_delay_ms": 1, "pair_capture_gap_ms": 5,
                "timestamp_ns": 1,
            },
            {
                "round_index": 0, "capture_role": "challenger", "output_delivered": 0,
                "pair_status": "challenger_won", "capture_valid_pair": 1,
                "abs_rel": .01, "a1": .99, "cell_id": "E08_G064",
                "control_decision_delay_ms": 1, "pair_capture_gap_ms": 5,
                "timestamp_ns": 2,
            },
        ]
        summary = logger._summary()
        self.assertEqual(summary["abs_rel"], .20)
        self.assertEqual(summary["a1"], .7)
        self.assertEqual(summary["output_coverage"], 1.0)

    def test_summary_counts_only_commit_events_as_switches(self) -> None:
        logger = CaptureLogger.__new__(CaptureLogger)
        logger.rows = []
        for round_index, status, event, observations in (
            (0, "ambiguous", "pending_started", 1),
            (1, "ambiguous", "pending_committed", 2),
            (2, "challenger_won", "immediate_commit", ""),
        ):
            for role, delivered in (("initial", 1), ("challenger", 0)):
                logger.rows.append({
                    "round_index": round_index,
                    "capture_role": role,
                    "output_delivered": delivered,
                    "pair_status": status,
                    "switch_event": event,
                    "pending_observation_count": observations,
                    "capture_valid_pair": 1,
                    "abs_rel": .1,
                    "a1": .9,
                    "cell_id": "E16_G064",
                    "control_decision_delay_ms": 1,
                    "pair_capture_gap_ms": 5,
                    "timestamp_ns": round_index * 2 + delivered,
                })
        summary = logger._summary()
        self.assertEqual(summary["switch_count"], 2)
        self.assertEqual(summary["pending_started_count"], 1)
        self.assertEqual(summary["pending_committed_count"], 1)
        self.assertEqual(summary["mean_pending_observation_count"], 1.5)


if __name__ == "__main__":
    unittest.main()
