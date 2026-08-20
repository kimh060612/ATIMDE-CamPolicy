import unittest
from pathlib import Path

import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import ExperimentConfig, PolicyConfig, SafetyPolicy
from ati_mde_control.local_search_update_experiment import LocalSearchUpdateExperiment
from ati_mde_control.pairwise_policy import PairwisePolicy
from ati_mde_control.types import PairStatus, SwitchEvent
from hardware.utils import ContextKey, QScore, SensorCell


def score(mu: float, std: float = 0.01) -> QScore:
    return QScore(
        mu + std,
        std,
        mu,
        {"mde_batch_size": 1, "mde_inference_ms": 1.0},
    )


class Provider:
    def __init__(self) -> None:
        self.current = ContextKey(0, 0)
        self.is_stable = True

    def get(self):
        return self.current

    def close(self):
        pass


class Camera:
    exposure_value_per_ms = 1000.0

    def __init__(
        self,
        events,
        provider,
        *,
        capture_gaps_us=None,
        effective=None,
        fail_capture=None,
        on_capture=None,
    ) -> None:
        self.events = events
        self.provider = provider
        self.capture_gaps_us = list(capture_gaps_us or [])
        self.effective = list(effective or [])
        self.fail_capture = fail_capture
        self.on_capture = on_capture
        self.active_cell = None
        self.capture_count = 0
        self.color_frame_number = 0
        self.depth_frame_number = 0
        self.color_timestamp_us = 0
        self.depth_timestamp_us = 0
        self.setting_effective = True
        self.sensor_settle_ms = 0.0

    def apply_cell(self, cell):
        self.active_cell = cell
        self.events.append(f"apply:{cell.cell_id}")
        raw = cell.exposure_ms * 1000
        return raw, raw, cell.gain

    def capture_rgbd(self):
        self.capture_count += 1
        cell_id = self.active_cell.cell_id
        self.events.append(f"capture:{cell_id}")
        if self.fail_capture == self.capture_count:
            raise TimeoutError("synthetic capture failure")
        gap = self.capture_gaps_us.pop(0) if self.capture_gaps_us else 1000
        self.color_timestamp_us += gap
        self.depth_timestamp_us = self.color_timestamp_us
        self.color_frame_number = self.depth_frame_number = self.capture_count
        self.setting_effective = self.effective.pop(0) if self.effective else True
        if self.on_capture is not None:
            self.on_capture(self.capture_count, self.provider)
        image = np.full((2, 2, 3), self.capture_count, dtype=np.uint8)
        depth = np.ones((2, 2), dtype=np.float32)
        return image, depth

    def close(self):
        pass


class Predictor:
    device = "cpu"

    def __init__(self, events, results, provider, on_predict=None) -> None:
        self.events = events
        self.results = list(results)
        self.provider = provider
        self.on_predict = on_predict
        self.calls = 0

    def predict_scores(self, image, context, exposure_us, gain):
        self.calls += 1
        exposure_ms = round(exposure_us / 1000.0)
        self.events.append(f"predict:E{exposure_ms:02d}_G{round(gain):03d}")
        if self.on_predict is not None:
            self.on_predict(self.calls, self.provider)
        return self.results.pop(0)

    def predict(self, *args, **kwargs):
        raise AssertionError("The local-search update path must use predict_scores().")

    def predict_batch(self, *args, **kwargs):
        raise AssertionError("The local-search update path must not use predict_batch().")


class Logger:
    def __init__(self) -> None:
        self.rows = []

    def record(self, frame, predicted, values):
        row = {
            "round_index": frame.round_index,
            "capture_role": frame.role,
            "cell_id": frame.cell.cell_id,
            **values,
        }
        self.rows.append(row)
        return row


class Evaluator:
    def evaluate_rows(self, rows):
        pass


def make_config(policy=None, max_gap_ms=100.0):
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
        max_pair_capture_gap_ms=max_gap_ms,
        camera_parameter_warn_ms=1000,
        mde_inference_warn_ms=1000,
        control_decision_warn_ms=1000,
        evaluation_alignment="metric",
        min_depth_m=0.2,
        max_depth_m=10,
        min_valid_depth_pixels=1,
    )


def make_experiment(
    results,
    *,
    policy_config=None,
    safety=None,
    max_gap_ms=100.0,
    camera_options=None,
    on_predict=None,
):
    events = []
    provider = Provider()
    config = make_config(policy_config, max_gap_ms)
    policy = PairwisePolicy(config.policy, safety or SafetyPolicy())
    camera = Camera(events, provider, **(camera_options or {}))
    predictor = Predictor(events, results, provider, on_predict)
    logger = Logger()
    runtime = LocalSearchUpdateExperiment(
        config,
        CaptureRunner(camera, provider, config.max_pair_capture_gap_ms),
        predictor,
        policy,
        logger,
        Evaluator(),
    )
    return runtime, events, provider, predictor, policy, logger


class LocalSearchUpdateExperimentTest(unittest.TestCase):
    def test_current_only_round_calls_predictor_exactly_once(self) -> None:
        runtime, events, _, predictor, policy, logger = make_experiment([score(0.05)])
        state = policy.state(ContextKey(0, 0))
        state.active_cell_id = SensorCell(16, 64).cell_id
        state.search_cycle_complete = True
        state.probe_pending = False

        result = runtime.run_round()

        self.assertEqual(predictor.calls, 1)
        self.assertEqual(events.count("capture:E16_G064"), 1)
        self.assertIsNone(result.challenger)
        self.assertEqual(logger.rows[0]["selected"], 1)
        self.assertEqual(logger.rows[0]["output_delivered"], 1)

    def test_current_cell_is_not_reapplied_when_camera_is_already_there(self) -> None:
        runtime, events, _, _, policy, _ = make_experiment([score(0.05)])
        current = SensorCell(16, 64)
        state = policy.state(ContextKey(0, 0))
        state.active_cell_id = current.cell_id
        state.search_cycle_complete = True
        state.probe_pending = False
        runtime.last_applied_cell = current
        runtime.capture_runner.camera.active_cell = current

        runtime.run_round()

        self.assertNotIn("apply:E16_G064", events)
        self.assertEqual(events[:2], ["capture:E16_G064", "predict:E16_G064"])

    def test_probe_round_scores_current_and_challenger_once_each(self) -> None:
        runtime, events, _, predictor, policy, _ = make_experiment(
            [score(0.20), score(0.10)]
        )
        state = policy.state(ContextKey(0, 0))
        state.active_cell_id = SensorCell(16, 64).cell_id
        state.search_cycle_complete = True
        state.probe_pending = False

        result = runtime.run_round()

        self.assertEqual(predictor.calls, 2)
        self.assertEqual(
            [event for event in events if event.startswith("predict:")],
            ["predict:E16_G064", "predict:E08_G064"],
        )
        self.assertEqual(result.decision.status, PairStatus.CHALLENGER_WON)

    def test_probe_trigger_does_not_recapture_current(self) -> None:
        runtime, events, _, _, policy, _ = make_experiment(
            [score(0.20), score(0.10)]
        )
        state = policy.state(ContextKey(0, 0))
        state.active_cell_id = SensorCell(16, 64).cell_id
        state.search_cycle_complete = True
        state.probe_pending = False

        runtime.run_round()

        self.assertEqual(
            events,
            [
                "apply:E16_G064",
                "capture:E16_G064",
                "predict:E16_G064",
                "apply:E08_G064",
                "capture:E08_G064",
                "predict:E08_G064",
            ],
        )

    def test_challenger_commit_leaves_camera_without_second_apply(self) -> None:
        runtime, events, _, _, _, logger = make_experiment(
            [score(0.20), score(0.10)]
        )

        result = runtime.run_round()

        self.assertEqual(result.decision.switch_event, SwitchEvent.IMMEDIATE_COMMIT)
        self.assertEqual(events.count("apply:E08_G064"), 1)
        self.assertEqual(runtime.last_applied_cell, SensorCell(8, 64))
        current_row, challenger_row = logger.rows
        self.assertEqual(
            (current_row["selected"], current_row["output_delivered"]), (0, 0)
        )
        self.assertEqual(
            (challenger_row["selected"], challenger_row["output_delivered"]),
            (1, 1),
        )

    def test_challenger_reject_restores_current_only_after_comparison(self) -> None:
        runtime, events, _, _, _, logger = make_experiment(
            [score(0.10), score(0.20)]
        )

        result = runtime.run_round()

        self.assertEqual(result.decision.status, PairStatus.CURRENT_WON)
        self.assertEqual(events[-1], "apply:E16_G064")
        self.assertEqual(events.count("apply:E16_G064"), 2)
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))
        current_row, challenger_row = logger.rows
        self.assertEqual(
            (current_row["selected"], current_row["output_delivered"]), (1, 1)
        )
        self.assertEqual(
            (challenger_row["selected"], challenger_row["output_delivered"]),
            (0, 0),
        )

    def test_pending_recheck_has_no_duplicate_current_inference(self) -> None:
        runtime, events, _, predictor, _, _ = make_experiment(
            [
                score(0.112, 0.03),
                score(0.10, 0.03),
                score(0.13, 0.005),
                score(0.10, 0.005),
            ]
        )

        first = runtime.run_round()
        second = runtime.run_round()

        self.assertEqual(first.decision.switch_event, SwitchEvent.PENDING_STARTED)
        self.assertEqual(second.decision.switch_event, SwitchEvent.PENDING_COMMITTED)
        self.assertEqual(predictor.calls, 4)
        predictions = [event for event in events if event.startswith("predict:")]
        self.assertEqual(
            predictions,
            [
                "predict:E16_G064",
                "predict:E08_G064",
                "predict:E16_G064",
                "predict:E08_G064",
            ],
        )

    def test_context_change_or_instability_never_commits_challenger(self) -> None:
        def change_after_challenger(call, provider):
            if call == 2:
                provider.current = ContextKey(1, 0)

        runtime, _, _, _, policy, _ = make_experiment(
            [score(0.20), score(0.10)], on_predict=change_after_challenger
        )
        result = runtime.run_round()
        self.assertEqual(result.decision.status, PairStatus.INVALID_PAIR)
        self.assertEqual(
            policy.state(ContextKey(0, 0)).active_cell_id, SensorCell(16, 64).cell_id
        )

        def destabilize_after_challenger(call, provider):
            if call == 2:
                provider.is_stable = False

        runtime, _, _, _, policy, _ = make_experiment(
            [score(0.20), score(0.10)], on_predict=destabilize_after_challenger
        )
        result = runtime.run_round()
        self.assertEqual(result.decision.status, PairStatus.INVALID_PAIR)
        self.assertEqual(
            policy.state(ContextKey(0, 0)).active_cell_id, SensorCell(16, 64).cell_id
        )

    def test_context_change_after_current_stops_probe_without_second_inference(self) -> None:
        def change_after_current(call, provider):
            if call == 1:
                provider.current = ContextKey(1, 0)

        runtime, events, _, predictor, policy, _ = make_experiment(
            [score(0.20)], on_predict=change_after_current
        )

        result = runtime.run_round()

        self.assertIsNone(result.challenger)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(sum(event.startswith("capture:") for event in events), 1)
        self.assertEqual(
            policy.state(ContextKey(0, 0)).active_cell_id, SensorCell(16, 64).cell_id
        )

    def test_large_capture_gap_is_logged_but_does_not_invalidate_pair(self) -> None:
        runtime, _, _, _, _, logger = make_experiment(
            [score(0.20), score(0.10)],
            max_gap_ms=1.0,
            camera_options={"capture_gaps_us": [1000, 500_000]},
        )

        result = runtime.run_round()

        self.assertEqual(result.decision.status, PairStatus.CHALLENGER_WON)
        self.assertEqual(logger.rows[0]["pair_capture_gap_ms"], 500.0)
        self.assertEqual(logger.rows[0]["capture_valid_pair"], 1)

    def test_ineffective_challenger_is_invalid_and_restores_current(self) -> None:
        runtime, events, _, predictor, _, _ = make_experiment(
            [score(0.20), score(0.10)],
            camera_options={"effective": [True, False]},
        )

        result = runtime.run_round()

        self.assertEqual(result.decision.status, PairStatus.INVALID_PAIR)
        self.assertEqual(predictor.calls, 2)
        self.assertEqual(events[-1], "apply:E16_G064")
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))

    def test_challenger_capture_failure_is_invalid_and_restores_current(self) -> None:
        runtime, events, _, predictor, _, logger = make_experiment(
            [score(0.20)], camera_options={"fail_capture": 2}
        )

        result = runtime.run_round()

        self.assertEqual(result.decision.status, PairStatus.INVALID_PAIR)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(events[-1], "apply:E16_G064")
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))
        self.assertEqual(logger.rows[0]["pair_invalid_reason"], "capture_failure:TimeoutError")

    def test_existing_safety_policy_still_selects_only_a_safe_neighbor(self) -> None:
        disabled = {"0,0": frozenset({SensorCell(8, 64).cell_id})}
        safety = SafetyPolicy(disabled_cells_by_context=disabled)
        runtime, events, _, _, _, _ = make_experiment(
            [score(0.20), score(0.10)], safety=safety
        )

        result = runtime.run_round()

        self.assertEqual(result.challenger.cell, SensorCell(32, 64))
        self.assertIn("apply:E32_G064", events)


if __name__ == "__main__":
    unittest.main()
