import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np

import orbbec_ati_risk_bandit_bidirectional_exposure_sync as entrypoint
from ati_mde_control.bidirectional_exposure_guard import (
    BidirectionalExposureGuard,
    BidirectionalExposureGuardConfig,
)
from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import SafetyPolicy
from ati_mde_control.risk_bandit_bidirectional_exposure_sync_experiment import (
    METHOD_NAME,
    RiskBanditBidirectionalExposureSyncExperiment,
)
from ati_mde_control.risk_bandit_policy import RiskBanditConfig
from ati_mde_control.saturation_guard import SaturationGuardedRiskBanditPolicy
from hardware.utils import ContextKey, QScore
from tests.test_risk_bandit_saturation_experiment import (
    Camera,
    Evaluator,
    Logger,
    Predictor,
    Provider,
    RecordingPolicy,
    experiment_config,
)


def make_experiment(*, effective=True, score=None):
    events = []
    temporary = TemporaryDirectory()
    provider = Provider()
    camera = Camera(
        events,
        [np.full((2, 2, 3), 100, np.uint8)],
        [effective],
    )
    predictor = Predictor(
        events,
        [score or QScore(0.25, 0.1, 0.2, {"mde_inference_ms": 1.0})],
    )
    logger = Logger(events)
    config = experiment_config(Path(temporary.name))
    policy = RecordingPolicy(
        RiskBanditConfig(), SafetyPolicy(), config.default_cell, events=events
    )
    guard = BidirectionalExposureGuard(
        BidirectionalExposureGuardConfig(), policy.safe_fallback
    )
    runtime = RiskBanditBidirectionalExposureSyncExperiment(
        config,
        CaptureRunner(camera, provider, config.max_pair_capture_gap_ms),
        predictor,
        policy,
        logger,
        Evaluator(events),
        guard,
    )
    runtime._test_temporary = temporary
    return runtime, events, provider, camera, predictor, policy, logger, guard


class RiskBanditBidirectionalExposureSyncExperimentTest(unittest.TestCase):
    def track(self, runtime):
        self.addCleanup(runtime._test_temporary.cleanup)
        return runtime

    def test_current_observation_precedes_selection_and_is_in_history(self) -> None:
        runtime, events, _, _, _, policy, _, guard = make_experiment(
            score=QScore(0.146, 9.0, 8.0)
        )
        self.track(runtime)
        original_add = policy.add_observation
        original_select = policy.select_from_candidates
        original_observe = guard.observe
        history_at_selection = []

        def add_observation(*args, **kwargs):
            events.append("gp_update")
            return original_add(*args, **kwargs)

        def select(*args, **kwargs):
            events.append("sync_select")
            history_at_selection.extend(policy.history)
            return original_select(*args, **kwargs)

        def observe(*args, **kwargs):
            events.append("guard")
            return original_observe(*args, **kwargs)

        policy.add_observation = add_observation
        policy.select_from_candidates = select
        guard.observe = observe
        result = runtime.run_round()

        self.assertLess(events.index("inference_complete"), events.index("gp_update"))
        self.assertLess(events.index("gp_update"), events.index("guard"))
        self.assertLess(events.index("guard"), events.index("sync_select"))
        self.assertLess(
            events.index("sync_select"),
            next(i for i, event in enumerate(events) if event.startswith("apply_next:")),
        )
        self.assertEqual(len(history_at_selection), 1)
        current_observation = history_at_selection[0]
        self.assertEqual(current_observation.q, 0.146)
        self.assertEqual(current_observation.context, result.frame.capture_context)
        self.assertEqual(current_observation.cell, result.frame.cell)
        self.assertEqual(current_observation.timestamp_ns, result.frame.timestamp_ns)
        self.assertEqual(result.gp_update_status, "updated")

    def test_one_capture_one_prediction_and_no_pair(self) -> None:
        runtime, events, _, camera, predictor, _, logger, _ = make_experiment()
        runtime.capture_runner.capture_pair = lambda *args, **kwargs: (
            (_ for _ in ()).throw(AssertionError("paired capture is forbidden"))
        )
        self.track(runtime).run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(len(logger.rows), 1)
        self.assertFalse(any("challenger" in event for event in events))

    def test_camera_next_apply_waits_for_inference_completion(self) -> None:
        runtime, events, _, camera, _, _, _, _ = make_experiment()
        self.track(runtime)
        started = threading.Event()
        release = threading.Event()
        failures = []

        class BlockingPredictor:
            device = "cpu"

            def predict_scores(self, *args):
                events.append("inference_start")
                started.set()
                if not release.wait(timeout=2):
                    raise RuntimeError("test inference release timed out")
                events.append("inference_complete")
                return QScore(0.2, 0.1, 0.1)

        runtime.predictor = BlockingPredictor()

        def run():
            try:
                runtime.run_round()
            except Exception as error:
                failures.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(started.wait(timeout=1))
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(camera.apply_count, 1)
        self.assertFalse(any(event.startswith("apply_next:") for event in events))
        release.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(camera.apply_count, 2)

    def test_capture_context_updates_gp_and_latest_context_selects(self) -> None:
        runtime, _, provider, _, predictor, policy, _, _ = make_experiment()
        self.track(runtime)
        capture_context = ContextKey(0, 0)
        selection_context = ContextKey(3, 2)
        provider.current = capture_context
        update_contexts = []
        selection_contexts = []
        original_predict = predictor.predict_scores
        original_add = policy.add_observation
        original_select = policy.select_from_candidates

        def predict(*args, **kwargs):
            score = original_predict(*args, **kwargs)
            provider.current = selection_context
            return score

        def add(context, *args, **kwargs):
            update_contexts.append(context)
            return original_add(context, *args, **kwargs)

        def select(context, *args, **kwargs):
            selection_contexts.append(context)
            return original_select(context, *args, **kwargs)

        predictor.predict_scores = predict
        policy.add_observation = add
        policy.select_from_candidates = select
        result = runtime.run_round()
        self.assertEqual(result.frame.capture_context, capture_context)
        self.assertEqual(update_contexts, [capture_context])
        self.assertEqual(selection_contexts, [selection_context])

    def test_ineffective_and_non_finite_scores_do_not_enter_history(self) -> None:
        ineffective, _, _, _, _, policy, _, _ = make_experiment(effective=False)
        ineffective_result = self.track(ineffective).run_round()
        self.assertEqual(ineffective_result.gp_update_status, "setting_ineffective")
        self.assertEqual(policy.history, ())

        non_finite, _, _, _, _, policy, _, _ = make_experiment(
            score=QScore(float("nan"), 0.1, 0.1)
        )
        non_finite_result = self.track(non_finite).run_round()
        self.assertEqual(non_finite_result.gp_update_status, "non_finite_score")
        self.assertEqual(policy.history, ())

    def test_predictor_failure_is_not_added_to_gp(self) -> None:
        runtime, _, _, _, predictor, policy, logger, _ = make_experiment()
        self.track(runtime)
        predictor.scores[:] = [RuntimeError("inference failed")]
        with self.assertRaisesRegex(RuntimeError, "risk prediction failed"):
            runtime.run_round()
        self.assertEqual(policy.history, ())
        self.assertEqual(len(logger.rows), 1)

    def test_finalize_reuses_fair_evaluation_path_without_worker(self) -> None:
        runtime, events, _, _, _, _, logger, _ = make_experiment()
        self.track(runtime).run_round()
        original_sidecar_write = runtime.saturation_logger.write

        def sidecar_write():
            path = original_sidecar_write()
            events.append("sidecar_write")
            return path

        runtime.saturation_logger.write = sidecar_write
        self.assertFalse(hasattr(runtime, "_executor"))
        self.assertFalse(hasattr(runtime, "_pending"))
        self.assertEqual(runtime.finalize(), Path("probing_modelv1.csv"))
        self.assertLess(events.index("evaluate"), events.index("common_write"))
        self.assertLess(events.index("common_write"), events.index("sidecar_write"))
        self.assertEqual(logger.rows[0]["abs_rel"], 0.1)


class RiskBanditBidirectionalExposureSyncEntrypointTest(unittest.TestCase):
    def test_method_name_and_builder_use_sync_runtime_and_fair_evaluator(self) -> None:
        self.assertNotIn("Delay-Aware", METHOD_NAME)
        calls = {}

        class FakeCamera:
            def __init__(self, **kwargs):
                calls["camera"] = self

        class FakePredictor:
            def __init__(self, *args):
                calls["predictor"] = self

        class FakeEvaluator:
            def __init__(self, predictor, config, precision):
                calls["evaluator_args"] = (predictor, config, precision)

        class FakeRuntime:
            def __init__(self, *args):
                self.args = args

        predictor_module = ModuleType("ati_mde_control.predictor")
        predictor_module.CameraErrorPredictor = FakePredictor
        sensor_module = ModuleType("hardware.sensor")
        sensor_module.OrbbecColorCamera = FakeCamera
        with TemporaryDirectory() as temporary:
            args = entrypoint.parse_args(
                [
                    "--camera-error-checkpoint",
                    "checkpoint.pt",
                    "--output-dir",
                    temporary,
                ]
            )
            with patch.dict(
                sys.modules,
                {
                    "ati_mde_control.predictor": predictor_module,
                    "hardware.sensor": sensor_module,
                },
            ), patch.object(
                entrypoint, "build_context_provider", return_value=object()
            ), patch.object(
                entrypoint, "CaptureRunner", return_value=object()
            ), patch.object(
                entrypoint, "CaptureLogger", return_value=object()
            ), patch.object(
                entrypoint, "FairDepthEvaluator", FakeEvaluator
            ), patch.object(
                entrypoint,
                "RiskBanditBidirectionalExposureSyncExperiment",
                FakeRuntime,
            ):
                runtime = entrypoint.build_experiment(args)
        predictor, _, precision = calls["evaluator_args"]
        self.assertIs(predictor, calls["predictor"])
        self.assertEqual(precision, "fp32")
        self.assertIsInstance(runtime.args[3], SaturationGuardedRiskBanditPolicy)
        self.assertIsInstance(runtime.args[6], BidirectionalExposureGuard)

    def test_new_sources_have_no_async_or_forbidden_paths(self) -> None:
        paths = (
            Path(
                "ati_mde_control/"
                "risk_bandit_bidirectional_exposure_sync_experiment.py"
            ),
            Path("orbbec_ati_risk_bandit_bidirectional_exposure_sync.py"),
        )
        forbidden = (
            "ThreadPoolExecutor",
            "Future",
            "capture_pair(",
            "nelder",
            "iqa",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
