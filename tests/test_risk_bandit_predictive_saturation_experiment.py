import csv
import json
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import SafetyPolicy
from ati_mde_control.predictive_saturation_guard import (
    PredictiveSaturationGuard,
    PredictiveSaturationGuardConfig,
)
from ati_mde_control.risk_bandit_policy import RiskBanditConfig
from ati_mde_control.risk_bandit_predictive_saturation_experiment import (
    PREDICTIVE_SATURATION_CSV_FIELDS,
    RiskBanditPredictiveSaturationExperiment,
)
from ati_mde_control.saturation_guard import (
    SaturationGuardedRiskBanditPolicy,
    ev_index,
)
from hardware.utils import ContextKey, QScore, SensorCell
from tests.test_risk_bandit_saturation_experiment import (
    Camera,
    Evaluator,
    Logger,
    Predictor,
    Provider,
    RecordingPolicy,
    experiment_config,
)
import orbbec_ati_risk_bandit_predictive_saturation as entrypoint
import orbbec_ati_risk_bandit_saturation as reactive_entrypoint


def make_experiment(
    *,
    images=None,
    effective=None,
    scores=None,
    safety=None,
    inference_release=None,
):
    events = []
    temporary = TemporaryDirectory()
    provider = Provider()
    images = images or [np.zeros((2, 2, 3), dtype=np.uint8)]
    effective = effective or [True] * len(images)
    scores = scores or [QScore(0.25, 7.0, 6.0, {"mde_inference_ms": 1.0})]
    camera = Camera(events, images, effective, inference_release)
    predictor = Predictor(events, scores, inference_release)
    logger = Logger(events)
    config = experiment_config(Path(temporary.name))
    policy = RecordingPolicy(
        RiskBanditConfig(),
        safety or SafetyPolicy(),
        config.default_cell,
        events=events,
    )
    guard = PredictiveSaturationGuard(PredictiveSaturationGuardConfig())
    runtime = RiskBanditPredictiveSaturationExperiment(
        config,
        CaptureRunner(camera, provider, config.max_pair_capture_gap_ms),
        predictor,
        policy,
        logger,
        Evaluator(events),
        guard,
    )
    runtime._test_temporary = temporary
    return runtime, events, camera, predictor, policy, logger, guard


class RiskBanditPredictiveSaturationExperimentTest(unittest.TestCase):
    def track(self, runtime):
        self.addCleanup(runtime._test_temporary.cleanup)
        self.addCleanup(runtime._shutdown_worker)
        return runtime

    def test_one_capture_one_prediction_no_pair_or_challenger(self) -> None:
        runtime, events, camera, predictor, _, logger, _ = make_experiment()
        runtime.capture_runner.capture_pair = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paired capture is forbidden")
        )
        self.track(runtime).run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(len(logger.rows), 1)
        self.assertEqual(logger.rows[0]["capture_role"], "initial")
        self.assertFalse(any("challenger" in event for event in events))

    def test_projection_adds_no_predictor_call_and_gp_target_remains_q(self) -> None:
        image = np.full((2, 2, 3), 124, np.uint8)
        score = QScore(0.146, 9.0, 8.0)
        runtime, _, _, predictor, policy, _, _ = make_experiment(
            images=[image], scores=[score]
        )
        result = self.track(runtime).run_round()
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(policy.update_scores, [score])
        self.assertEqual(policy.history[0].q, 0.146)
        self.assertEqual(result.score.q, 0.146)

    def test_gp_selects_only_from_final_predictive_candidates(self) -> None:
        runtime, _, _, _, policy, _, _ = make_experiment(
            images=[np.full((2, 2, 3), 124, np.uint8)]
        )
        result = self.track(runtime).run_round()
        allowed = result.candidate_filter.candidates
        self.assertEqual(policy.allowed_history[0], allowed)
        self.assertIn(result.decision.selected_cell, allowed)

    def test_ineffective_hard_frame_updates_neither_gp_nor_quarantine(self) -> None:
        runtime, _, _, _, policy, _, guard = make_experiment(
            images=[np.full((2, 2, 3), 255, np.uint8)], effective=[False]
        )
        result = self.track(runtime).run_round()
        self.assertEqual(result.gp_update_status, "setting_ineffective")
        self.assertEqual(policy.history, ())
        self.assertIsNone(guard.quarantine(ContextKey(0, 0), 0))

    def test_event_order_and_maximum_one_pending_inference(self) -> None:
        release = threading.Event()
        runtime, events, _, predictor, _, _, guard = make_experiment(
            images=[np.zeros((2, 2, 3), np.uint8)] * 2,
            scores=[QScore(0.2, 0.1, 0.1), QScore(0.3, 0.1, 0.1)],
            inference_release=release,
        )
        self.track(runtime)
        original_observe = guard.observe
        original_filter = guard.filter_candidates

        def observe(*args, **kwargs):
            events.append("observed_saturation")
            return original_observe(*args, **kwargs)

        def filter_candidates(*args, **kwargs):
            events.append("candidate_projection")
            return original_filter(*args, **kwargs)

        guard.observe = observe
        guard.filter_candidates = filter_candidates
        runtime._on_inference_start = lambda: events.append("inference_start")
        runtime.run_round()
        release.clear()
        runtime.run_round()
        ordered = [
            next(index for index, event in enumerate(events) if event.startswith("capture:")),
            events.index("inference_start"),
            events.index("observed_saturation"),
            events.index("candidate_projection"),
            next(index for index, event in enumerate(events) if event.startswith("apply_next:")),
            events.index("inference_complete"),
        ]
        self.assertEqual(ordered, sorted(ordered))
        self.assertEqual(predictor.max_active, 1)
        self.assertIsNone(runtime._pending)

    def test_sidecar_indices_schema_and_compact_sorted_rejections(self) -> None:
        current = SensorCell(4, 16)
        runtime, _, _, _, _, logger, _ = make_experiment(
            images=[np.full((2, 2, 3), 125, np.uint8)]
        )
        runtime.current_cell = current
        runtime._pre_applied = runtime._apply_cell(current)
        self.track(runtime).run_round()
        sidecar = runtime.saturation_logger.rows[0]
        self.assertEqual(
            (logger.rows[0]["round_index"], logger.rows[0]["capture_index"]),
            (sidecar["round_index"], sidecar["capture_index"]),
        )
        rejected = json.loads(sidecar["projected_rejected_cell_ids"])
        step_rejected = json.loads(sidecar["ev_step_rejected_cell_ids"])
        self.assertEqual(rejected, sorted(rejected))
        self.assertEqual(step_rejected, sorted(step_rejected))
        self.assertNotIn(" ", sidecar["projected_rejected_cell_ids"])

    def test_finalize_stops_worker_then_evaluates_and_writes_sidecar(self) -> None:
        runtime, events, _, _, _, logger, _ = make_experiment()
        self.track(runtime).run_round()
        original_shutdown = runtime._executor.shutdown
        original_sidecar_write = runtime.saturation_logger.write

        def shutdown(*args, **kwargs):
            result = original_shutdown(*args, **kwargs)
            events.append("worker_shutdown")
            return result

        def sidecar_write():
            path = original_sidecar_write()
            events.append("sidecar_write")
            return path

        runtime._executor.shutdown = shutdown
        runtime.saturation_logger.write = sidecar_write
        self.assertEqual(runtime.finalize(), Path("probing_modelv1.csv"))
        self.assertLess(events.index("worker_shutdown"), events.index("evaluate"))
        self.assertLess(events.index("evaluate"), events.index("common_write"))
        self.assertLess(events.index("common_write"), events.index("sidecar_write"))
        self.assertEqual(logger.rows[0]["abs_rel"], 0.1)
        path = runtime.config.output_dir / "predictive_saturation_guard.csv"
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.assertEqual(tuple(reader.fieldnames), PREDICTIVE_SATURATION_CSV_FIELDS)
            self.assertEqual(len(list(reader)), 1)


class RiskBanditPredictiveSaturationEntrypointTest(unittest.TestCase):
    def test_cli_defaults_validation_and_reactive_defaults_are_unchanged(self) -> None:
        args = entrypoint.parse_args(["--camera-error-checkpoint", "checkpoint.pt"])
        expected = {
            "sat_pixel_threshold": 250.0,
            "sat_soft_clip_ratio": 0.60,
            "sat_secondary_clip_ratio": 0.85,
            "sat_hard_clip_ratio": 0.89,
            "sat_hard_mean_luminance": 243.0,
            "sat_recovery_clip_ratio": 0.50,
            "sat_recovery_frames": 3,
            "sat_quarantine_rounds": 60,
            "sat_min_ev_drop_stops": 1,
            "sat_max_upward_ev_stops": 1,
            "sat_projected_pixel_threshold": 250.0,
            "sat_projected_hard_clip_ratio": 0.90,
        }
        self.assertEqual({name: getattr(args, name) for name in expected}, expected)
        reactive = reactive_entrypoint.parse_args(
            ["--camera-error-checkpoint", "checkpoint.pt"]
        )
        self.assertEqual(reactive.sat_soft_clip_ratio, 0.80)
        self.assertEqual(reactive.sat_hard_clip_ratio, 0.90)
        self.assertEqual(reactive.sat_quarantine_rounds, 30)

        invalid = (
            ("--sat-soft-clip-ratio", "0.5"),
            ("--sat-hard-clip-ratio", "nan"),
            ("--sat-projected-pixel-threshold", "0"),
            ("--sat-projected-hard-clip-ratio", "1.1"),
            ("--sat-max-upward-ev-stops", "-1"),
            ("--sat-quarantine-rounds", "0"),
            ("--evaluation-alignment", "metric"),
        )
        for name, value in invalid:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                entrypoint.parse_args(
                    ["--camera-error-checkpoint", "checkpoint.pt", name, value]
                )

    def test_builder_uses_shared_fair_evaluator_and_predictive_runtime(self) -> None:
        calls = {}

        class FakeCamera:
            def __init__(self, **kwargs):
                calls["camera"] = self

        class FakePredictor:
            def __init__(self, *args):
                calls["predictor"] = self

        class FakeEvaluator:
            def __init__(self, predictor, config, precision):
                calls["evaluator"] = self
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
                entrypoint, "RiskBanditPredictiveSaturationExperiment", FakeRuntime
            ):
                runtime = entrypoint.build_experiment(args)
        predictor, config, precision = calls["evaluator_args"]
        self.assertIs(predictor, calls["predictor"])
        self.assertIs(runtime.args[5], calls["evaluator"])
        self.assertEqual(config.safety_path, entrypoint.DEFAULT_SAFETY_CONFIG)
        self.assertEqual(precision, "fp32")
        self.assertIsInstance(runtime.args[3], SaturationGuardedRiskBanditPolicy)
        self.assertIsInstance(runtime.args[6], PredictiveSaturationGuard)

    def test_new_sources_have_no_forbidden_import_or_pair_call(self) -> None:
        paths = (
            Path("ati_mde_control/predictive_saturation_guard.py"),
            Path("ati_mde_control/risk_bandit_predictive_saturation_experiment.py"),
            Path("orbbec_ati_risk_bandit_predictive_saturation.py"),
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            imports = [
                line.lower()
                for line in source.splitlines()
                if line.startswith("import ") or line.startswith("from ")
            ]
            self.assertFalse(
                any("iqa" in line or "nelder" in line for line in imports), path
            )
            self.assertNotIn("capture_pair(", source)


if __name__ == "__main__":
    unittest.main()
