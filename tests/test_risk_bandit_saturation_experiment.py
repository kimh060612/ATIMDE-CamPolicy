import csv
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import ExperimentConfig, PolicyConfig, SafetyPolicy
from ati_mde_control.risk_bandit_policy import RiskBanditConfig
from ati_mde_control.risk_bandit_saturation_experiment import (
    RiskBanditSaturationExperiment,
    SATURATION_CSV_FIELDS,
)
from ati_mde_control.saturation_guard import (
    SaturationGuard,
    SaturationGuardConfig,
    SaturationGuardedRiskBanditPolicy,
    ev_index,
)
from hardware.utils import ContextKey, QScore, SensorCell
import orbbec_ati_risk_bandit_saturation as entrypoint


class Provider:
    is_stable = True

    def __init__(self) -> None:
        self.current = ContextKey(0, 0)

    def get(self):
        return self.current

    def close(self):
        pass


class Camera:
    exposure_value_per_ms = 1000.0

    def __init__(self, events, images, effective, inference_release=None) -> None:
        self.events = events
        self.images = list(images)
        self.effective = list(effective)
        self.inference_release = inference_release
        self.active_cell = None
        self.apply_count = 0
        self.capture_count = 0
        self.setting_effective = True
        self.sensor_settle_ms = 0.0

    def apply_cell(self, cell):
        self.apply_count += 1
        self.active_cell = cell
        self.events.append(
            f"apply_{'next' if self.capture_count else 'initial'}:{cell.cell_id}"
        )
        if self.capture_count and self.inference_release is not None:
            self.inference_release.set()
        raw = cell.exposure_ms * 1000
        self.sensor_settle_ms = 2.0
        return raw, raw, cell.gain

    def capture_rgbd(self):
        self.capture_count += 1
        self.events.append(f"capture:{self.active_cell.cell_id}")
        self.color_frame_number = self.depth_frame_number = self.capture_count
        self.color_timestamp_us = self.depth_timestamp_us = self.capture_count * 1000
        self.setting_effective = self.effective.pop(0)
        return self.images.pop(0), np.ones((2, 2), dtype=np.float32)

    def close(self):
        pass


class Predictor:
    device = "cpu"

    def __init__(self, events, scores, inference_release=None) -> None:
        self.events = events
        self.scores = list(scores)
        self.inference_release = inference_release
        self.calls = []
        self.active = 0
        self.max_active = 0

    def predict_scores(self, image, context, exposure_us, gain):
        self.calls.append((image, context, exposure_us, gain))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.inference_release is not None:
                if not self.inference_release.wait(timeout=2):
                    raise RuntimeError("next camera apply did not start")
            score = self.scores.pop(0)
            if isinstance(score, Exception):
                raise score
            self.events.append("inference_complete")
            return score
        finally:
            self.active -= 1

    def predict(self, *args, **kwargs):
        raise AssertionError("v2 must use predict_scores().")

    def predict_batch(self, *args, **kwargs):
        raise AssertionError("v2 must not run candidate or depth-head inference.")


class RecordingPolicy(SaturationGuardedRiskBanditPolicy):
    def __init__(self, *args, events, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events = events
        self.allowed_history = []
        self.update_scores = []

    def select_from_candidates(self, context, current_cell, timestamp_ns, candidates):
        self.events.append("gp_select")
        self.allowed_history.append(tuple(candidates))
        return super().select_from_candidates(
            context, current_cell, timestamp_ns, candidates
        )

    def add_observation(self, context, cell, timestamp_ns, score):
        self.update_scores.append(score)
        return super().add_observation(context, cell, timestamp_ns, score)


class Logger:
    def __init__(self, events) -> None:
        self.events = events
        self.rows = []

    def record(self, frame, score, values):
        self.events.append("common_record")
        row = {
            "round_index": frame.round_index,
            "capture_index": frame.capture_index,
            "capture_role": frame.role,
            "output_delivered": int(frame.role == "initial"),
            "cell_id": frame.cell.cell_id,
            "q": "" if score is None else score.q,
            "abs_rel": "",
            "a1": "",
            "valid_depth_pixels": "",
            "evaluation_inference_ms": "",
            **values,
        }
        self.rows.append(row)
        return row

    def write(self):
        self.events.append("common_write")
        return Path("probing_modelv1.csv")


class Evaluator:
    def __init__(self, events) -> None:
        self.events = events

    def evaluate_rows(self, rows):
        self.events.append("evaluate")
        for row in rows:
            row.update(
                abs_rel=0.1,
                a1=0.9,
                valid_depth_pixels=4,
                evaluation_inference_ms=1.0,
            )


def experiment_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        checkpoint_path=Path("unused"),
        output_dir=output_dir,
        model_size="small",
        device="cpu",
        precision="fp32",
        local_files_only=True,
        q_uncertainty_weight=1.645,
        policy=PolicyConfig(),
        safety_path=None,
        default_cell=SensorCell(16, 64),
        max_rounds=10,
        round_interval_ms=0,
        max_pair_capture_gap_ms=100,
        camera_parameter_warn_ms=1000,
        mde_inference_warn_ms=1000,
        control_decision_warn_ms=1000,
        evaluation_alignment="scale_shift_inverse",
        min_depth_m=0.1,
        max_depth_m=10,
        min_valid_depth_pixels=1,
    )


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
    guard = SaturationGuard(SaturationGuardConfig())
    runtime = RiskBanditSaturationExperiment(
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


class RiskBanditSaturationExperimentTest(unittest.TestCase):
    def track(self, runtime):
        self.addCleanup(runtime._test_temporary.cleanup)
        self.addCleanup(runtime._shutdown_worker)
        return runtime

    def test_one_capture_one_score_prediction_and_no_challenger(self) -> None:
        runtime, events, camera, predictor, _, logger, _ = make_experiment()
        self.track(runtime).run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(len(logger.rows), 1)
        self.assertEqual(logger.rows[0]["capture_role"], "initial")
        self.assertEqual(logger.rows[0]["output_delivered"], 1)
        self.assertFalse(any("challenger" in event for event in events))

    def test_pair_capture_method_is_never_called(self) -> None:
        runtime, _, _, _, _, _, _ = make_experiment()
        runtime.capture_runner.capture_pair = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paired capture is forbidden")
        )
        self.track(runtime).run_round()

    def test_luminance_does_not_add_prediction_or_change_q_target(self) -> None:
        white = np.full((2, 2, 3), 255, dtype=np.uint8)
        score = QScore(0.146, 10.0, 9.0)
        runtime, _, _, predictor, policy, _, _ = make_experiment(
            images=[white], scores=[score]
        )
        result = self.track(runtime).run_round()
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(policy.update_scores, [score])
        self.assertEqual(policy.history[0].q, 0.146)
        self.assertEqual(result.score.q, 0.146)
        self.assertGreater(result.observation.metrics.mean_luminance, 245)

    def test_ineffective_hard_frame_does_not_create_long_term_quarantine(self) -> None:
        runtime, _, _, _, policy, _, guard = make_experiment(
            images=[np.full((2, 2, 3), 255, dtype=np.uint8)],
            effective=[False],
        )
        result = self.track(runtime).run_round()
        self.assertEqual(result.gp_update_status, "setting_ineffective")
        self.assertEqual(policy.history, ())
        self.assertIsNone(guard.quarantine(ContextKey(0, 0), 0))

    def test_hard_frame_selects_at_least_one_ev_stop_lower(self) -> None:
        runtime, _, _, _, _, _, _ = make_experiment(
            images=[np.full((2, 2, 3), 255, dtype=np.uint8)]
        )
        result = self.track(runtime).run_round()
        self.assertLessEqual(
            ev_index(result.decision.selected_cell), ev_index(result.frame.cell) - 1
        )

    def test_gp_receives_only_guard_allowed_candidates(self) -> None:
        runtime, _, _, _, policy, _, _ = make_experiment(
            images=[np.full((2, 2, 3), 255, dtype=np.uint8)]
        )
        result = self.track(runtime).run_round()
        self.assertEqual(policy.allowed_history[0], result.candidate_filter.candidates)
        self.assertIn(result.decision.selected_cell, policy.allowed_history[0])

    def test_event_order_and_single_pending_worker(self) -> None:
        release = threading.Event()
        images = [np.zeros((2, 2, 3), dtype=np.uint8)] * 2
        scores = [QScore(0.2, 0.1, 0.1), QScore(0.3, 0.1, 0.1)]
        runtime, events, _, predictor, _, _, guard = make_experiment(
            images=images, scores=scores, inference_release=release
        )
        self.track(runtime)
        original_observe = guard.observe

        def observe(*args, **kwargs):
            events.append("luminance_guard")
            return original_observe(*args, **kwargs)

        guard.observe = observe
        runtime._on_inference_start = lambda: events.append("inference_start")
        runtime.run_round()
        release.clear()
        runtime.run_round()
        first_apply = next(
            index for index, event in enumerate(events) if event.startswith("apply_next:")
        )
        ordered = [
            next(index for index, event in enumerate(events) if event.startswith("capture:")),
            events.index("inference_start"),
            events.index("luminance_guard"),
            first_apply,
            events.index("inference_complete"),
        ]
        self.assertEqual(ordered, sorted(ordered))
        self.assertEqual(predictor.max_active, 1)
        self.assertIsNone(runtime._pending)

    def test_common_and_sidecar_rows_share_round_and_capture_indices(self) -> None:
        runtime, _, _, _, _, logger, _ = make_experiment()
        result = self.track(runtime).run_round()
        sidecar = runtime.saturation_logger.rows[0]
        self.assertEqual(
            (logger.rows[0]["round_index"], logger.rows[0]["capture_index"]),
            (sidecar["round_index"], sidecar["capture_index"]),
        )
        self.assertEqual(sidecar["selected_next_cell"], result.decision.selected_cell.cell_id)

    def test_finalize_stops_worker_then_evaluates_common_and_sidecar(self) -> None:
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
        self.assertTrue(runtime._worker_closed)
        self.assertEqual(logger.rows[0]["abs_rel"], 0.1)
        sidecar_path = runtime.config.output_dir / "saturation_guard.csv"
        with sidecar_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self.assertEqual(tuple(reader.fieldnames), SATURATION_CSV_FIELDS)
            self.assertEqual(len(list(reader)), 1)


class RiskBanditSaturationEntrypointTest(unittest.TestCase):
    def test_cli_defaults_and_validation(self) -> None:
        args = entrypoint.parse_args(["--camera-error-checkpoint", "checkpoint.pt"])
        self.assertEqual(args.sat_pixel_threshold, 250.0)
        self.assertEqual(args.sat_soft_clip_ratio, 0.80)
        self.assertEqual(args.sat_secondary_clip_ratio, 0.85)
        self.assertEqual(args.sat_hard_clip_ratio, 0.90)
        self.assertEqual(args.sat_hard_mean_luminance, 245.0)
        self.assertEqual(args.sat_recovery_clip_ratio, 0.70)
        self.assertEqual(args.sat_recovery_frames, 3)
        self.assertEqual(args.sat_quarantine_rounds, 30)
        self.assertEqual(args.sat_min_ev_drop_stops, 1)
        invalid = (
            ("--sat-pixel-threshold", "nan"),
            ("--sat-pixel-threshold", "256"),
            ("--sat-soft-clip-ratio", "0.7"),
            ("--sat-secondary-clip-ratio", "0.95"),
            ("--sat-hard-clip-ratio", "1.1"),
            ("--sat-hard-mean-luminance", "inf"),
            ("--sat-recovery-frames", "0"),
            ("--sat-quarantine-rounds", "0"),
            ("--sat-min-ev-drop-stops", "0"),
            ("--evaluation-alignment", "metric"),
        )
        for name, value in invalid:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                entrypoint.parse_args(
                    ["--camera-error-checkpoint", "checkpoint.pt", name, value]
                )

    def test_builder_connects_exact_shared_evaluator_and_score_predictor(self) -> None:
        calls = {}

        class FakeCamera:
            def __init__(self, **kwargs):
                calls["camera"] = self

        class FakePredictor:
            def __init__(self, *args):
                calls["predictor"] = self

        class FakeEvaluator:
            def __init__(self, predictor, config, precision):
                calls["evaluator_instance"] = self
                calls["evaluator"] = (predictor, config, precision)

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
                entrypoint, "RiskBanditSaturationExperiment", FakeRuntime
            ):
                runtime = entrypoint.build_experiment(args)
        predictor, config, precision = calls["evaluator"]
        self.assertIs(predictor, calls["predictor"])
        self.assertIs(runtime.args[2], predictor)
        self.assertIs(runtime.args[5], calls["evaluator_instance"])
        self.assertEqual(config.safety_path, entrypoint.DEFAULT_SAFETY_CONFIG)
        self.assertEqual(precision, "fp32")
        self.assertIsInstance(runtime.args[3], SaturationGuardedRiskBanditPolicy)
        self.assertIsInstance(runtime.args[6], SaturationGuard)

    def test_new_path_has_no_forbidden_algorithm_import_or_pair_call(self) -> None:
        paths = (
            Path("ati_mde_control/saturation_guard.py"),
            Path("ati_mde_control/risk_bandit_saturation_experiment.py"),
            Path("orbbec_ati_risk_bandit_saturation.py"),
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
