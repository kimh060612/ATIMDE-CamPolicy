import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.brightness_safety import (
    BrightnessGuardConfig,
    BrightnessGuardMode,
)
from ati_mde_control.config import ExperimentConfig, PolicyConfig, SafetyPolicy
from ati_mde_control.full_depth_predictor import FullDepthBatchPrediction
from ati_mde_control.nelder_mead_experiment import RiskNelderMeadExperiment
from ati_mde_control.nelder_mead_policy import (
    ContextualRiskNelderMeadPolicy,
    RiskNelderMeadConfig,
)
from hardware.utils import ContextKey, QScore, SensorCell
import orbbec_ati_neldermead as entrypoint


class Provider:
    is_stable = True

    def __init__(self):
        self.current = ContextKey(0, 0)

    def get(self):
        return self.current

    def close(self):
        pass


class Camera:
    exposure_value_per_ms = 1000.0

    def __init__(self, events, provider, on_capture=None, image_value=0):
        self.events = events
        self.provider = provider
        self.on_capture = on_capture
        self.image_value = image_value
        self.capture_count = 0
        self.setting_effective = True
        self.sensor_settle_ms = 0.0

    def apply_cell(self, cell):
        self.active_cell = cell
        self.events.append(f"apply:{cell.cell_id}")
        raw = cell.exposure_ms * 1000
        return raw, raw, cell.gain

    def capture_rgbd(self):
        self.capture_count += 1
        self.events.append(f"capture:{self.active_cell.cell_id}")
        if self.on_capture:
            self.on_capture(self.provider)
        self.color_frame_number = self.depth_frame_number = self.capture_count
        self.color_timestamp_us = self.depth_timestamp_us = self.capture_count * 1000
        return (
            np.full((2, 2, 3), self.image_value, np.uint8),
            np.ones((2, 2), np.float32),
        )

    def close(self):
        pass


class Predictor:
    def __init__(self, events, scores):
        self.events = events
        self.scores = list(scores)
        self.calls = []

    def predict_batch(self, images, contexts, exposure_us_values, gains):
        self.events.append("predict_batch")
        self.calls.append((contexts[0], exposure_us_values[0], gains[0]))
        score = self.scores.pop(0)
        depth = np.full(images[0].shape[:2], len(self.calls), dtype=np.float32)
        return FullDepthBatchPrediction((score,), (depth,))

    def predict(self, *args, **kwargs):
        raise AssertionError("Nelder-Mead path must use predict_batch().")

    def predict_scores(self, *args, **kwargs):
        raise AssertionError("Nelder-Mead path must run the full depth head.")


class Logger:
    def __init__(self, events):
        self.events = events
        self.rows = []

    def record(self, frame, score, values):
        self.events.append("log")
        row = {
            "capture_role": frame.role,
            "output_delivered": int(frame.role == "initial"),
            "cell_id": frame.cell.cell_id,
            "q": score.q,
            "abs_rel": "",
            **values,
        }
        self.rows.append(row)
        return row

    def write(self):
        self.events.append("write")
        return Path("result.csv")


class Evaluator:
    def __init__(self, events):
        self.events = events

    def evaluate_rows(self, rows):
        self.events.append("evaluate")
        for row in rows:
            row["abs_rel"] = 0.1


def config(output_dir=Path("unused")):
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
    scores,
    safety=None,
    on_capture=None,
    *,
    image_value=0,
    brightness_guard_config=None,
):
    events = []
    temporary = TemporaryDirectory()
    provider = Provider()
    camera = Camera(events, provider, on_capture, image_value)
    predictor = Predictor(events, scores)
    logger = Logger(events)
    cfg = config(Path(temporary.name))
    policy = ContextualRiskNelderMeadPolicy(
        RiskNelderMeadConfig(simplex_tolerance=0.0),
        safety or SafetyPolicy(),
        cfg.default_cell,
    )
    runtime = RiskNelderMeadExperiment(
        cfg,
        CaptureRunner(camera, provider, cfg.max_pair_capture_gap_ms),
        predictor,
        policy,
        logger,
        Evaluator(events),
        brightness_guard_config,
    )
    runtime._test_temporary = temporary
    return runtime, events, provider, camera, predictor, policy, logger


class RiskNelderMeadExperimentTest(unittest.TestCase):
    def test_round_captures_and_predicts_once_then_observes_q(self):
        runtime, events, _, camera, predictor, policy, logger = make_experiment(
            [QScore(0.25, 9.0, 8.0)]
        )
        result = runtime.run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(policy.best_risk(ContextKey(0, 0)), 0.25)
        self.assertEqual(result.update_status, "updated")
        self.assertEqual(logger.rows[0]["capture_role"], "initial")
        self.assertEqual(logger.rows[0]["output_delivered"], 1)
        self.assertEqual(logger.rows[0]["selected"], 1)
        self.assertEqual(events, [
            "apply:E16_G064", "capture:E16_G064", "predict_batch", "log"
        ])

    def test_round_saves_full_resolution_raw_depth(self):
        runtime, _, _, _, _, _, _ = make_experiment([QScore(0.25, 0.1, 0.2)])
        runtime.run_round()
        paths = list((runtime.config.output_dir / "depth_pred_raw").glob("*.npy"))
        self.assertEqual(len(paths), 1)
        prediction = np.load(paths[0], allow_pickle=False)
        self.assertEqual(prediction.shape, (2, 2))
        self.assertEqual(prediction.dtype, np.float32)

    def test_capture_cells_always_follow_motion_light_safety(self):
        safety = SafetyPolicy(
            max_exposure_ms_by_motion=(8, 32, 32, 32, 32),
            allowed_gains_by_light=(
                (16, 32),
                (16, 32, 64, 128),
                (16, 32, 64, 128),
            ),
        )
        scores = [QScore(0.5 - index * 0.01, 0.1, 0.1) for index in range(10)]
        runtime, _, _, _, _, _, logger = make_experiment(scores, safety)
        for _ in range(10):
            runtime.run_round()
        safe_ids = {cell.cell_id for cell in safety.safe_cells(ContextKey(0, 0))}
        self.assertTrue(all(row["cell_id"] in safe_ids for row in logger.rows))

    def test_context_change_during_capture_does_not_update_wrong_simplex(self):
        changed = ContextKey(4, 2)

        def change(provider):
            provider.current = changed

        runtime, _, provider, _, predictor, policy, _ = make_experiment(
            [QScore(0.2, 0.1, 0.1), QScore(0.3, 0.1, 0.1)],
            on_capture=change,
        )
        first = runtime.run_round()
        self.assertEqual(first.update_status, "context_changed")
        self.assertIsNone(policy.best_risk(ContextKey(0, 0)))
        runtime.capture_runner.camera.on_capture = None
        second = runtime.run_round()
        self.assertEqual(second.update_status, "updated")
        self.assertIn(second.frame.cell, policy.safe_cells(changed))
        self.assertEqual(predictor.calls[0][0], changed)
        self.assertEqual(provider.current, changed)

    def test_finalize_evaluates_before_write(self):
        runtime, events, _, _, _, _, _ = make_experiment(
            [QScore(0.2, 0.1, 0.1)]
        )
        runtime.run_round()
        self.assertEqual(runtime.finalize(), Path("result.csv"))
        self.assertLess(events.index("evaluate"), events.index("write"))

    def test_log_mode_records_brightness_without_changing_nelder_mead_cells(self):
        scores = [QScore(0.3, 0.1, 0.1), QScore(0.2, 0.1, 0.1)]
        legacy = make_experiment(list(scores), image_value=255)[0]
        logged, _, _, _, _, _, logger = make_experiment(
            list(scores),
            image_value=255,
            brightness_guard_config=BrightnessGuardConfig(
                mode=BrightnessGuardMode.LOG
            ),
        )
        legacy_cells = [legacy.run_round().frame.cell for _ in range(2)]
        logged_cells = [logged.run_round().frame.cell for _ in range(2)]
        self.assertEqual(logged_cells, legacy_cells)
        self.assertEqual(logger.rows[0]["brightness_state"], "severe_over")

    def test_enforce_mode_schedules_severe_over_recovery_next_round(self):
        runtime, _, _, _, _, _, logger = make_experiment(
            [QScore(0.3, 0.1, 0.1), QScore(9.0, 8.0, 1.0)],
            image_value=255,
            brightness_guard_config=BrightnessGuardConfig(
                mode=BrightnessGuardMode.ENFORCE
            ),
        )
        first = runtime.run_round()
        second = runtime.run_round()
        self.assertEqual(first.frame.cell, SensorCell(16, 64))
        self.assertEqual(second.frame.cell, SensorCell(8, 64))
        self.assertEqual(second.operation, "brightness_recovery")
        self.assertEqual(logger.rows[0]["brightness_force_recovery"], 1)


class RiskNelderMeadEntrypointTest(unittest.TestCase):
    def test_cli_uses_default_safety_envelope_and_validates_simplex(self):
        args = entrypoint.parse_args(["--camera-error-checkpoint", "checkpoint.pt"])
        self.assertEqual(args.safety_config, entrypoint.DEFAULT_SAFETY_CONFIG)
        self.assertEqual(args.simplex_restart_frames, 60)
        self.assertEqual(args.simplex_tolerance, 0.02)
        with self.assertRaises(SystemExit):
            entrypoint.parse_args([
                "--camera-error-checkpoint", "checkpoint.pt",
                "--simplex-restart-frames", "2",
            ])

    def test_bundled_safety_envelope_constrains_all_context_initial_cells(self):
        args = entrypoint.parse_args(["--camera-error-checkpoint", "checkpoint.pt"])
        safety = SafetyPolicy.from_json(args.safety_config)
        policy = ContextualRiskNelderMeadPolicy(
            RiskNelderMeadConfig(),
            safety,
            SensorCell(args.initial_exposure_ms, args.initial_gain),
        )
        for motion_state in range(5):
            for light_state in range(3):
                context = ContextKey(motion_state, light_state)
                self.assertIn(policy.next_cell(context), safety.safe_cells(context))

    def test_builder_uses_camera_predictor_safety_and_fair_evaluator(self):
        calls = {}

        class FakeCamera:
            def __init__(self, **kwargs):
                pass

        class FakePredictor:
            def __init__(self, *args):
                calls["predictor"] = self
                calls["predictor_args"] = args

        class FakeEvaluator:
            def __init__(self, predictor, cfg, precision):
                calls["evaluator"] = (predictor, cfg, precision)

        class FakeRuntime:
            def __init__(self, *args):
                self.args = args

        predictor_module = ModuleType("ati_mde_control.full_depth_predictor")
        predictor_module.CameraErrorFullDepthPredictor = FakePredictor
        sensor_module = ModuleType("hardware.sensor")
        sensor_module.OrbbecColorCamera = FakeCamera
        with TemporaryDirectory() as temporary:
            args = entrypoint.parse_args([
                "--camera-error-checkpoint", "checkpoint.pt",
                "--output-dir", temporary,
                "--q-uncertainty-weight", "1.645",
            ])
            with patch.dict(sys.modules, {
                "ati_mde_control.full_depth_predictor": predictor_module,
                "hardware.sensor": sensor_module,
            }), patch.object(
                entrypoint, "build_context_provider", return_value=object()
            ), patch.object(
                entrypoint, "CaptureRunner", return_value=object()
            ), patch.object(
                entrypoint, "CaptureLogger", return_value=object()
            ), patch.object(
                entrypoint, "FairDepthEvaluator", FakeEvaluator
            ), patch.object(
                entrypoint, "RiskNelderMeadExperiment", FakeRuntime
            ):
                runtime = entrypoint.build_experiment(args)
        predictor, cfg, precision = calls["evaluator"]
        self.assertIs(predictor, calls["predictor"])
        self.assertEqual(calls["predictor_args"][4], 1.645)
        self.assertEqual(cfg.safety_path, entrypoint.DEFAULT_SAFETY_CONFIG)
        self.assertEqual(precision, "fp32")
        self.assertIsInstance(runtime.args[3], ContextualRiskNelderMeadPolicy)


if __name__ == "__main__":
    unittest.main()
