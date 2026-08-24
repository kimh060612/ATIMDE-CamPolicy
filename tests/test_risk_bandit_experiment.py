import math
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
from ati_mde_control.logging import CaptureLogger as RealCaptureLogger
from ati_mde_control.risk_bandit_experiment import RiskBanditExperiment
from ati_mde_control.risk_bandit_policy import (
    GPNumericalError,
    RiskBanditConfig,
    RiskBanditPolicy,
)
from hardware.utils import ContextKey, QScore, SensorCell
import orbbec_ati_risk_bandit as entrypoint


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
        effective=True,
        on_capture=None,
        fail_apply_on=None,
        fail_capture_on=None,
        image_value=None,
    ) -> None:
        self.events = events
        self.provider = provider
        self.effective = effective
        self.on_capture = on_capture
        self.fail_apply_on = fail_apply_on
        self.fail_capture_on = fail_capture_on
        self.image_value = image_value
        self.active_cell = None
        self.capture_count = 0
        self.apply_count = 0
        self.color_frame_number = 0
        self.depth_frame_number = 0
        self.color_timestamp_us = 0
        self.depth_timestamp_us = 0
        self.setting_effective = True
        self.sensor_settle_ms = 0.0

    def apply_cell(self, cell):
        self.apply_count += 1
        if self.apply_count == self.fail_apply_on:
            raise RuntimeError("synthetic camera apply failure")
        self.active_cell = cell
        self.sensor_settle_ms = 7.0
        self.events.append(f"apply:{cell.cell_id}")
        raw = cell.exposure_ms * 1000
        return raw, raw, cell.gain

    def capture_rgbd(self):
        self.capture_count += 1
        if self.capture_count == self.fail_capture_on:
            raise TimeoutError("synthetic capture failure")
        self.events.append(f"capture:{self.active_cell.cell_id}")
        if self.on_capture is not None:
            self.on_capture(self.capture_count, self.provider)
        self.color_frame_number = self.depth_frame_number = self.capture_count
        self.color_timestamp_us = self.depth_timestamp_us = self.capture_count * 1000
        self.setting_effective = (
            self.effective[self.capture_count - 1]
            if isinstance(self.effective, list)
            else self.effective
        )
        value = self.capture_count if self.image_value is None else self.image_value
        image = np.full((2, 2, 3), value, dtype=np.uint8)
        return image, np.ones((2, 2), dtype=np.float32)

    def close(self):
        pass


class Predictor:
    device = "cpu"

    def __init__(self, events, scores=None) -> None:
        self.events = events
        self.scores = list(scores or [QScore(0.4, 0.2, 0.2, {"mde_inference_ms": 1.0})])
        self.calls = 0

    def predict_batch(self, images, contexts, exposure_us_values, gains):
        self.calls += 1
        self.events.append("predict_start")
        score = self.scores.pop(0)
        if isinstance(score, BaseException):
            raise score
        self.events.append("predict_complete")
        depth = np.full(images[0].shape[:2], self.calls, dtype=np.float32)
        return FullDepthBatchPrediction((score,), (depth,))

    def predict(self, *args, **kwargs):
        raise AssertionError("Risk bandit must use predict_batch().")

    def predict_scores(self, *args, **kwargs):
        raise AssertionError("Risk bandit must run the full depth head.")


class RecordingPolicy(RiskBanditPolicy):
    def __init__(self, *args, events, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events
        self.selection_history_lengths = []

    def select_action(self, *args, **kwargs):
        self.selection_history_lengths.append(len(self.history))
        return super().select_action(*args, **kwargs)

    def add_observation(self, *args, **kwargs):
        result = super().add_observation(*args, **kwargs)
        self.events.append("gp_update")
        return result


class Logger:
    def __init__(self, events) -> None:
        self.events = events
        self.rows = []

    def record(self, frame, score, values):
        self.events.append("logging")
        row = {
            "round_index": frame.round_index,
            "capture_role": frame.role,
            "output_delivered": int(frame.role == "initial"),
            "cell_id": frame.cell.cell_id,
            "camera_bias": "" if score is None else score.mu,
            "std": "" if score is None else score.uncertainty,
            "q": "" if score is None else score.q,
            "requested_exposure_raw": frame.requested_exposure_raw,
            "actual_exposure_raw": frame.actual_exposure_raw,
            "actual_gain": frame.actual_gain,
            "camera_parameter_ms": frame.camera_parameter_ms,
            "sensor_settle_ms": frame.sensor_settle_ms,
            "abs_rel": "",
            **values,
        }
        self.rows.append(row)
        return row

    def write(self):
        self.events.append("logger_write")
        return Path("result.csv")


class Evaluator:
    def __init__(self, events) -> None:
        self.events = events

    def evaluate_rows(self, rows):
        self.events.append("evaluate")
        for row in rows:
            row["abs_rel"] = 0.1


def experiment_config(output_dir=Path("unused")) -> ExperimentConfig:
    return ExperimentConfig(
        checkpoint_path=Path("unused"),
        output_dir=output_dir,
        model_size="small",
        device="cpu",
        precision="fp32",
        local_files_only=True,
        q_uncertainty_weight=1.0,
        policy=PolicyConfig(),
        safety_path=None,
        default_cell=SensorCell(16, 64),
        max_rounds=10,
        round_interval_ms=0,
        max_pair_capture_gap_ms=100.0,
        camera_parameter_warn_ms=1000,
        mde_inference_warn_ms=1000,
        control_decision_warn_ms=1000,
        evaluation_alignment="scale_shift_inverse",
        min_depth_m=0.2,
        max_depth_m=10,
        min_valid_depth_pixels=1,
    )


def make_experiment(
    *, scores=None, safety=None, camera_options=None, brightness_guard_config=None
):
    events = []
    temporary = TemporaryDirectory()
    provider = Provider()
    camera = Camera(events, provider, **(camera_options or {}))
    predictor = Predictor(events, scores)
    logger = Logger(events)
    config = experiment_config(Path(temporary.name))
    policy = RecordingPolicy(
        RiskBanditConfig(),
        safety or SafetyPolicy(),
        config.default_cell,
        events=events,
    )
    runtime = RiskBanditExperiment(
        config,
        CaptureRunner(camera, provider, config.max_pair_capture_gap_ms),
        predictor,
        policy,
        logger,
        Evaluator(events),
        brightness_guard_config,
    )
    runtime._test_temporary = temporary
    return runtime, events, provider, camera, predictor, policy, logger


class RiskBanditExperimentTest(unittest.TestCase):
    def track(self, runtime):
        self.addCleanup(runtime._test_temporary.cleanup)
        return runtime

    def test_one_capture_one_prediction_and_no_challenger(self) -> None:
        runtime, events, _, camera, predictor, _, logger = make_experiment()
        self.track(runtime).run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(len(logger.rows), 1)
        self.assertEqual(logger.rows[0]["capture_role"], "initial")
        self.assertEqual(logger.rows[0]["output_delivered"], 1)
        self.assertEqual(logger.rows[0]["selected"], 1)
        self.assertFalse(any("challenger" in str(value) for value in logger.rows[0].values()))
        self.assertEqual(sum(item.startswith("capture:") for item in events), 1)

    def test_capture_pair_is_never_called(self) -> None:
        runtime, _, _, _, _, _, _ = make_experiment()
        runtime.capture_runner.capture_pair = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("capture_pair must not be called")
        )
        self.track(runtime).run_round()

    def test_surrogate_scores_all_cells_without_more_predictor_calls(self) -> None:
        runtime, _, _, _, predictor, policy, _ = make_experiment(
            scores=[
                QScore(0.4, 0.2, 0.2),
                QScore(0.3, 0.1, 0.2),
            ]
        )
        self.track(runtime)
        runtime.run_round()
        second = runtime.run_round()
        self.assertEqual(predictor.calls, 2)
        self.assertEqual(len(second.decision.candidates), 16)
        self.assertEqual(policy.selection_history_lengths, [0, 1])

    def test_gp_uses_capture_context_even_when_context_changes_and_is_unstable(self) -> None:
        changed = ContextKey(1, 2)

        def change_context(_, provider):
            provider.current = changed
            provider.is_stable = False

        runtime, _, _, _, _, policy, _ = make_experiment(
            camera_options={"on_capture": change_context}
        )
        result = self.track(runtime).run_round()
        self.assertEqual(result.gp_update_status, "updated")
        self.assertEqual(policy.history[0].context, changed)
        self.assertEqual(policy.history[0].cell, result.frame.cell)

    def test_ineffective_and_non_finite_frames_do_not_update_gp(self) -> None:
        runtime, _, _, _, _, policy, logger = make_experiment(
            camera_options={"effective": [False, True]},
            scores=[QScore(0.4, 0.2, 0.2), QScore(math.nan, 0.2, 0.2)],
        )
        self.track(runtime)
        first = runtime.run_round()
        second = runtime.run_round()
        self.assertEqual(first.gp_update_status, "setting_ineffective")
        self.assertEqual(second.gp_update_status, "non_finite_score")
        self.assertEqual(policy.history, ())
        self.assertEqual(len(logger.rows), 2)

    def test_pre_applied_metadata_is_preserved_on_next_capture(self) -> None:
        runtime, _, _, _, _, _, logger = make_experiment()
        self.track(runtime).run_round()
        row = logger.rows[0]
        self.assertEqual(row["requested_exposure_raw"], 16_000)
        self.assertEqual(row["actual_exposure_raw"], 16_000)
        self.assertEqual(row["actual_gain"], 64)
        self.assertEqual(row["sensor_settle_ms"], 7.0)
        self.assertGreaterEqual(row["camera_parameter_ms"], 0.0)

    def test_real_capture_logger_has_primary_only_semantics(self) -> None:
        runtime, _, _, _, _, _, _ = make_experiment()
        self.track(runtime)
        with TemporaryDirectory() as temporary:
            runtime.logger = RealCaptureLogger(Path(temporary))
            runtime.run_round()
            self.assertEqual(len(runtime.logger.rows), 1)
            row = runtime.logger.rows[0]
            self.assertEqual(row["capture_role"], "initial")
            self.assertEqual(row["output_delivered"], 1)
            self.assertEqual(row["selected"], 1)
            self.assertEqual(row["pair_status"], "not_probed")
            self.assertEqual(row["active_cell_before"], row["cell_id"])
            self.assertEqual(runtime.logger._summary()["probe_overhead"], 0.0)

    def test_full_depth_inference_is_synchronous_before_next_apply(self) -> None:
        runtime, events, _, camera, _, _, _ = make_experiment()
        original_apply = camera.apply_cell

        def assert_prediction_finished(cell):
            if camera.apply_count:
                self.assertIn("predict_start", events)
                self.assertIn("predict_complete", events)
                events.append("post_inference_apply")
            return original_apply(cell)

        camera.apply_cell = assert_prediction_finished
        self.track(runtime).run_round()
        ordered = [
            next(index for index, event in enumerate(events) if event.startswith("capture:")),
            events.index("predict_start"),
            events.index("predict_complete"),
            events.index("post_inference_apply"),
            events.index("gp_update"),
            events.index("logging"),
        ]
        self.assertEqual(ordered, sorted(ordered))

    def test_no_background_executor_or_threading_is_used(self) -> None:
        runtime, _, _, _, predictor, _, _ = make_experiment(
            scores=[QScore(0.4, 0.2, 0.2), QScore(0.3, 0.1, 0.2)]
        )
        self.track(runtime)
        runtime.run_round()
        runtime.run_round()
        self.assertEqual(predictor.calls, 2)
        source = Path("ati_mde_control/risk_bandit_experiment.py").read_text()
        self.assertNotIn("ThreadPoolExecutor", source)
        self.assertNotIn("import threading", source)

    def test_predictor_failure_logs_frame_without_gp_update(self) -> None:
        runtime, _, _, camera, _, policy, logger = make_experiment(
            scores=[RuntimeError("synthetic predictor failure")]
        )
        with self.assertRaisesRegex(RuntimeError, "risk prediction failed"):
            runtime.run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(policy.history, ())
        self.assertEqual(len(logger.rows), 1)

    def test_capture_and_camera_failures_do_not_add_extra_capture(self) -> None:
        runtime, _, _, camera, predictor, _, logger = make_experiment(
            camera_options={"fail_capture_on": 1}
        )
        with self.assertRaisesRegex(TimeoutError, "synthetic capture failure"):
            runtime.run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(logger.rows, [])

        runtime, _, _, camera, predictor, _, logger = make_experiment(
            camera_options={"fail_apply_on": 2}
        )
        with self.assertRaisesRegex(RuntimeError, "camera apply failed"):
            runtime.run_round()
        self.assertEqual(camera.capture_count, 1)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(len(logger.rows), 1)

    def test_keyboard_interrupt_propagates_without_background_work(self) -> None:
        runtime, _, _, _, _, _, _ = make_experiment(scores=[KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            runtime.run_round()

    def test_finalize_evaluates_then_writes(self) -> None:
        runtime, events, _, _, _, _, _ = make_experiment()
        runtime.run_round()
        path = runtime.finalize()
        self.assertEqual(path, Path("result.csv"))
        self.assertLess(events.index("evaluate"), events.index("logger_write"))

    def test_each_capture_saves_one_full_resolution_raw_depth(self) -> None:
        runtime, _, _, _, _, _, _ = make_experiment()
        self.track(runtime).run_round()
        paths = list((runtime.config.output_dir / "depth_pred_raw").glob("*.npy"))
        self.assertEqual(len(paths), 1)
        prediction = np.load(paths[0], allow_pickle=False)
        self.assertEqual(prediction.shape, (2, 2))
        self.assertEqual(prediction.dtype, np.float32)

    def test_newly_unsafe_cell_is_replaced_before_next_capture(self) -> None:
        safety = SafetyPolicy(max_exposure_ms_by_motion=(32, 8, 32, 32, 32))
        runtime, events, provider, _, _, _, _ = make_experiment(
            safety=safety,
            scores=[QScore(0.4, 0.2, 0.2), QScore(0.3, 0.1, 0.2)],
        )
        self.track(runtime)
        runtime.run_round()
        provider.current = ContextKey(1, 0)
        second = runtime.run_round()
        self.assertIn(second.frame.cell, safety.safe_cells(ContextKey(1, 0)))
        second_capture_index = [
            index for index, item in enumerate(events) if item.startswith("capture:")
        ][1]
        self.assertTrue(events[second_capture_index - 1].startswith("apply:"))

    def test_gp_numerical_failure_uses_safe_fallback_without_extra_capture(self) -> None:
        runtime, _, provider, camera, _, policy, _ = make_experiment(
            scores=[QScore(0.4, 0.2, 0.2), QScore(0.3, 0.1, 0.2)]
        )
        self.track(runtime)
        runtime.run_round()
        captures_before = camera.capture_count
        with patch.object(
            policy,
            "posterior_predictions",
            side_effect=GPNumericalError("synthetic"),
        ):
            result = runtime.run_round()
        self.assertEqual(camera.capture_count, captures_before + 1)
        self.assertEqual(result.decision.status, "gp_numerical_fallback")
        self.assertIn(
            result.decision.selected_cell,
            policy.safety_policy.safe_cells(provider.current),
        )

    def test_log_mode_records_brightness_without_changing_legacy_decision(self) -> None:
        scores = [QScore(0.4, 0.2, 0.2), QScore(0.3, 0.1, 0.2)]
        legacy = make_experiment(
            scores=list(scores), camera_options={"image_value": 255}
        )[0]
        runtime, _, _, _, _, _, logger = make_experiment(
            scores=list(scores),
            camera_options={"image_value": 255},
            brightness_guard_config=BrightnessGuardConfig(
                mode=BrightnessGuardMode.LOG
            ),
        )
        self.track(legacy)
        self.track(runtime)
        with patch(
            "ati_mde_control.risk_bandit_experiment.time.time_ns",
            return_value=1_000_000_000,
        ):
            legacy_decisions = [legacy.run_round().decision for _ in range(2)]
            logged_decisions = [runtime.run_round().decision for _ in range(2)]
        self.assertEqual(
            [item.selected_cell for item in logged_decisions],
            [item.selected_cell for item in legacy_decisions],
        )
        self.assertEqual(
            [item.status for item in logged_decisions],
            [item.status for item in legacy_decisions],
        )
        self.assertEqual(logger.rows[0]["brightness_state"], "severe_over")
        self.assertEqual(logger.rows[0]["brightness_guard_mode"], "log")

    def test_enforce_mode_applies_severe_over_recovery_before_learned_score(self) -> None:
        runtime, _, _, camera, _, _, logger = make_experiment(
            camera_options={"image_value": 255},
            brightness_guard_config=BrightnessGuardConfig(
                mode=BrightnessGuardMode.ENFORCE
            ),
        )
        result = self.track(runtime).run_round()
        self.assertEqual(result.decision.status, "brightness_recovery")
        self.assertEqual(result.decision.selected_cell, SensorCell(8, 64))
        self.assertEqual(camera.active_cell, SensorCell(8, 64))
        self.assertEqual(logger.rows[0]["brightness_force_recovery"], 1)


class RiskBanditEntrypointTest(unittest.TestCase):
    def test_cli_defaults_and_metric_alignment_rejection(self) -> None:
        args = entrypoint.parse_args(["--camera-error-checkpoint", "checkpoint.pt"])
        self.assertEqual(args.safety_config, entrypoint.DEFAULT_SAFETY_CONFIG)
        self.assertEqual(args.bandit_window_size, 48)
        self.assertEqual(args.bandit_exploration_beta, 1.0)
        self.assertEqual(args.bandit_switch_penalty, 0.005)
        self.assertEqual(args.evaluation_precision, "fp32")
        with self.assertRaises(SystemExit):
            entrypoint.parse_args([
                "--camera-error-checkpoint", "checkpoint.pt",
                "--evaluation-alignment", "metric",
            ])

    def test_default_safety_envelope_limits_every_motion_light_context(self) -> None:
        args = entrypoint.parse_args(["--camera-error-checkpoint", "checkpoint.pt"])
        safety = SafetyPolicy.from_json(args.safety_config)
        policy = RiskBanditPolicy(
            RiskBanditConfig(), safety, SensorCell(16, 64)
        )
        expected_exposure_limits = (16, 32, 32, 16, 8)
        expected_gains = (
            {16, 32, 64},
            {16, 32, 64},
            {16, 32, 64, 128},
        )
        for motion_state, exposure_limit in enumerate(expected_exposure_limits):
            for light_state, allowed_gains in enumerate(expected_gains):
                context = ContextKey(motion_state, light_state)
                safe = policy.safe_cells(context)
                self.assertTrue(safe)
                self.assertTrue(
                    all(cell.exposure_ms <= exposure_limit for cell in safe)
                )
                self.assertEqual({cell.gain for cell in safe}, allowed_gains)
                selected = policy.select_action(
                    context, SensorCell(32, 128), 0
                ).selected_cell
                self.assertIn(selected, safe)

    def test_explicit_safety_config_overrides_default(self) -> None:
        custom = Path("/tmp/custom_safety_envelope.json")
        args = entrypoint.parse_args([
            "--camera-error-checkpoint", "checkpoint.pt",
            "--safety-config", str(custom),
        ])
        self.assertEqual(args.safety_config, custom)

    def test_cli_rejects_invalid_bandit_values(self) -> None:
        invalid_options = (
            ("--bandit-window-size", "0"),
            ("--bandit-exploration-beta", "nan"),
            ("--bandit-switch-penalty", "-0.1"),
            ("--bandit-exposure-length-scale", "0"),
            ("--bandit-motion-cross-correlation", "1"),
            ("--bandit-jitter", "inf"),
        )
        for name, value in invalid_options:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                entrypoint.parse_args([
                    "--camera-error-checkpoint", "checkpoint.pt", name, value
                ])

    def test_builder_constructs_the_shared_fair_depth_evaluator(self) -> None:
        calls = {}

        class FakeCamera:
            def __init__(self, **kwargs):
                calls["camera_kwargs"] = kwargs

        class FakePredictor:
            def __init__(self, *args):
                calls["predictor_args"] = args

        class FakeEvaluator:
            def __init__(self, predictor, config, precision):
                calls["evaluator_args"] = (predictor, config, precision)

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
                "--evaluation-precision", "fp32",
            ])
            provider = object()
            logger = object()
            runner = object()
            with patch.dict(
                sys.modules,
                {
                    "ati_mde_control.full_depth_predictor": predictor_module,
                    "hardware.sensor": sensor_module,
                },
            ), patch.object(
                entrypoint, "build_context_provider", return_value=provider
            ), patch.object(
                entrypoint, "CaptureLogger", return_value=logger
            ), patch.object(
                entrypoint, "CaptureRunner", return_value=runner
            ), patch.object(
                entrypoint, "FairDepthEvaluator", FakeEvaluator
            ), patch.object(
                entrypoint, "RiskBanditExperiment", FakeRuntime
            ):
                runtime = entrypoint.build_experiment(args)

        predictor, config, precision = calls["evaluator_args"]
        self.assertIs(predictor, runtime.args[2])
        self.assertIs(config, runtime.args[0])
        self.assertEqual(precision, "fp32")
        self.assertIs(runtime.args[1], runner)
        self.assertIs(runtime.args[4], logger)
        self.assertIsInstance(runtime.args[5], FakeEvaluator)

    def test_new_sources_have_no_iqa_or_nelder_mead_import(self) -> None:
        paths = (
            Path("ati_mde_control/risk_bandit_policy.py"),
            Path("ati_mde_control/risk_bandit_experiment.py"),
            Path("orbbec_ati_risk_bandit.py"),
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            import_lines = [
                line.lower() for line in source.splitlines()
                if line.startswith("import ") or line.startswith("from ")
            ]
            self.assertFalse(any("iqa" in line or "nelder" in line for line in import_lines))


if __name__ == "__main__":
    unittest.main()
