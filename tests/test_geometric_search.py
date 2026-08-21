import unittest
from pathlib import Path

import numpy as np

from ati_mde_control.capture_runner import CaptureRunner
from ati_mde_control.config import ExperimentConfig, PolicyConfig, SafetyPolicy
from ati_mde_control.geometric_search import (
    GeometricProposalKind,
    symmetric_axis_candidates,
)
from ati_mde_control.geometric_search_policy import GeometricSearchPolicy
from ati_mde_control.geometric_search_update_experiment import (
    GeometricSearchUpdateExperiment,
)
from ati_mde_control.types import PairStatus
from hardware.utils import ContextKey, QScore, SensorCell


def score(mu: float, std: float = 0.01) -> QScore:
    return QScore(
        mu + std,
        std,
        mu,
        {"mde_batch_size": 1, "mde_inference_ms": 1.0},
    )


class GeometricSearchPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ContextKey(0, 0)
        self.policy = GeometricSearchPolicy(PolicyConfig(), SafetyPolicy())
        self.current = self.policy.committed_cell(
            self.context, SensorCell(16, 64)
        )

    def test_symmetric_candidates_ignore_feasible_span(self) -> None:
        context = ContextKey(2, 1)
        safe = SafetyPolicy().safe_cells(context)
        current = SensorCell(8, 64)

        even = symmetric_axis_candidates(
            current, "exposure", safe, {current.cell_id}, episode_id=0
        )
        odd = symmetric_axis_candidates(
            current, "exposure", safe, {current.cell_id}, episode_id=1
        )

        self.assertEqual(even, [SensorCell(16, 64), SensorCell(4, 64)])
        self.assertEqual(odd, [SensorCell(4, 64), SensorCell(16, 64)])

    def test_symmetric_gain_candidates_alternate_by_episode(self) -> None:
        context = ContextKey(0, 0)
        safety = SafetyPolicy(
            max_exposure_ms_by_motion=(16, 32, 32, 16, 8),
            allowed_gains_by_light=(
                (16, 32, 64),
                (16, 32, 64, 128),
                (16, 32, 64, 128),
            ),
        )
        safe = safety.safe_cells(context)
        current = SensorCell(8, 32)

        even = symmetric_axis_candidates(
            current, "gain", safe, {current.cell_id}, episode_id=0
        )
        odd = symmetric_axis_candidates(
            current, "gain", safe, {current.cell_id}, episode_id=1
        )

        self.assertEqual(even, [SensorCell(8, 64), SensorCell(8, 16)])
        self.assertEqual(odd, [SensorCell(8, 16), SensorCell(8, 64)])

    def test_episode_initialization_alternates_an_equal_exposure_tie(self) -> None:
        context = ContextKey(0, 1)
        safety = SafetyPolicy(
            max_exposure_ms_by_motion=(16, 32, 32, 16, 8)
        )
        policy = GeometricSearchPolicy(PolicyConfig(), safety)
        current = policy.committed_cell(context, SensorCell(8, 64))

        first = policy.proposal_after_current(context, current, score(0.30))
        self.assertIn(first.cell_id, policy.state(context).tested_cell_ids)
        self.assertNotIn(
            SensorCell(4, 64).cell_id, policy.state(context).tested_cell_ids
        )
        policy.resolve(
            context, current, score(0.30), first, score(0.40), round_index=0
        )
        self.assertIsNone(
            policy.proposal_after_current(context, current, score(0.01))
        )
        second = policy.proposal_after_current(context, current, score(0.30))

        self.assertEqual(first, SensorCell(16, 64))
        self.assertEqual(second, SensorCell(4, 64))
        self.assertIn(second.cell_id, policy.state(context).tested_cell_ids)
        self.assertEqual(policy.state(context).episode_id, 2)

    def test_symmetric_candidates_filter_unsafe_and_evaluated_cells(self) -> None:
        current = SensorCell(8, 64)
        safe = {
            current,
            SensorCell(4, 64),
            # E16 is deliberately unsafe; E32 is not an adjacent candidate.
            SensorCell(32, 64),
        }

        candidates = symmetric_axis_candidates(
            current, "exposure", safe, {current.cell_id}, episode_id=0
        )

        self.assertEqual(candidates, [SensorCell(4, 64)])

        evaluated = symmetric_axis_candidates(
            current,
            "exposure",
            safe | {SensorCell(16, 64)},
            {current.cell_id, SensorCell(16, 64).cell_id},
            episode_id=0,
        )
        self.assertEqual(evaluated, [SensorCell(4, 64)])

    def _build_simplex(self):
        # Keep these reflection/expansion tests on the historical negative
        # geometry; neutral initialization itself is covered above for both
        # episode parities.
        self.policy.state(self.context).episode_id = 1
        exposure = self.policy.proposal_after_current(
            self.context, self.current, score(0.30)
        )
        self.assertEqual(exposure, SensorCell(8, 64))
        first = self.policy.resolve(
            self.context,
            self.current,
            score(0.30),
            exposure,
            score(0.20),
            0,
        )
        self.assertEqual(first.status, PairStatus.CHALLENGER_WON)

        gain = self.policy.proposal_after_current(
            self.context, exposure, score(0.20)
        )
        self.assertEqual(gain, SensorCell(16, 32))
        second = self.policy.resolve(
            self.context,
            exposure,
            score(0.20),
            gain,
            score(0.25),
            1,
        )
        self.assertEqual(second.status, PairStatus.CURRENT_WON)
        return exposure

    def test_initialization_uses_two_axis_neighbors_one_at_a_time(self) -> None:
        active = self._build_simplex()
        state = self.policy.state(self.context)

        self.assertEqual(len(state.vertices), 3)
        self.assertEqual(
            {vertex.cell for vertex in state.vertices},
            {SensorCell(16, 64), SensorCell(8, 64), SensorCell(16, 32)},
        )
        reflection = self.policy.proposal_after_current(
            self.context, active, score(0.20)
        )
        self.assertEqual(reflection, SensorCell(8, 32))
        self.assertEqual(
            state.pending_proposal.kind, GeometricProposalKind.REFLECTION
        )

    def test_reflection_win_defers_expansion_to_next_opportunity(self) -> None:
        active = self._build_simplex()
        reflection = self.policy.proposal_after_current(
            self.context, active, score(0.20)
        )
        decision = self.policy.resolve(
            self.context,
            active,
            score(0.20),
            reflection,
            score(0.15),
            2,
        )

        state = self.policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CHALLENGER_WON)
        self.assertIsNone(state.pending_proposal)
        self.assertEqual(state.deferred_kind, GeometricProposalKind.EXPANSION)

        expansion = self.policy.proposal_after_current(
            self.context, reflection, score(0.15)
        )
        self.assertEqual(expansion, SensorCell(4, 16))
        self.assertEqual(
            state.pending_proposal.kind, GeometricProposalKind.EXPANSION
        )

    def test_reflection_reject_and_ambiguity_defer_contraction(self) -> None:
        for challenger_mu, expected_status in (
            (0.30, PairStatus.CURRENT_WON),
            (0.195, PairStatus.AMBIGUOUS),
        ):
            with self.subTest(status=expected_status):
                policy = GeometricSearchPolicy(PolicyConfig(), SafetyPolicy())
                self.policy = policy
                self.current = policy.committed_cell(
                    self.context, SensorCell(16, 64)
                )
                active = self._build_simplex()
                reflection = policy.proposal_after_current(
                    self.context, active, score(0.20)
                )
                decision = policy.resolve(
                    self.context,
                    active,
                    score(0.20),
                    reflection,
                    score(challenger_mu),
                    2,
                )
                state = policy.state(self.context)
                self.assertEqual(decision.status, expected_status)
                self.assertEqual(
                    state.deferred_kind, GeometricProposalKind.CONTRACTION
                )
                self.assertIsNone(state.pending_edge_from_id)
                contraction = policy.proposal_after_current(
                    self.context, active, score(0.20)
                )
                self.assertIsNotNone(contraction)
                self.assertEqual(
                    state.pending_proposal.kind,
                    GeometricProposalKind.CONTRACTION,
                )

    def test_pairwise_margin_uses_combined_uncertainty(self) -> None:
        challenger = self.policy.proposal_after_current(
            self.context, self.current, score(0.20, 0.03)
        )

        decision = self.policy.resolve(
            self.context,
            self.current,
            score(0.20, 0.03),
            challenger,
            score(0.18, 0.04),
            0,
        )

        self.assertAlmostEqual(decision.delta_mu, 0.02)
        self.assertAlmostEqual(decision.pair_std, 0.05)
        self.assertAlmostEqual(decision.effective_margin, 0.0225)
        self.assertEqual(decision.status, PairStatus.AMBIGUOUS)
        self.assertEqual(
            self.policy.state(self.context).pending_edge_from_id, None
        )

    def test_contraction_failure_resets_all_episode_observations(self) -> None:
        active = self._build_simplex()
        reflection = self.policy.proposal_after_current(
            self.context, active, score(0.20)
        )
        self.policy.resolve(
            self.context,
            active,
            score(0.20),
            reflection,
            score(0.30),
            2,
        )
        contraction = self.policy.proposal_after_current(
            self.context, active, score(0.20)
        )
        decision = self.policy.resolve(
            self.context,
            active,
            score(0.20),
            contraction,
            score(0.30),
            3,
        )

        state = self.policy.state(self.context)
        self.assertEqual(decision.status, PairStatus.CURRENT_WON)
        self.assertFalse(state.episode_active)
        self.assertEqual(state.vertices, [])
        self.assertEqual(state.tested_cell_ids, set())
        self.assertEqual(state.last_geometric_reason, "contraction_failure")

    def test_good_enough_current_score_ends_episode_without_a_probe(self) -> None:
        first = self.policy.proposal_after_current(
            self.context, self.current, score(0.30)
        )
        self.policy.resolve(
            self.context,
            self.current,
            score(0.30),
            first,
            score(0.20),
            0,
        )

        challenger = self.policy.proposal_after_current(
            self.context, first, score(0.11)
        )

        state = self.policy.state(self.context)
        self.assertIsNone(challenger)
        self.assertFalse(state.episode_active)
        self.assertEqual(state.vertices, [])
        self.assertEqual(state.last_geometric_reason, "good_enough_risk")

    def test_existing_periodic_interval_can_reprobe_good_enough_risk(self) -> None:
        policy = GeometricSearchPolicy(
            PolicyConfig(periodic_reprobe_interval=2), SafetyPolicy()
        )
        current = policy.committed_cell(self.context, SensorCell(16, 64))

        self.assertIsNone(
            policy.proposal_after_current(self.context, current, score(0.05))
        )
        self.assertIsNotNone(
            policy.proposal_after_current(self.context, current, score(0.05))
        )

    def test_context_change_discards_previous_episode_simplex(self) -> None:
        self.policy.on_context(self.context)
        self.assertIsNotNone(
            self.policy.proposal_after_current(
                self.context, self.current, score(0.30)
            )
        )

        self.policy.on_context(ContextKey(1, 0))

        old = self.policy.state(self.context)
        self.assertFalse(old.episode_active)
        self.assertEqual(old.vertices, [])
        self.assertEqual(old.tested_cell_ids, set())

    def test_every_projected_proposal_is_safe_and_untested(self) -> None:
        disabled = {"0,0": frozenset({SensorCell(8, 64).cell_id})}
        safety = SafetyPolicy(disabled_cells_by_context=disabled)
        policy = GeometricSearchPolicy(PolicyConfig(), safety)
        current = policy.committed_cell(self.context, SensorCell(16, 64))

        challenger = policy.proposal_after_current(
            self.context, current, score(0.30)
        )

        state = policy.state(self.context)
        self.assertEqual(challenger, SensorCell(32, 64))
        self.assertIn(challenger, safety.safe_cells(self.context))
        self.assertEqual(
            len(state.tested_cell_ids), len(set(state.tested_cell_ids))
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

    def __init__(self, events, provider) -> None:
        self.events = events
        self.provider = provider
        self.active_cell = None
        self.capture_count = 0
        self.color_timestamp_us = 0
        self.depth_timestamp_us = 0
        self.color_frame_number = 0
        self.depth_frame_number = 0
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
        self.color_timestamp_us += 1000
        self.depth_timestamp_us = self.color_timestamp_us
        self.color_frame_number = self.depth_frame_number = self.capture_count
        image = np.full((2, 2, 3), self.capture_count, dtype=np.uint8)
        return image, np.ones((2, 2), dtype=np.float32)

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
        del image, context
        self.calls += 1
        exposure_ms = round(exposure_us / 1000.0)
        self.events.append(f"predict:E{exposure_ms:02d}_G{round(gain):03d}")
        if self.on_predict is not None:
            self.on_predict(self.calls, self.provider)
        return self.results.pop(0)

    def predict(self, *args, **kwargs):
        raise AssertionError("Geometric update must use one single-frame score call.")

    def predict_batch(self, *args, **kwargs):
        raise AssertionError("Geometric update must not batch or recapture current.")


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
        del rows


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        checkpoint_path=Path("unused"),
        output_dir=Path("unused"),
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
        evaluation_alignment="metric",
        min_depth_m=0.2,
        max_depth_m=10,
        min_valid_depth_pixels=1,
    )


def make_experiment(results, *, safety=None, on_predict=None):
    events = []
    provider = Provider()
    config = make_config()
    policy = GeometricSearchPolicy(config.policy, safety or SafetyPolicy())
    predictor = Predictor(events, results, provider, on_predict)
    camera = Camera(events, provider)
    runtime = GeometricSearchUpdateExperiment(
        config,
        CaptureRunner(camera, provider, config.max_pair_capture_gap_ms),
        predictor,
        policy,
        Logger(),
        Evaluator(),
    )
    return runtime, events, provider, predictor, policy


class GeometricSearchUpdateExperimentTest(unittest.TestCase):
    def test_current_only_round_has_exactly_one_inference(self) -> None:
        runtime, events, _, predictor, _ = make_experiment([score(0.05)])

        result = runtime.run_round()

        self.assertEqual(predictor.calls, 1)
        self.assertEqual(sum(event.startswith("capture:") for event in events), 1)
        self.assertIsNone(result.challenger)

    def test_probe_reuses_current_and_has_exactly_two_inferences(self) -> None:
        runtime, events, _, predictor, _ = make_experiment(
            [score(0.20), score(0.10)]
        )

        result = runtime.run_round()

        self.assertEqual(predictor.calls, 2)
        self.assertEqual(sum(event.startswith("capture:") for event in events), 2)
        self.assertEqual(
            [event for event in events if event.startswith("predict:")],
            ["predict:E16_G064", "predict:E32_G064"],
        )
        self.assertEqual(result.decision.status, PairStatus.CHALLENGER_WON)

    def test_commit_does_not_reapply_the_already_active_challenger(self) -> None:
        runtime, events, _, _, _ = make_experiment(
            [score(0.20), score(0.10)]
        )

        runtime.run_round()

        self.assertEqual(events.count("apply:E32_G064"), 1)
        self.assertEqual(runtime.last_applied_cell, SensorCell(32, 64))

    def test_reject_restores_current_only_after_the_comparison(self) -> None:
        runtime, events, _, _, _ = make_experiment(
            [score(0.20), score(0.30)]
        )

        result = runtime.run_round()

        self.assertEqual(result.decision.status, PairStatus.CURRENT_WON)
        self.assertEqual(events[-1], "apply:E16_G064")
        self.assertEqual(runtime.last_applied_cell, SensorCell(16, 64))

    def test_unsafe_initial_neighbor_is_never_applied(self) -> None:
        disabled = {"0,0": frozenset({SensorCell(8, 64).cell_id})}
        runtime, events, _, _, _ = make_experiment(
            [score(0.20), score(0.10)],
            safety=SafetyPolicy(disabled_cells_by_context=disabled),
        )

        result = runtime.run_round()

        self.assertEqual(result.challenger.cell, SensorCell(32, 64))
        self.assertNotIn("apply:E08_G064", events)

    def test_context_change_after_current_prevents_challenger_inference(self) -> None:
        def change_context(call, provider):
            if call == 1:
                provider.current = ContextKey(1, 0)

        runtime, events, _, predictor, policy = make_experiment(
            [score(0.20)], on_predict=change_context
        )

        result = runtime.run_round()

        self.assertIsNone(result.challenger)
        self.assertEqual(predictor.calls, 1)
        self.assertEqual(sum(event.startswith("capture:") for event in events), 1)
        self.assertEqual(policy.state(ContextKey(0, 0)).vertices, [])


if __name__ == "__main__":
    unittest.main()
