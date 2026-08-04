from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hardware.utils import QScore

from .context import LIGHT_STATE_TO_LABEL, MOTION_STATE_TO_LABEL
from .types import CapturedFrame


CSV_FIELDS = (
    "record_type", "round_index", "capture_index", "timestamp_ns", "capture_role",
    "output_delivered", "motion_state", "motion_label", "light_state", "lighting_label",
    "cell_id", "exposure_ms", "exposure_us_model", "gain", "requested_exposure_raw",
    "actual_exposure_raw", "actual_gain", "active_cell_before", "active_cell_after",
    "search_axis", "search_direction", "search_cycle_complete", "cycle_had_switch",
    "probe_pending_before", "probe_pending_after", "rounds_since_valid_probe",
    "challenger_cell_id", "pair_status", "delta_mu", "pair_std", "effective_margin",
    "pair_mode", "switch_event", "round_reason",
    "pair_capture_gap_ms", "capture_valid_pair", "pair_invalid_reason",
    "edge_valid_count", "edge_invalid_count", "edge_invalid_cooldown",
    "edge_ambiguous_count", "edge_ambiguous_cooldown", "consecutive_invalid_pairs",
    "force_current_only_rounds", "forced_current_only", "camera_bias", "std", "q",
    "mde_batch_size", "mde_inference_ms", "camera_parameter_ms",
    "control_decision_delay_ms", "offload_risk", "offload_requested", "selected",
    "initial_inference_count",
    "pending_switch_from_id", "pending_switch_to_id", "pending_switch_wins",
    "switch_confirmations_required", "switch_confirmation_age", "rollback_cell_id",
    "rollback_verification_pending", "post_switch_dwell_remaining",
    "rollback_verification_age", "committed_switch", "rollback_applied",
    "image_path", "depth_path", "abs_rel", "a1", "valid_depth_pixels",
    "evaluation_inference_ms", "evaluation_error", "probe_overhead", "valid_pair_rate",
    "invalid_pair_rate", "ambiguous_pair_rate", "switch_count", "switch_precision",
    "harmful_switch_rate", "rollback_count", "confirmation_pending_count",
    "parameter_occupancy", "mean_decision_latency_ms",
    "pair_gap_p50_ms", "pair_gap_p95_ms", "effective_output_rate_hz", "output_coverage",
)


class CaptureLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        self.depth_dir = output_dir / "depth_gt"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []

    def record(self, frame: CapturedFrame, score: QScore | None, values: dict[str, Any]) -> dict[str, Any]:
        stem = f"round_{frame.round_index:04d}_capture_{frame.capture_index:05d}_{frame.cell.cell_id}_{frame.timestamp_ns}"
        image_path, depth_path = self.image_dir / f"{stem}.png", self.depth_dir / f"{stem}.npy"
        if not cv2.imwrite(str(image_path), frame.image):
            raise OSError(f"Failed to save RGB image: {image_path}")
        np.save(depth_path, np.ascontiguousarray(frame.depth_m, dtype=np.float32))
        row = {field: "" for field in CSV_FIELDS}
        row.update(
            record_type="capture",
            round_index=frame.round_index,
            capture_index=frame.capture_index,
            timestamp_ns=frame.timestamp_ns,
            capture_role=frame.role,
            output_delivered=int(frame.role == "initial"),
            motion_state=frame.latched_context.motion_state,
            motion_label=MOTION_STATE_TO_LABEL[frame.latched_context.motion_state],
            light_state=frame.latched_context.light_state,
            lighting_label=LIGHT_STATE_TO_LABEL[frame.latched_context.light_state],
            cell_id=frame.cell.cell_id,
            exposure_ms=frame.cell.exposure_ms,
            exposure_us_model=frame.exposure_us,
            gain=frame.cell.gain,
            requested_exposure_raw=frame.requested_exposure_raw,
            actual_exposure_raw=frame.actual_exposure_raw,
            actual_gain=frame.actual_gain,
            camera_parameter_ms=frame.camera_parameter_ms,
            initial_inference_count=1,
            image_path=str(image_path),
            depth_path=str(depth_path),
        )
        if score is not None:
            row.update(
                camera_bias=score.mu,
                std=score.uncertainty,
                q=score.q,
                mde_batch_size=int(score.extra.get("mde_batch_size", 1)),
                mde_inference_ms=score.extra.get("mde_inference_ms", ""),
            )
        row.update(values)
        self.rows.append(row)
        return row

    def write(self) -> Path:
        path = self.output_dir / "probing_modelv1.csv"
        temporary = path.with_suffix(".csv.tmp")
        summary = {field: "" for field in CSV_FIELDS}
        summary.update(record_type="summary", **self._summary())
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
            writer.writerow(summary)
        os.replace(temporary, path)
        return path

    def _summary(self) -> dict[str, Any]:
        initial = [row for row in self.rows if row["output_delivered"] == 1]
        probes = [row for row in self.rows if row["capture_role"] == "challenger"]
        pair_rows = [row for row in initial if row["pair_status"] not in ("", "not_probed")]
        valid = [row for row in pair_rows if row["capture_valid_pair"] == 1]
        switched = [row for row in valid if row.get("switch_event") == "committed"]
        comparisons = []
        for row in switched:
            probe = next((item for item in probes if item["round_index"] == row["round_index"]), None)
            if probe and row["abs_rel"] != "" and probe["abs_rel"] != "":
                comparisons.append(float(probe["abs_rel"]) - float(row["abs_rel"]))
        gaps = np.array([float(row["pair_capture_gap_ms"]) for row in pair_rows], dtype=float)
        times = [int(row["timestamp_ns"]) for row in initial]
        duration = (max(times) - min(times)) / 1e9 if len(times) > 1 else 0.0
        mean = lambda values: float(np.mean(values)) if values else ""
        rate = lambda count, total: count / total if total else 0.0
        primary_abs_rel = [float(row["abs_rel"]) for row in initial if row["abs_rel"] != ""]
        primary_a1 = [float(row["a1"]) for row in initial if row["a1"] != ""]
        return {
            "capture_index": len(self.rows),
            "abs_rel": mean(primary_abs_rel),
            "a1": mean(primary_a1),
            "probe_overhead": rate(len(probes), len(initial)),
            "valid_pair_rate": rate(len(valid), len(pair_rows)),
            "invalid_pair_rate": rate(len(pair_rows) - len(valid), len(pair_rows)),
            "ambiguous_pair_rate": rate(sum(row["pair_status"] == "ambiguous" for row in valid), len(valid)),
            "switch_count": len(switched),
            "rollback_count": sum(row.get("switch_event") == "rolled_back" for row in initial),
            "confirmation_pending_count": sum(
                row.get("switch_event") == "confirmation_pending" for row in initial
            ),
            "switch_precision": rate(sum(delta < 0 for delta in comparisons), len(comparisons)),
            "harmful_switch_rate": rate(sum(delta > 0 for delta in comparisons), len(comparisons)),
            "parameter_occupancy": json.dumps(Counter(row["cell_id"] for row in initial), sort_keys=True),
            "mean_decision_latency_ms": mean([float(row["control_decision_delay_ms"]) for row in initial if row["control_decision_delay_ms"] != ""]),
            "pair_gap_p50_ms": float(np.percentile(gaps, 50)) if gaps.size else "",
            "pair_gap_p95_ms": float(np.percentile(gaps, 95)) if gaps.size else "",
            "effective_output_rate_hz": (len(initial) - 1) / duration if duration > 0 else "",
            "output_coverage": rate(len(initial), len({row["round_index"] for row in self.rows})),
        }
