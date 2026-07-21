import csv
import json
import math
import os
from pathlib import Path
from typing import Any

class ExperimentLogger:
    FIELDNAMES = [
        "timestamp_ns",
        "cycle_index",
        "slot_index",
        "phase",
        "motion_state",
        "motion_label",
        "light_state",
        "lighting_label",
        "cell_id",
        "exposure_ms",
        "gain",
        "requested_exposure_raw",
        "actual_exposure_raw",
        "actual_gain",
        "camera_parameter_ms",
        "mde_inference_ms",
        "control_decision_delay_ms",
        "delay_warning",
        "delay_reasons",
        "q",
        "uncertainty",
        "mu",
        "ema_before",
        "ema_after",
        "count_after",
        "best_before",
        "best_after",
        "challenger_after",
        "challenger_streak_after",
        "cycle_committed",
        "discard_reason",
        "offload_requested",
        "image_path",
        "depth_path",
        "extra_json",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

