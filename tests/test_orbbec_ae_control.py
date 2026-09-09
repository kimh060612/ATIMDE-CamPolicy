import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import orbbec_ae_control as ae


class _Camera:
    def __init__(self, **kwargs):
        self.exposure_value_per_ms = kwargs["exposure_value_per_ms"]
        self.index = 0
        self.closed = False

    def capture_rgbd(self):
        self.index += 1
        self.color_frame_number = self.depth_frame_number = self.index
        self.color_timestamp_us = self.index * 1000
        self.depth_timestamp_us = self.color_timestamp_us + 10
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        return np.zeros((2, 2, 3), np.uint8), depth

    def read_settings(self):
        return True, 80, 32

    def close(self):
        self.closed = True


class _Predictor:
    def __init__(self, _args):
        pass

    def infer(self, _image_path, output_path):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.save(output_path, 1.0 / depth)
        return 1.5


class AutoExposureControlTest(unittest.TestCase):
    def test_capture_mde_evaluation_share_each_csv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_dir=Path(directory),
                num_frames=2,
                exposure_value_per_ms=10.0,
                frame_timeout_ms=1,
                warmup_frames=0,
                capture_interval_ms=0.0,
                depth_alignment="scale_shift_inverse",
                min_depth_m=1e-3,
                max_depth_m=10.0,
                min_valid_depth_pixels=1,
            )
            with (
                patch.object(ae, "DefaultOrbbecCamera", _Camera),
                patch.object(ae, "DepthAnythingV2Small", _Predictor),
            ):
                report, captured, evaluated = ae._run(args)

            self.assertEqual((captured, evaluated), (2, 2))
            with report.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                [row["record_type"] for row in rows],
                ["capture", "capture", "summary"],
            )
            self.assertTrue(all(row["abs_rel"] == "0.0" for row in rows[:2]))
            self.assertTrue(
                all(row["rgbd_timestamp_gap_us"] == "10" for row in rows[:2])
            )
            self.assertTrue(all(row["auto_exposure"] == "1" for row in rows[:2]))


if __name__ == "__main__":
    unittest.main()
