import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

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
    def test_non_ae_color_appearance_is_fixed_before_capture(self):
        camera = ae.DefaultOrbbecCamera.__new__(ae.DefaultOrbbecCamera)
        camera.device = Mock()
        camera.device.is_property_supported.return_value = True
        camera.device.get_bool_property.return_value = False
        expected_int_values = {
            ae.OBPropertyID.OB_PROP_COLOR_WHITE_BALANCE_INT: 4600,
            ae.OBPropertyID.OB_PROP_COLOR_BRIGHTNESS_INT: 0,
            ae.OBPropertyID.OB_PROP_COLOR_GAMMA_INT: 300,
            ae.OBPropertyID.OB_PROP_COLOR_SATURATION_INT: 64,
            ae.OBPropertyID.OB_PROP_COLOR_SHARPNESS_INT: 50,
        }
        camera.device.get_int_property.side_effect = expected_int_values.__getitem__

        camera._configure_fixed_color_appearance()

        expected_bool_calls = [
            call.set_bool_property(
                ae.OBPropertyID.OB_PROP_COLOR_HDR_BOOL, False
            ),
            call.set_bool_property(
                ae.OBPropertyID.OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
                False,
            ),
        ]
        expected_int_calls = [
            call.set_int_property(property_id, value)
            for property_id, value in expected_int_values.items()
        ]
        self.assertEqual(
            [
                method_call
                for method_call in camera.device.method_calls
                if method_call[0] == "set_bool_property"
            ],
            expected_bool_calls,
        )
        self.assertEqual(
            [
                method_call
                for method_call in camera.device.method_calls
                if method_call[0] == "set_int_property"
            ],
            expected_int_calls,
        )
        awb_call_index = camera.device.method_calls.index(expected_bool_calls[1])
        white_balance_call_index = camera.device.method_calls.index(
            expected_int_calls[0]
        )
        self.assertLess(awb_call_index, white_balance_call_index)

    def test_auto_exposure_is_enabled_then_eight_frames_are_settled(self):
        camera = ae.DefaultOrbbecCamera.__new__(ae.DefaultOrbbecCamera)
        camera.device = Mock()
        camera.device.is_property_supported.return_value = True
        camera.device.get_bool_property.return_value = True
        camera.pipeline = Mock()

        camera._enable_auto_exposure_and_settle(
            settle_frames=8,
            settle_timeout=2.0,
        )

        auto_exposure_property = (
            ae.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
        )
        self.assertEqual(
            camera.device.method_calls,
            [
                call.is_property_supported(
                    auto_exposure_property,
                    ae.OBPermissionType.PERMISSION_WRITE,
                ),
                call.set_bool_property(auto_exposure_property, True),
                call.get_bool_property(auto_exposure_property),
            ],
        )
        self.assertEqual(camera.pipeline.wait_for_frames.call_count, 8)

    def test_frame_metadata_takes_precedence_over_static_device_properties(self):
        camera = ae.DefaultOrbbecCamera.__new__(ae.DefaultOrbbecCamera)
        camera.device = Mock()
        camera.device.get_bool_property.return_value = True
        camera.device.get_int_property.side_effect = [156, 16]
        camera.device_exposure_raw = None
        camera.device_gain = None

        metadata_values = {
            ae.OBFrameMetadataType.AUTO_EXPOSURE: 1,
            ae.OBFrameMetadataType.EXPOSURE: 55,
            ae.OBFrameMetadataType.GAIN: 16,
        }
        color_frame = Mock()
        color_frame.has_metadata.side_effect = metadata_values.__contains__
        color_frame.get_metadata_value.side_effect = metadata_values.__getitem__

        camera._record_frame_settings(color_frame)
        settings = camera.read_settings()

        self.assertEqual(settings, (True, 55, 16))
        self.assertEqual(camera.device_exposure_raw, 156)
        self.assertEqual(camera.device_gain, 16)
        self.assertEqual(camera.auto_exposure_source, "frame_metadata")
        self.assertEqual(camera.exposure_source, "frame_metadata")
        self.assertEqual(camera.gain_source, "frame_metadata")

    def test_capture_mde_evaluation_share_each_csv_row(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_dir=Path(directory),
                num_frames=2,
                exposure_value_per_ms=10.0,
                camera_settle_frames=8,
                camera_settle_timeout=2.0,
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
