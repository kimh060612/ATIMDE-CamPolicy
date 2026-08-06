import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import hardware.sensor as sensor
from ati_mde_control.capture_runner import CaptureRunner
from hardware.utils import ContextKey, SensorCell


class Frame:
    def __init__(self, number, timestamp_us, *, depth=False):
        self.number = number
        self.timestamp_us = timestamp_us
        self.depth = depth

    def get_frame_number(self):
        return self.number

    def get_timestamp_us(self):
        return self.timestamp_us

    def get_width(self):
        return 1

    def get_height(self):
        return 1

    def get_format(self):
        return "rgb"

    def get_data(self):
        return np.array([1], np.uint16) if self.depth else np.array([1, 2, 3], np.uint8)

    def get_depth_scale(self):
        return 1.0


class Frameset:
    def __init__(self, number, timestamp_us=None):
        timestamp_us = number * 1000 if timestamp_us is None else timestamp_us
        self.color = Frame(number, timestamp_us)
        self.depth = Frame(number, timestamp_us, depth=True)

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


class LegacyFrame:
    def __init__(self, timestamp_ms):
        self.timestamp_ms = timestamp_ms

    def get_timestamp(self):
        return self.timestamp_ms


class LegacyFrameset:
    def __init__(self, timestamp_ms):
        self.color = self.depth = LegacyFrame(timestamp_ms)

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


class Pipeline:
    def __init__(self, frames, *, continuous=False):
        self.frames = list(frames)
        self.continuous = continuous
        self.calls = 0
        self.timeouts = []

    def wait_for_frames(self, timeout_ms):
        self.calls += 1
        self.timeouts.append(timeout_ms)
        if self.frames:
            return self.frames.pop(0)
        return Frameset(self.calls) if self.continuous else None


class Device:
    def __init__(self, *, readback_matches=True):
        self.values = {}
        self.readback_matches = readback_matches
        self.set_calls = []

    def set_int_property(self, property_id, value):
        self.set_calls.append((property_id, value))
        self.values[property_id] = value

    def get_int_property(self, property_id):
        value = self.values[property_id]
        return value if self.readback_matches else value + 1


class Provider:
    is_stable = True

    def get(self):
        return ContextKey(0, 0)


def camera(frames, *, settle_frames=2, readback_matches=True, continuous=False):
    result = sensor.OrbbecColorCamera.__new__(sensor.OrbbecColorCamera)
    result.exposure_value_per_ms = 10.0
    result.settle_frames = settle_frames
    result.frame_timeout_ms = 1
    result.pipeline = Pipeline(frames, continuous=continuous)
    result.device = Device(readback_matches=readback_matches)
    result.align_filter = SimpleNamespace(process=lambda frameset: frameset)
    result.frame_sequence = None
    result.color_frame_number = result.depth_frame_number = None
    result.color_timestamp_us = result.depth_timestamp_us = None
    result.setting_effective = False
    result.sensor_settle_ms = 0.0
    result._last_metadata = None
    result._capture_safe = True
    result._readback_matches = False
    result._settled_frames = 0
    result._pending_cell = None
    result._verified_active_cell = None
    result._actual_exposure = None
    result._actual_gain = None
    return result


class Profiles:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.requests = []
        self.default_requested = False

    def get_video_stream_profile(self, width, height, frame_format, fps):
        self.requests.append((width, height, frame_format, fps))
        if self.error:
            raise self.error
        return self.result

    def get_default_video_stream_profile(self):
        self.default_requested = True
        return "default"


class ProfilePipeline:
    def __init__(self, profiles):
        self.profiles = profiles

    def get_stream_profile_list(self, sensor_type):
        return self.profiles[sensor_type]


class RunnerCamera:
    exposure_value_per_ms = 10.0
    sensor_settle_ms = 0.0
    setting_effective = True

    def __init__(self):
        self.index = 0

    def apply_cell(self, cell):
        return cell.exposure_ms * 10, cell.exposure_ms * 10, cell.gain

    def capture_rgbd(self):
        self.index += 1
        self.color_frame_number = self.depth_frame_number = self.index
        self.color_timestamp_us = self.depth_timestamp_us = (100_000, 125_000)[self.index - 1]
        return np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.float32)


class NoTimestampCamera(RunnerCamera):
    def capture_rgbd(self):
        image, depth = super().capture_rgbd()
        self.color_timestamp_us = self.depth_timestamp_us = None
        return image, depth


class DepthTimestampOnlyCamera(RunnerCamera):
    def capture_rgbd(self):
        image, depth = super().capture_rgbd()
        self.color_timestamp_us = None
        return image, depth


class SensorFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.properties = SimpleNamespace(
            OB_PROP_COLOR_EXPOSURE_INT="exposure",
            OB_PROP_COLOR_GAIN_INT="gain",
        )
        self.format = SimpleNamespace(RGB="rgb", BGR="bgr", YUYV="yuyv", MJPG="mjpg", I420="i420")
        self.patches = (
            patch.object(sensor, "OBPropertyID", self.properties, create=True),
            patch.object(sensor, "OBFormat", self.format, create=True),
        )
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def test_exact_640x480_30_rgbd_profiles_are_selected(self):
        profiles = {
            "color": Profiles("color-profile"),
            "depth": Profiles("depth-profile"),
        }
        pipeline = ProfilePipeline(profiles)
        self.assertEqual(
            sensor._required_video_profile(pipeline, "color", "rgb", "color"),
            "color-profile",
        )
        self.assertEqual(
            sensor._required_video_profile(pipeline, "depth", "y16", "depth"),
            "depth-profile",
        )
        self.assertEqual(profiles["color"].requests, [(640, 480, "rgb", 30)])
        self.assertEqual(profiles["depth"].requests, [(640, 480, "y16", 30)])

    def test_missing_profile_fails_without_default_fallback(self):
        profiles = Profiles(error=RuntimeError("not found"))
        with self.assertRaisesRegex(RuntimeError, "Required depth profile 640x480@30"):
            sensor._required_video_profile(
                ProfilePipeline({"depth": profiles}), "depth", "y16", "depth"
            )
        self.assertFalse(profiles.default_requested)

    def test_stale_queue_is_drained_and_two_fresh_frames_are_settled(self):
        cam = camera([
            Frameset(100), Frameset(101), None, None,
            Frameset(102), Frameset(103), Frameset(104),
        ])
        cam.apply_cell(SensorCell(8, 64))
        image, _ = cam.capture_rgbd()
        self.assertEqual(image.shape, (1, 1, 3))
        self.assertEqual(cam.color_frame_number, 104)
        self.assertTrue(cam.setting_effective)

    def test_verified_active_cell_is_not_reapplied_or_settled(self):
        cam = camera([
            None, None, Frameset(1), Frameset(2), Frameset(3), Frameset(4),
        ])
        cell = SensorCell(8, 64)
        cam.apply_cell(cell)
        cam.capture_rgbd()
        calls_after_first_capture = cam.pipeline.calls
        writes_after_first_capture = list(cam.device.set_calls)

        requested, actual_exposure, actual_gain = cam.apply_cell(cell)
        cam.capture_rgbd()

        self.assertEqual(cam.device.set_calls, writes_after_first_capture)
        self.assertEqual(cam.pipeline.calls, calls_after_first_capture + 1)
        self.assertEqual((requested, actual_exposure, actual_gain), (80, 80, 64))
        self.assertEqual(cam.color_frame_number, 4)
        self.assertEqual(cam.sensor_settle_ms, 0.0)
        self.assertTrue(cam.setting_effective)

    def test_legacy_device_timestamp_is_used_when_us_api_is_absent(self):
        metadata = sensor.frame_metadata(LegacyFrameset(12))
        self.assertEqual(metadata.color_timestamp_us, 12_000)
        self.assertFalse(sensor.metadata_is_fresh(None, metadata))

    def test_timestamp_only_metadata_is_fresh(self):
        previous = sensor.FrameMetadata(None, None, 1_000, 1_000)
        current = sensor.FrameMetadata(None, None, 2_000, 2_000)

        self.assertTrue(sensor.metadata_is_fresh(previous, current))


    def test_duplicate_timestamp_is_rejected_without_frame_number(self):
        previous = sensor.FrameMetadata(None, None, 2_000, 2_000)
        current = sensor.FrameMetadata(None, None, 2_000, 2_000)

        self.assertFalse(sensor.metadata_is_fresh(previous, current))


    def test_decreasing_timestamp_is_rejected_without_frame_number(self):
        previous = sensor.FrameMetadata(None, None, 2_000, 2_000)
        current = sensor.FrameMetadata(None, None, 1_000, 1_000)

        self.assertFalse(sensor.metadata_is_fresh(previous, current))


    def test_metadata_without_number_or_timestamp_is_rejected(self):
        metadata = sensor.FrameMetadata(None, None, None, None)

        self.assertFalse(sensor.metadata_is_fresh(None, metadata))

    def test_duplicate_and_decreasing_numbers_are_rejected(self):
        cam = camera([
            Frameset(101), None, None,
            Frameset(101), Frameset(100), Frameset(102),
            Frameset(102), Frameset(99), Frameset(103), Frameset(104),
        ])
        cam.apply_cell(SensorCell(8, 64))
        cam.capture_rgbd()
        self.assertEqual(cam.color_frame_number, 104)

    def test_matching_readback_does_not_make_undrained_stale_queue_effective(self):
        cam = camera([], continuous=True)
        cam.apply_cell(SensorCell(8, 64))
        self.assertFalse(cam.setting_effective)
        with self.assertRaises(RuntimeError):
            cam.capture_rgbd()

    def test_wrong_readback_keeps_fresh_capture_ineffective(self):
        cam = camera([None, None, Frameset(1)], settle_frames=0, readback_matches=False)
        cam.apply_cell(SensorCell(8, 64))
        cam.capture_rgbd()
        self.assertEqual(cam.color_frame_number, 1)
        self.assertFalse(cam.setting_effective)

    def test_matching_readback_cannot_make_stale_aligned_frame_effective(self):
        cam = camera([None, None, Frameset(6)], settle_frames=0)
        cam._last_metadata = sensor.FrameMetadata(5, 5, 5_000, 5_000)
        cam.align_filter = SimpleNamespace(process=lambda frameset: Frameset(5))
        cam.apply_cell(SensorCell(8, 64))
        with patch("hardware.sensor.time.monotonic", side_effect=(0, 0, 0, 0, 2)):
            with self.assertRaises(TimeoutError):
                cam.capture_rgbd()
        self.assertFalse(cam.setting_effective)
        self.assertIsNone(cam.color_frame_number)

    def test_pair_gap_uses_device_timestamp(self):
        runner = CaptureRunner(RunnerCamera(), Provider(), 100.0)
        with patch("ati_mde_control.capture_runner.time.time_ns", side_effect=(1, 9_000_000_001)):
            pair = runner.capture_pair(
                SensorCell(4, 64), SensorCell(8, 64), ContextKey(0, 0), 0
            )
        self.assertEqual(pair.gap_ms, 25.0)
        self.assertTrue(pair.valid)

    def test_pair_without_device_timestamp_is_invalid(self):
        runner = CaptureRunner(NoTimestampCamera(), Provider(), 100.0)
        pair = runner.capture_pair(
            SensorCell(4, 64), SensorCell(8, 64), ContextKey(0, 0), 0
        )
        self.assertIsNone(pair.gap_ms)
        self.assertFalse(pair.valid)
        self.assertEqual(pair.invalid_reason, "device_timestamp_unavailable")

    def test_pair_gap_does_not_fallback_to_depth_timestamp(self):
        runner = CaptureRunner(DepthTimestampOnlyCamera(), Provider(), 100.0)
        pair = runner.capture_pair(
            SensorCell(4, 64), SensorCell(8, 64), ContextKey(0, 0), 0
        )
        self.assertIsNone(pair.gap_ms)
        self.assertFalse(pair.valid)
        self.assertEqual(pair.invalid_reason, "device_timestamp_unavailable")

    def test_continuously_full_queue_stops_at_drain_limit(self):
        cam = camera([], settle_frames=0, continuous=True)
        cam.apply_cell(SensorCell(8, 64))
        self.assertEqual(cam.pipeline.calls, sensor.MAX_QUEUE_DRAIN_FRAMES)
        self.assertEqual(set(cam.pipeline.timeouts), {sensor.QUEUE_DRAIN_TIMEOUT_MS})
        self.assertFalse(cam._capture_safe)

    def test_frame_arriving_in_drain_to_command_race_is_not_captured(self):
        cam = camera([
            Frameset(100), None,
            Frameset(101), None,
            Frameset(102),
        ], settle_frames=0)
        cam.apply_cell(SensorCell(8, 64))
        cam.capture_rgbd()
        self.assertEqual(cam.color_frame_number, 102)


if __name__ == "__main__":
    unittest.main()
