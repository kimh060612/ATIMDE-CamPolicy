import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import cv2
import numpy as np

from motion_classifier import assign_motion_label
from hardware.utils import *

try:
    from pyorbbecsdk import (  # type: ignore
        AlignFilter,
        Config,
        OBError,
        OBFormat,
        OBFrameAggregateOutputMode,
        OBPermissionType,
        OBPropertyID,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )
except ImportError as exc:  # pragma: no cover - depends on target machine
    raise SystemExit(
        "pyorbbecsdk is not installed. For SDK v2, install the official "
        "package/build appropriate for your platform (the PyPI package is "
        "typically named pyorbbecsdk2 but imports as pyorbbecsdk)."
    ) from exc


EXPOSURE_MS_VALUES: tuple[int, ...] = (4, 8, 16, 32)
GAIN_VALUES: tuple[int, ...] = (16, 32, 64, 128)
MOTION_STATE_COUNT = 5
LIGHT_STATE_COUNT = 3
DEFAULT_CELLS_PER_CYCLE = 4
MOTION_LABEL_TO_STATE = {
    "stop": 0,
    "slow": 1,
    "fast": 2,
    "rotate": 3,
    "spin": 4,
}
LIGHT_LABEL_TO_STATE = {"normal": 0, "dim": 1, "dark": 2}
MOTION_STATE_TO_LABEL = {value: key for key, value in MOTION_LABEL_TO_STATE.items()}
LIGHT_STATE_TO_LABEL = {value: key for key, value in LIGHT_LABEL_TO_STATE.items()}

class ContextProvider:
    def get(self) -> ContextKey:
        raise NotImplementedError

    def close(self) -> None:
        pass

class RosMotionContextProvider(ContextProvider):
    """Classify motion from ROS 2 IMU and wheel-odometry topics.

    This is an embedded rclpy node, so the script does not need to be installed
    as a ROS 2 package.  The caller must still source a ROS 2 environment that
    provides rclpy, sensor_msgs, and nav_msgs.
    """

    def __init__(
        self,
        *,
        light_state: int,
        imu_topic: str,
        odom_topic: str,
        sensor_timeout_sec: float,
    ) -> None:
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Imu
        except ImportError as exc:
            raise RuntimeError(
                "ROS motion input requires rclpy, sensor_msgs, and nav_msgs. "
                "Source the ROS 2 installation/workspace before running this script."
            ) from exc

        if sensor_timeout_sec <= 0:
            raise ValueError("sensor_timeout_sec must be positive.")
        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node("orbbec_motion_context")
        self._light_state = light_state
        self._sensor_timeout_sec = sensor_timeout_sec
        self._linear_speed: Optional[float] = None
        self._linear_acceleration: Optional[float] = None
        self._angular_speed: Optional[float] = None
        self._imu_received_at: Optional[float] = None
        self._odom_received_at: Optional[float] = None
        self._last_context = ContextKey(0, light_state)
        self._warned_waiting = False

        def on_imu(message: Any) -> None:
            # Use planar acceleration to avoid treating gravity on the IMU Z
            # axis as vehicle acceleration; yaw rate is rotation about Z.
            self._linear_acceleration = math.hypot(
                message.linear_acceleration.x,
                message.linear_acceleration.y,
            )
            self._angular_speed = abs(float(message.angular_velocity.z))
            self._imu_received_at = time.monotonic()

        def on_odom(message: Any) -> None:
            twist = message.twist.twist
            self._linear_speed = math.hypot(twist.linear.x, twist.linear.y)
            self._odom_received_at = time.monotonic()

        self._imu_subscription = self._node.create_subscription(
            Imu, imu_topic, on_imu, qos_profile_sensor_data
        )
        self._odom_subscription = self._node.create_subscription(
            Odometry, odom_topic, on_odom, qos_profile_sensor_data
        )
        print(
            f"[ROS] motion input imu={imu_topic} odom={odom_topic} "
            f"light_state={light_state}"
        )

    def get(self) -> ContextKey:
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        now = time.monotonic()
        fresh = (
            self._imu_received_at is not None
            and self._odom_received_at is not None
            and now - self._imu_received_at <= self._sensor_timeout_sec
            and now - self._odom_received_at <= self._sensor_timeout_sec
        )
        if not fresh:
            if not self._warned_waiting:
                print("[WARNING] Waiting for fresh /imu and /odom data; using stop.")
                self._warned_waiting = True
            self._last_context = ContextKey(0, self._light_state)
            return self._last_context

        label = assign_motion_label(
            float(self._linear_speed),
            float(self._linear_acceleration),
            float(self._angular_speed),
        )
        self._last_context = ContextKey(
            MOTION_LABEL_TO_STATE[label], self._light_state
        )
        self._warned_waiting = False
        return self._last_context

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
            

class FixedContextProvider(ContextProvider):
    def __init__(self, context: ContextKey) -> None:
        context.validate()
        self._context = context

    def get(self) -> ContextKey:
        return self._context


class JsonContextProvider(ContextProvider):
    """Read motion/light from a JSON file that may be updated externally.

    Expected content:

        {"motion_state": 0, "light_state": 0}

    Invalid/transient partial writes fall back to the last valid context.
    External writers should still prefer atomic file replacement.
    """

    def __init__(self, path: Path, fallback: ContextKey) -> None:
        fallback.validate()
        self.path = path
        self.last_valid = fallback

    def get(self) -> ContextKey:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            context = ContextKey(
                motion_state=int(payload["motion_state"]),
                light_state=int(payload["light_state"]),
            )
            context.validate()
            self.last_valid = context
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        return self.last_valid

