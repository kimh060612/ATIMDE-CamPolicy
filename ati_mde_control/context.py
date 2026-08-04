from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from hardware.utils import ContextKey


MOTION_LABEL_TO_STATE = {"fast": 0, "slow": 1, "stop": 2, "rotate": 3, "spin": 4}
LIGHT_LABEL_TO_STATE = {"normal": 0, "dim": 1, "dark": 2}
MOTION_STATE_TO_LABEL = {value: key for key, value in MOTION_LABEL_TO_STATE.items()}
LIGHT_STATE_TO_LABEL = {value: key for key, value in LIGHT_LABEL_TO_STATE.items()}


class ContextProvider(Protocol):
    def get(self) -> ContextKey: ...
    def close(self) -> None: ...


class RosTrainingContextAdapter:
    HARDWARE_STATE_TO_LABEL = {0: "stop", 1: "slow", 2: "fast", 3: "rotate", 4: "spin"}

    def __init__(self, provider: ContextProvider) -> None:
        self.provider = provider

    def get(self) -> ContextKey:
        context = self.provider.get()
        return ContextKey(MOTION_LABEL_TO_STATE[self.HARDWARE_STATE_TO_LABEL[context.motion_state]], context.light_state)

    def close(self) -> None:
        self.provider.close()


class DebouncedContextProvider:
    def __init__(self, provider: ContextProvider, debounce_ms: float, minimum_samples: int) -> None:
        if debounce_ms < 0 or minimum_samples < 1:
            raise ValueError("Context debounce requires non-negative time and positive samples.")
        self.provider = provider
        self.debounce_sec = debounce_ms / 1000.0
        self.minimum_samples = minimum_samples
        self.committed = provider.get()
        self.committed.validate()
        self.candidate: ContextKey | None = None
        self.candidate_since = 0.0
        self.candidate_samples = 0

    def get(self) -> ContextKey:
        observed = self.provider.get()
        observed.validate()
        now = time.monotonic()
        if observed == self.committed:
            self.candidate = None
            self.candidate_samples = 0
        elif observed != self.candidate:
            self.candidate = observed
            self.candidate_since = now
            self.candidate_samples = 1
        else:
            self.candidate_samples += 1
            if self.candidate_samples >= self.minimum_samples and now - self.candidate_since >= self.debounce_sec:
                self.committed = observed
                self.candidate = None
                self.candidate_samples = 0
        return self.committed

    @property
    def is_stable(self) -> bool:
        return self.candidate is None

    def close(self) -> None:
        self.provider.close()


def build_context_provider(args) -> DebouncedContextProvider:
    from hardware import robot

    light_state = LIGHT_LABEL_TO_STATE[args.lighting_state]
    fallback = ContextKey(args.motion_state, light_state)
    fallback.validate()
    if args.context_file:
        provider: ContextProvider = robot.JsonContextProvider(Path(args.context_file), fallback)
    elif args.motion_source == "ros":
        provider = RosTrainingContextAdapter(robot.RosMotionContextProvider(
            light_state=light_state,
            imu_topic=args.imu_topic,
            odom_topic=args.odom_topic,
            sensor_timeout_sec=args.ros_sensor_timeout_sec,
        ))
    else:
        provider = robot.FixedContextProvider(fallback)
    return DebouncedContextProvider(provider, args.context_debounce_ms, args.context_debounce_samples)
