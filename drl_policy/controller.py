"""Pretrained DRL exposure policy for live camera frames."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from drl_policy.agent import Actor


STATE_SHAPE = (4, 84, 84)


def exposure_for_ev(
    exposure_ms: float,
    ev: float,
    minimum_ms: float,
    maximum_ms: float,
) -> float:
    """Apply EV to an exposure quantity: -1 EV doubles it."""
    if not all(map(math.isfinite, (exposure_ms, ev, minimum_ms, maximum_ms))):
        raise ValueError("Exposure and EV values must be finite.")
    if exposure_ms <= 0 or minimum_ms <= 0 or maximum_ms < minimum_ms:
        raise ValueError("Require 0 < minimum exposure <= maximum exposure.")
    return min(
        max(exposure_ms * 2.0 ** (-np.clip(ev, -2.0, 2.0)), minimum_ms),
        maximum_ms,
    )


class DRLExposureController:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.device = device
        self.actor = Actor(STATE_SHAPE, 1, 512).to(device)
        self.actor.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        self.actor.eval()
        self.state: np.ndarray | None = None

    def observe(self, image_bgr: np.ndarray) -> None:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected an HxWx3 BGR image.")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.resize(gray, STATE_SHAPE[1:][::-1])
        self.state = (
            np.repeat(gray[None], STATE_SHAPE[0], axis=0)
            if self.state is None
            else np.concatenate((self.state[1:], gray[None]), axis=0)
        )

    def action(self) -> tuple[float, float]:
        if self.state is None:
            raise RuntimeError("Observe a frame before requesting an action.")
        state = (
            torch.from_numpy(np.ascontiguousarray(self.state))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            actor_action, _ = self.actor(state, deterministic=True, with_logprob=False)
        normalized_action = float(actor_action.item())
        # The training environment used t_next=t*2^(2a). Express the same
        # command as conventional EV, where negative EV lengthens exposure.
        ev = float(np.clip(-2.0 * normalized_action, -2.0, 2.0))
        return ev, normalized_action


def _self_check() -> None:
    assert exposure_for_ev(10.0, -1.0, 0.1, 100.0) == 20.0
    assert exposure_for_ev(10.0, 2.0, 0.1, 100.0) == 2.5
    assert exposure_for_ev(10.0, -2.0, 0.1, 20.0) == 20.0


if __name__ == "__main__":
    _self_check()
    print("DRL exposure conversion check passed.")
