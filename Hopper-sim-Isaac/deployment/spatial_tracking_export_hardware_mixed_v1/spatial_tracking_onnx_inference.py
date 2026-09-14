#!/usr/bin/env python3
"""Minimal stateful inference check for the 52-D spatial-tracking actor."""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort


class SpatialTrackingPolicy:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 256), dtype=np.float32)
        self.c = np.zeros((1, 1, 256), dtype=np.float32)

    def step(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, 52)
        actions, self.h, self.c = self.session.run(
            ["actions", "h_out", "c_out"],
            {"obs": obs, "h_in": self.h, "c_in": self.c},
        )
        actions = np.clip(actions[0], -1.0, 1.0)
        motor_targets = np.clip(0.5 * actions + 0.5, 0.0, 1.0)
        return actions, motor_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    policy = SpatialTrackingPolicy(args.model)
    actions, motor_targets = policy.step(np.zeros(52, dtype=np.float32))
    print("actions:", actions)
    print("normalized motor targets F1..F4:", motor_targets)


if __name__ == "__main__":
    main()
