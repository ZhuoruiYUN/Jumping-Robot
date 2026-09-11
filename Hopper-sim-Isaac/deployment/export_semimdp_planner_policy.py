"""Export the 69-D two-hop Semi-MDP planner actor to ONNX."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn


class HopActor(nn.Module):
    def __init__(self, obs_dim: int = 69, action_dim: int = 4):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.second_hop_adapter = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        long_hop = obs[:, -2:-1]
        return torch.clamp(self.actor(obs) + long_hop * self.second_hop_adapter(obs), -1.0, 1.0)


class CheckpointCompatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(69, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 4),
        )
        self.second_hop_adapter = nn.Sequential(
            nn.Linear(69, 64),
            nn.ELU(),
            nn.Linear(64, 4),
        )
        self.critic = nn.Sequential(
            nn.Linear(69, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((4,), math.log(0.20)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = data["model_state_dict"]

    compat = CheckpointCompatModel()
    missing, unexpected = compat.load_state_dict(state, strict=False)
    allowed_missing = {key for key in missing if key.startswith("critic.") or key == "log_std"}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")

    actor = HopActor()
    actor.actor.load_state_dict(compat.actor.state_dict())
    actor.second_hop_adapter.load_state_dict(compat.second_hop_adapter.state_dict())
    actor.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    obs = torch.zeros(1, 69, dtype=torch.float32)
    torch.onnx.export(
        actor,
        obs,
        output,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"Exported {output}")


if __name__ == "__main__":
    main()
