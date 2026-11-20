"""Stage 3b-D: matched continuation of C; only first-two touchdown weights differ."""

import json

import torch

from .artifacts import ROOT, sha
from .foot_touchdown_reward import FootTouchdownTracker
from .point_energy_margin import contract as source_contract, reward_contract


SOURCE = ROOT / "outputs/benchmark37_stage3b_point_energy4cm/train_seed42_20261027_064630_499968Z/model_199.pt"
SOURCE_SHA256 = "78e408ec60fd864e73c224d706b0ca57d60401bfad902f253c4071be0984b6a1"
ARMS = ("control", "focus")
VARIANTS = {arm: f"real_release_first_two_{arm}" for arm in ARMS}
SERIES = {arm: f"benchmark37_stage3b_first_two_{arm}" for arm in ARMS}
ITERATIONS = 200


def multiplier(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown first-two arm: {arm}")
    return 1.0 if arm == "control" else 2.0


def contract(arm):
    return dict(
        name="first_two_touchdown_weight_v1", arm=arm,
        source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=source_contract(),
        early_touchdown_multiplier=multiplier(arm), later_touchdown_multiplier=1.0,
        counted_events="confirmed-flight touchdown attempts; count invalid/missed/death touchdowns too; "
                       "initial drop and unconfirmed contact bounce do not consume a jump index",
        reward="retain original foot<=10cm, apex>=1m, tilt<=15deg, valid flight and alive gates; "
               "multiply only the original foot_touchdown term on attempt indices 1 and 2",
        reset="per-environment attempt index cleared only on episode reset; keep history across touchdowns",
        unchanged="C energy target 1.04m, actor/score target 1.0m, 37D actor/critic, edge radius sampling, "
                  "release, SysID physics, all other rewards, PPO and point_short15 scoring",
        training_iterations=ITERATIONS,
        initialization="same pinned C weights and normalizers; fresh optimizer and iteration 0; "
                       "768 environments x 200 updates in each arm; no smoke checkpoint continuation",
        code_sha256=sha(ROOT / "benchmark37/first_two_focus.py"),
    )


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    m = json.loads((path.parent / "manifest.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or m.get("mode") != "train" or m.get("status") != "complete" or m.get("smoke")
            or m.get("variant") != "real_release_point_edge_energy4cm"
            or m.get("source_lock") != source_lock or m.get("physics_lock") != physics_lock
            or m.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256
            or m.get("task_intervention") != source_contract()
            or m.get("reward_intervention") != reward_contract()):
        raise ValueError("First-two A/B requires the pinned completed Stage 3b-C model and matching contracts")
    return m


class FirstTwoTracker(FootTouchdownTracker):
    def __init__(self, num_envs, device, arm):
        super().__init__(num_envs, device)
        self.arm = arm
        self.factor = multiplier(arm)
        self.attempt_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Buckets: first, second, third-and-later; cumulative across resets.
        self.attempt_counts = torch.zeros(3, dtype=torch.long, device=device)
        self.rewarded_counts = torch.zeros_like(self.attempt_counts)
        self.base_reward_by_slot = torch.zeros(3, device=device)
        self.extra_reward_by_slot = torch.zeros(3, device=device)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.attempt_index[env_ids] = 0

    def step(self, q, z, vz, foot_xy, target_xy, tilt_deg, died):
        # Mirror Episode.sample numbering: a confirmed flight ending at contact
        # consumes an index even if too short, below apex, off target, or fatal.
        attempt = (self.phase == self.FLIGHT) & (q > 0.002)
        self.attempt_index += attempt.long()
        base = super().step(q, z, vz, foot_xy, target_xy, tilt_deg, died)
        extra = torch.where(attempt & (self.attempt_index <= 2),
                            base * (self.factor - 1.0), torch.zeros_like(base))
        for i in range(3):
            slot = attempt & ((self.attempt_index == i + 1) if i < 2 else (self.attempt_index >= 3))
            self.attempt_counts[i] += slot.sum()
            self.rewarded_counts[i] += (slot & (base > 0)).sum()
            self.base_reward_by_slot[i] += torch.where(slot, base, 0.0).sum()
            self.extra_reward_by_slot[i] += torch.where(slot, extra, 0.0).sum()
        self.reward_sum += extra.sum()
        return base + extra

    def audit(self):
        return dict(arm=self.arm, multiplier=self.factor, slots=["first", "second", "later"],
                    attempt_counts=self.attempt_counts.tolist(), rewarded_counts=self.rewarded_counts.tolist(),
                    base_reward_by_slot=self.base_reward_by_slot.tolist(),
                    extra_reward_by_slot=self.extra_reward_by_slot.tolist())
