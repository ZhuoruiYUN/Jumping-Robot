"""Stage 3a: one isolated foot-center touchdown reward on the frozen stationary task.

The reward uses simulator link truth only during training. Actor observations stay 37-D.
"""

import json

import torch

from .artifacts import ROOT, sha
from .pre_touch_control import contract as pretouch_contract


VARIANT = "real_release_pretouch_foot_event"
SERIES = "benchmark37_stage3a_foot_event"
SOURCE = ROOT / "outputs/benchmark37_stage2e_pretouch/train_seed42_20261026_032020_799352Z/model_199.pt"
SOURCE_SHA256 = "b61719e3f30d31c6bf959226d109dd93e5685e1aad5b89eaf35fac1e9808c97c"
ITERATIONS = 200
REWARD_SCALE = 4.0
FOOT_WIDTH_M = 0.05
FOOT_LIMIT_M = 0.10
APEX_MIN_M = 1.0
TILT_LIMIT_DEG = 15.0


def contract():
    return dict(
        name="stationary_foot_center_touchdown_reward_v1",
        source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=pretouch_contract(),
        observation="unchanged 37D; fixed stationary target; no future landing point",
        reward="once per confirmed post-release flight at actual SpringLeg sphere-center touchdown; "
               "4*exp(-0.5*(foot_xy_error/0.05m)^2), gated by foot error<=10cm, "
               "apex>=1m, tilt<=15deg and no death",
        initial_drop="first contact after release receives no touchdown reward",
        reset="clear event phase per episode; retain history through contacts within an episode",
        release="root-link z 0.8..1.2m; fixed +/-5deg roll/pitch with 20% upright; XY/yaw zero; "
                "policy enabled immediately after release; no new release perturbation",
        unchanged="SysID physics, motor delay/lag and DR, all existing reward terms, 37D actor, PPO and 15s horizon",
        training_iterations=ITERATIONS,
        initialization="weights and normalization from pinned pre-touch model; fresh optimizer and iteration zero",
        code_sha256=sha(ROOT / "benchmark37/foot_touchdown_reward.py"),
    )


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    selected = json.loads((ROOT / "benchmark37/stationary_pretouch_candidate.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or selected.get("checkpoint") != str(path.relative_to(ROOT))
            or selected.get("checkpoint_sha256") != SOURCE_SHA256
            or not selected.get("status", "").startswith("stage2c_and_stage2d_upright_long301_passed")
            or manifest.get("mode") != "train" or manifest.get("status") != "complete"
            or manifest.get("smoke") or manifest.get("variant") != "real_release_pretouch_goal10"
            or manifest.get("source_lock") != source_lock or manifest.get("physics_lock") != physics_lock
            or manifest.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256
            or manifest.get("task_intervention") != pretouch_contract()
            or manifest.get("reward_intervention", {}).get("code_sha256")
            != sha(ROOT / "benchmark37/energy_margin.py")):
        raise ValueError("Foot-event reward requires the pinned, completed pre-touch stationary model")
    return manifest


class FootTouchdownTracker:
    """Vectorized copy of the evaluator's drop/ground/candidate/flight event boundaries."""

    DROP, GROUND, CANDIDATE, FLIGHT = range(4)

    def __init__(self, num_envs, device, foot_limit_m=FOOT_LIMIT_M):
        self.foot_limit_m = foot_limit_m
        self.phase = torch.full((num_envs,), self.DROP, device=device, dtype=torch.long)
        self.air_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.peak = torch.zeros(num_envs, device=device)
        self.apex_seen = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.touchdowns = torch.zeros((), device=device, dtype=torch.long)
        self.qualified = torch.zeros((), device=device, dtype=torch.long)
        self.reward_sum = torch.zeros((), device=device)

    def reset(self, env_ids):
        self.phase[env_ids] = self.DROP
        self.air_steps[env_ids] = 0
        self.peak[env_ids] = 0
        self.apex_seen[env_ids] = False

    def step(self, q, z, vz, foot_xy, target_xy, tilt_deg, died):
        contact = q > 0.002
        takeoff = (q <= 0.001) & (vz > 0)
        before = self.phase.clone()
        drop_touch = (before == self.DROP) & contact
        launch = (before == self.GROUND) & takeoff
        airborne = (before == self.CANDIDATE) | (before == self.FLIGHT)
        touch = airborne & contact
        self.peak = torch.where(airborne, torch.maximum(self.peak, z), self.peak)
        self.apex_seen |= airborne & (vz <= 0)
        valid = touch & (before == self.FLIGHT) & (self.air_steps >= 5) & self.apex_seen & ~died
        foot_error = torch.linalg.vector_norm(foot_xy - target_xy, dim=-1)
        qualified = (valid & (self.peak >= APEX_MIN_M) & (tilt_deg <= TILT_LIMIT_DEG)
                     & (foot_error <= self.foot_limit_m))
        reward = torch.where(qualified,
                             REWARD_SCALE * torch.exp(-0.5 * torch.square(foot_error / FOOT_WIDTH_M)),
                             torch.zeros_like(foot_error))
        self.touchdowns += valid.sum()
        self.qualified += qualified.sum()
        self.reward_sum += reward.sum()
        self.air_steps = torch.where(launch, 1, self.air_steps)
        self.peak = torch.where(launch, z, self.peak)
        self.apex_seen = torch.where(launch, False, self.apex_seen)
        continuing = airborne & ~contact
        self.air_steps += continuing.long()
        self.phase = torch.where(drop_touch | touch, self.GROUND, self.phase)
        self.phase = torch.where(launch, self.CANDIDATE, self.phase)
        self.phase = torch.where(continuing & (self.air_steps >= 3), self.FLIGHT, self.phase)
        self.air_steps = torch.where(touch, 0, self.air_steps)
        self.apex_seen = torch.where(touch, False, self.apex_seen)
        return reward
