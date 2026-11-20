"""Stage 3b-A: one fixed XY landing point, first-hop scoring, unchanged 37-D policy."""

import json
import math

import torch

from .artifacts import ROOT, sha
from .collector import Collector
from .foot_touchdown_reward import contract as foot_contract
from .metrics import distribution


VARIANT = "real_release_point_short"
SERIES = "benchmark37_stage3b_point_short"
SOURCE = ROOT / "outputs/benchmark37_stage3a_foot_event/train_seed42_20261026_053258_136967Z/model_199.pt"
SOURCE_SHA256 = "ae4262c6a8d2c9ba75c9c4c4c58e7e9ebfec5dc2e667be1a18199421fa5a15e1"
ITERATIONS = 200
MAX_RADIUS_M = 0.40
EVAL_RADII_M = (0.20, 0.40)
DIR_COUNT = 8
BUCKET_COUNT = 1 + len(EVAL_RADII_M) * DIR_COUNT
STATIONARY_TRAIN_FRACTION = 0.10


def contract():
    return dict(
        name="first_hop_short_point_target_v1",
        source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=foot_contract(),
        intervention="only change target XY on reset; target held fixed through the 15s episode",
        target="world XY displacement from release root link: 10% zero; otherwise radius 0..0.4m "
               "and uniform heading during training",
        observation="unchanged 37D: existing pos_error[10:13] = R(root)^T*(target_world-root_world); "
                    "five past four-motor actions remain at [17:37]",
        reward="unchanged Stage 3a rewards; foot event still uses actual SpringLeg link center",
        release="root-link z 0.8..1.2m; motors stopped at reset and policy active after release; "
                "roll/pitch +/-5deg with 20% upright during training",
        evaluation="17 fixed buckets: stationary, 8 headings at 0.2m and 8 headings at 0.4m; "
                   "16 episodes per bucket; first post-drop jump is primary; all later jumps reported separately",
        horizon="15s as source; one target per episode, not first-hop termination",
        training_iterations=ITERATIONS,
        initialization="all 37D actor/critic/normalizer weights from pinned Stage 3a; fresh optimizer, iteration 0",
        code_sha256=sha(ROOT / "benchmark37/point_target_short.py"),
    )


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    pin = json.loads((ROOT / "benchmark37/stationary_foot_event_candidate.json").read_text())
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or pin.get("checkpoint") != str(path.relative_to(ROOT))
            or pin.get("checkpoint_sha256") != SOURCE_SHA256
            or pin.get("status") != "approved_for_stage3b_single_hop_simulation_development"
            or manifest.get("mode") != "train" or manifest.get("status") != "complete"
            or manifest.get("smoke") or manifest.get("variant") != "real_release_pretouch_foot_event"
            or manifest.get("source_lock") != source_lock or manifest.get("physics_lock") != physics_lock
            or manifest.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256
            or manifest.get("task_intervention") != foot_contract()):
        raise ValueError("Short-point warm start requires pinned, complete Stage 3a stationary model")
    return manifest


def bucket_name(bucket_id):
    if bucket_id == 0:
        return "stationary"
    radial_index, heading_index = divmod(bucket_id - 1, DIR_COUNT)
    return f"r{int(EVAL_RADII_M[radial_index]*100):02d}cm_heading{heading_index*45:03d}deg"


def reset_target(env, env_ids, seed, evaluation):
    """Set one target relative to the *actual* release root XY; use independent RNG."""
    if not len(env_ids):
        return
    if not hasattr(env, "_point_target_rng"):
        env._point_target_rng = torch.Generator(device=env.device).manual_seed(int(seed) + 830027)
        env._point_target_reset_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._point_target_bucket = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._point_target_displacement_w = torch.zeros(env.num_envs, 2, device=env.device)
    n = len(env_ids)
    if evaluation:
        bucket = (env_ids + 16 * env._point_target_reset_counts[env_ids]) % BUCKET_COUNT
        radial_index = torch.div((bucket - 1).clamp_min(0), DIR_COUNT, rounding_mode="floor")
        heading_index = (bucket - 1).clamp_min(0) % DIR_COUNT
        radii = torch.tensor(EVAL_RADII_M, device=env.device)
        radius = torch.where(bucket == 0, 0.0, radii[radial_index])
        angle = heading_index * (2.0 * math.pi / DIR_COUNT)
    else:
        radius = torch.rand(n, device=env.device, generator=env._point_target_rng) * MAX_RADIUS_M
        angle = torch.rand(n, device=env.device, generator=env._point_target_rng) * (2.0 * math.pi)
        stationary = torch.rand(n, device=env.device, generator=env._point_target_rng) < STATIONARY_TRAIN_FRACTION
        radius = torch.where(stationary, 0.0, radius)
        bucket = torch.full((n,), -1, device=env.device, dtype=torch.long)
    displacement = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)
    root_xy = env._robot.data.root_link_pos_w[env_ids, :2]
    env._desired_pos_w[env_ids, :2] = root_xy + displacement
    env._prev_distance_to_goal_xy[env_ids] = radius
    env._point_target_displacement_w[env_ids] = displacement
    env._point_target_bucket[env_ids] = bucket
    env._point_target_reset_counts[env_ids] += 1


def audit_target(env, evaluation):
    delta = env._desired_pos_w[:, :2] - env._robot.data.root_link_pos_w[:, :2]
    err = torch.linalg.vector_norm(delta - env._point_target_displacement_w, dim=-1)
    radius = torch.linalg.vector_norm(delta, dim=-1)
    if torch.any(err > 1e-5) or torch.any(radius > MAX_RADIUS_M + 1e-5):
        raise RuntimeError("Point target does not match the release-root-relative 0..0.4m contract")
    if evaluation and not torch.equal(env._point_target_bucket, torch.arange(env.num_envs, device=env.device) % BUCKET_COUNT):
        raise RuntimeError("Point evaluation did not initialize the fixed bucket schedule")
    return dict(radius_min_m=float(radius.min()), radius_max_m=float(radius.max()),
                target_error_max_m=float(err.max()), initial_bucket_ids=env._point_target_bucket.tolist())


class PointCollector(Collector):
    """Retain original episode diagnostics and add first-jump target-bucket reporting."""

    def record_reset_states(self):
        super().record_reset_states()
        pending = [i for i, e in enumerate(self.episodes)
                   if e.steps == 0 and self.counts[i] < self.p["episodes_per_env"]]
        if not pending:
            return
        disp = self.env._point_target_displacement_w[pending].detach().cpu().tolist()
        buckets = self.env._point_target_bucket[pending].detach().cpu().tolist()
        for i, xy, bucket in zip(pending, disp, buckets):
            self.episodes[i].start.update(target_displacement_world_xy_m=xy, target_bucket_id=bucket,
                                          target_bucket=bucket_name(bucket))

    def result(self):
        out = super().result()
        out["task"] = "one fixed 0..0.4m XY target per 15s episode; first post-drop jump is primary"
        buckets = {}
        for bucket_id in range(BUCKET_COUNT):
            rows = [r for r in self.rows if r["start"]["target_bucket_id"] == bucket_id]
            first = [r["events"][0] for r in rows if r["events"]]
            buckets[bucket_name(bucket_id)] = dict(
                episodes=len(rows), first_completed=len(first),
                first_strict=sum(r["first_strict"] for r in rows),
                first_strict_rate=sum(r["first_strict"] for r in rows)/len(rows) if rows else None,
                first_foot_error_m=distribution([e["foot_error_m"] for e in first]),
                first_apex_ge_1m=sum(e["valid_flight"] and e["apex_ge_1m"] for e in first),
                first_tilt_deg=distribution([e["touchdown_tilt_deg"] for e in first]),
                first_foot_gt_10cm=sum(e["foot_error_m"] > .10 for e in first),
                physical_deaths=sum(r["died"] for r in rows),
            )
        out["point_target_buckets"] = buckets
        out["point_target_note"] = "First post-drop jump is scored before any later corrections; "
        out["point_target_note"] += "15s episode may contain further jumps to the same target"
        return out
