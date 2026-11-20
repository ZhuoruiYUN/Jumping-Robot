"""Fixed-case evaluation intervention; never enabled during training.

Reward terms are differences of the *actual* episode accumulators surrounding
the complete reward chain. No reward formula or simulator dynamics is replaced.
"""
import csv
import json
import math
from types import SimpleNamespace

import torch

from .artifacts import ROOT, sha, write_json
from .two_ground_targets import _ensure, _set_active_goal, _rotate_world_to_body


CHECKPOINTS = {
    "stage4c_A": (
        "outputs/benchmark37_stage4c_smooth_control/train_seed42_20261028_112425_724647Z/model_49.pt",
        "955be180780eb9e0f6faf0727febbb73b0ea6ce8900876b73e7c2810dfb9d9c1"),
    "stage4d_control": (
        "outputs/benchmark37_stage4d_foot_tail_control/train_seed42_20261028_134843_882013Z/model_49.pt",
        "97ff6287b9c831cfe035e92cf997926b81d2c87147b085c519f0e0a1c68aac65"),
}


def cases():
    patterns = [("stationary", 0., 0.), ("short20", .2, 0.), ("short40", .4, 0.)]
    patterns += [(f"long{int(r*100)}_{turn}", r, float(turn))
                 for r in (.6, .8) for turn in (0, 90, 180)]
    result = []
    for name, radius, turn in patterns:
        for height in (.8, 1., 1.2):
            for heading in (0., 90., 180., 270.):
                # Four stationary entries intentionally provide four DR draws.
                def vector(angle):
                    return [radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))]
                result.append(dict(case_id=len(result), pattern=name, release_height_m=height,
                                   heading_deg=heading, turn_deg=turn, edge_m=radius,
                                   first_edge_xy_m=vector(heading), next_edge_xy_m=vector(heading + turn)))
    return result


def contract():
    return dict(name="fixed_two_ground_reward_diagnostic_v1", cases=cases(),
                code_sha256=sha(__file__), episode_count=len(cases()), horizon_s=15.,
                scope="diagnostic grid, not the random-target benchmark and not a promotion score",
                release="root-link z=0.8/1.0/1.2m, upright yaw=0, zero linear/angular velocity; motors reset OFF, policy immediately enabled",
                randomization="source mass/inertia/motor-tau DR and observation noise retained; first quota episode only; reset RNG fork prevents discarded resets shifting active rows",
                queue="fixed P_t and P_(t+1); strict hits advance, misses retry; after two strict route advances both points hold the original P_(t+1)",
                rewards="actual accumulator deltas after full reward chain, before auto-reset; no reward changes; float32 accumulation cancellation audited against returned reward",
                windows="initial drop includes first-touch sample; first attempt starts next step through its confirmed touchdown; second attempt through next confirmed touchdown; incomplete windows retained on death/timeout",
                phases="initial_drop, contact(q>0.002), ascent(vz>0), descent(vz<=0); touchdown reward belongs to contact",
                second_attempt="true next point only if first attempt strictly hit; otherwise retry; missing attempts count as failures",
                limitation="reward magnitude and kinematic association do not establish causal policy-gradient conflict; power remains an uncalibrated proxy")


def validate_checkpoint(path):
    path = path.resolve(strict=True)
    for label, (relative, expected) in CHECKPOINTS.items():
        if path == (ROOT / relative).resolve() and sha(path) == expected:
            manifest = json.loads((path.parent / "manifest.json").read_text())
            if manifest.get("status") != "complete" or manifest.get("smoke"):
                break
            return label
    # New arms use completed, hashed training provenance rather than assuming a
    # future timestamp. Historical pinned checkpoints above remain unchanged.
    from .flight_power_ab import validate_trained_checkpoint
    try:
        return "stage4e_power_" + validate_trained_checkpoint(path)
    except ValueError:
        pass
    from .apex_deficit_ab import validate_trained_checkpoint as validate_deficit
    try:
        return "stage4f_apex_"+validate_deficit(path)
    except ValueError:
        pass
    from .first_hop_margin_ab import validate_trained_checkpoint as validate_margin
    try:
        return "stage4g_margin_"+validate_margin(path)
    except ValueError:
        pass
    raise ValueError("Fixed reward diagnostic requires a pinned historical or completed matched A/B checkpoint")


def checkpoint_digest(label, path):
    if label in CHECKPOINTS:
        return CHECKPOINTS[label][1]
    if label.startswith("stage4g_margin_"):
        from .first_hop_margin_ab import validate_trained_checkpoint as validate_margin
        if label != "stage4g_margin_"+validate_margin(path):
            raise ValueError("Stage4g label does not match checkpoint")
        return sha(path)
    if label.startswith("stage4f_apex_"):
        from .apex_deficit_ab import validate_trained_checkpoint as validate_deficit
        if label != "stage4f_apex_"+validate_deficit(path):
            raise ValueError("Stage4f label does not match checkpoint")
        return sha(path)
    from .flight_power_ab import validate_trained_checkpoint
    if label != "stage4e_power_"+validate_trained_checkpoint(path):
        raise ValueError("Stage4e diagnostic label does not match checkpoint")
    return sha(path)


def comparison_order(labels):
    historical = tuple(CHECKPOINTS)
    flight_power = ("stage4e_power_control", "stage4e_power_relaxed")
    apex_deficit = ("stage4f_apex_control", "stage4f_apex_penalty")
    first_margin = ("stage4g_margin_control", "stage4g_margin_margin3cm")
    for pair in (historical,flight_power,apex_deficit,first_margin):
        if len(labels) == 2 and set(labels) == set(pair):
            return pair
    raise ValueError("Comparison requires a matched historical, Stage4e or Stage4f pair")


def reset_rng_context(device):
    device = torch.device(device)
    return torch.random.fork_rng(devices=([device.index if device.index is not None else torch.cuda.current_device()]
                                          if device.type == "cuda" else []))


def reset_pair(env, ids, seed):
    if not len(ids):
        return
    _ensure(env, seed)
    if env.num_envs != len(cases()):
        raise ValueError("Fixed-case protocol requires exactly 108 environments")
    table = cases()
    chosen = [table[int(i)] for i in ids.detach().cpu().tolist()]
    pose = torch.cat((env._robot.data.root_link_pos_w[ids], env._robot.data.root_quat_w[ids]), -1).clone()
    pose[:, 2] = torch.tensor([c["release_height_m"] for c in chosen], device=env.device)
    pose[:, 3:] = 0
    pose[:, 3] = 1
    env._robot.write_root_pose_to_sim(pose, ids)
    env._robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), ids)
    start = pose[:, :2]
    env._two_ground_pt_xy_w[ids] = start + torch.tensor([c["first_edge_xy_m"] for c in chosen], device=env.device)
    env._two_ground_pt1_xy_w[ids] = env._two_ground_pt_xy_w[ids] + torch.tensor(
        [c["next_edge_xy_m"] for c in chosen], device=env.device)
    env._two_ground_route_index[ids] = 0
    env._two_ground_total_resets += len(ids)
    _set_active_goal(env, ids)


def advance_on_hit(env, hit):
    ids = torch.nonzero(hit, as_tuple=False).flatten()
    if not len(ids):
        return
    # Fixed two-point case: P_(t+1) remains in place after being promoted.
    env._two_ground_pt_xy_w[ids] = env._two_ground_pt1_xy_w[ids].clone()
    env._two_ground_route_index[ids] += 1
    env._two_ground_total_advances += len(ids)
    _set_active_goal(env, ids)


def fixed_api(source):
    result = SimpleNamespace(**{k: getattr(source, k) for k in dir(source) if not k.startswith("__")})
    result.reset_pair, result.advance_on_hit = reset_pair, advance_on_hit
    return result


def velocity_components(q, velocity_w):
    horizontal = velocity_w.clone()
    horizontal[:, 2] = 0
    vertical = velocity_w - horizontal
    h = _rotate_world_to_body(q, horizontal)[:, :2]
    v = _rotate_world_to_body(q, vertical)[:, :2]
    # Cross term can be negative. These are an identity, not independent costs.
    return h.square().sum(-1), v.square().sum(-1), 2 * (h * v).sum(-1)


KINEMATIC_FIELDS = ["spring_q_m", "root_z_m", "vz_m_s", "world_vxy_m_s", "body_vxy_sq",
                    "body_vxy_from_world_horizontal_sq", "body_vxy_from_world_vertical_sq",
                    "body_vxy_cross", "tilt_deg", "foot_error_m", "root_error_m", "route_index",
                    "P_t_x", "P_t_y", "P_next_x", "P_next_y", "foot_x", "foot_y",
                    "vx_m_s", "vy_m_s"]


def read_kinematics(env):
    d = env._robot.data
    h, v, cross = velocity_components(d.root_quat_w, d.root_lin_vel_w)
    lateral = d.root_lin_vel_b[:, :2].square().sum(-1)
    if not torch.allclose(h + v + cross, lateral, atol=1e-4, rtol=1e-4):
        raise RuntimeError("Root body/world velocity conventions differ; stop reward decomposition")
    foot = d.body_link_pos_w[:, env._foot_body_id, :2]
    goal = env._two_ground_pt_xy_w
    tilt = torch.rad2deg(torch.acos((1 - 2*d.root_quat_w[:, 1:3].square().sum(-1)).clamp(-1, 1)))
    return torch.stack((d.joint_pos[:, env._spring_joint_id], d.root_link_pos_w[:, 2],
        d.root_lin_vel_w[:, 2], torch.linalg.vector_norm(d.root_lin_vel_w[:, :2], dim=-1),
        lateral, h, v, cross, tilt, torch.linalg.vector_norm(foot-goal, dim=-1),
        torch.linalg.vector_norm(d.root_link_pos_w[:, :2]-goal, dim=-1),
        env._two_ground_route_index.float(), goal[:, 0], goal[:, 1],
        env._two_ground_pt1_xy_w[:, 0], env._two_ground_pt1_xy_w[:, 1],
        foot[:, 0], foot[:, 1], d.root_lin_vel_w[:, 0], d.root_lin_vel_w[:, 1]), -1)


def reward_deltas(before, after, reward):
    if set(before) != set(after):
        raise RuntimeError("Reward accumulator keys changed inside reward callback")
    names = list(before)
    values = torch.stack([after[k] - before[k] for k in names], -1)
    error = (values.sum(-1) - reward).abs()
    if not torch.isfinite(values).all() or not torch.isfinite(reward).all() or error.max() > 5e-4:
        raise RuntimeError(f"Reward decomposition does not reconcile: max error {error.max().item()}")
    return names, values, error


def instrument_collector(base):
    class FixedRewardCollector(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reward_file = (self.folder / "fixed_reward_steps.csv").open("w", newline="")
            self.reward_writer = None
            self.pending_reward = []
            self.reward_error_max = 0.
            self.reward_samples = 0
            self.case_table = cases()
            self.initial_state = None

        def record_reset_states(self):
            super().record_reset_states()
            if self.initial_state is not None:
                return
            e, d = self.env, self.env._robot.data
            tensors = dict(root_link_pos_w=d.root_link_pos_w, root_quat_w=d.root_quat_w,
                           root_com_velocity_w=d.root_com_lin_vel_w, root_angular_velocity_w=d.root_ang_vel_w,
                           joint_pos=d.joint_pos, joint_vel=d.joint_vel,
                           mass_multiplier=e.dr_mass_multi, inertia_multiplier=e.dr_inertia_multi,
                           motor_tau=e.dr_tm, motor_state=e._motor_u,
                           action_history=torch.stack(tuple(e._action_history), dim=1),
                           P_t=e._two_ground_pt_xy_w, P_next=e._two_ground_pt1_xy_w)
            self.initial_state = {k: v.detach().cpu().tolist() for k, v in tensors.items()}
            for i, ep in enumerate(self.episodes):
                ep.start["fixed_case"] = self.case_table[i]
            write_json(self.folder / "fixed_initial_states.json", self.initial_state)
            write_json(self.folder / "fixed_case_contract.json", contract())

        def sample(self, died, timeout):
            # Hold the old Episode references: base.sample replaces terminated
            # episodes before _get_rewards is called by DirectRLEnv.
            refs = list(self.episodes)
            active = [(i, len(ep.events), ep.first_touch_step is None)
                      for i, ep in enumerate(refs) if self.counts[i] == 0 and len(ep.events) < 2]
            signals = read_kinematics(self.env).detach().cpu().tolist() if active else None
            super().sample(died, timeout)
            self.pending_reward = []
            for i, attempt_before, drop in active:
                ep, s = refs[i], signals[i]
                phase = "initial_drop" if drop else "contact" if s[0] > .002 else "ascent" if s[2] > 0 else "descent"
                self.pending_reward.append((i, ep.steps, 0 if drop else attempt_before+1,
                                            phase, len(ep.events) > attempt_before, s))

        def capture_reward(self, before, reward):
            names, terms, errors = reward_deltas(before, self.env._episode_sums, reward)
            self.reward_error_max = max(self.reward_error_max, float(errors.max()))
            if not self.pending_reward:
                return
            indices = [item[0] for item in self.pending_reward]
            values = torch.cat((terms, reward[:, None], errors[:, None]), -1)[indices].detach().cpu().tolist()
            if self.reward_writer is None:
                fields = ["env_id", "step", "attempt", "phase", "touchdown_event"] + KINEMATIC_FIELDS
                fields += ["r/"+name for name in names] + ["returned_reward", "reconciliation_abs_error"]
                self.reward_writer = csv.writer(self.reward_file)
                self.reward_writer.writerow(fields)
            for (i, step, attempt, phase, event, s), row in zip(self.pending_reward, values):
                self.reward_writer.writerow([i, step, attempt, phase, int(event)] + s + row)
            self.reward_samples += len(values)
            self.pending_reward = []

        def result(self):
            self.reward_file.flush()
            result = super().result()
            result["task"] = "fixed two-ground diagnostic grid; only first two attempts used for reward analysis"
            result["two_ground"]["case_note"] = "Fixed 108-case grid; paired initial states must be verified across policies. Targets hold at the second point after strict advancement."
            result["fixed_case_rewards"] = dict(contract=contract(), samples=self.reward_samples,
                reconciliation_max_abs_error=self.reward_error_max,
                initial_state_sha256=sha(self.folder / "fixed_initial_states.json"),
                promotion="not a promotion evaluation; fixed grid differs from random full-range benchmark")
            return result

        def close(self):
            self.reward_file.close()
            super().close()
    return FixedRewardCollector
