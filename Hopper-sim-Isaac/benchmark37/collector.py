"""Passive evaluation only: called before DirectRLEnv auto-reset."""
import json
import torch
from .metrics import Episode, summarize, distribution
from .geometry_diagnostics import GeometryAudit


def audit_asset(env, asset_path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(asset_path))
    prims = list(stage.Traverse())
    body = next(p for p in prims if p.GetName() == "Body")
    leg = next(p for p in prims if p.GetName() == "SpringLeg")
    if leg.GetTypeName() != "Sphere":
        raise RuntimeError("Foot origin is not a sphere center; review geometry before scoring")
    cache = UsdGeom.XformCache()
    rest = cache.GetLocalToWorldTransform(body).GetInverse().Transform(
        cache.GetLocalToWorldTransform(leg).ExtractTranslation())
    attributes = {}
    for prim in prims:
        values = {}
        for a in prim.GetAttributes():
            name = a.GetName()
            if any(k in name.lower() for k in ("mass", "inertia", "centerofmass", "limit", "stiffness", "damping", "friction", "restitution", "radius", "axis", "localpos")):
                values[name] = str(a.Get())
        if values:
            attributes[str(prim.GetPath())] = values
    robot = env._robot
    masses = robot.root_physx_view.get_masses().to(env.device)
    inertias = robot.root_physx_view.get_inertias().to(env.device)
    com = (masses[..., None] * robot.data.body_com_pos_w).sum(1) / masses.sum(1, keepdim=True)
    return dict(asset_attributes=attributes, body_names=robot.body_names,
                runtime_link_masses_kg=masses[0].tolist(),
                runtime_link_inertias_kgm2=inertias[0].tolist(),
                runtime_joint_stiffness=robot.root_physx_view.get_dof_stiffnesses()[0].tolist(),
                runtime_joint_damping=robot.root_physx_view.get_dof_dampings()[0].tolist(),
                runtime_joint_limits=robot.root_physx_view.get_dof_limits()[0].tolist(),
                root_link_world=robot.data.root_link_pos_w[0].tolist(),
                root_com_world=robot.data.root_com_pos_w[0].tolist(),
                whole_vehicle_com_world=com[0].tolist(),
                foot_rest_body_m=list(rest), foot_prim_type=str(leg.GetTypeName()),
                note="COM and exact foot are evaluation truth only; never added to actor")


def validate_sysid_asset(audit):
    """Fail before learning if composed PhysX mass/inertia misses the q=0 proxy."""
    if audit["body_names"] != ["Body", "SpringLeg"]:
        raise RuntimeError(f"Unexpected SysID body order: {audit['body_names']}")
    masses = audit["runtime_link_masses_kg"]
    matrices = audit["runtime_link_inertias_kgm2"]
    expected_masses = (.145, .010)
    expected_diagonals = ((.00036765048272165417, .00032697633812165414, .00054),
                          (.00001, .00001, .00001))
    for actual, expected in zip(masses, expected_masses):
        if abs(actual - expected) > 1e-5:
            raise RuntimeError(f"Composed SysID link mass {actual} differs from {expected}")
    for matrix, expected in zip(matrices, expected_diagonals):
        if len(matrix) != 9:
            raise RuntimeError("Unexpected PhysX inertia layout; inspect asset_audit.json")
        for actual, target in zip((matrix[0], matrix[4], matrix[8]), expected):
            if abs(actual - target) > 1e-6:
                raise RuntimeError(f"Composed SysID link inertia {actual} differs from {target}")
    d = audit["foot_rest_body_m"][2]
    reduced_mass = masses[0] * masses[1] / sum(masses)
    composed = (matrices[0][0] + matrices[1][0] + reduced_mass * d * d,
                matrices[0][4] + matrices[1][4] + reduced_mass * d * d,
                matrices[0][8] + matrices[1][8])
    for actual, target in zip(composed, (.001016642785, .0009759686404, .00055)):
        if abs(actual - target) > 1e-6:
            raise RuntimeError(f"Composed SysID q=0 whole inertia {actual} differs from {target}")
    return dict(q0_whole_mass_kg=sum(masses), q0_whole_inertia_kgm2=composed,
                interpretation="q=0 two-link proxy; not measured link inertias at compressed leg")


class Collector:
    def __init__(self, env, protocol, folder, audit, sysid_support=False, real_release=False,
                 pre_touch_control=False):
        self.env, self.p, self.folder = env, protocol, folder
        self.sysid_support = sysid_support
        self.real_release = real_release
        self.pre_touch_control = pre_touch_control
        self.motor_samples = self.motor_below = self.motor_above = 0
        self.release_forced_off_samples = self.release_off_violations = 0
        self.pre_touch_motor_samples = self.pre_touch_nonzero_steps = 0
        self.pre_touch_max_motor_u = 0.0
        self.motor_active_samples = self.motor_active_below = self.motor_active_above = 0
        self.rows = []
        self.counts = [0] * env.num_envs
        self.episodes = [Episode(protocol) for _ in self.counts]
        self.previous_u = env._motor_u.clone()
        self.leg_id = env._robot.find_bodies("SpringLeg")[0][0]
        self.rest = torch.tensor(audit["foot_rest_body_m"], device=env.device)
        self.file = (folder / "episodes.jsonl").open("w")
        self.trace = (folder / "trace_first_episodes.jsonl").open("w")
        self.geometry_file = (folder / "geometry_diagnostics.jsonl").open("w")
        self.geometry = [GeometryAudit() for _ in self.counts]

    @property
    def complete(self):
        return min(self.counts) >= self.p["episodes_per_env"]

    def record_reset_states(self):
        """Read actual reset state before the first action, without sampling observations."""
        pending = [i for i, e in enumerate(self.episodes)
                   if e.steps == 0 and self.counts[i] < self.p["episodes_per_env"]]
        if not pending:
            return
        env, d = self.env, self.env._robot.data
        state = torch.stack([d.root_link_pos_w[:, 2], d.joint_pos[:, env._spring_joint_id],
                             d.joint_vel[:, env._spring_joint_id], env._desired_pos_w[:, 2],
                             env._motor_u.abs().amax(-1)], dim=1)[pending].detach().cpu().tolist()
        for i, a in zip(pending, state):
            self.episodes[i].start = dict(root_z_m=a[0], joint_q_m=a[1], joint_qdot_m_s=a[2],
                                          target_z_m=a[3], motor_state_absmax=a[4],
                                          snapshot_phase="reset_before_first_action")
        if hasattr(env, "_release_tilt_rp"):
            from benchmark37.release_tilt import audit_tilt
            audit_tilt(env, pending)
            rp = torch.rad2deg(env._release_tilt_rp[pending]).detach().cpu().tolist()
            quats = d.root_quat_w[pending].detach().cpu().tolist()
            for i, angles, quat in zip(pending, rp, quats):
                self.episodes[i].start.update(release_roll_pitch_deg=angles, root_quaternion_wxyz=quat)

    def sample(self, died, timeout):
        from isaaclab.utils.math import quat_apply
        env = self.env
        d = env._robot.data
        q = d.joint_pos[:, env._spring_joint_id]
        offset = self.rest.expand(env.num_envs, 3).clone()
        offset[:, 2] += q
        estimate = d.root_link_pos_w + quat_apply(d.root_quat_w, offset)
        foot = d.body_link_pos_w[:, self.leg_id]
        tilt = torch.rad2deg(torch.acos((1-2*(d.root_quat_w[:, 1:3]**2).sum(-1)).clamp(-1, 1)))
        alpha = env._dt_policy / (env.dr_tm + env._dt_policy)
        duty = (env._motor_u - (1-alpha)*self.previous_u) / alpha
        release_pending = (env._release_pending if self.real_release else
                           torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        data = torch.stack([
            d.root_link_pos_w[:, 2], q, d.root_com_lin_vel_w[:, 2], env._desired_pos_w[:, 2],
            torch.linalg.vector_norm(foot-estimate, dim=-1), duty.abs().amax(-1),
            env._motor_u.abs().amax(-1), env._thrust[:, 0, 2],
            torch.linalg.vector_norm(foot[:, :2]-env._desired_pos_w[:, :2], dim=-1),
            torch.linalg.vector_norm(d.root_link_pos_w[:, :2]-env._desired_pos_w[:, :2], dim=-1),
            tilt, d.root_com_lin_vel_w[:, 0], d.root_com_lin_vel_w[:, 1],
            died.float(), timeout.float(), foot[:, 2],
            (env._motor_u < .2437326064).sum(-1).float(),
            (env._motor_u > .7331279315).sum(-1).float(),
            release_pending.float(),
        ], dim=1).detach().cpu().tolist()
        # Evaluation-only link truth, never passed to policy. Capture the boundaries
        # around geometry failures even outside the two legacy full-step traces.
        geometry_data = torch.cat((d.root_link_pos_w, d.root_quat_w, foot, estimate,
                                   d.body_link_quat_w[:, self.leg_id],
                                   d.joint_vel[:, env._spring_joint_id:env._spring_joint_id+1],
                                   d.root_ang_vel_w), dim=1).detach().cpu().tolist()
        self.previous_u.copy_(env._motor_u)
        self.previous_u[died | timeout] = 0
        for i, a in enumerate(data):
            if self.counts[i] >= self.p["episodes_per_env"]:
                continue
            if self.sysid_support:
                self.motor_samples += 4
                self.motor_below += int(a[16])
                self.motor_above += int(a[17])
                if self.real_release and bool(a[18]):
                    if self.pre_touch_control:
                        self.pre_touch_motor_samples += 4
                        self.pre_touch_nonzero_steps += int(a[6] > 1e-6)
                        self.pre_touch_max_motor_u = max(self.pre_touch_max_motor_u, a[6])
                        self.motor_active_samples += 4
                        self.motor_active_below += int(a[16])
                        self.motor_active_above += int(a[17])
                    else:
                        self.release_forced_off_samples += 4
                        self.release_off_violations += int(a[5] > 1e-6 or a[6] > 1e-6
                                                           or abs(a[7]) > 1e-6)
                else:
                    self.motor_active_samples += 4
                    self.motor_active_below += int(a[16])
                    self.motor_active_above += int(a[17])
            s = dict(zip(("z", "q", "vz", "target_z", "foot_formula_error", "duty_max",
                          "motor_max", "force", "foot_error", "root_error", "tilt"), a[:11]))
            s.update(velocity=[a[11], a[12], a[2]], died=bool(a[13]))
            g = geometry_data[i]
            self.geometry[i].sample(dict(
                step=self.episodes[i].steps+1, phase_before_sample=self.episodes[i].phase,
                error_m=a[4], tilt_deg=a[10], spring_q_m=a[1], spring_qdot_m_s=g[17],
                died=bool(a[13]), timed_out=bool(a[14]), release_pending=bool(a[18]),
                root_link_world=g[:3], root_quaternion_wxyz=g[3:7], foot_center_world=g[7:10],
                estimated_foot_world=g[10:13], leg_quaternion_wxyz=g[13:17],
                root_com_velocity_world=s["velocity"], root_angular_velocity_world=g[18:21]))
            if i < 2 and self.counts[i] == 0:
                self.trace.write(json.dumps(dict(env_id=i, step=self.episodes[i].steps+1,
                                                foot_center_z_world=a[15],
                                                release_pending=bool(a[18]), **s), allow_nan=False) + "\n")
            self.episodes[i].sample(s)
            if a[13] or a[14]:
                row = self.episodes[i].finish(bool(a[13]), bool(a[14]), env.step_dt)
                row.update(env_id=i, episode_index=self.counts[i])
                self.file.write(json.dumps(row, allow_nan=False) + "\n")
                self.file.flush()
                self.rows.append(row)
                self.geometry_file.write(json.dumps(dict(env_id=i, episode_index=self.counts[i],
                                                         **self.geometry[i].result()), allow_nan=False) + "\n")
                self.geometry_file.flush()
                self.geometry[i] = GeometryAudit()
                self.counts[i] += 1
                self.episodes[i] = Episode(self.p)

    def result(self):
        result = summarize(self.rows)
        if hasattr(self.env, "_release_tilt_rp"):
            result["initial_release_attitude"] = dict(
                audited_episodes=len(self.rows),
                roll_deg=distribution([r["start"]["release_roll_pitch_deg"][0] for r in self.rows]),
                pitch_deg=distribution([r["start"]["release_roll_pitch_deg"][1] for r in self.rows]),
                note="actual quaternion checked against requested angles before each episode's first action")
        result.update(complete_quotas=self.complete, per_env_episodes=self.counts,
                      foot_formula_within_1mm=result["foot_formula_max_error_m"] <= .001,
                      task=("stage 2c real-height release, stationary fixed target; no directional claims"
                            if self.real_release else "original stationary fixed target; no directional claims"),
                      touchdown_detector="spring compression proxy, not measured contact force")
        if self.sysid_support:
            result["sysid_motor_w_exposure"] = dict(
                fitted_support=[.2437326064, .7331279315],
                motor_step_samples=self.motor_samples,
                below_support=self.motor_below, above_support=self.motor_above,
                outside_support_fraction=(self.motor_below + self.motor_above) / self.motor_samples,
                note="outside range is polynomial extrapolation, not a measured actuator model",
            )
            if self.real_release:
                result["sysid_motor_w_exposure"].update(
                    active_motor_samples=self.motor_active_samples,
                    active_below_support=self.motor_active_below,
                    active_above_support=self.motor_active_above,
                    active_outside_support_fraction=(self.motor_active_below + self.motor_active_above)
                    / max(1, self.motor_active_samples),
                    note=("all post-release motor steps are active-policy exposure"
                          if self.pre_touch_control else
                          "forced-off zero is intentional; use active_outside_support_fraction for fitted-model exposure"),
                )
                if self.pre_touch_control:
                    result["pre_touch_control"] = dict(
                        enabled=True, motor_samples=self.pre_touch_motor_samples,
                        nonzero_actuator_steps=self.pre_touch_nonzero_steps,
                        max_motor_u=self.pre_touch_max_motor_u,
                        note="Release-pending marks the initial drop for measurement; actions are not gated")
                else:
                    result["sysid_motor_w_exposure"]["forced_off_motor_samples"] = self.release_forced_off_samples
                    result["release_gate_off_violations"] = self.release_off_violations
        return result

    def close(self):
        self.file.close()
        self.trace.close()
        self.geometry_file.close()
