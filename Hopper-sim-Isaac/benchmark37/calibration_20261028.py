"""Versioned spring/air physics update; task, actor and actuator code are inherited."""

import json

import torch

from .artifacts import ROOT, sha, verify_sysid_sources

PARAMS_PATH = ROOT / "benchmark37/calibration_20261028.json"
ASSET_PATH = ROOT / "benchmark37/HopperCalibrated20261028.usda"
PARAMS = json.loads(PARAMS_PATH.read_text())
REVISION = PARAMS["revision"]
SERIES = "benchmark37_stage4a_calibrated_20261028"
SOURCE = ROOT / "outputs/benchmark37_stage4a_two_ground_short/train_seed42_20261027_090744_980836Z/model_199.pt"
SOURCE_SHA256 = "6a239712638c23ec61140f5206ef5e0d4d203711dd097e0c91e8c74f062db36a"


def physics_contract():
    return dict(revision=REVISION, parameters=PARAMS,
                files={str(path.relative_to(ROOT)): sha(path) for path in
                       (PARAMS_PATH, ASSET_PATH, ROOT / "benchmark37/calibration_20261028.py")},
                base_sysid_lock=verify_sysid_sources())


def configure(cfg):
    """Override an instantiated config (configclass fields are not class attrs)."""
    cfg.robot = cfg.robot.replace(spawn=cfg.robot.spawn.replace(
        usd_path=str(ASSET_PATH), rigid_props=cfg.robot.spawn.rigid_props.replace(linear_damping=0.0)))
    cfg.sim.dt = PARAMS["timing"]["physics_dt_s"]
    cfg.decimation = PARAMS["timing"]["decimation"]
    cfg.sim.render_interval = cfg.decimation
    if abs(cfg.sim.dt * cfg.decimation - .01) > 1e-12:
        raise RuntimeError("Calibration must retain the 100Hz actuator/policy contract")


def axial_force(v):
    air = PARAMS["air"]
    descending = air["descending_saturation_N_whole"] * torch.tanh(
        (v / air["descending_transition_speed_m_s"]).square())
    ascending = -air["ascending_quadratic_Ns2_m2_whole"] * v.square()
    return torch.where(v < 0, descending, ascending)


def air_force_body(link_velocities_body):
    """(N,2,3), Body/SpringLeg COM velocity in root Body axes; returns newton."""
    masses = link_velocities_body.new_tensor([PARAMS["mass_kg"][name] for name in ("Body", "SpringLeg")])
    rates = link_velocities_body.new_tensor([PARAMS["air"]["lateral_damping_rate_s_inv"][name]
                                            for name in ("Body", "SpringLeg")])
    force = -link_velocities_body * (masses * rates)[None, :, None]
    force[..., 2] = axial_force(link_velocities_body[..., 2]) * (masses / masses.sum())[None, :]
    return force


def rotate(q, v, inverse=False):
    """wxyz quaternion rotation with tensor broadcasting."""
    vector = -q[..., 1:] if inverse else q[..., 1:]
    t = 2 * torch.cross(vector, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(vector, t, dim=-1)


class CalibratedAirMixin:
    def _apply_action(self):
        if self._robot.body_names != ["Body", "SpringLeg"]:
            raise RuntimeError("Calibrated air assumes Body/SpringLeg order")
        q = self._robot.data.root_quat_w[:, None, :].expand(-1, 2, -1)
        velocity = rotate(q, self._robot.data.body_com_lin_vel_w, inverse=True)
        air_b = air_force_body(velocity)
        force_b = air_b.clone()
        force_b[:, 0:1] += self._thrust
        torque_b = torch.zeros_like(force_b)
        torque_b[:, 0:1] = self._moment
        # Replace all link wrenches each physics substep; do not accumulate air.
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=[0, 1], forces=rotate(q, force_b), torques=rotate(q, torque_b), is_global=True)
        if not hasattr(self, "_calibration_air_calls"):
            self._calibration_air_calls = 0
            self._calibration_air_max_n = torch.zeros((), device=self.device)
            self._calibration_air_max_power_w = torch.zeros((), device=self.device)
        self._calibration_air_calls += 1
        self._calibration_air_max_n = torch.maximum(
            self._calibration_air_max_n, torch.linalg.vector_norm(air_b.sum(1), dim=-1).max())
        self._calibration_air_max_power_w = torch.maximum(
            self._calibration_air_max_power_w, (air_b * velocity).sum((-1, -2)).max())


def runtime_air_audit(env):
    result = dict(physics_substep_calls=getattr(env, "_calibration_air_calls", 0),
                max_net_air_force_n=float(getattr(env, "_calibration_air_max_n", 0)),
                max_air_power_w=float(getattr(env, "_calibration_air_max_power_w", 0)),
                physics_dt_s=env.sim.cfg.dt, policy_dt_s=env.step_dt)
    if (abs(result["physics_dt_s"] - .005) > 1e-12 or abs(result["policy_dt_s"] - .01) > 1e-12
            or result["max_air_power_w"] > 1e-6):
        raise RuntimeError("Calibrated runtime violates timing or dissipative-air contract")
    return result


def validate_source(path, source_lock, physics_lock):
    from .two_ground_targets import VARIANT, contract
    path = path.resolve(strict=True)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or manifest.get("status") != "complete" or manifest.get("mode") != "train"
            or manifest.get("smoke") or manifest.get("physics_revision")
            or manifest.get("variant") != VARIANT or manifest.get("task_intervention") != contract()
            or manifest.get("source_lock") != source_lock or manifest.get("physics_lock") != physics_lock
            or manifest.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256):
        raise ValueError("Calibrated training requires the pinned completed 41D Stage4a source")
    return manifest


def initialize_runner(runner, path):
    if runner.alg.optimizer.state:
        raise RuntimeError("Physics adaptation needs a fresh optimizer")
    saved = torch.load(path, map_location=runner.device, weights_only=False)["model_state_dict"]
    runner.alg.policy.load_state_dict(saved, strict=True)
    runner.current_learning_iteration = 0
    return dict(source_checkpoint=str(path), source_sha256=sha(path), obs_dim=41,
                transfer="all policy/critic/normalizer tensors copied exactly", new_iteration=0,
                optimizer_state_entries=len(runner.alg.optimizer.state))


def validate_asset(audit):
    masses = audit["runtime_link_masses_kg"]
    inertias = audit["runtime_link_inertias_kgm2"]
    if audit["body_names"] != ["Body", "SpringLeg"]:
        raise RuntimeError("Unexpected calibrated body ordering")
    for actual, expected in zip(masses, (.1456, .01)):
        if abs(actual - expected) > 1e-6:
            raise RuntimeError("Runtime calibrated masses disagree with config")
    d = audit["foot_rest_body_m"]
    if max(abs(d[i]) for i in (0, 1)) > 1e-6 or abs(d[2] + .273) > 1e-6:
        raise RuntimeError("Runtime asset rest-foot geometry differs from calibrated geometry")
    reduced_mass = masses[0] * masses[1] / sum(masses)
    whole = [inertias[0][j] + inertias[1][j] + (reduced_mass * d[2]**2 if j != 8 else 0)
             for j in (0, 4, 8)]
    for actual, target in zip(whole, PARAMS["whole_q0_inertia_kgm2"]):
        if abs(actual - target) > 1e-6:
            raise RuntimeError("Runtime calibrated inertia proxy disagrees with config")
    attrs = audit["asset_attributes"]
    expected = {"physics:upperLimit": .087, "drive:linear:physics:stiffness": 560.,
                "drive:linear:physics:damping": .6}
    for key, value in expected.items():
        if abs(float(attrs["/Drone/center_spring_joint"][key]) - value) > 1e-5:
            raise RuntimeError(f"Composed spring mismatch: {key}")
    if (len(audit["runtime_joint_stiffness"]) != 1
            or abs(audit["runtime_joint_stiffness"][0] - 560.) > 1e-4
            or abs(audit["runtime_joint_damping"][0] - .6) > 1e-5
            or abs(audit["runtime_joint_limits"][0][0]) > 1e-6
            or abs(audit["runtime_joint_limits"][0][1] - .087) > 1e-6):
        raise RuntimeError("Live PhysX spring stiffness/damping/travel do not match calibration")
    for name in ("Body", "SpringLeg"):
        if float(attrs[f"/Drone/{name}"]["physxRigidBody:linearDamping"]) != 0:
            raise RuntimeError("PhysX linear damping must be zero to avoid double-counting air")
    return dict(q0_whole_mass_kg=sum(masses), q0_whole_inertia_kgm2=whole,
                root_A_to_ground_q0_m=.285, spring=PARAMS["spring"],
                compression_travel_m=.087,
                interpretation=PARAMS["inertia_scope"])
