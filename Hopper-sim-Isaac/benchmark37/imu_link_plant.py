"""Opt-in Isaac plant calibrated from imu_link hops, with native yaw actuation."""

import math
from collections import deque

import torch

from .yaw_thrust_feedback import _thrust_from_duty


VERSION = "imu_link_split_or_shared_timing_v1"


class ImuLinkPlant:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.rows = torch.arange(env.num_envs, device=env.device)
        self.rng = torch.Generator(device=env.device).manual_seed(env.cfg.seed + 620734)
        self.collective_u = torch.zeros_like(env._motor_u)
        self.attitude_u = torch.zeros_like(env._motor_u)
        if config.get("version") != VERSION or not config.get("profiles"):
            raise ValueError("Missing imu_link plant fit")
        vlo, vhi = config["bus_voltage_range_v"]
        spring = config["spring"]
        if not (3.0 <= vlo <= vhi <= 4.3 and 450. <= spring["stiffness_n_m"] <= 700.
                and .2 <= spring["damping_ns_m"] <= 1.2):
            raise ValueError("Invalid imu_link bus voltage or spring calibration")
        rows = []
        for profile in config["profiles"]:
            motor, attitude = profile["actuator"], profile["attitude"]
            values = [motor[k] for k in ("tau_s", "delay_ticks", "thrust_scale",
                      "air_ascend_scale", "air_descend_scale", "voltage_reference_v")]
            values += [attitude[k] for k in ("tau_s", "delay_ticks", "roll_scale", "pitch_scale")]
            bounds = [(.015, .25), (0., 7.), (.25, 2.), (0., 8.), (0., 8.), (3., 5.),
                      (.015, .25), (0., 7.), (.15, 2.), (.15, 2.)]
            if not all(math.isfinite(x) and lo <= x <= hi for x, (lo, hi) in zip(values, bounds)):
                raise ValueError("imu_link profile outside the supported fit envelope")
            rows.append(values)
        self.parameters = torch.tensor(rows, device=env.device)
        history_length = math.floor(max(max(row[1], row[7]) for row in rows)) + 2
        self.history = deque((torch.zeros_like(env._motor_u) for _ in range(history_length)),
                             maxlen=history_length)
        self.profile = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.voltage_gain = torch.ones(env.num_envs, device=env.device)
        self.mass_scale = torch.ones_like(self.voltage_gain)
        if hasattr(env, 'rigid_scales'):
            self.mass_scale.copy_(env.rigid_scales[:, 0])
        self.reset(self.rows)

    def reset(self, ids):
        self.collective_u[ids] = 0.
        self.attitude_u[ids] = 0.
        for duty in self.history:
            duty[ids] = 0.
        profiles = torch.randint(len(self.parameters), (len(ids),), generator=self.rng, device=self.env.device)
        self.profile[ids] = profiles
        lo, hi = self.config["bus_voltage_range_v"]
        voltage = torch.empty(len(ids), device=self.env.device).uniform_(lo, hi, generator=self.rng)
        self.voltage_gain[ids] = voltage / self.parameters[profiles, 5]

    def _filter(self, u, stacked, tau, delay, voltage_gain):
        floor_delay = delay.floor().long()
        last = stacked.shape[1] - 1
        duty = stacked[self.rows, last - floor_delay].lerp(
            stacked[self.rows, last - floor_delay - 1], (delay - floor_delay)[:, None])
        alpha = -torch.expm1(-self.env.step_dt / tau)
        u.lerp_((duty * voltage_gain[:, None]).clamp(0., 1.), alpha[:, None])

    def step(self):
        env = self.env
        self.history.append(.5 * (env._actions + 1.))
        stacked = torch.stack(tuple(self.history), dim=1)
        tau, delay, _, _, _, _, moment_tau, moment_delay, roll, pitch = self.parameters[self.profile].unbind(-1)
        self._filter(self.collective_u, stacked, tau, delay, self.voltage_gain)
        self._filter(self.attitude_u, stacked, moment_tau, moment_delay, torch.ones_like(self.voltage_gain))
        env._thrust[:, 0, 2] = self.collective_scale * _thrust_from_duty(self.collective_u).sum(-1)
        f1, f2, f3, f4 = (_thrust_from_duty(self.attitude_u) / env.dr_mass_multi).unbind(-1)
        env._moment[:, 0, 0] = roll * .0813 * (f1 + f4 - f2 - f3) / env.dr_inertia_multi[:, 0]
        env._moment[:, 0, 1] = pitch * .0813 * (f1 + f2 - f3 - f4) / env.dr_inertia_multi[:, 1]
        # Native motor state owns the yaw torque. The power model uses the calibrated duty.
        env._motor_u_history[:, -1, :] = self.collective_u
        env._soc_tracker += (self.collective_u - env._motor_u).sum(-1, keepdim=True) / 10000.

    @property
    def collective_scale(self):
        return self.parameters[self.profile, 2] / self.env.dr_mass_multi[:, 0]

    def alter_air_force(self, force_body, velocity_body):
        ascent, descent = self.parameters[self.profile, 3:5].unbind(-1)
        force_body[..., 2] *= torch.where(velocity_body[..., 2] >= 0.,
                                         ascent[:, None], descent[:, None])

    def install_spring(self):
        values = self.config["spring"]
        robot = self.env._robot
        joint = self.env._spring_joint_id
        stiffness = self.collective_u.new_full((len(self.rows), 1), values["stiffness_n_m"])
        damping = self.collective_u.new_full((len(self.rows), 1), values["damping_ns_m"])
        robot.write_joint_stiffness_to_sim(stiffness, joint_ids=[joint])
        robot.write_joint_damping_to_sim(damping, joint_ids=[joint])
        view = robot.root_physx_view
        torch.testing.assert_close(view.get_dof_stiffnesses()[:, joint].cpu(), stiffness[:, 0].cpu())
        torch.testing.assert_close(view.get_dof_dampings()[:, joint].cpu(), damping[:, 0].cpu())
        return {"physics_fit": VERSION, "profiles": [p["name"] for p in self.config["profiles"]],
                "spring_stiffness_n_m": values["stiffness_n_m"], "spring_damping_ns_m": values["damping_ns_m"],
                "yaw_torque": "original V73 native delay, state, allocation and inertia path",
                "spring_physx_readback": True}
