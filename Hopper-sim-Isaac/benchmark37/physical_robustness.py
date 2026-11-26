"""Training-only planar centre-of-mass variation and a single airborne push."""

import math

import torch


VERSION = 'planar_com_and_impulse_v1'


def validate(config):
    radius = config['com_radius_m']
    probability = config['impulse_probability']
    speed_lo, speed_hi = config['impulse_speed_m_s']
    time_lo, time_hi = config['impulse_flight_time_s']
    if (config['version'] != VERSION or not 0. < radius <= .005
            or not 0. <= probability <= 1.
            or not 0. < speed_lo <= speed_hi <= .20
            or not .05 <= time_lo <= time_hi <= .40):
        raise ValueError('Invalid planar COM or airborne impulse configuration')


def polar_xy(radius, turn):
    angle = math.tau * turn
    return radius[:, None] * torch.stack((angle.cos(), angle.sin()), dim=-1)


class PlanarPhysicalRobustness:
    """Change Body COM in PhysX and inject a bounded planar velocity step."""

    def __init__(self, env, config):
        validate(config)
        self.env = env
        self.config = config
        self.view = env._robot.root_physx_view
        if env._robot.body_names != ['Body', 'SpringLeg']:
            raise ValueError('Planar COM variation requires Body and SpringLeg links')
        nominal_coms = self.view.get_coms()
        self.nominal_body_xy = nominal_coms[:, 0, :2].clone()
        masses = self.view.get_masses()
        # Existing rigid-body DR multiplies every link mass by the same factor.
        self.body_fraction = masses[:, 0] / masses.sum(dim=-1)
        self.com_rng = torch.Generator(device='cpu').manual_seed(env.cfg.seed + 752051)
        self.push_rng = torch.Generator(device=env.device).manual_seed(env.cfg.seed + 752052)
        self.body_shift = torch.zeros((env.num_envs, 3), device=env.device)
        self.push_xy = torch.zeros_like(self.body_shift[:, :2])
        self.push_pending = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.push_flight = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.push_step = torch.zeros_like(self.push_flight)
        self.flight_count = torch.zeros_like(self.push_flight)
        self.pushes = torch.zeros((), device=env.device, dtype=torch.long)
        self.max_push_m_s = torch.zeros((), device=env.device)
        self.max_other_velocity_delta = torch.zeros_like(self.max_push_m_s)
        self.max_com_m = 0.
        self.com_resets = 0

    def reset(self, ids):
        """Replace the previous offset relative to the original asset, never accumulate it."""
        ids_cpu = ids.cpu()
        n = len(ids_cpu)
        if not n:
            return
        com_draw = torch.rand((n, 2), generator=self.com_rng)
        assembly_xy = polar_xy(self.config['com_radius_m'] * com_draw[:, 0].sqrt(), com_draw[:, 1])
        body_xy = assembly_xy / self.body_fraction[ids_cpu, None]
        coms = self.view.get_coms().clone()
        # Keep the current principal axes. Existing inertia DR can also set them.
        coms[ids_cpu, 0, :2] = self.nominal_body_xy[ids_cpu] + body_xy
        self.view.set_coms(coms, ids_cpu)
        data = self.env._robot.data
        for buffer in ('_body_com_pose_b', '_body_com_pose_w', '_body_link_vel_w',
                       '_body_com_state_w', '_body_link_state_w', '_root_com_pose_w',
                       '_root_link_vel_w', '_root_com_state_w', '_root_link_state_w'):
            getattr(data, buffer).timestamp = -1.
        shift = coms[ids_cpu, 0, :2] - self.nominal_body_xy[ids_cpu]
        self.body_shift[ids, :2] = shift.to(self.env.device)
        self.max_com_m = max(self.max_com_m, float(assembly_xy.norm(dim=-1).max()))
        self.com_resets += n

        push_draw = torch.rand((n, 4), device=self.env.device, generator=self.push_rng)
        speed_lo, speed_hi = self.config['impulse_speed_m_s']
        time_lo, time_hi = self.config['impulse_flight_time_s']
        self.push_xy[ids] = polar_xy(speed_lo + (speed_hi - speed_lo) * push_draw[:, 0], push_draw[:, 1])
        self.push_pending[ids] = push_draw[:, 2] < self.config['impulse_probability']
        self.push_flight[ids] = torch.randint(1, 3, (n,), device=self.env.device, generator=self.push_rng)
        self.push_step[ids] = ((time_lo + (time_hi - time_lo) * push_draw[:, 3]) / self.env.step_dt).round().long()
        self.flight_count[ids] = 0

    def maybe_push(self):
        event = self.env._foot_event
        data = self.env._robot.data
        self.flight_count += (event.phase == event.CANDIDATE) & (event.air_steps == 1)
        eligible = (self.push_pending & (self.flight_count == self.push_flight)
                    & (event.phase == event.FLIGHT) & (event.air_steps == self.push_step)
                    & ~event.apex_seen & (data.joint_pos[:, self.env._spring_joint_id] <= .001)
                    & (data.root_com_lin_vel_w[:, 2] > .2))
        ids = eligible.nonzero().flatten()
        if not len(ids):
            return
        before = data.root_com_vel_w[ids].clone()
        after = before.clone()
        # Apply a single equivalent free-air linear impulse with no yaw or vertical input.
        after[:, :2] += self.push_xy[ids]
        self.env._robot.write_root_com_velocity_to_sim(after, env_ids=ids)
        data._root_link_vel_w.timestamp = -1.
        self.push_pending[ids] = False
        self.pushes += len(ids)
        self.max_push_m_s = torch.maximum(self.max_push_m_s, (after[:, :2] - before[:, :2]).norm(dim=-1).max())
        self.max_other_velocity_delta = torch.maximum(
            self.max_other_velocity_delta, (after[:, 2:] - before[:, 2:]).abs().amax())

    def audit(self):
        actual_xy = self.view.get_coms()[:, 0, :2] - self.nominal_body_xy
        torch.testing.assert_close(actual_xy, self.body_shift[:, :2].cpu(), rtol=0., atol=1.e-6)
        readback = (actual_xy * self.body_fraction[:, None]).norm(dim=-1).max()
        if float(readback) > self.config['com_radius_m'] + 1.e-6:
            raise RuntimeError('PhysX COM exceeds the requested planar radius')
        return dict(config=self.config, com_resets=self.com_resets, com_max_sample_m=self.max_com_m,
                    com_max_physx_readback_m=float(readback), impulses=int(self.pushes),
                    impulse_max_m_s=float(self.max_push_m_s),
                    max_vertical_or_angular_delta=float(self.max_other_velocity_delta))
