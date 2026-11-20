"""OBS65 state/history layout with direct-motor memory, followed by ring8.

This is NOT the CTBR OBS65 interface. All slices describe pre-normalization
values; the recurrent actor and critic receive the same 73 values.
"""
import torch

from .ring_geometry import rotate

VERSION = 'obs65_direct_motor_plus_ring8_v1'
DIM = 73
SLICES = {
    'target_errors_body_xy_m': [0, 6],
    'velocity_A_body_m_s': [6, 9],
    'quaternion_wxyz': [9, 13],
    'angular_velocity_body_rad_s': [13, 16],
    'velocity_A_world_m_s': [16, 19],
    'state_history_old': [19, 32],
    'state_history_new': [32, 45],
    'accepted_motor_action': [45, 49],
    'action_history_old': [49, 53],
    'action_history_new': [53, 57],
    'apex_height_error_m': [57, 58],
    'apex_history_m': [58, 61],
    'landing_error_history_m': [61, 64],
    'shadow_contact': [64, 65],
    'ring_center_body_m': [65, 68],
    'ring_normal_body': [68, 71],
    'ring_inner_radius_m': [71, 72],
    'ring_pass_state': [72, 73],
}


def observation_dim(course):
    version = course.get('observation_version', 'stage4l41_plus_ring8_v1')
    if version in ('direct_motor_no_event_history_crossing_state72_v1',
                   'direct_motor_no_event_history_crossing_state72_history40_20_v2'):
        return 72
    if version=='direct_motor_crossing_quaternion73_history40_20_v1':
        return 73
    if version == VERSION:
        return DIM
    if version == 'stage4l41_plus_ring8_v1':
        return 49
    raise ValueError(f'Unknown observation contract: {version}')


def contract():
    return dict(
        version=VERSION, dimension=DIM, actor_critic_same=True, slices=SLICES,
        source_manual='OBS65_观测接口交接手册.md (2026-11-06)',
        reference_point='A = root link, not COM',
        target_slots='P_t, P_next, P_next; all relative to current A, full 3D inverse rotation then XY, metres',
        state_history_order='v_A_body[3], omega_body[3], quaternion_wxyz[4], v_A_world[3]; oldest then newest',
        timing='assemble before inference; commit that snapshot at next _pre_physics_step before accepting the new action; nominal histories -20/-10ms',
        action_interface='direct motors F1,F2,F3,F4; clamp(policy_output,-1,1) accepted into existing delay queue; NOT CTBR, not delayed command or motor state',
        action_history_timing='current previous accepted policy command; old/new snapshots from previous policy observations; zero until first action accepted',
        initialization='all observation histories and recent action zero; physical motor delay queue remains OFF=-1',
        events='confirmed post-drop hop apex on first descending A velocity; valid confirmed landing stores foot-link XY error before target advance; no strict-success gate',
        landing_measurement='simulated SpringLeg sphere-centre XY proxy used by existing reward, not a PhysX contact-patch sensor',
        shadow_contact='clip((0.015-(A_z_ground-0.285*u_z))/u_z,0,0.086)>0.002; finite state and abs(u_z)>=0.2; observation only',
        noise='Gaussian std0.01 on measured A position, world velocity, body angular velocity and quaternion components, then normalize/canonicalize quaternion; derive body values from same sample; no noise on history placeholders/actions/events/radius/status',
        ring='relative center3 + normal3 in current observed body frame + inner radius1 + pending0.5/passed1; no future trajectory',
        normalization='existing actor/critic empirical normalizers once, starting fresh for this contract',
        checkpoint='fresh73 policy, or strict same-version73 checkpoint;49D and CTBR65 checkpoints rejected',
    )


def canonical_quaternion(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.where(q[:, :1] < 0, -q, q)


def inverse_rotate(q, v):
    inverse = q.clone()
    inverse[..., 1:] *= -1
    return rotate(inverse, v)


def shadow_contact(position, q, ground_z):
    uz = 1 - 2 * q[:, 1:3].square().sum(-1)
    valid = torch.isfinite(position).all(-1) & torch.isfinite(q).all(-1) & (uz.abs() >= .2)
    denominator = torch.where(valid, uz, torch.ones_like(uz))
    compression = ((.015 - (position[:, 2] - ground_z - .285 * uz)) / denominator).clamp(0, .086)
    return (valid & (compression > .002)).float()[:, None]


def assemble(position, q, velocity_world, omega_body, targets_xy, ground_z,
             apex_target_ground, memory, ring_center, ring_normal, radius, passed):
    """Return obs and the current history snapshot (which is not committed yet)."""
    q = canonical_quaternion(q)
    velocity_body = inverse_rotate(q, velocity_world)
    errors = torch.nn.functional.pad(targets_xy - position[:, None, :2], (0, 1))
    errors = inverse_rotate(q[:, None, :].expand(-1, 3, -1), errors)[..., :2].flatten(1)
    current = torch.cat((velocity_body, q, omega_body, velocity_world), -1)
    snapshot = torch.cat((velocity_body, omega_body, q, velocity_world), -1)
    ring = torch.cat((inverse_rotate(q, ring_center - position), inverse_rotate(q, ring_normal),
                      torch.full_like(position[:, :1], radius), torch.where(passed, 1., .5)[:, None]), -1)
    obs = torch.cat((errors, current, memory.states.flatten(1), memory.recent_action,
                     memory.actions.flatten(1), (apex_target_ground - (position[:, 2] - ground_z))[:, None],
                     memory.apices, memory.landings, shadow_contact(position, q, ground_z), ring), -1)
    assert obs.shape == (position.shape[0], DIM)
    return obs, snapshot


class ObservationMemory:
    def __init__(self, n, device):
        self.states = torch.zeros(n, 2, 13, device=device)
        self.actions = torch.zeros(n, 2, 4, device=device)
        self.recent_action = torch.zeros(n, 4, device=device)
        self.apices = torch.zeros(n, 3, device=device)
        self.landings = torch.zeros(n, 3, device=device)
        self.apex_recorded = torch.zeros(n, device=device, dtype=torch.bool)

    def reset(self, ids):
        for tensor in (self.states, self.actions, self.recent_action, self.apices, self.landings, self.apex_recorded):
            tensor[ids] = 0

    def commit(self, snapshot):
        self.states[:, 0] = self.states[:, 1].clone()
        self.states[:, 1] = snapshot
        self.actions[:, 0] = self.actions[:, 1].clone()
        self.actions[:, 1] = self.recent_action

    @staticmethod
    def push_event(history, mask, value):
        history[mask] = torch.cat((history[mask, 1:], value[mask, None]), -1)

    def events(self, phase, air_steps, spring_q, height, vz, peak, foot_error, alive):
        # DROP=0, GROUND=1, CANDIDATE=2, FLIGHT=3 in inherited tracker.
        self.apex_recorded[phase <= 1] = False
        confirmed = (phase == 3) & (air_steps >= 5)
        apex = confirmed & (vz <= 0) & ~self.apex_recorded & alive
        self.push_event(self.apices, apex, torch.maximum(peak, height))
        self.apex_recorded |= apex
        landing = confirmed & (spring_q > .002) & self.apex_recorded & alive
        self.push_event(self.landings, landing, foot_error)


def wrap_env(base):
    """Observation-only adapter, installed outside task reward/geometry classes."""
    class RingObs73Env(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if observation_dim(self.course) != DIM or self.cfg.observation_space != DIM:
                raise ValueError('OBS73 requires explicit matching course and cfg')
            self.obs73_memory = ObservationMemory(self.num_envs, self.device)
            self._obs73_snapshot = None

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if hasattr(self, 'obs73_memory'):
                ids = self._robot._ALL_INDICES if ids is None else ids
                self.obs73_memory.reset(ids)
                self._obs73_snapshot = None

        def _get_observations(self):
            d = self._robot.data
            # Coherent measured sample; stored histories are never re-noised or
            # re-rotated with the current attitude. Repeated reads do not commit.
            p = d.root_link_pos_w + .01 * torch.randn_like(d.root_link_pos_w)
            q = canonical_quaternion(d.root_quat_w + .01 * torch.randn_like(d.root_quat_w))
            v = d.root_link_lin_vel_w + .01 * torch.randn_like(d.root_link_lin_vel_w)
            w = d.root_ang_vel_b + .01 * torch.randn_like(d.root_ang_vel_b)
            targets = torch.stack((self._two_ground_pt_xy_w, self._two_ground_pt1_xy_w,
                                   self._two_ground_pt1_xy_w), 1)
            ground = self._terrain.env_origins[:, 2]
            obs, self._obs73_snapshot = assemble(
                p, q, v, w, targets, ground, self._desired_pos_w[:, 2] - ground,
                self.obs73_memory, self.ring.center, self.ring.normal,
                self.course['inner_radius_m'], self.ring.passed)
            return {'policy': obs}

        def _pre_physics_step(self, actions):
            if self._obs73_snapshot is None:
                self._get_observations()
            # Inference has finished; save the state and the action that existed
            # BEFORE it. Only then accept this inference's new motor command.
            self.obs73_memory.commit(self._obs73_snapshot)
            self._obs73_snapshot = None
            super()._pre_physics_step(actions)
            self.obs73_memory.recent_action.copy_(self._actions)

        def _get_rewards(self):
            d = self._robot.data
            f = self._foot_event
            ground = self._terrain.env_origins[:, 2]
            alive = (d.root_link_pos_w[:, 2] >= .05) & (d.root_quat_w[:, 1:3].square().sum(-1) <= .5)
            self.obs73_memory.events(
                f.phase, f.air_steps, d.joint_pos[:, self._spring_joint_id],
                d.root_link_pos_w[:, 2] - ground, d.root_link_lin_vel_w[:, 2], f.peak - ground,
                torch.linalg.vector_norm(d.body_link_pos_w[:, self._foot_body_id, :2] - self._two_ground_pt_xy_w, dim=-1),
                alive)
            return super()._get_rewards()

    return RingObs73Env


def validate_checkpoint_course(path, course):
    """Check metadata before opening Isaac or creating output artifacts."""
    import json
    previous = json.loads((path.parent / 'ring_config.json').read_text())
    if observation_dim(previous) != observation_dim(course) or previous.get('observation_version') != course.get('observation_version'):
        raise ValueError('Checkpoint observation contract differs from requested course; use a matching version or an explicit migration, never direct resume.')
