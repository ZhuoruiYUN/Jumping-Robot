"""Independent OBS73 vertical PPO foundation, trained without any teacher.

Robot/motor physics and PPO are inherited. Only reset curriculum and task
rewards differ. Strict1m/10cm/15deg scoring always remains separate.
"""
from collections import deque
import math

import torch


class VerticalCurriculum:
    def __init__(self, config, evaluation=False):
        self.config = config
        self.evaluation = evaluation
        self.level = len(config['levels']) - 1 if evaluation else 0
        self.updates = 0
        self.window = deque(maxlen=config['window_updates'])
        self.promotions = 0
        self.updates_without_valid_landing = 0

    @property
    def current(self):
        return self.config['levels'][self.level]

    def record_update(self, counts):
        if self.evaluation:
            return False
        self.window.append([int(value) for value in counts])
        self.updates += 1
        self.updates_without_valid_landing = 0 if counts[0] else self.updates_without_valid_landing + 1
        limit = self.config.get('max_updates_without_valid_landing')
        if limit and self.updates_without_valid_landing >= limit:
            raise RuntimeError(f'Vertical training stopped: {limit} consecutive updates without a valid post-drop hop landing. '
                               'Do not extend this run; inspect the policy/reward before further training.')
        valid, good, deaths = map(sum, zip(*self.window))
        ready = (self.level < len(self.config['levels']) - 1
                 and self.updates >= self.config['window_updates']
                 and valid >= self.config['minimum_valid_landings']
                 and good / max(1, valid) >= self.config['promotion_target_landing_rate']
                 and deaths == 0)
        if ready:
            self.level += 1
            self.promotions += 1
            self.updates = 0
            self.window.clear()
        return ready

    def state(self):
        return dict(level=self.level, level_updates=self.updates, window=list(self.window),
                    promotions=self.promotions, target_apex_m=self.current['target_apex_m'],
                    updates_without_valid_landing=self.updates_without_valid_landing)

    def restore(self, state):
        level = state['level']
        if self.evaluation or not 0 <= level < len(self.config['levels']):
            raise ValueError('Invalid training curriculum state')
        self.level = level
        self.updates = state['level_updates']
        self.window.clear()
        self.window.extend(state['window'])
        self.promotions = state['promotions']
        self.updates_without_valid_landing = state.get('updates_without_valid_landing', 0)


def native_landing_reward(peak, target, foot_error, tilt_deg, valid, config):
    # Smooth shaping, never a relaxed benchmark success flag. No underheight
    # penalty against1m while the curriculum deliberately requests a lower hop.
    quality = torch.exp(-.5 * ((peak - target) / config['apex_width_m']).square())
    quality *= torch.exp(-.5 * (foot_error / config['foot_width_m']).square())
    quality *= torch.exp(-.5 * (tilt_deg / config['tilt_width_deg']).square())
    return torch.where(valid, config['scale'] * quality, 0.)


def startup_precision_bonus(completed_landings, valid, strict, foot_payment, flight_quality, config):
    """Modestly reweight early precise touchdowns, retaining whole-flight quality.

    The existing episode landing count supplies attempt indices. Initial drop,
    repeated contact and later landings cannot collect this event payment.
    """
    early = valid & strict & (completed_landings < config['landings'])
    return torch.where(early, (config['multiplier']-1.)*foot_payment.clamp(min=0.)
                       * flight_quality.clamp(0., 1.), 0.)


def attitude_costs(quaternion, omega_body, phase, physical, dt, config):
    """Post-drop tilt and roll/pitch rate costs, integrated in control time.

    Penalize the entire cycle, including preparation for the next launch.
    No thrust/action cost: equal motor thrust for a higher jump is unaffected.
    Quaternion cone tilt also covers roll without double-counting yaw.
    """
    tilt = torch.acos((1. - 2. * quaternion[:, 1:3].square().sum(-1)).clamp(-1., 1.))
    active = (phase != 0) & ~physical
    return {
        'vertical_tilt': torch.where(active, -config['tilt_weight'] * tilt.square() * dt, 0.),
        'vertical_roll_pitch_rate': torch.where(active,
            -config['roll_pitch_rate_weight'] * omega_body[:, :2].square().sum(-1) * dt, 0.),
    }


def flight_braking_costs(quaternion, omega_body, phase, spring, spring_rate,
                         physical, foot_clearance, root_vz, dt, config):
    """Discourage outward rotation and high rates before the next impact.

    This state-only cost does not penalize corrective motor commands. Contact
    impulses, spring motion and the initial release are excluded. No memory,
    action filter, controller or observation is added.
    """
    active = (((phase == 2) | (phase == 3)) & ~physical & (foot_clearance > 0)
              & (spring <= config['free_spring_q_max_m'])
              & (spring_rate.abs() <= config['free_spring_qdot_max_m_s']))
    w, x, y, z = quaternion.unbind(-1)
    gradient = torch.stack((2*(y*z+w*x), 2*(w*y-x*z)), -1)
    sine = torch.linalg.vector_norm(gradient, dim=-1)
    cosine = (1-2*(x.square()+y.square())).clamp(-1., 1.)
    tilt_deg = torch.rad2deg(torch.atan2(sine, cosine))
    outward_rate = (gradient*omega_body[:, :2]).sum(-1).clamp(min=0.)/sine.clamp(min=1e-6)
    tilt_weight = ((tilt_deg-config['outward_start_tilt_deg'])
                   / (config['outward_full_tilt_deg']-config['outward_start_tilt_deg'])).clamp(0., 1.)
    outward = -config['outward_rate_weight']*tilt_weight*outward_rate.square()*dt
    # At a3m/s descent, the final30cm gives roughly100ms to brake. Use actual
    # foot geometry for the reward; the policy still receives its existing73D.
    landing_weight = (1-foot_clearance/config['prelanding_clearance_m']).clamp(0., 1.)
    excess_rate = (torch.linalg.vector_norm(omega_body[:, :2], dim=-1)
                   - config['prelanding_rate_free_rad_s']).clamp(min=0.)
    landing = -config['prelanding_rate_weight']*landing_weight*excess_rate.square()*dt
    return {
        'vertical_outward_rotation': torch.where(active, outward, 0.),
        'vertical_prelanding_rate': torch.where(active & (root_vz < 0), landing, 0.),
    }


def near_upright_braking_cost(quaternion, omega_body, phase, spring, spring_rate,
                              physical, foot_clearance, dt, config):
    """Brake fast upright crossings without constraining large-angle recovery.

    Use the roll/pitch rate norm on both sides of upright, including exactly
    upright: a restoring-only sign gate would disappear at the crossing where
    braking is needed. The cone-angle taper removes this extra cost outside
    the small-angle region. No motor command or physics is modified.
    """
    w, x, y, z = quaternion.unbind(-1)
    gradient = torch.stack((2*(y*z+w*x), 2*(w*y-x*z)), -1)
    sine = torch.linalg.vector_norm(gradient, dim=-1)
    cosine = (1-2*(x.square()+y.square())).clamp(-1., 1.)
    tilt_deg = torch.rad2deg(torch.atan2(sine, cosine))
    proximity = ((config['zero_tilt_deg']-tilt_deg)
                 / (config['zero_tilt_deg']-config['full_tilt_deg'])).clamp(0., 1.)
    excess = (torch.linalg.vector_norm(omega_body[:, :2], dim=-1)
              - config['free_rate_rad_s']).clamp(min=0.)
    active = (((phase == 2) | (phase == 3)) & ~physical & (foot_clearance > 0)
              & (spring <= config['free_spring_q_max_m'])
              & (spring_rate.abs() <= config['free_spring_qdot_max_m_s']))
    return torch.where(active, -config['weight']*proximity*excess.square()*dt, 0.)


def pitch_corridor_cost(quaternion, phase, spring, spring_rate, physical, dt, config):
    """Persistent free-flight angle cost, including a held pitch bias."""
    w, x, y, z = quaternion.unbind(-1)
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1., 1.))
    excess = (pitch.abs()-math.radians(config['free_angle_deg'])).clamp(min=0.)
    active = (((phase == 2) | (phase == 3)) & ~physical
              & (spring <= config['free_spring_q_max_m'])
              & (spring_rate.abs() <= config['free_spring_qdot_max_m_s']))
    return torch.where(active, -config['weight']*excess.square()*dt, 0.)


def leg_swing_cost(foot_velocity_world, root_velocity_world, phase, spring,
                   physical, dt, config):
    """Penalize lateral foot motion relative to the root during free flight.

    Subtract translation before selecting world XY. Vertical spring motion is
    excluded, as are the initial drop and spring-compression contact samples.
    """
    relative_xy = foot_velocity_world[:, :2] - root_velocity_world[:, :2]
    active = ((phase == 2) | (phase == 3)) & (spring <= config['free_spring_q_max_m']) & ~physical
    return torch.where(active, -config['weight'] * relative_xy.square().sum(-1) * dt, 0.)


def rotation_work_cost(moment_body, omega_body, phase, spring, physical, dt, config, quaternion=None):
    """Cost actual motor work that accelerates free-flight roll/pitch.

    Reuse the ring task's positive-work kernel on each axis separately, so
    braking on one axis cannot cancel acceleration on the other. Moments are
    applied Body-frame motor torques after delay/lag, rather than new actions.
    """
    from .ring_targeted_adjustments import pitch_work_cost
    active = ((phase == 2) | (phase == 3)) & (spring <= config['free_spring_q_max_m']) & ~physical
    work_config = {'pitch_work_weight': config['weight']}
    axis_cost = torch.stack([pitch_work_cost(moment_body[:, axis], omega_body[:, axis], dt, work_config)
                             for axis in (0, 1)], -1)
    recovery = config.get('upright_recovery')
    if recovery is not None:
        if quaternion is None:
            raise ValueError('Upright recovery work relief requires the current wxyz quaternion')
        w, x, y, z = quaternion.unbind(-1)
        # Gradient of 1-Rzz with respect to body roll/pitch rotation. Its
        # dot product with body omega is the instantaneous tilt-potential rate.
        gradient = torch.stack((2*(y*z+w*x), 2*(w*y-x*z)), -1)
        axis_motion = gradient*omega_body[:, :2]
        restoring = ((axis_motion < 0) & (gradient*moment_body[:, :2] < 0)
                     & (axis_motion.sum(-1, keepdim=True) < 0))
        tilt = torch.rad2deg(torch.acos((1-2*(x.square()+y.square())).clamp(-1., 1.)))
        ramp = ((tilt-recovery['start_tilt_deg'])
                / (recovery['full_tilt_deg']-recovery['start_tilt_deg'])).clamp(0., 1.)
        # Keep a residual positive-work cost; neither this relief nor braking
        # yields a positive reward. Small-angle oscillations retain full cost.
        axis_cost = axis_cost*(1-recovery['relief_fraction']*ramp[:, None]*restoring)
    cost = axis_cost.sum(-1)
    return torch.where(active, cost, 0.)


def yaw_costs(quaternion, yaw_rate_body, phase, physical, dt, config):
    """Hold the stationary heading with a shortest-angle error, in radians."""
    w, x, y, z = quaternion.unbind(-1)
    heading = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y.square() + z.square()))
    difference = heading - config['target_heading_rad']
    error = torch.atan2(torch.sin(difference), torch.cos(difference))
    active = (phase != 0) & ~physical
    return {
        'vertical_yaw_heading': torch.where(active, -config['heading_weight'] * error.square() * dt, 0.),
        'vertical_yaw_rate': torch.where(active, -config['rate_weight'] * yaw_rate_body.square() * dt, 0.),
    }, error, active


def xy_energy_reward_delta(root_z, root_vz, target_z, distance, gravity,
                           energy_error_scale, xy_scale, goal_scale, dt, config):
    """Remove the XY reward for powered escape above the requested hop energy.

    Reuse the native energy kernel/width. Energy deficits keep the original XY
    shaping, so compression and early takeoff are not suppressed. The source
    terms are evaluated here only to compute an exact additive replacement.
    """
    energy = gravity * root_z.clamp(min=0.) + .5 * root_vz.square()
    ceiling = gravity * (target_z + config['energy_margin_m']).clamp(min=0.)
    weight = torch.exp(-(energy - ceiling).clamp(min=0.) / energy_error_scale)
    xy = xy_scale * torch.exp(-distance / .5) * dt
    old_goal = goal_scale * (distance < .35).to(distance.dtype) * dt
    goal = goal_scale * (distance < config['root_goal_radius_m']).to(distance.dtype) * dt
    return {'distance_to_xy': (weight - 1.) * xy,
            'goal_bonus': weight * goal - old_goal}, weight


def env_class(evaluation=False, evaluation_level=None):
    from .obs73_stationary import env_class as stationary_class
    from .release_tilt import quaternion
    from .ring_post_stability import terms
    from .foot_touchdown_reward import FootTouchdownTracker
    from .energy_margin import energy_reward_delta
    from .point_energy_margin import extra_energy_reward_delta
    from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnv

    class Obs73VerticalEnv(stationary_class()):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.vertical_config = self.course['vertical_pretraining']
            self.position_plateau_m = self.vertical_config.get('position_reward_plateau_m', 0.)
            if (not math.isfinite(self.position_plateau_m) or not 0 <= self.position_plateau_m < .1
                    or (self.position_plateau_m and not self.vertical_config.get('reference_flight_reward'))):
                raise ValueError('Position plateau requires reference rewards and radius below strict10cm')
            self._foot_event = FootTouchdownTracker(self.num_envs,self.device,self.landing_radius_m)
            self.vertical_curriculum = VerticalCurriculum(self.vertical_config, evaluation)
            self.response_envelope = None
            if self.vertical_config.get('response_robustness'):
                from .obs73_response_robustness import ResponseEnvelope
                self.response_envelope = ResponseEnvelope(self.num_envs, self.device,
                    self.vertical_config['response_robustness'], self.cfg.seed, evaluation)
            self.evaluation_at_training_difficulty = evaluation_level is not None
            if evaluation_level is not None:
                if not evaluation or not 0 <= evaluation_level < len(self.vertical_config['levels']):
                    raise ValueError('A frozen evaluation level requires evaluation=True and a valid level')
                self.vertical_curriculum.level = evaluation_level
            self._vertical_rng = torch.Generator(device=self.device).manual_seed(int(self.cfg.seed) + 731109)
            self._vertical_episode_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._vertical_pending = torch.zeros(3, dtype=torch.long, device=self.device)
            self._vertical_last_update = -1
            self.vertical_target_hits = torch.zeros((), device=self.device)
            self.vertical_valid_landings = self.vertical_target_hits.clone()
            self._episode_sums['vertical_native_landing'] = torch.zeros(self.num_envs, device=self.device)
            self.attitude_config = self.vertical_config.get('attitude_refinement')
            if self.attitude_config is not None:
                for key in ('tilt_weight', 'roll_pitch_rate_weight'):
                    if not math.isfinite(self.attitude_config[key]) or self.attitude_config[key] < 0:
                        raise ValueError(f'Invalid vertical attitude coefficient: {key}')
                for key in ('vertical_tilt', 'vertical_roll_pitch_rate'):
                    self._episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
            self.flight_braking_config = self.vertical_config.get('flight_braking')
            if self.flight_braking_config is not None:
                c = self.flight_braking_config
                for key in ('outward_rate_weight', 'prelanding_rate_weight', 'outward_start_tilt_deg',
                            'outward_full_tilt_deg', 'prelanding_clearance_m', 'prelanding_rate_free_rad_s',
                            'free_spring_q_max_m', 'free_spring_qdot_max_m_s'):
                    if not math.isfinite(c[key]) or c[key] <= 0:
                        raise ValueError(f'Invalid flight braking setting: {key}')
                if not c['outward_start_tilt_deg'] < c['outward_full_tilt_deg'] < 90:
                    raise ValueError('Invalid outward rotation tilt ramp')
                for name in ('vertical_outward_rotation', 'vertical_prelanding_rate'):
                    self._episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self.near_upright_braking_config = self.vertical_config.get('near_upright_braking')
            if self.near_upright_braking_config is not None:
                c = self.near_upright_braking_config
                for key in ('weight', 'full_tilt_deg', 'zero_tilt_deg', 'free_rate_rad_s',
                            'free_spring_q_max_m', 'free_spring_qdot_max_m_s'):
                    if not math.isfinite(c[key]) or c[key] <= 0:
                        raise ValueError(f'Invalid near-upright braking setting: {key}')
                if not c['full_tilt_deg'] < c['zero_tilt_deg'] < 90:
                    raise ValueError('Invalid near-upright braking angle taper')
                self._episode_sums['vertical_near_upright_braking'] = torch.zeros(self.num_envs, device=self.device)
            self.pitch_corridor_config = self.vertical_config.get('pitch_corridor')
            if self.pitch_corridor_config is not None:
                for key in ('weight', 'free_angle_deg', 'free_spring_q_max_m', 'free_spring_qdot_max_m_s'):
                    if not math.isfinite(self.pitch_corridor_config[key]) or self.pitch_corridor_config[key] <= 0:
                        raise ValueError(f'Invalid pitch corridor setting: {key}')
                if self.pitch_corridor_config['free_angle_deg'] >= 90:
                    raise ValueError('Pitch corridor must be smaller than90deg')
                self._episode_sums['vertical_pitch_corridor'] = torch.zeros(self.num_envs, device=self.device)
            self.leg_swing_config = self.vertical_config.get('leg_swing_penalty')
            if self.leg_swing_config is not None:
                for key in ('weight', 'free_spring_q_max_m'):
                    if not math.isfinite(self.leg_swing_config[key]) or self.leg_swing_config[key] <= 0:
                        raise ValueError(f'Invalid vertical leg-swing setting: {key}')
                self._episode_sums['vertical_leg_swing'] = torch.zeros(self.num_envs, device=self.device)
            self.rotation_work_config = self.vertical_config.get('rotation_work_penalty')
            if self.rotation_work_config is not None:
                for key in ('weight', 'free_spring_q_max_m'):
                    if not math.isfinite(self.rotation_work_config[key]) or self.rotation_work_config[key] <= 0:
                        raise ValueError(f'Invalid vertical rotation-work setting: {key}')
                recovery = self.rotation_work_config.get('upright_recovery')
                if recovery is not None:
                    if (not all(math.isfinite(recovery[key]) for key in
                                ('start_tilt_deg', 'full_tilt_deg', 'relief_fraction'))
                            or not 0 < recovery['start_tilt_deg'] < recovery['full_tilt_deg'] < 90
                            or not 0 < recovery['relief_fraction'] < 1):
                        raise ValueError('Invalid upright recovery work relief')
                self._episode_sums['vertical_rotation_work'] = torch.zeros(self.num_envs, device=self.device)
            self.yaw_config = self.vertical_config.get('yaw_refinement')
            if self.yaw_config is not None:
                for key in ('heading_weight', 'rate_weight'):
                    if not math.isfinite(self.yaw_config[key]) or self.yaw_config[key] < 0:
                        raise ValueError(f'Invalid vertical yaw coefficient: {key}')
                if not math.isfinite(self.yaw_config['target_heading_rad']):
                    raise ValueError('Invalid vertical yaw reference')
                for key in ('vertical_yaw_heading', 'vertical_yaw_rate'):
                    self._episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
                # Integrated heading/rate squares, elapsed time, peak heading.
                self.vertical_episode_yaw = torch.zeros(self.num_envs, 4, device=self.device)
            self.jump_cycle = None
            if 'jump_cycle_reward' in self.vertical_config:
                from .obs73_jump_cycle import JumpCycleReward
                self.jump_cycle = JumpCycleReward(self.num_envs, self.device,
                    self.vertical_config['jump_cycle_reward'], position_plateau_m=self.position_plateau_m)
            self.reference_reward = self.vertical_config.get('reference_flight_reward')
            if self.reference_reward is not None:
                if self.jump_cycle is None:
                    raise ValueError('Reference flight rewards require the event-only XY/goal cycle')
                for key in ('energy_margin_m', 'height_width_m', 'apex_height_width_m',
                            'apex_speed_width_m_s', 'apex_deficit_scale'):
                    if not math.isfinite(self.reference_reward[key]) or self.reference_reward[key] <= 0:
                        raise ValueError(f'Invalid reference reward setting: {key}')
                margin = self.reference_reward.get('height_reward_margin_m', 0.)
                if not math.isfinite(margin) or margin < 0:
                    raise ValueError('Invalid reference height reward margin')

            self.flight_quality = None
            if 'flight_quality' in self.vertical_config:
                from .obs73_flight_quality import FlightQuality
                if self.reference_reward is None or self.jump_cycle is None:
                    raise ValueError('Flight quality requires reference flight rewards and event-only XY/goal')
                self.flight_quality = FlightQuality(self.num_envs, self.device, self.vertical_config['flight_quality'])
                self._episode_sums['vertical_flight_quality_discount'] = torch.zeros(self.num_envs, device=self.device)
                if self.vertical_config['flight_quality'].get('settled_pitch_refinement'):
                    if self.vertical_config['flight_quality']['settled_pitch_refinement'].get('mode') == 'bounded_cost':
                        self._episode_sums['vertical_settled_pitch_cost'] = torch.zeros(self.num_envs, device=self.device)
                    # completed, strict, pitch-span sum, span>10 count, pitch-rate RMS sum
                    self.settled_episode_flight = torch.zeros(self.num_envs, 5, device=self.device)
                    limits = self.vertical_config.get('settled_evaluation')
                    if limits:
                        if limits['warmup_landings'] != self.vertical_config['flight_quality']['settled_pitch_refinement']['warmup_landings']:
                            raise ValueError('Reward and evaluation settling windows differ')
                        if not isinstance(limits['minimum_later_landings_per_episode'], int) or limits['minimum_later_landings_per_episode'] < 1:
                            raise ValueError('Invalid minimum settled landing count')
            self.startup_precision_config = self.vertical_config.get('startup_precision_reward')
            if self.startup_precision_config is not None:
                c = self.startup_precision_config
                if self.flight_quality is None or self.reference_reward is None:
                    raise ValueError('Startup precision requires reference foot rewards and whole-flight quality')
                if not isinstance(c['landings'], int) or c['landings'] < 1:
                    raise ValueError('Invalid startup precision landing count')
                if not math.isfinite(c['multiplier']) or c['multiplier'] <= 1.:
                    raise ValueError('Invalid startup precision multiplier')
                self._episode_sums['vertical_startup_precision_bonus'] = torch.zeros(self.num_envs, device=self.device)
            self.stable_zip_reward = None
            if 'stable_zip_reward' in self.vertical_config:
                from .obs73_stable_zip_reward import StableZipReward
                if self.jump_cycle is not None or self.reference_reward is not None or self.attitude_config is not None or self.leg_swing_config is not None or self.rotation_work_config is not None:
                    raise ValueError('Stable ZIP rewards replace legacy event/reference/extra stability rewards')
                if self.yaw_config and (self.yaw_config['heading_weight'] or self.yaw_config['rate_weight']):
                    raise ValueError('Stable ZIP mode uses its own yaw reward; legacy yaw costs must be zero')
                self.stable_zip_reward = StableZipReward(self.num_envs,self.device,self.vertical_config['stable_zip_reward'])
                for name in ('survival','xy','height','yaw','attitude','horizontal_velocity','power','synergy',
                             'anti_synergy','lazy_ground','vertical_energy','apex','angular_velocity',
                             'xy_progress','goal_bonus','flight_power','hover','action_rate','termination'):
                    self._episode_sums['stable_zip_'+name] = torch.zeros(self.num_envs,device=self.device)

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if not hasattr(self, 'vertical_curriculum'):
                return
            ids = self._robot._ALL_INDICES if ids is None else ids
            physics = self.course.get('rigid_body_randomization')
            if physics:
                from .flight_plane import set_rigid_scales
                if not hasattr(self,'rigid_scales'):
                    self.rigid_scales = torch.ones(self.num_envs,2)
                    self.rigid_rng = torch.Generator().manual_seed(self.cfg.seed + 47001)
                scales = torch.empty(len(ids),2).uniform_(*physics['factor_range'], generator=self.rigid_rng)
                set_rigid_scales(self, ids, scales)
                self.dr_mass_multi[ids] = 1.
                self.dr_inertia_multi[ids] = 1.
            if self.response_envelope is not None:
                self.response_envelope.reset(ids, self.dr_inertia_multi)
            c = self.vertical_curriculum.current
            self._vertical_episode_level[ids] = self.vertical_curriculum.level
            d = self._robot.data
            pose = torch.cat((d.root_link_pos_w[ids], d.root_quat_w[ids]), -1).clone()
            low, high = c['release_height_m']
            pose[:, 2] = self._terrain.env_origins[ids, 2] + low + (high - low) * torch.rand(
                len(ids), generator=self._vertical_rng, device=self.device)
            rp = (2 * torch.rand(len(ids), 2, generator=self._vertical_rng, device=self.device) - 1) * math.radians(c['tilt_limit_deg'])
            upright = torch.rand(len(ids), generator=self._vertical_rng, device=self.device) < c['upright_fraction']
            rp[upright] = 0.
            pose[:, 3:7] = quaternion(rp)
            self._robot.write_root_pose_to_sim(pose, ids)
            self._release_tilt_rp[ids] = rp
            self._desired_pos_w[ids, 2] = self._terrain.env_origins[ids, 2] + c['target_apex_m']
            if self.jump_cycle is not None:
                self.jump_cycle.reset(ids)
            if self.flight_quality is not None:
                self.flight_quality.reset(ids)
            if hasattr(self, 'settled_episode_flight'):
                self.settled_episode_flight[ids] = 0.
            if self.yaw_config is not None:
                self.vertical_episode_yaw[ids] = 0.
            if self.stable_zip_reward is not None:
                distance = torch.linalg.vector_norm(d.root_pos_w[ids,:2]-self._desired_pos_w[ids,:2],dim=-1)
                self.stable_zip_reward.reset(ids,distance)

        def _get_rewards(self):
            d = self._robot.data
            f = self._foot_event
            z = d.root_link_pos_w[:, 2]
            q = d.root_quat_w
            spring = d.joint_pos[:, self._spring_joint_id]
            vz = d.root_com_lin_vel_w[:, 2]
            tilt = torch.rad2deg(torch.acos((1 - 2 * q[:, 1:3].square().sum(-1)).clamp(-1, 1)))
            physical = (z < .05) | (q[:, 1:3].square().sum(-1) > .5)
            airborne = ((f.phase == f.CANDIDATE) | (f.phase == f.FLIGHT)) & ~self.reset_buf
            contact = spring > .002
            valid = (f.phase == f.FLIGHT) & contact & (f.air_steps >= 5) & (f.apex_seen | (vz <= 0)) & ~physical
            peak = torch.maximum(f.peak, z)
            foot_error = torch.linalg.vector_norm(d.body_link_pos_w[:, self._foot_body_id, :2] - self._two_ground_pt_xy_w, dim=-1)
            target = self._desired_pos_w[:, 2]
            native_good = valid & (peak >= target - self.vertical_config['training_height_tolerance_m']) & (foot_error <= self.landing_radius_m) & (tilt <= 15.)
            if self.stable_zip_reward is not None:
                reward = self.stable_zip_reward.step(self,contact,physical,valid,native_good,peak)
                energy = extra = xy_delta = torch.zeros_like(reward)
                xy_weight = torch.ones_like(reward)
            else:
                extra = native_landing_reward(peak, target, foot_error, tilt, valid, self.vertical_config['landing_reward'])
                # Baseline dense rewards plus its established4cm energy margin.
                # Stage4l K50/strict-first-two bonuses and post-ring costs are not
                # part of this independent cold-start reward.
                event_names = ('energy_tracking', 'distance_to_z', 'apex', 'distance_to_xy', 'goal_bonus')
                before = {name: self._episode_sums[name].clone() for name in event_names} if self.jump_cycle is not None else {}
                if self.position_plateau_m:
                    previous_xy_distance = self._prev_distance_to_goal_xy.clone()
                reward = QuadhopperEnv._get_rewards(self)
                if self.position_plateau_m:
                    from .obs73_jump_cycle import position_cost_deltas
                    distance = torch.linalg.vector_norm(d.root_pos_w[:, :2]-self._desired_pos_w[:, :2], dim=-1)
                    deltas = position_cost_deltas(distance, previous_xy_distance, self.position_plateau_m,
                        self.cfg.xy_error_penalty_scale, self.cfg.xy_progress_reward_scale, self.step_dt)
                    for name, value in deltas.items():
                        reward = reward + value
                        self._episode_sums[name] += value
                energy_args = (d.root_pos_w[:, 2], d.root_lin_vel_w[:, 2], target,
                               abs(self.sim.cfg.gravity[2]), self.cfg.energy_tracking_reward_scale,
                               self.cfg.vertical_energy_error_scale, self.step_dt)
                energy = energy_reward_delta(*energy_args) + extra_energy_reward_delta(*energy_args)
                self._episode_sums['energy_tracking'] += energy
                self._episode_sums['vertical_native_landing'] += extra
                xy_delta = torch.zeros_like(reward)
                xy_weight = torch.ones_like(reward)
                if self.jump_cycle is None and 'dense_xy_energy_gate' in self.vertical_config:
                    distance = torch.linalg.vector_norm(d.root_pos_w[:, :2] - self._desired_pos_w[:, :2], dim=-1)
                    replacements, xy_weight = xy_energy_reward_delta(
                        d.root_pos_w[:, 2], d.root_lin_vel_w[:, 2], target, distance,
                        abs(self.sim.cfg.gravity[2]), self.cfg.vertical_energy_error_scale,
                        self.cfg.distance_to_xy_reward_scale, self.cfg.goal_bonus_scale,
                        self.step_dt, self.vertical_config['dense_xy_energy_gate'])
                    for name, delta in replacements.items():
                        self._episode_sums[name] += delta
                        xy_delta += delta
                if self.jump_cycle is not None:
                    distance = torch.linalg.vector_norm(d.root_pos_w[:, :2] - self._desired_pos_w[:, :2], dim=-1)
                    foot_bonus_mask = None
                    if self.course.get('moving_foot_goal_bonus'):
                        route = self.omni_route
                        lanes = self._robot._ALL_INDICES
                        edge = route.points[lanes,route.index] - route.points[lanes,(route.index-1).clamp_min(0)]
                        foot_bonus_mask = edge.ne(0).any(dim=-1)
                        guard = getattr(self,'key_guard',None)
                        if guard is not None:
                            foot_bonus_mask &= ~guard.mask
                    replacements, cycle_landing, apex_event = self.jump_cycle.step(
                        f.phase, f.air_steps, spring, z, vz, peak, target, distance,
                        foot_error, tilt, physical, valid, abs(self.sim.cfg.gravity[2]),
                        self.cfg.vertical_energy_error_scale,
                        {name: getattr(self.cfg, 'goal_bonus_scale' if name == 'goal_bonus' else name + '_reward_scale') for name in event_names},
                        self.vertical_config['landing_reward'], self.landing_radius_m, foot_bonus_mask)
                    if self.reference_reward is not None:
                        from .obs73_reference_reward import flight_terms, landing_terms
                        replacements.update(flight_terms(d.root_pos_w[:, 2], d.root_lin_vel_w[:, 2], target,
                            abs(self.sim.cfg.gravity[2]), self.cfg.vertical_energy_error_scale,
                            {name: getattr(self.cfg, name+'_reward_scale') for name in ('energy_tracking','distance_to_z','apex')},
                            self.step_dt, self.reference_reward))
                        cycle_landing, deficit = landing_terms(peak, target, foot_error, tilt, valid,
                            self.vertical_config['landing_reward'], self.reference_reward,
                            position_plateau_m=self.position_plateau_m, foot_limit_m=self.landing_radius_m)
                        reward = reward + deficit
                        self._episode_sums['apex_deficit'] += deficit
                    for name, new in replacements.items():
                        xy_delta += new - (self._episode_sums[name] - before[name])
                        self._episode_sums[name].copy_(before[name] + new)
                    self._episode_sums['vertical_native_landing'] += cycle_landing - extra
                    extra = cycle_landing
                    self.extras.setdefault('log', {}).update({
                        'Vertical/apex_events_total': self.jump_cycle.apex_count,
                        'Vertical/apex_event_count': apex_event.sum(),
                        'Vertical/cycle_landing_reward_mean': cycle_landing.mean(),
                        ('Vertical/continuous_energy_reward_mean' if self.reference_reward is not None
                         else 'Vertical/energy_progress_mean'): replacements['energy_tracking'].mean(),
                    })
            if self.flight_quality is not None:
                payment = replacements['distance_to_xy'] + replacements['goal_bonus'] + extra
                quality = self.flight_quality.step(q, d.root_ang_vel_b, f.phase, spring,
                    d.joint_vel[:, self._spring_joint_id], physical, valid, payment,
                    completed_landings=self.stationary_episode_pitch[:, 0])
                reward = reward + quality['discount']
                self._episode_sums['vertical_flight_quality_discount'] += quality['discount']
                if 'vertical_settled_pitch_cost' in self._episode_sums:
                    reward = reward + quality['settled_pitch_cost']
                    self._episode_sums['vertical_settled_pitch_cost'] += quality['settled_pitch_cost']
                    self.extras.setdefault('log', {})['Settled/pitch_cost_mean'] = quality['settled_pitch_cost'].mean()
                self.extras.setdefault('log', {}).update({
                    'Vertical/flight_quality_discount_mean': quality['discount'].mean(),
                    'Vertical/flight_quality_at_landing': quality['quality'].sum()/valid.sum().clamp(min=1),
                    'Vertical/quality_landing_events': valid.sum(),
                    'Vertical/quality_tracked_landings': quality['complete'].sum(),
                })
                if hasattr(self, 'settled_episode_flight'):
                    settled = quality['settled']
                    self.settled_episode_flight += torch.stack((settled.float(),
                        (settled & native_good).float(),
                        torch.where(settled, quality['pitch_span_deg'], 0.),
                        (settled & (quality['pitch_span_deg'] > 10.)).float(),
                        torch.where(settled, quality['pitch_rate_rms_rad_s'], 0.)), -1)
                    totals = self.settled_episode_flight.sum(0);denominator = totals[0].clamp(min=1.)
                    self.extras.setdefault('log', {}).update({
                        'Settled/completed_flights': totals[0],
                        'Settled/strict_landing_rate': totals[1]/denominator,
                        'Settled/mean_pitch_span_deg': totals[2]/denominator,
                        'Settled/pitch_span_gt10_rate': totals[3]/denominator,
                        'Settled/mean_pitch_rate_rms_rad_s': totals[4]/denominator,
                    })
            if self.startup_precision_config is not None:
                bonus = startup_precision_bonus(self.stationary_episode_pitch[:, 0],
                    valid, native_good, extra, quality['quality'], self.startup_precision_config)
                reward = reward + bonus
                self._episode_sums['vertical_startup_precision_bonus'] += bonus
                self.extras.setdefault('log', {})['Vertical/startup_precision_bonus_mean'] = bonus.mean()
            # Preserve the established strict scorer for metrics only. Its
            # fine-tuning reward/penalty outputs are deliberately not applied.
            if self.attitude_config is not None:
                costs = attitude_costs(q, d.root_ang_vel_b, f.phase, physical, self.step_dt, self.attitude_config)
                for name, value in costs.items():
                    reward = reward + value
                    self._episode_sums[name] += value
            if self.flight_braking_config is not None:
                clearance = d.body_link_pos_w[:, self._foot_body_id, 2]-self._terrain.env_origins[:, 2]-.012
                costs = flight_braking_costs(q, d.root_ang_vel_b, f.phase, spring,
                    d.joint_vel[:, self._spring_joint_id], physical, clearance,
                    d.root_link_lin_vel_w[:, 2], self.step_dt, self.flight_braking_config)
                for name, value in costs.items():
                    reward = reward + value
                    self._episode_sums[name] += value
                    self.extras.setdefault('log', {})['Vertical/'+name+'_mean'] = value.mean()
            if self.near_upright_braking_config is not None:
                clearance = d.body_link_pos_w[:, self._foot_body_id, 2]-self._terrain.env_origins[:, 2]-.012
                cost = near_upright_braking_cost(q, d.root_ang_vel_b, f.phase, spring,
                    d.joint_vel[:, self._spring_joint_id], physical, clearance,
                    self.step_dt, self.near_upright_braking_config)
                reward = reward + cost
                self._episode_sums['vertical_near_upright_braking'] += cost
                self.extras.setdefault('log', {})['Vertical/near_upright_braking_cost_mean'] = cost.mean()
            if self.pitch_corridor_config is not None:
                cost = pitch_corridor_cost(q, f.phase, spring, d.joint_vel[:, self._spring_joint_id],
                    physical, self.step_dt, self.pitch_corridor_config)
                reward = reward + cost
                self._episode_sums['vertical_pitch_corridor'] += cost
                self.extras.setdefault('log', {})['Vertical/pitch_corridor_cost_mean'] = cost.mean()
            if self.leg_swing_config is not None:
                swing = leg_swing_cost(d.body_link_lin_vel_w[:, self._foot_body_id],
                    d.root_link_lin_vel_w, f.phase, spring, physical, self.step_dt, self.leg_swing_config)
                reward = reward + swing
                self._episode_sums['vertical_leg_swing'] += swing
            if self.rotation_work_config is not None:
                work = rotation_work_cost(self._moment[:, 0], d.root_ang_vel_b,
                    f.phase, spring, physical, self.step_dt, self.rotation_work_config, quaternion=q)
                reward = reward + work
                self._episode_sums['vertical_rotation_work'] += work
                self.extras.setdefault('log', {})['Vertical/rotation_work_penalty_mean'] = work.mean()
            if self.yaw_config is not None:
                costs, yaw_error, yaw_active = yaw_costs(q, d.root_ang_vel_b[:, 2],
                    f.phase, physical, self.step_dt, self.yaw_config)
                for name, value in costs.items():
                    reward = reward + value
                    self._episode_sums[name] += value
                elapsed = yaw_active.to(z.dtype) * self.step_dt
                self.vertical_episode_yaw[:, :3] += torch.stack((yaw_error.square() * elapsed,
                    d.root_ang_vel_b[:, 2].square() * elapsed, elapsed), -1)
                self.vertical_episode_yaw[:, 3] = torch.maximum(self.vertical_episode_yaw[:, 3],
                    torch.where(yaw_active, yaw_error.abs(), 0.))
                duration = self.vertical_episode_yaw[:, 2].sum().clamp(min=self.step_dt)
                self.extras.setdefault('log', {}).update({
                    'Vertical/yaw_error_rms_deg': torch.rad2deg(torch.sqrt(self.vertical_episode_yaw[:, 0].sum()/duration)),
                    'Vertical/yaw_rate_rms_rad_s': torch.sqrt(self.vertical_episode_yaw[:, 1].sum()/duration),
                })
            f.step(spring, z, vz, d.body_link_pos_w[:, self._foot_body_id, :2],
                   self._two_ground_pt_xy_w, tilt, physical)
            _, pitch = terms(q, d.root_ang_vel_b[:, 1], torch.zeros_like(physical), self.step_dt,
                             dict(pitch_weight=0., pitch_rate_weight=0.))
            self.stationary_stats.step(pitch, d.root_ang_vel_b[:, 1], airborne, contact, valid)
            span = torch.rad2deg(self.stationary_stats.high - self.stationary_stats.low)
            rms = torch.sqrt(self.stationary_stats.rate_sq / self.stationary_stats.samples.clamp(min=1.))
            self.stationary_episode_pitch += torch.stack((valid.float(), torch.where(valid, span, 0.),
                (valid & (span > 10)).float(), torch.where(valid, rms, 0.)), -1)
            current_level = self._vertical_episode_level == self.vertical_curriculum.level
            self._vertical_pending += torch.stack(((valid & current_level).sum(),
                (native_good & current_level).sum(), (physical & current_level).sum()))
            self.vertical_valid_landings += valid.sum()
            self.vertical_target_hits += native_good.sum()
            update = int(self.common_step_counter) // 256
            if self.common_step_counter % 256 == 0 and update != self._vertical_last_update:
                promoted = self.vertical_curriculum.record_update(self._vertical_pending.cpu().tolist())
                self._vertical_pending.zero_()
                self._vertical_last_update = update
                if promoted:
                    print(f'VERTICAL_CURRICULUM level={self.vertical_curriculum.level} target_apex_m={self.vertical_curriculum.current["target_apex_m"]}', flush=True)
            logs = {key.replace('PostRing/', 'Vertical/'): value for key, value in self.stationary_stats.log().items()}
            logs.update({'Vertical/curriculum_level': self.vertical_curriculum.level,
                         'Vertical/current_target_m': self.vertical_curriculum.current['target_apex_m'],
                         'Vertical/xy_energy_gate_mean': xy_weight.mean(),
                         'Vertical/target_qualified_landings_total': self.vertical_target_hits,
                         'Vertical/valid_landings_total': self.vertical_valid_landings,
                         'Vertical/root_height_mean_m': (z-self._terrain.env_origins[:,2]).mean(),
                         'Vertical/valid_apex_mean_m': torch.where(valid,peak-self._terrain.env_origins[:,2],0.).sum()/valid.sum().clamp(min=1),
                         'Vertical/strict1m_landings_total': f.qualified})
            if self.jump_cycle is not None or self.stable_zip_reward is not None:
                logs.pop('Vertical/xy_energy_gate_mean')
            self.extras.setdefault('log', {}).update(logs)
            if self.response_envelope is not None:
                self.extras['log'].update(self.response_envelope.log())
            return reward + energy + extra + xy_delta

    return Obs73VerticalEnv
