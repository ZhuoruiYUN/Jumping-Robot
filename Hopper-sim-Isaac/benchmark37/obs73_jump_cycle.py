"""Event rewards for the vertical task; no actuator or observation changes."""
import torch


def plateau_distance(distance, radius=0.):
    # Keep the original reward at the boundary/outside it. Do not shift all
    # errors by5cm or change the measured error used by strict success gates.
    return distance.clamp(min=radius) if radius else distance


def position_cost_deltas(distance, previous, radius, error_scale, progress_scale, dt):
    """Exact replacement of the inherited XY cost/progress inside a plateau."""
    shaped = plateau_distance(distance, radius)
    old_progress = (previous-distance).clamp(-.1, .1)
    new_progress = (plateau_distance(previous, radius)-shaped).clamp(-.1, .1)
    return {
        'xy_error_p': error_scale*(shaped.clamp(max=1.)-distance.clamp(max=1.))*dt,
        'xy_progress': progress_scale*(new_progress-old_progress),
    }


class JumpCycleReward:
    def __init__(self, n, device, config, position_plateau_m=0.):
        self.config = config
        self.position_plateau_m = position_plateau_m
        self.launch_height = torch.zeros(n, device=device)
        self.apex_paid = torch.zeros(n, device=device, dtype=torch.bool)
        self.previous_potential = torch.zeros(n, device=device)
        self.potential_ready = torch.zeros(n, device=device, dtype=torch.bool)
        self.apex_count = torch.zeros((), device=device)

    def reset(self, ids):
        self.launch_height[ids] = 0.
        self.apex_paid[ids] = False
        self.previous_potential[ids] = 0.
        self.potential_ready[ids] = False

    def step(self, phase, air_steps, spring, z, vz, peak, target, distance,
             foot_error, tilt, physical, valid, gravity, energy_width, scales,
             landing_config, foot_limit_m=.1, foot_bonus_mask=None):
        # Use the existing drop/ground/confirmed-flight detector BEFORE it
        # advances. Initial release and repeated stationary apex samples pay0.
        launch = (phase == 1) & (spring <= .001) & (vz > 0) & ~physical
        self.launch_height[launch] = z[launch]
        self.apex_paid[launch] = False
        apex = ((phase == 3) & (air_steps >= 5) & (spring <= .002)
                & (vz <= 0) & ~self.apex_paid & ~physical)
        self.apex_paid |= apex
        self.apex_count += apex.sum()
        reward_target = target + self.config.get('apex_margin_m', 0.)
        desired_rise = (reward_target - self.launch_height).clamp(min=.05)
        rise = ((peak - self.launch_height) / desired_rise).clamp(0., 1.).square()
        height_quality = torch.exp(-.5 * ((peak - reward_target) / landing_config['apex_width_m']).square())
        tilt_quality = torch.exp(-.5 * (tilt / landing_config['tilt_width_deg']).square())
        foot_quality = torch.exp(-.5 * (plateau_distance(foot_error, self.position_plateau_m)
                                        / landing_config['foot_width_m']).square())
        apex_quality = height_quality * rise * tilt_quality
        landing_quality = apex_quality * foot_quality

        # Potential-based shaping uses the SAME gamma as PPO, including the
        # zero terminal potential on physical death. It supplies a dense signal
        # without paying a positive energy reward every hover sample. Timeouts
        # retain their potential, matching PPO's timeout value bootstrapping.
        energy = gravity * z.clamp(min=0.) + .5 * vz.square()
        energy_target = gravity * (target + self.config['energy_margin_m']).clamp(min=0.)
        potential = scales['energy_tracking'] * torch.exp(-(energy - energy_target).abs() / energy_width)
        previous = torch.where(self.potential_ready, self.previous_potential, potential)
        terminal_potential = torch.where(physical, 0., potential)
        energy_progress = self.config['gamma'] * terminal_potential - previous
        self.previous_potential.copy_(potential)
        self.potential_ready.fill_(True)

        complete = valid
        if self.config.get('strict_goal_bonus', False):
            complete = valid & (peak >= target) & (foot_error <= foot_limit_m) & (tilt <= 15.)
        in_bonus_circle = distance < foot_limit_m
        if foot_bonus_mask is not None:
            in_bonus_circle = torch.where(foot_bonus_mask, foot_error <= foot_limit_m, in_bonus_circle)
        terms = {
            'energy_tracking': energy_progress,
            'distance_to_z': torch.where(apex, scales['distance_to_z'] * apex_quality, 0.),
            'apex': torch.where(apex, scales['apex'] * apex_quality, 0.),
            'distance_to_xy': torch.where(valid, scales['distance_to_xy']
                * torch.exp(-plateau_distance(distance, self.position_plateau_m) / .5) * landing_quality, 0.),
            'goal_bonus': torch.where(complete & in_bonus_circle, scales['goal_bonus'] * landing_quality, 0.),
        }
        landing = torch.where(complete, landing_config['scale'] * landing_quality, 0.)
        return terms, landing, apex
