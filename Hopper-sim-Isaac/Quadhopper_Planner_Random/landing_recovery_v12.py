"""Reward-only near-ground state; no simulator imports or actuator changes."""
import math
import torch


SETTINGS_V12 = {
    'prelanding_clearance_m': 0.18,
    'descent_speed_m_s': 0.10,
    'prelanding_tilt_deadband_rad': math.radians(20),
    'prelanding_rate_deadband_rad_s': 1.5,
    'prelanding_tilt_scale': -6.0,
    'prelanding_rate_scale': -2.0,
    'recovery_clearance_m': 0.035,
    'recovery_max_window_s': 0.40,
    'release_upward_speed_m_s': 0.10,
    'release_confirmation_steps': 2,
}


def prelanding_mask(clearance, vertical_speed, moving, cycle_active, touchdown,
                    initial_drop, recovery_remaining, settings=SETTINGS_V12):
    return (moving & cycle_active & ~touchdown & ~initial_drop
            & (recovery_remaining == 0)
            & (clearance <= settings['prelanding_clearance_m'])
            & (vertical_speed < -settings['descent_speed_m_s']))


def prelanding_costs(quat_wxyz, body_rates, settings=SETTINGS_V12):
    # Total body-Z tilt, independent of yaw. Do not impose a pitch trajectory.
    tilt = torch.acos((1 - 2 * quat_wxyz[:, 1:3].square().sum(-1)).clamp(-1, 1))
    tilt_cost = ((tilt - settings['prelanding_tilt_deadband_rad']).clamp_min(0)
                 / math.radians(10)).clamp_max(3).square()
    rate_cost = (body_rates[:, :2].abs() - settings['prelanding_rate_deadband_rad_s']).clamp(
        0, 3).square().sum(-1)
    return tilt_cost, rate_cost


class NearGroundRecovery:
    """Bounded moving-touchdown latch, independent of spring liftoff events.

    Release requires foot-sphere clearance AND upward root velocity for two
    consecutive samples. This is a reward heuristic, not a contact sensor or
    a replacement for the task's cycle accounting. Repeated spring contacts
    do not refresh an ongoing window. The latch owns no random state.
    """
    def __init__(self, num_envs, device, step_dt, settings=SETTINGS_V12):
        if step_dt <= 0:
            raise ValueError('step_dt must be positive')
        self.settings = dict(settings)
        self.window_steps = max(1, round(settings['recovery_max_window_s'] / step_dt))
        self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.release_count = torch.zeros_like(self.remaining)

    def reset(self, ids):
        self.remaining[ids] = 0
        self.release_count[ids] = 0

    def update(self, touchdown, completed_stationary, initial_drop, clearance, vertical_speed):
        s = self.settings
        start = touchdown & ~completed_stationary & ~initial_drop & (self.remaining == 0)
        self.remaining = torch.where(start, torch.full_like(self.remaining, self.window_steps),
                                     (self.remaining - 1).clamp_min(0))
        cancel = initial_drop | (touchdown & completed_stationary)
        self.remaining[cancel] = 0
        release_sample = ((self.remaining > 0) & ~touchdown
                          & (clearance > s['recovery_clearance_m'])
                          & (vertical_speed > s['release_upward_speed_m_s']))
        self.release_count = torch.where(release_sample, self.release_count + 1,
                                         torch.zeros_like(self.release_count))
        released = self.release_count >= s['release_confirmation_steps']
        self.remaining[released] = 0
        self.release_count[self.remaining == 0] = 0
        active = (self.remaining > 0) & (clearance <= s['recovery_clearance_m'])
        return active, released
