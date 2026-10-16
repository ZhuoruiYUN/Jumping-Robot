"""Reward-only attitude forecast and bounded post-touchdown recovery.

Quaternion convention: body-to-world, wxyz. Rates: body frame, rad/s.
No simulator, RNG, contact events, motor commands or desired path are changed.
"""
import math
import torch


SETTINGS_V14 = {
    'prediction_model': 'constant_body_rate_quaternion_sampled_peak',
    'prediction_horizon_max_s': 0.30,
    'prediction_samples': 4,
    'recovery_clearance_m': 0.035,
    'recovery_max_window_s': 0.40,
    'release_upward_speed_m_s': 0.10,
    'release_confirmation_steps': 2,
    'release_tilt_max_rad': math.radians(20),
    'release_rp_rate_norm_max_rad_s': 1.2,
}


def body_tilt(quat_wxyz):
    q = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.acos((1 - 2 * q[..., 1:3].square().sum(-1)).clamp(-1, 1))


def predicted_touchdown_tilt(quat_wxyz, angular_velocity_b, clearance, vz,
                             settings=SETTINGS_V14):
    """Sample q(t)=q(0)*exp(omega_body*t/2), including the current attitude.

The sampled peak catches rotation through upright without a signed tilt-rate
linearization. Four samples reduce endpoint aliasing at fast rates; this is a
constant-rate risk heuristic, not a dynamics prediction or safety guarantee.
"""
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[-1] != 4:
        raise ValueError('expected (N,4) wxyz quaternions')
    if angular_velocity_b.shape != (quat_wxyz.shape[0], 3):
        raise ValueError('expected matching (N,3) body angular rates')
    if clearance.shape != vz.shape or clearance.shape != (quat_wxyz.shape[0],):
        raise ValueError('expected matching (N,) clearance and vertical speed')
    q = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tilt = body_tilt(q)
    horizon = (clearance.clamp_min(0) / (-vz).clamp_min(.20)).clamp(
        0, settings['prediction_horizon_max_s'])
    fractions = torch.arange(1, settings['prediction_samples'] + 1,
                             device=q.device, dtype=q.dtype) / settings['prediction_samples']
    half_t = horizon[:, None, None] * fractions[None, :, None] * .5
    half_angle = angular_velocity_b.norm(dim=-1)[:, None, None] * half_t
    # sinc avoids division by zero and has the correct zero-rate limit.
    delta_xyz = angular_velocity_b[:, None, :] * half_t * torch.sinc(half_angle / math.pi)
    delta_w = torch.cos(half_angle)
    qw, qxyz = q[:, None, :1], q[:, None, 1:]
    # Right multiplication is required for BODY-frame angular velocity.
    future_xyz = (qw * delta_xyz + delta_w * qxyz
                  + torch.linalg.cross(qxyz.expand_as(delta_xyz), delta_xyz, dim=-1))
    future_w = qw * delta_w - (qxyz * delta_xyz).sum(-1, keepdim=True)
    future = torch.cat((future_w, future_xyz), -1)
    forecast = torch.maximum(tilt, body_tilt(future).amax(dim=-1))
    return tilt, forecast, horizon


class AttitudeRecovery:
    """A reward latch, not a physical contact or flight-state estimator.

Active near the foot, or while tilt / roll-pitch rate is unsafe in the bounded
post-touchdown window (including ascent). Safe ascent releases after two
consecutive samples. Recontacts cannot extend a live window indefinitely.
"""
    def __init__(self, num_envs, device, step_dt, settings=SETTINGS_V14):
        if step_dt <= 0:
            raise ValueError('step_dt must be positive')
        self.settings = dict(settings)
        self.window_steps = max(1, round(settings['recovery_max_window_s'] / step_dt))
        self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.release_count = torch.zeros_like(self.remaining)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.attitude_safe = torch.zeros_like(self.active)

    def reset(self, ids):
        self.remaining[ids] = 0
        self.release_count[ids] = 0
        self.active[ids] = False
        self.attitude_safe[ids] = False

    def update(self, touchdown, completed_stationary, initial_drop, clearance,
               vertical_speed, quat_wxyz, body_rates):
        s = self.settings
        start = touchdown & ~completed_stationary & ~initial_drop & (self.remaining == 0)
        self.remaining = torch.where(start, torch.full_like(self.remaining, self.window_steps),
                                     (self.remaining - 1).clamp_min(0))
        self.remaining[initial_drop | (touchdown & completed_stationary)] = 0
        self.attitude_safe = ((body_tilt(quat_wxyz) <= s['release_tilt_max_rad'])
                             & (body_rates[:, :2].norm(dim=-1)
                                <= s['release_rp_rate_norm_max_rad_s']))
        release_sample = ((self.remaining > 0) & ~touchdown & self.attitude_safe
                          & (clearance > s['recovery_clearance_m'])
                          & (vertical_speed > s['release_upward_speed_m_s']))
        self.release_count = torch.where(release_sample, self.release_count + 1,
                                         torch.zeros_like(self.release_count))
        released = self.release_count >= s['release_confirmation_steps']
        self.remaining[released] = 0
        self.release_count[self.remaining == 0] = 0
        self.active = ((self.remaining > 0)
                       & ((clearance <= s['recovery_clearance_m']) | ~self.attitude_safe))
        return self.active, released
