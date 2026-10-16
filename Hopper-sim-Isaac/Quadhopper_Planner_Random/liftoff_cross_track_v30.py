"""V30 task-only liftoff cross-track reward; no Isaac, actuator or RNG access."""

import torch


DEADBAND_M_S = 0.10
NORMALIZER_M_S = 0.10
MAX_NORMALIZED_EXCESS = 2.0
MIN_TARGET_DISTANCE_M = 1e-4


def liftoff_cross_track_term(foot_xy, target_xy, body_velocity_xy,
                             liftoff, scale):
    """Return bounded event penalty and measured lateral speed.

    Direction is the actual foot-to-current-target vector at liftoff. The
    along-target component is unconstrained. A coincident target is ignored.
    """
    if (foot_xy.ndim != 2 or foot_xy.shape[-1] != 2
            or target_xy.shape != foot_xy.shape
            or body_velocity_xy.shape != foot_xy.shape
            or liftoff.shape != foot_xy.shape[:1]):
        raise ValueError('V30 liftoff cross-track tensor shapes disagree')
    if scale > 0:
        raise ValueError('V30 penalty scale must be non-positive')
    displacement = target_xy - foot_xy
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    lateral_speed = torch.abs(
        (body_velocity_xy[:, 0] * -displacement[:, 1]
         + body_velocity_xy[:, 1] * displacement[:, 0])
        / distance.clamp_min(MIN_TARGET_DISTANCE_M)
    )
    valid = liftoff.bool() & (distance > MIN_TARGET_DISTANCE_M)
    excess = ((lateral_speed - DEADBAND_M_S).clamp_min(0.0)
              / NORMALIZER_M_S).clamp(max=MAX_NORMALIZED_EXCESS)
    reward = scale * excess.square() * valid.to(excess.dtype)
    return reward, lateral_speed, valid
