"""Yaw-preserving horizontal basis reversal for the quaternion OBS73 actor."""

import torch

from .ab_phase_reference import FrozenActor


OPPOSITE_MOTORS = (2, 3, 0, 1)


def rotate_obs73_half_turn(observation):
    """Mirror horizontal motion in both frames, preserving yaw and vertical motion.

    The attitude is conjugated by a half-turn about Z, so quaternion X/Y
    components flip while W/Z keep their signs. Each motor history sample
    is permuted in the same order used for sending the mirrored actions.
    """
    if observation.shape[-1] != 73:
        raise ValueError('Expected quaternion OBS73 with 40/20 ms state history')
    result = observation.clone()
    for begin,end in ((0,6),(6,8),(10,12),(13,15),(16,18),
                      (59,61),(63,65),(66,68),(69,71)):
        result[..., begin:end].neg_()
    # Each history is velocity(body), angular velocity(body), quaternion, velocity(world).
    for start in (19, 32):
        for begin,end in ((0,2),(3,5),(7,9),(10,12)):
            result[..., start+begin:start+end].neg_()
    for start in (45, 49, 53):
        result[..., start:start+4] = observation[..., [start+i for i in OPPOSITE_MOTORS]]
    return result


def choose_basis(current_edge, preview_edge, previous):
    """Latch the approach sign when a moving edge is commanded, including retries."""
    has_current = current_edge.square().sum(-1) > 1.e-8
    edge = torch.where(has_current[:, None], current_edge, preview_edge)
    main_axis = edge.abs().argmax(dim=-1, keepdim=True)
    sign = edge.gather(-1, main_axis).squeeze(-1) < 0.
    return torch.where(edge.square().sum(-1) > 1.e-8, sign, previous)


class AxisCanonicalActor:
    """Keep both actor histories warm on the same physical trajectory."""

    def __init__(self, native, policy, route):
        self.native = native
        self.policy = policy
        self.mirror = FrozenActor(policy, policy.state_dict())
        self.route = route
        self.lanes = torch.arange(len(route.index), device=route.index.device)
        self.use_mirror = torch.zeros_like(route.index, dtype=torch.bool)

    def __call__(self, observations):
        raw = self.policy.get_actor_obs(observations)
        canonical = self.mirror(rotate_obs73_half_turn(raw))[..., OPPOSITE_MOTORS]
        direct = self.native(observations)
        points, index = self.route.points, self.route.index
        current = points[self.lanes, index]
        previous = points[self.lanes, (index - 1).clamp_min(0)]
        preview = points[self.lanes, index + 1]
        self.use_mirror = choose_basis(current - previous, preview - current, self.use_mirror)
        return torch.where(self.use_mirror[:, None], canonical, direct)

    def reset(self, dones):
        self.mirror.reset(dones)
        self.use_mirror &= ~dones.bool().reshape_as(self.use_mirror)
