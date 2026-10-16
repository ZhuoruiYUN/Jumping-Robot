"""Deployable observation history and foot kinematics; no Isaac dependency."""
import torch

BASE_DIM = 52
FRAME_DIM = 56
PAST_FRAMES = 5
OBS_DIM = FRAME_DIM * (PAST_FRAMES + 1)
LAYOUT = 'history_foot_v19'
PROTOCOL = 'physical_1m_two_circle_history_foot_v19'


def rotate_wxyz(quat, vector):
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    uv = torch.linalg.cross(quat[..., 1:], vector, dim=-1)
    return vector + 2 * (quat[..., :1] * uv + torch.linalg.cross(quat[..., 1:], uv, dim=-1))


def foot_offset_world(quat, spring_position, rest_offset):
    """Body link origin -> foot sphere center; +joint q retracts the leg."""
    offset = torch.as_tensor(rest_offset, device=quat.device, dtype=quat.dtype)
    offset = offset.expand(quat.shape[0], 3).clone()
    offset[:, 2] += spring_position
    return rotate_wxyz(quat, offset)


def observation_frame(base_obs, target_delta_xy_w, rest_offset, spring_travel):
    # Reuse the same measured/noisy orientation and spring samples as the
    # original 52-D input, rather than expose clean simulation leg state.
    offset = foot_offset_world(base_obs[:, 6:10],
                               base_obs[:, 15].clamp(0, spring_travel), rest_offset)
    foot_error = target_delta_xy_w - offset[:, :2]
    return torch.cat((base_obs, offset[:, :2], foot_error.clamp(-1, 1)), dim=-1)


class ObservationHistory:
    """One cached observation per control tick, reset independently per env.

    Layout is current,t-1,...,t-5. Repeat the first measured frame after reset;
    touchdown/waypoint changes do not reset the buffer.
    """
    def __init__(self, num_envs, device):
        self.frames = torch.zeros(num_envs, PAST_FRAMES + 1, FRAME_DIM, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ticks = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        self.ready[ids] = False
        self.ticks[ids] = -1
        self.frames[ids] = 0

    def update(self, frame, tick):
        fresh = self.ticks != tick
        first = fresh & ~self.ready
        advance = fresh & self.ready
        self.frames[advance, 1:] = self.frames[advance, :-1].clone()
        self.frames[advance, 0] = frame[advance]
        self.frames[first] = frame[first, None, :].expand(-1, PAST_FRAMES + 1, -1)
        self.ready[fresh] = True
        self.ticks[fresh] = tick
        return self.frames.flatten(1).clone()


def expand_from_spatial52(state):
    """Zero-pad BOTH LSTM encoders; preserve the initial actor/critic outputs."""
    if any('obs_normalizer.' in name for name in state):
        raise ValueError('v19 transfer requires the unnormalized v18 spatial52 policy')
    result = {name: value.clone() for name, value in state.items()}
    for name in ('memory_a.rnn.weight_ih_l0', 'memory_c.rnn.weight_ih_l0'):
        old = result[name]
        if old.shape[1] != BASE_DIM:
            raise ValueError(f'Expected 52-D source encoder: {name}, {old.shape}')
        result[name] = old.new_zeros(old.shape[0], OBS_DIM)
        result[name][:, :BASE_DIM] = old
    return result
