"""Single-touchdown P2 counterfactual with a verified common prefix.

Installed only for evaluation. The command queue is edited after its normal
advance and before the environment replans or builds the next observation.
"""

import hashlib
import json
import math

import torch

from .failure_trace import plant_values


MODES = ('original', 'left90', 'right90', 'reverse')


def rotated_target(p1, p2, mode):
    if mode not in MODES:
        raise ValueError(f'unknown P2 branch mode: {mode}')
    delta = p2 - p1
    if mode == 'original':
        changed = delta
    elif mode == 'left90':
        changed = torch.stack((-delta[1], delta[0]))
    elif mode == 'right90':
        changed = torch.stack((delta[1], -delta[0]))
    else:
        changed = -delta
    return p1 + changed


def _numbers(value):
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class ClosedLoopP2Probe:
    def __init__(self, env, policy, env_id, branch_step, mode,
                 expected_sha256=None, episode_getter=None):
        if mode not in MODES or not 0 <= env_id < env.num_envs or branch_step < 1:
            raise ValueError('invalid closed-loop P2 branch configuration')
        if expected_sha256 is not None and (len(expected_sha256) != 64 or
                                            any(c not in '0123456789abcdef' for c in expected_sha256)):
            raise ValueError('expected state hash must be a lowercase SHA256 hex digest')
        self.env = env
        self.policy = policy
        self.env_id = env_id
        self.branch_step = branch_step
        self.mode = mode
        self.expected_sha256 = expected_sha256
        self.episode_getter = episode_getter
        self.original_advance = None
        self.observation = None
        self.action = None
        self.branch = None

    def install(self):
        if self.original_advance is not None:
            raise RuntimeError('P2 branch already installed')
        self.original_advance = self.env.commands.advance
        self.env.commands.advance = self._advance

    def begin_step(self, observation, action):
        self.observation = observation
        self.action = action

    def after_step(self, observation, reset_pending):
        if self.branch is None or 'observation_after_branch' in self.branch:
            return
        if int(self.env.common_step_counter) != self.branch_step:
            return
        values = _numbers(observation[self.env_id])
        if len(values) != 336:
            raise RuntimeError(f'P2 branch expected 336 policy observations, got {len(values)}')
        self.branch['observation_after_branch'] = values
        self.branch['reset_at_branch'] = bool(reset_pending[self.env_id])
        if self.branch['reset_at_branch']:
            raise RuntimeError('P2 branch target environment reset at the branch touchdown')

    def _snapshot(self, p1, p2):
        env_id = self.env_id
        values = plant_values(self.env)
        state = {
            'step': int(self.env.common_step_counter),
            'episode_id': int(self.episode_getter()) if self.episode_getter else None,
            'plant': {name: _numbers(tensor[env_id]) for name, tensor in sorted(values.items())},
            'anchor_xy_w': _numbers(self.env.commands.anchor_w[env_id]),
            'p1_xy_w': _numbers(p1), 'p2_xy_w': _numbers(p2),
            'route_index': int(self.env.commands.route_index[env_id]),
            'cycle_index': int(self.env.commands.cycle_index[env_id]),
            'observation_before_touchdown': _numbers(self.observation[env_id]),
            'action_before_touchdown': _numbers(self.action[env_id]),
            'cpu_rng_sha256': hashlib.sha256(
                torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        }
        if torch.cuda.is_available():
            state['cuda_rng_sha256'] = [hashlib.sha256(
                rng.cpu().numpy().tobytes()).hexdigest()
                for rng in torch.cuda.get_rng_state_all()]
        hidden = self.policy.memory_a.hidden_states
        if hidden is not None:
            if not isinstance(hidden, tuple):
                hidden = (hidden,)
            state['actor_hidden_after_action'] = [_numbers(tensor[:, env_id]) for tensor in hidden]
        return state

    def _advance(self, env_ids):
        self.original_advance(env_ids)
        if self.branch is not None or int(self.env.common_step_counter) != self.branch_step:
            return
        if not bool((env_ids == self.env_id).any()):
            return
        if self.observation is None or self.action is None:
            raise RuntimeError('P2 branch missing the pre-touchdown policy input')
        p1 = self.env.commands.targets_w[self.env_id, 0].clone()
        p2 = self.env.commands.targets_w[self.env_id, 1].clone()
        distance = float(torch.linalg.vector_norm(p2 - p1))
        if not math.isfinite(distance) or distance < .10 or distance > .50 + 1e-4:
            raise RuntimeError(f'P2 branch sampled an invalid radius: {distance}')
        state = self._snapshot(p1, p2)
        digest = _digest(state)
        if self.expected_sha256 is not None and digest != self.expected_sha256:
            raise RuntimeError(f'P2 branch state mismatch: expected {self.expected_sha256}, got {digest}')
        changed = rotated_target(p1, p2, self.mode)
        if abs(float(torch.linalg.vector_norm(changed-p1)) - distance) > 1e-5:
            raise RuntimeError('P2 branch changed target distance')
        if self.mode != 'original':
            self.env.commands.targets_w[self.env_id, 1].copy_(changed)
        self.branch = dict(
            env_id=self.env_id, episode_id=state['episode_id'], step=self.branch_step,
            mode=self.mode, state_sha256=digest, state=state,
            original_p1_xy_w=_numbers(p1), original_p2_xy_w=_numbers(p2),
            executed_p2_xy_w=_numbers(changed), p2_radius_m=distance,
        )

    def finish(self, rows):
        if self.branch is None:
            raise RuntimeError(f'P2 branch did not trigger at env {self.env_id}, step {self.branch_step}')
        if 'observation_after_branch' not in self.branch:
            raise RuntimeError('P2 branch did not capture the first post-branch policy observation')
        selected = sorted((row for row in rows if row['env_id'] == self.env_id and
                           row['episode_id'] == self.branch['episode_id']),
                          key=lambda row: row['liftoff_step'])
        preceding = next((row for row in selected if row['touchdown_step'] == self.branch_step), None)
        following = [row for row in selected if row['liftoff_step'] > self.branch_step]
        first = following[0] if following else None
        second = following[1] if len(following) > 1 else None
        if preceding is None:
            raise RuntimeError('P2 branch has no completed pre-branch touchdown')
        if first is not None:
            target = (first['target_x_m'], first['target_y_m'])
            if math.dist(target, self.branch['original_p1_xy_w']) > 1e-4:
                raise RuntimeError('P2 branch changed the first post-branch target P1')
            next_target = (first['next_target_x_m'], first['next_target_y_m'])
            if math.dist(next_target, self.branch['executed_p2_xy_w']) > 1e-4:
                raise RuntimeError('P2 branch did not reach the policy next-target input')
        return dict(branch=self.branch, preceding_touchdown=preceding,
                    first_post_branch_hop=first, second_post_branch_hop=second,
                    first_hop_completed=first is not None,
                    second_hop_completed=second is not None,
                    interpretation='First P1 landing has matched starting state and target. '
                    'The following liftoff is part of the closed-loop response; '
                    'later random route samples are not guaranteed identical.')
