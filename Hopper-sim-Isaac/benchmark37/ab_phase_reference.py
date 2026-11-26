"""Train a single AB actor with phase-specific, frozen recurrent action references."""
from copy import deepcopy

import torch


def motor_advantage(actions, mean, sigma, reference):
    """Centered actor-only signal for Gaussian exploration in normalized motor commands."""
    scale = sigma.clamp_min(1.e-6)
    direction = (reference.clamp(-1, 1) - mean.clamp(-1, 1)) / scale
    # Limit each reference to one standard deviation along the four-motor direction.
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.)
    noise = (actions - mean) / scale
    return (direction * noise).sum(-1)


class FrozenActor(torch.nn.Module):
    """A teacher sees the student's observation history, including steps before handoff."""
    def __init__(self, policy, weights):
        super().__init__()
        self.normalizer = deepcopy(policy.actor_obs_normalizer)
        self.rnn = deepcopy(policy.memory_a.rnn)
        self.head = deepcopy(policy.actor)
        for module, prefix in [(self.normalizer, 'actor_obs_normalizer.'), (self.rnn, 'memory_a.rnn.'),
                               (self.head, 'actor.')]:
            module.load_state_dict({k[len(prefix):]: v for k,v in weights.items() if k.startswith(prefix)}, strict=True)
        self.rnn.flatten_parameters()
        self.requires_grad_(False)
        self.eval()
        self.hidden = None

    def forward(self, obs):
        out, self.hidden = self.rnn(self.normalizer(obs).unsqueeze(0), self.hidden)
        return self.head(out.squeeze(0))

    def reset(self, dones):
        for state in self.hidden:
            state.masked_fill_(dones.bool().reshape(1, -1, 1), 0.)


class PhaseReferencePPO:
    """Delegate PPO and augment its normalized actor advantage after computing task returns."""
    def __init__(self, ppo, raw, config, precision_checkpoint, recovery_checkpoint):
        self.ppo = ppo
        self.raw = raw
        self.ab = raw.omni_route.ab_pairs
        if not self.ab.all_lanes or ppo.normalize_advantage_per_mini_batch:
            raise ValueError('Phase references require full AB and rollout-normalized advantages')
        self.minimum_edge = config['long_edge_m']
        self.max_steps = config['startup_steps']
        if not (0 < self.minimum_edge <= .5 and isinstance(self.max_steps, int) and self.max_steps > 0):
            raise ValueError('Invalid AB startup reference window')
        self.lanes = torch.arange(raw.num_envs, device=raw.device)
        self.teachers = []
        for path in (precision_checkpoint, recovery_checkpoint):
            state = torch.load(path, map_location=raw.device, weights_only=False)['model_state_dict']
            self.teachers.append(FrozenActor(ppo.policy, state))
        self.guide = torch.zeros_like(ppo.storage.advantages)
        self.totals = torch.zeros(5, dtype=torch.float64, device=raw.device)
        self.last_metrics = {}
        self.direction_guard = None

    def __getattr__(self, name):
        return getattr(self.ppo, name)

    @torch.no_grad()
    def act(self, obs):
        action = self.ppo.act(obs)
        state = self.ppo.policy.get_actor_obs(obs)
        precision, recovery = [teacher(state) for teacher in self.teachers]
        points = self.raw.omni_route.points
        a = self.ab.a_index
        edge = (points[self.lanes,a] - points[self.lanes,a-1]).norm(dim=-1)
        start = (~self.ab.seen_a & (self.raw.episode_length_buf < self.max_steps) & (edge >= self.minimum_edge))
        reference = torch.where(start[:, None], recovery, precision)
        transition = self.ppo.transition
        if self.direction_guard is not None and self.direction_guard.startup_fit:
            self.direction_guard.record_startup(start, transition.action_mean, recovery)
        guide = motor_advantage(action, transition.action_mean, transition.action_sigma, reference)
        self.guide[self.ppo.storage.step, :, 0].copy_(guide)
        mse = (transition.action_mean.clamp(-1,1) - reference.clamp(-1,1)).square().mean(-1)
        self.totals += torch.stack((start.sum(), (~start).sum(), (mse*start).sum(),
                                    (mse*~start).sum(),guide.square().sum()))
        return action

    def process_env_step(self, obs, rewards, dones, extras):
        for teacher in self.teachers:
            teacher.reset(dones)
        self.ppo.process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        self.ppo.compute_returns(obs)
        storage = self.ppo.storage
        if storage.step != len(self.guide):
            raise RuntimeError('Incomplete reference rollout')
        storage.advantages.add_(self.guide)
        starts, other, err_start, err_other, energy = self.totals.tolist()
        self.last_metrics = dict(start_frames=starts, precision_frames=other,
            start_command_rms=(err_start/max(1,starts))**.5, precision_command_rms=(err_other/max(1,other))**.5,
            guide_rms=(energy/(starts+other))**.5)
        self.totals.zero_()

    def update(self):
        if self.direction_guard is None:
            return self.ppo.update()
        previous = [p.detach().clone() for p in self.direction_guard.parameters]
        result = self.ppo.update()
        self.last_metrics.update(self.direction_guard.project(previous))
        return result
