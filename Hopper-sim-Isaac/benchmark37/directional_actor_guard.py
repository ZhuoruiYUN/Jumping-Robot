"""Bound recurrent motor-command drift on independent, direction-stratified AB histories."""
import numpy as np
import torch


class DirectionalActorGuard:
    def __init__(self, policy, path, budget_multiplier):
        self.policy = policy
        self.readout_space = None
        self.startup_fit = False
        self.parameters = tuple(policy.memory_a.rnn.parameters()) + tuple(policy.actor.parameters())
        device = self.parameters[0].device
        if not 0 < budget_multiplier <= 2:
            raise ValueError('Invalid directional command budget multiplier')
        with np.load(path, allow_pickle=False) as bank:
            self.obs = torch.as_tensor(bank['obs'], device=device)
            labels = torch.as_tensor(bank['labels'], dtype=torch.long, device=device)
            self.selected = labels >= 0
            self.groups = labels[self.selected]
            self.counts = self.groups.bincount(minlength=16)
            self.target = torch.as_tensor(bank['reference'], device=device)[self.selected]
            self.budget = torch.as_tensor(bank['short_rms'], device=device) * budget_multiplier
        if (self.counts.shape != (16,) or self.counts.min() < 100 or self.budget.shape != (16, 2)
                or not torch.all(torch.isfinite(self.budget) & (self.budget > 0))):
            raise ValueError('Incomplete directional motor/yaw probe')
        self.yaw = self.obs.new_tensor([-0.25, 0.25, -0.25, 0.25])
        # RSL still applies the checkpoint normalization in act() and act_inference().
        # Pin actor statistics so that the accepted policy stays feasible during rollouts.
        initial = self.ratios()
        if not torch.isfinite(initial).all() or initial.max().item() > .02:
            raise ValueError('Reference bank does not reproduce the initial actor on its complete histories')
        policy.actor_obs_normalization = False

    @torch.no_grad()
    def protect_moving_features(self, protected_energy, startup_fit=False):
        """Keep the parent recurrent controller, and preserve common moving-state readout directions."""
        if not 0 < protected_energy < 1:
            raise ValueError('Expected a fraction of the reference feature energy')
        head = self.policy.actor
        readout = head[-1]
        if not isinstance(readout, torch.nn.Linear) or readout.bias is None:
            raise ValueError('Expected a direct linear motor readout with bias')
        hidden, _ = self.policy.memory_a.rnn(self.policy.actor_obs_normalizer(self.obs))
        features = hidden[self.selected]
        for layer in list(head.children())[:-1]:
            features = layer(features)
        # Include the bias as a constant input. Give each of the 16 motion groups equal weight.
        features = torch.cat((features, torch.ones_like(features[:, :1])), dim=1).double()
        features = features / self.counts[self.groups, None].double().sqrt()
        values, vectors = torch.linalg.eigh(features.T @ features)
        energy = values.flip(0).clamp_min(0).cumsum(0)
        protected = int(torch.searchsorted(energy, protected_energy * energy[-1]).item()) + 1
        remaining = vectors[:, :len(values)-protected]
        if not remaining.shape[1]:
            raise ValueError('Moving feature constraint leaves no trainable readout direction')
        self.readout_space = (remaining @ remaining.T).to(dtype=readout.weight.dtype)
        self.policy.memory_a.rnn.requires_grad_(False)
        head.requires_grad_(False)
        readout.requires_grad_(True)
        self.parameters = (readout.weight, readout.bias)
        self.startup_fit = startup_fit
        if startup_fit:
            self.feature_gram = torch.zeros_like(self.readout_space)
            self.target_cross = readout.weight.new_zeros(readout.out_features, len(values))
            self.start_count = readout.weight.new_zeros(())
        return dict(protected_directions=protected, trainable_directions=remaining.shape[1],
                    feature_energy=float(energy[protected-1] / energy[-1]))

    @torch.no_grad()
    def record_startup(self, start, mean, reference):
        # RSL's last hidden state is the feature produced by this same act() call.
        hidden = self.policy.memory_a.hidden_states[0][-1]
        x = hidden[start]
        for layer in list(self.policy.actor.children())[:-1]:
            x = layer(x)
        x = torch.cat((x, x.new_ones(len(x), 1)), dim=1)
        action = mean[start]
        target = action + (reference[start].clamp(-1, 1) - action.clamp(-1, 1))
        self.feature_gram.add_(x.T @ x)
        self.target_cross.add_(target.T @ x)
        self.start_count.add_(len(x))

    @torch.no_grad()
    def fit_startup(self, previous):
        """Take a least-squares descent step on this rollout's startup teacher commands."""
        weight, bias = self.parameters
        current = torch.cat((weight, bias[:, None]), dim=1)
        prior = torch.cat((previous[0], previous[1][:, None]), dim=1)
        count = self.start_count.clamp_min(1.)
        gram = self.feature_gram / count
        residual = self.target_cross / count - current @ gram
        direction = residual @ self.readout_space
        slope = (direction * residual).sum().clamp_min(0)
        curvature = ((direction @ gram) * direction).sum().clamp_min(1.e-20)
        # Limit the extra step to the size PPO actually accepted in the free subspace.
        step = torch.minimum(slope / curvature, (current - prior).norm() / direction.norm().clamp_min(1.e-12))
        change = step * direction
        weight.add_(change[:, :-1])
        bias.add_(change[:, -1])
        gain = (2 * step * slope - step.square() * curvature) / weight.shape[0]
        metrics = dict(guard_fit_frames=self.start_count.item(), guard_fit_mse_decrease=gain.item())
        self.feature_gram.zero_()
        self.target_cross.zero_()
        self.start_count.zero_()
        return metrics

    @torch.no_grad()
    def restrict_readout(self, previous):
        weight, bias = self.parameters
        change = torch.cat((weight - previous[0], (bias - previous[1])[:, None]), dim=1)
        accepted = change @ self.readout_space
        weight.copy_(previous[0] + accepted[:, :-1])
        bias.copy_(previous[1] + accepted[:, -1])
        return float(accepted.norm() / change.norm().clamp_min(1.e-12))

    @torch.no_grad()
    def ratios(self):
        hidden, _ = self.policy.memory_a.rnn(self.policy.actor_obs_normalizer(self.obs))
        error = self.policy.actor(hidden[self.selected]).clamp(-1, 1) - self.target
        squared = torch.stack((error.square().mean(dim=-1), (error @ self.yaw).square()), dim=-1)
        sums = squared.new_zeros((16, 2)).index_add_(0, self.groups, squared)
        return (sums / self.counts[:, None]).sqrt() / self.budget

    @torch.no_grad()
    def project(self, previous):
        """Keep the full critic/Adam update; project only the recurrent deterministic actor."""
        readout_metrics = {}
        if self.readout_space is not None:
            readout_metrics['guard_readout_fraction'] = self.restrict_readout(previous)
            if self.startup_fit:
                readout_metrics.update(self.fit_startup(previous))
        ratios = self.ratios()
        proposed = ratios.max().item()
        if not torch.isfinite(ratios).all():
            raise FloatingPointError('Nonfinite recurrent command drift')
        fraction = 1.
        if proposed > 1.:
            candidate = [p.detach().clone() for p in self.parameters]
            for fraction in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.):
                for parameter, prior, new in zip(self.parameters, previous, candidate, strict=True):
                    parameter.copy_(torch.lerp(prior, new, fraction))
                ratios = self.ratios()
                if ratios.max().item() <= 1.:
                    break
            else:
                raise RuntimeError('Previously accepted directional actor is outside its budget')
        return dict(**readout_metrics, guard_proposed_ratio=proposed, guard_step_fraction=fraction,
                    guard_motor_ratio=ratios[:, 0].max().item(), guard_yaw_ratio=ratios[:, 1].max().item())
