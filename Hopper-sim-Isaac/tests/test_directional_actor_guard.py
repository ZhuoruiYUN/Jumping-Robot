import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmark37.directional_actor_guard import DirectionalActorGuard


class TestDirectionalActorGuard(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(68)
        self.policy = SimpleNamespace(memory_a=SimpleNamespace(rnn=torch.nn.LSTM(3, 5)),
                                      actor=torch.nn.Linear(5, 4), actor_obs_normalizer=torch.nn.Identity(),
                                      actor_obs_normalization=True, critic_obs_normalization=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'bank.npz'
        obs = torch.randn(103, 16, 3)
        with torch.no_grad():
            reference = self.policy.actor(self.policy.memory_a.rnn(obs)[0]).clamp(-1, 1)
        labels = np.tile(np.arange(16), (103, 1))
        labels[:3] = -1
        self.data = dict(obs=obs.numpy(), labels=labels, reference=reference.numpy(),
                         short_rms=np.full((16, 2), .04, dtype=np.float32))
        np.savez(self.path, **self.data)

    def test_warm_history_and_backtrack_in_worst_yaw_group(self):
        self.data['short_rms'][15, 1] = .01
        np.savez(self.path, **self.data)
        guard = DirectionalActorGuard(self.policy, self.path, 1.)
        self.assertFalse(self.policy.actor_obs_normalization)
        self.assertTrue(self.policy.critic_obs_normalization)
        previous = [p.detach().clone() for p in guard.parameters]
        self.assertLess(guard.ratios().max().item(), .001)
        self.assertEqual(guard.project(previous)['guard_step_fraction'], 1.)
        with torch.no_grad():
            self.policy.actor.bias.add_(torch.tensor([-.12, .12, -.12, .12]))
        report = guard.project(previous)
        self.assertGreater(report['guard_proposed_ratio'], 5.)
        self.assertGreater(report['guard_step_fraction'], 0.)
        self.assertLess(report['guard_step_fraction'], 1.)
        self.assertLessEqual(report['guard_yaw_ratio'], 1.)
        self.assertLessEqual(report['guard_motor_ratio'], 1.)

    def test_readout_keeps_recurrent_weights_and_protected_feature_directions(self):
        # A Sequential subclass with required constructor arguments matches RSL's MLP interface.
        class Head(torch.nn.Sequential):
            def __init__(self, readout):
                super().__init__(torch.nn.Identity(), readout)

        self.policy.actor = Head(self.policy.actor)
        guard = DirectionalActorGuard(self.policy, self.path, 1.)
        report = guard.protect_moving_features(.95)
        self.assertGreater(report['protected_directions'], 0)
        self.assertGreater(report['trainable_directions'], 0)
        self.assertFalse(any(p.requires_grad for p in self.policy.memory_a.rnn.parameters()))
        self.assertTrue(all(p.requires_grad for p in guard.parameters))
        frozen = [p.detach().clone() for p in self.policy.memory_a.rnn.parameters()]
        previous = [p.detach().clone() for p in guard.parameters]
        with torch.no_grad():
            for p in guard.parameters:
                p.add_(.03 * torch.randn_like(p))
        metrics = guard.project(previous)
        self.assertGreater(metrics['guard_readout_fraction'], 0.)
        self.assertLess(metrics['guard_readout_fraction'], 1.)
        weight, bias = guard.parameters
        diff = torch.cat((weight-previous[0], (bias-previous[1])[:,None]), dim=1)
        torch.testing.assert_close(diff, diff @ guard.readout_space, atol=1.e-6, rtol=1.e-5)
        for initial, actual in zip(frozen, self.policy.memory_a.rnn.parameters()):
            self.assertTrue(torch.equal(initial, actual))

    def test_startup_fit_reduces_only_selected_teacher_errors(self):
        self.policy.actor = torch.nn.Sequential(self.policy.actor)
        guard = DirectionalActorGuard(self.policy, self.path, 1.)
        guard.protect_moving_features(.95, startup_fit=True)
        hidden = torch.randn(80,5) * .1
        self.policy.memory_a.hidden_states = (hidden[None], torch.zeros_like(hidden)[None])
        start = torch.arange(80) < 40
        mean = self.policy.actor(hidden).detach()
        target = (mean + .08 * torch.randn_like(mean)).clamp(-1,1)
        target[~start] = float('nan')  # Out-of-phase samples never enter the startup fit.
        guard.record_startup(start, mean, target)
        previous = [p.detach().clone() for p in guard.parameters]
        with torch.no_grad():
            for p in guard.parameters:
                p.add_(.01 * torch.randn_like(p))
        guard.restrict_readout(previous)
        before = (self.policy.actor(hidden[start])-target[start]).square().mean()
        report = guard.fit_startup(previous)
        after = (self.policy.actor(hidden[start])-target[start]).square().mean()
        self.assertGreater(report['guard_fit_mse_decrease'], 0.)
        self.assertAlmostEqual(report['guard_fit_mse_decrease'], float((before-after).detach()), places=6)
        self.assertEqual(report['guard_fit_frames'], 40)
        self.assertEqual(guard.feature_gram.count_nonzero(), 0)
        self.assertEqual(guard.target_cross.count_nonzero(), 0)
        guard.record_startup(~torch.ones_like(start), mean, target)
        self.assertEqual(guard.fit_startup(previous)['guard_fit_frames'], 0)

    def test_wrong_parent_rejected_before_normalizer_is_pinned(self):
        self.data['reference'] += .2
        np.savez(self.path, **self.data)
        with self.assertRaisesRegex(ValueError, 'initial actor'):
            DirectionalActorGuard(self.policy, self.path, 1.)
        self.assertTrue(self.policy.actor_obs_normalization)


if __name__ == '__main__':
    unittest.main()
