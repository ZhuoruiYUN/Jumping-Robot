"""Check recurrent counterfactuals against the same actual actor pass."""

import unittest
from types import SimpleNamespace

import torch

from Quadhopper_Planner_Random.next_hop_probe import probe_actor


class NextHopProbeTest(unittest.TestCase):
    def test_probe_keeps_live_vector_batch_shape(self):
        class BatchSensitiveRNN(torch.nn.Module):
            def forward(self, inputs, hidden):
                output = inputs[..., 39:41].repeat_interleave(2, dim=-1)
                if inputs.shape[1] != 3:
                    output = output + 2e-5
                return output, hidden

        policy = SimpleNamespace(
            actor_obs_normalizer=torch.nn.Identity(),
            memory_a=SimpleNamespace(rnn=BatchSensitiveRNN()), actor=torch.nn.Identity())
        observations = torch.zeros(3, 336)
        observations[:, 39:41] = torch.tensor([[1., .2], [0., 1.], [-.5, .8]])
        hidden = (torch.zeros(1, 3, 4), torch.zeros(1, 3, 4))
        live_actions = policy.actor(policy.memory_a.rnn(observations.unsqueeze(0), hidden)[0].squeeze(0))
        rows = probe_actor(policy, observations, hidden, [0], live_actions)
        self.assertEqual(rows[0]['variant'], 'actual')
        self.assertEqual(rows[0]['action_delta_l2'], 0.0)

    def test_probe_matches_live_actor_without_mutating_hidden_state(self):
        torch.manual_seed(7)
        memory = torch.nn.LSTM(336, 12, 1)
        actor = torch.nn.Linear(12, 4)
        policy = SimpleNamespace(
            actor_obs_normalizer=torch.nn.Identity(),
            memory_a=SimpleNamespace(rnn=memory), actor=actor)
        observations = torch.randn(3, 336)
        observations[:, 39:41] = torch.tensor([[1.5, .5], [-.7, 1.], [.8, -.8]])
        hidden = (torch.randn(1, 3, 12), torch.randn(1, 3, 12))
        original_hidden = tuple(value.clone() for value in hidden)
        with torch.no_grad():
            actual, _ = memory(observations.unsqueeze(0), hidden)
            live_actions = actor(actual.squeeze(0))
            rows = probe_actor(policy, observations, hidden, [0, 2], live_actions)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row['action_delta_l2'] < 1e-6 for row in rows
                            if row['variant'] == 'actual'))
        self.assertEqual(rows[2]['segment_body_normalized'], [-.5, 1.5])
        self.assertEqual(rows[6]['segment_body_normalized'], [0.0, 0.0])
        for before, after in zip(original_hidden, hidden):
            self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
