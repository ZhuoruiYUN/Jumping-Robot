import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
import torch
from rsl_rl.modules import ActorCriticRecurrent
from Quadhopper_Planner_Random.spatial_path import hop_curve, project_path, expand_policy_inputs


class SpatialPathTest(unittest.TestCase):
    def setUp(self):
        self.start = torch.tensor([[0., 0., .38]])
        self.landing = torch.tensor([[.20, 0., .38]])
        self.following = torch.tensor([[.37, .10, .38]])
        self.curve = hop_curve(self.start, self.landing, self.following, torch.ones(1))

    def test_endpoints_apex_and_nonovershoot(self):
        torch.testing.assert_close(self.curve[:, 0], self.start)
        torch.testing.assert_close(self.curve[:, -1], self.landing)
        self.assertAlmostEqual(self.curve[0, 16, 2].item(), 1.)
        self.assertLessEqual(self.curve[..., 2].max().item(), 1.)
        self.assertGreaterEqual(self.curve[..., 2].min().item(), .38 - 1e-6)
        self.assertTrue(torch.all(self.curve[:, 1:17, 2] >= self.curve[:, :16, 2]))
        self.assertTrue(torch.all(self.curve[:, 17:, 2] <= self.curve[:, 16:-1, 2]))

    def test_geometry_translation_and_branch_selection(self):
        for index, descending in ((5, False), (26, True)):
            position = self.curve[:, index]
            phase = torch.tensor([descending])
            carrot, tangent, distance, progress = project_path(position, self.curve, phase)
            self.assertLess(distance.item(), 1e-6)
            self.assertAlmostEqual(tangent.norm().item(), 1., places=5)
            self.assertEqual(tangent[0, 2] < 0, descending)
            shift = torch.tensor([[4., -7., 0.]])
            c2, t2, d2, progress2 = project_path(position + shift, self.curve + shift[:, None], phase)
            torch.testing.assert_close(c2 - shift, carrot, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(t2, tangent, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(d2, distance, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(progress, progress2)

    def test_next_point_changes_landing_tangent_without_moving_target(self):
        other = self.following.clone()
        other[:, 1] *= -1
        curve = hop_curve(self.start, self.landing, other, torch.ones(1))
        torch.testing.assert_close(curve[:, -1], self.curve[:, -1])
        self.assertGreater((curve[:, -2] - self.curve[:, -2]).norm().item(), 1e-5)

    def test_zero_distance_commands_are_finite(self):
        curve = hop_curve(self.start, self.start, self.start, torch.ones(1))
        for phase in (False, True):
            outputs = project_path(self.start, curve, torch.tensor([phase]))
            for x in outputs:
                self.assertTrue(torch.isfinite(x).all())

    def test_minimum_progress_prevents_carrot_regression(self):
        phase = torch.tensor([False])
        early = self.curve[:, 2]
        unrestricted = project_path(early, self.curve, phase, lookahead_m=0.0)[0]
        held = project_path(early, self.curve, phase, lookahead_m=0.0,
                            minimum_fraction=torch.tensor([0.40]))[0]
        self.assertGreater((held - self.start).norm().item(),
                           (unrestricted - self.start).norm().item())

    def test_padding_preserves_real_policy_and_critic(self):
        source = Path(__file__).resolve().parents[1] / 'outputs/tracking_ab/compare_v2/preview_seed42/model_70.pt'
        state = torch.load(source, map_location='cpu', weights_only=False)['model_state_dict']
        padded = expand_policy_inputs(state)
        for key in ('memory_a.rnn.weight_ih_l0', 'memory_c.rnn.weight_ih_l0'):
            torch.testing.assert_close(padded[key][:, :43], state[key], atol=0, rtol=0)
            self.assertEqual(padded[key][:, 43:].count_nonzero(), 0)
        with redirect_stdout(io.StringIO()):
            old = ActorCriticRecurrent({'policy': torch.zeros(4, 43)},
                {'policy': ['policy'], 'critic': ['policy']}, 4,
                actor_hidden_dims=[256, 128], critic_hidden_dims=[256, 256])
            new = ActorCriticRecurrent({'policy': torch.zeros(4, 52)},
                {'policy': ['policy'], 'critic': ['policy']}, 4,
                actor_hidden_dims=[256, 128], critic_hidden_dims=[256, 256])
        old.load_state_dict(state)
        new.load_state_dict(padded)
        with torch.no_grad():
            for _ in range(8):
                x = torch.randn(4, 43)
                y = torch.cat((x, torch.randn(4, 9)), -1)
                torch.testing.assert_close(old.act_inference({'policy': x}), new.act_inference({'policy': y}), atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(old.evaluate({'policy': x}), new.evaluate({'policy': y}), atol=2e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
