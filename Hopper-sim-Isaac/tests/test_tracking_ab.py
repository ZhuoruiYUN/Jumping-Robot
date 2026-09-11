import copy
import ast
import io
import math
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from rsl_rl.modules import ActorCriticRecurrent
from Quadhopper_Planner_Random.tracking_metrics import TrackingMetrics
from Quadhopper_Planner_Random.waypoint_command import TwoHopRandomCommand


class TrackingABTest(unittest.TestCase):
    def test_resets_preserve_totals_and_failed_pairs(self):
        m = TrackingMetrics(2, 'cpu')
        ids = torch.tensor([0, 1])
        def observe(short, error, height):
            m.observe(ids, torch.full((2,), short), torch.tensor(error),
                      torch.zeros(2), torch.zeros(2), torch.tensor(height), torch.ones(2))
        observe(True, [0.04, 0.04], [1., 1.])
        m.reset(torch.tensor([1]))  # death after first hop must lower pair rate
        observe(False, [0.04, 0.04], [1., 1.])
        m.reset(ids)
        s = m.summary()
        self.assertEqual(s['touchdowns'], 4)
        self.assertEqual(s['pairs_started'], 2)
        self.assertEqual(s['pairs_completed'], 1)
        self.assertEqual(s['pair_valid_10cm'], 0.5)
        self.assertEqual(s['valid_hit_10cm'], 1.)

    def test_low_height_is_xy_hit_but_not_valid_hit(self):
        m = TrackingMetrics(1, 'cpu')
        m.observe(torch.tensor([0]), torch.tensor([True]), torch.tensor([0.04]),
                  torch.zeros(1), torch.zeros(1), torch.tensor([0.60]), torch.ones(1))
        s = m.summary()
        self.assertEqual(s['xy_hit_5cm'], 1.)
        self.assertEqual(s['valid_hit_5cm'], 0.)
        self.assertEqual(s['height_ok_rate'], 0.)

    def test_queue_honors_preview_at_both_phase_boundaries(self):
        torch.manual_seed(42)
        c = TwoHopRandomCommand(128, 'cpu', .1, .2, .1, .2, 58, math.pi/6)
        ids = torch.arange(128)
        c.reset(ids, torch.zeros(128, 3), False)
        for _ in range(6):
            promised = c.lookahead()[1].clone()
            c.advance(ids)
            torch.testing.assert_close(c.lookahead()[0], promised, rtol=0, atol=0)
            a = c.lookahead()[0] - c.anchor_w
            b = c.lookahead()[1] - c.lookahead()[0]
            turn = torch.atan2(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0], (a*b).sum(1)).abs()
            self.assertTrue(torch.all(turn <= math.pi/6 + 1e-5))

    def test_touchdown_hook_latches_target_before_parent_advances(self):
        # Execute the production subclass with a minimal parent in place of
        # Isaac physics. This checks event ordering without requiring a GPU.
        class Parent:
            def _update_cycle_events(self):
                self.target.add_(0.20)
                self.planner.positions_w.add_(0.20)

        source = Path(__file__).resolve().parents[1] / 'Quadhopper_Planner_Random/tracking_ab_env.py'
        cls = next(n for n in ast.parse(source.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'TrackingABEnv')
        namespace = dict(torch=torch, PlannerRandomTwoHopEnv=Parent)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        env = object.__new__(namespace['TrackingABEnv'])
        env.target = torch.tensor([[0.20, 0.]])
        env.commands = SimpleNamespace(lookahead=lambda: (env.target, None))
        env.planner = SimpleNamespace(positions_w=torch.zeros(1, 1, 3))
        env.cfg = SimpleNamespace(apex_tolerance=0.15, collect_tracking_metrics=True)
        env._robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.17, 0., 0.38]])))
        env._valid_apex_touchdown_event = torch.tensor([True])
        env._settled_apex_error = torch.tensor([0.30])
        env._settled_apex_height = torch.tensor([1.30])
        env._settled_apex_target = torch.ones(1)
        env._touchdown_event = env._setup_touchdown_event = torch.tensor([True])
        env.tracking_metrics = TrackingMetrics(1, 'cpu')
        env._update_cycle_events()
        result = env.tracking_metrics.summary()
        self.assertAlmostEqual(result['xy_error_m'], 0.03)
        self.assertAlmostEqual(result['along_error_m'], -0.03)
        self.assertEqual(result['xy_hit_5cm'], 1.)
        self.assertEqual(result['valid_hit_5cm'], 0.)
        self.assertFalse(env._valid_apex_touchdown_event.item())

    def test_practical_ab_preserves_learned_preview_pathway(self):
        root = Path(__file__).resolve().parents[1]
        matches = list((root / 'logs/rsl_rl').glob('*landingxy_pairrestart*/2026-09-07_22-13-09/model_10.pt'))
        self.assertEqual(len(matches), 1)
        state = torch.load(matches[0], map_location='cpu', weights_only=False)['model_state_dict']
        self.assertGreater(state['memory_a.rnn.weight_ih_l0'][:, 39:41].norm(), 0.1)
        self.assertGreater(state['memory_c.rnn.weight_ih_l0'][:, 39:41].norm(), 0.4)
        obs = {'policy': torch.randn(8, 43)}
        with redirect_stdout(io.StringIO()):
            actor = ActorCriticRecurrent(obs, {'policy': ['policy'], 'critic': ['policy']},
                4, actor_hidden_dims=[256, 128], critic_hidden_dims=[256, 256])
        actor.load_state_dict(state)
        masked = copy.deepcopy(actor)
        action_differences = []
        value_differences = []
        with torch.no_grad():
            for _ in range(8):
                x = torch.randn(8, 43)
                y = x.clone()
                y[:, 39:41] = 0
                action_differences.append((actor.act_inference({'policy': x}) -
                    masked.act_inference({'policy': y})).abs().max().item())
                value_differences.append((actor.evaluate({'policy': x}) -
                    masked.evaluate({'policy': y})).abs().max().item())
        self.assertGreater(max(action_differences), 1e-5)
        self.assertGreater(max(value_differences), 1e-5)


if __name__ == '__main__':
    unittest.main()
