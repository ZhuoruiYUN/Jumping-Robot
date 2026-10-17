"""CPU checks for the changed task contract; no simulator launch."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from Quadhopper_Planner_Random.path_free_stage2 import (
    SETTINGS, SETTINGS_V3, PROTOCOL, PROTOCOL_V3,
    apply_settings, endpoint_terms, manifest_flags)
from Quadhopper_Planner_Random.history_foot import ObservationHistory

ROOT = Path(__file__).resolve().parents[1]


class ParentStub:
    def _update_cycle_events(self):
        self.target.add_(100)  # Simulates touchdown consuming the command.


def task_class():
    # Exercise actual task methods without importing Isaac/starting its app.
    tree = ast.parse((ROOT / 'Quadhopper_Planner_Random/path_free_stage2_env.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    ns = {'torch': torch, 'HistoryFootEnv': ParentStub, 'endpoint_terms': endpoint_terms}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<task-under-test>', 'exec'), ns)
    return ns['PathFreeStage2Env']


class PathFreeStage2Test(unittest.TestCase):
    def test_no_reward_for_non_touchdown_or_insufficient_height(self):
        cfg = NS(**SETTINGS)
        error = torch.tensor([0., 0., .05, .10])
        landing, _, _ = endpoint_terms(error, torch.tensor([False, True, True, True]),
                                      torch.zeros(4), .01, cfg)
        self.assertEqual(landing[0].item(), 0)
        self.assertEqual(landing[1].item(), 100)
        self.assertGreater(landing[2].item(), landing[3].item())

    def test_energy_integral_is_time_step_invariant_and_never_positive(self):
        cfg = NS(**SETTINGS)
        def cost(dt):
            return endpoint_terms(torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                                  torch.tensor([100., -3.]), dt, cfg)[1]
        torch.testing.assert_close(cost(.01) * 100, cost(.02) * 50)
        torch.testing.assert_close(cost(.01), torch.tensor([-.01, 0.]))

    def test_wider_foot_reward_preserves_center_and_height_gate(self):
        control = NS(**SETTINGS_V3)
        wide = NS(**{**SETTINGS_V3, 'stage2_foot_precision_width_m': .09})
        error = torch.tensor([0., .10, .15, .15])
        valid = torch.tensor([True, True, True, False])
        power = torch.zeros(4)
        old, _, _ = endpoint_terms(error, valid, power, .01, control)
        new, _, _ = endpoint_terms(error, valid, power, .01, wide)
        self.assertEqual(old[0].item(), new[0].item())
        self.assertGreater(new[1].item(), old[1].item())
        self.assertGreater(new[2].item(), old[2].item())
        self.assertEqual(new[3].item(), 0)
        manifest = {'protocol': PROTOCOL_V3, 'v3_accounting_fix': True,
                    'foot_pair_v33': True, 'path_free_stage2': {
                        'stage2_foot_precision_width_m': .09}}
        flags = manifest_flags(manifest)
        self.assertEqual(flags[-2:], ['--stage2-foot-precision-width-m', '0.09'])

    def test_opt_in_single_hop_10cm_bonus_matches_strict_foot_gate(self):
        control = NS(**SETTINGS_V3)
        bonus = NS(**SETTINGS_V3, stage2_foot_hit_10cm_scale=20.0)
        error = torch.tensor([.05, .099, .10, .15, .02])
        valid = torch.tensor([True, True, True, True, False])
        power = torch.zeros(5)
        old, _, _ = endpoint_terms(error, valid, power, .01, control)
        new, _, _ = endpoint_terms(error, valid, power, .01, bonus)
        torch.testing.assert_close(new - old, torch.tensor([20., 20., 0., 0., 0.]))

    def test_single_hop_10cm_bonus_restores_only_when_recorded(self):
        base = {'protocol': PROTOCOL_V3, 'v3_accounting_fix': True,
                'foot_pair_v33': True, 'path_free_stage2': dict(SETTINGS_V3)}
        self.assertNotIn('--stage2-foot-hit-10cm-scale', manifest_flags(base))
        base['path_free_stage2']['stage2_foot_hit_10cm_scale'] = 20.0
        self.assertEqual(manifest_flags(base)[-2:],
                         ['--stage2-foot-hit-10cm-scale', '20.0'])
        base['foot_pair_v33'] = False
        with self.assertRaises(ValueError):
            manifest_flags(base)

    def test_touchdown_uses_consumed_target_and_height_gate(self):
        env = object.__new__(task_class())
        env.cfg = NS(**SETTINGS)
        env.target = torch.zeros(3, 2)
        env.commands = NS(lookahead=lambda ids=None: (env.target, env.target))
        env.foot_position_w = lambda: torch.tensor([[.03, .04, 0.]]).repeat(3, 1)
        env._touchdown_event = torch.tensor([True, True, False])
        env._initial_drop_pending = torch.zeros(3, dtype=torch.bool)
        env._settled_apex_height = torch.tensor([1., .80, 1.])
        env._settled_apex_target = torch.ones(3)
        env._update_cycle_events()
        torch.testing.assert_close(env._stage2_foot_error, torch.full((3,), .05))
        self.assertEqual(env._stage2_valid_touchdown.tolist(), [True, False, False])

    def test_strict_stage3_landing_gate_requires_apex_at_or_above_command(self):
        env = object.__new__(task_class())
        env.cfg = NS(**SETTINGS_V3)
        env.target = torch.zeros(2, 2)
        env.commands = NS(lookahead=lambda ids=None: (env.target, env.target))
        env.foot_position_w = lambda: torch.zeros(2, 3)
        env._touchdown_event = torch.tensor([True, True])
        env._initial_drop_pending = torch.zeros(2, dtype=torch.bool)
        env._settled_apex_height = torch.tensor([.999, 1.000])
        env._settled_apex_target = torch.ones(2)
        env._update_cycle_events()
        self.assertEqual(env._stage2_valid_touchdown.tolist(), [False, True])

    def test_replan_does_not_require_a_curve_solver(self):
        env = object.__new__(task_class())
        pos = torch.tensor([[0., 0., .28], [10., 0., .28]])
        target = pos[:, :2] + torch.tensor([.25, 0.])
        env._robot = NS(data=NS(root_pos_w=pos))
        env.commands = NS(lookahead=lambda ids: (target[ids], target[ids] + .1))
        env._height_commands = lambda ids: (torch.ones(len(ids)), torch.ones(len(ids)))
        env._normalize_xy = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        env._active_target_height = torch.zeros(2)
        env._cycle_takeoff_direction_w = torch.zeros(2, 2)
        env._cycle_next_direction_w = torch.zeros(2, 2)
        env._cycle_time = torch.ones(2)
        env.planner = NS(**{k: torch.full((2, 33, 3), -999.) for k in (
            'positions_w', 'next_positions_w', 'velocities_w', 'next_velocities_w')})
        env._replan(torch.tensor([1]))
        torch.testing.assert_close(env.planner.positions_w[1], pos[1].expand(33, 3))
        self.assertTrue((env.planner.positions_w[0] == -999).all())
        carrot, tangent, _, axis, _ = env._spatial_guidance()
        extra = torch.cat((carrot - pos, tangent, axis), -1)
        self.assertEqual(extra.count_nonzero().item(), 0)
        frame = torch.randn(2, 56)
        frame[:, 43:52] = extra
        history = ObservationHistory(2, 'cpu')
        for tick in range(8):
            obs = history.update(frame, tick).reshape(2, 6, 56)
            self.assertEqual(obs[:, :, 43:52].count_nonzero().item(), 0)

    def test_contract_and_scales(self):
        cfg = NS()
        apply_settings(cfg)
        self.assertFalse(cfg.use_raibert_hermite_guidance)
        for name in ('path_tracking_reward_scale', 'path_penalty_scale',
                     'spatial_progress_reward_scale', 'landing_hint_penalty_scale',
                     'projected_landing_reward_scale', 'projected_landing_penalty_scale'):
            self.assertEqual(getattr(cfg, name), 0)
        self.assertEqual(manifest_flags({'protocol': PROTOCOL}), ['--path-free-stage2'])
        self.assertEqual(manifest_flags({'protocol': PROTOCOL_V3}), ['--path-free-stage2-v3'])
        self.assertEqual(manifest_flags({'protocol': 'old'}), [])

    def test_sysid_manifest_restores_the_new_plant(self):
        manifest = {
            'protocol': PROTOCOL_V3,
            'v3_accounting_fix': True,
            'foot_pair_v33': {'enabled': True},
            'sysid_20261011_plant': True,
        }
        self.assertEqual(manifest_flags(manifest)[:4], [
            '--path-free-stage2-v3', '--v3-accounting-fix',
            '--foot-pair-v33', '--sysid-20261011-plant',
        ])
        manifest['foot_pair_v33'] = False
        with self.assertRaises(ValueError):
            manifest_flags(manifest)

    def test_eval_and_visualization_restore_path_free_flag(self):
        import run_directional_eval
        import run_two_circle_eval
        manifest = dict(protocol=PROTOCOL, stability_profile='two_circle_v18',
                        observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12, .3],
                                              long_hop_radius_m=[.3, .5]),
                        reverse_turn_halfwidth_deg=30, max_turn_angle_deg=180,
                        zero_hop_probability=.12, reverse_turn_probability=.65,
                        policy_target_root_z_m=1., target_tolerance_m=.1)
        for module, extra in ((run_directional_eval, ['--scenario', 'reverse_only']),
                              (run_two_circle_eval, ['--visualize'])):
            with patch('sys.argv', ['test', '/tmp/stage2_test_model.pt', '--dry-run'] + extra), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'exists', return_value=False), \
                 patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                 patch.object(module.subprocess, 'run') as run:
                module.main()
                self.assertTrue(run.called)
                for call in run.call_args_list:
                    cmd = call.args[0]
                    self.assertIn('--path-free-stage2', cmd)
                    self.assertIn('--dry-run', cmd)
                    self.assertNotIn('--raibert-hermite-guidance', cmd)


if __name__ == '__main__':
    unittest.main()
