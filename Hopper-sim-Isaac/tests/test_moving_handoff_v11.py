"""Reward-only regression checks; no simulator or training is launched."""
import ast
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import symtable
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/moving_handoff.py'))
CONTRACT = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/path_free_stage2.py'))
RECOVERY = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/contact_recovery.py'))


class MovingHandoffTests(unittest.TestCase):
    def test_reward_helpers_are_not_shadowed_by_local_tensors(self):
        path = ROOT / 'Quadhopper_Planner_Random/spatial_tracking_env.py'
        source = path.read_text()
        tree = ast.parse(source)
        helpers = {alias.asname or alias.name for node in tree.body
                   if isinstance(node, ast.ImportFrom) and node.module == 'moving_handoff'
                   for alias in node.names}
        symbols = symtable.symtable(source, str(path), 'exec')
        cls = next(child for child in symbols.get_children()
                   if child.get_name() == 'SpatialTrackingEnv')
        rewards = next(child for child in cls.get_children()
                       if child.get_name() == '_get_rewards')
        for name in helpers & set(rewards.get_identifiers()):
            self.assertTrue(rewards.lookup(name).is_global(), name)

    def test_actual_launch_reward_expression_with_contact_cost_tensor(self):
        # Execute the environment's real reward expression without Isaac; the
        # preceding touchdown cost must not replace the callable launch helper.
        tree = ast.parse((ROOT / 'Quadhopper_Planner_Random/spatial_tracking_env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'SpatialTrackingEnv')
        rewards = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                       and n.name == '_get_rewards')
        statement = next(n for n in rewards.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name)
                         and n.targets[0].id == 'moving_launch_reward')
        for enabled in (False, True):
            cfg = SimpleNamespace(moving_launch_rate_deadband_rad_s=.8,
                                  moving_launch_rate_penalty_scale=-8. if enabled else 0.)
            data = SimpleNamespace(root_ang_vel_b=torch.tensor([[1.8, -1.8, 0.],
                                                                [1.8, -1.8, 0.]]))
            namespace = dict(HANDOFF, self=SimpleNamespace(
                cfg=cfg, step_dt=.01, _robot=SimpleNamespace(data=data)),
                moving_contact_rate_cost=torch.ones(2),
                moving_launch_active=torch.tensor([enabled, False]))
            exec(compile(ast.Module(body=[statement], type_ignores=[]), '<reward>', 'exec'), namespace)
            torch.testing.assert_close(namespace['moving_launch_reward'],
                                       torch.tensor([-.16 if enabled else 0., 0.]))

    def test_contact_both_phases_opt_in_and_initial_drop_excluded(self):
        touchdown = torch.tensor([True, True, True, True, False])
        final = torch.tensor([False, True, True, True, False])
        stationary = torch.tensor([False, False, True, False, False])
        drop = torch.tensor([False, False, False, True, False])
        fn = HANDOFF['moving_contact_event']
        self.assertEqual(fn(touchdown, final, stationary, drop).tolist(),
                         [False, True, False, False, False])
        self.assertEqual(fn(touchdown, final, stationary, drop, all_hops=True).tolist(),
                         [True, True, False, False, False])

    def test_launch_phase_masks(self):
        # extension, compression, early flight, later flight, drop, zero command,
        # touchdown with a stale timer, non-active flight.
        active = HANDOFF['moving_launch_mask'](
            torch.tensor([.01, .01, 0., 0., 0., 0., .01, 0.]),
            torch.tensor([-.1, .1, 0., 0., 0., 0., -.1, 0.]),
            torch.tensor([False, False, True, True, True, True, False, False]),
            torch.tensor([0, 20, 20, 0, 20, 20, 20, 20]),
            torch.tensor([False, False, False, False, True, False, False, False]),
            torch.tensor([False, False, False, False, False, True, False, False]),
            torch.tensor([False, False, False, False, False, False, True, False]),
        )
        self.assertEqual(active.tolist(), [True, False, True, False, False, False, False, False])

    def test_rate_cost_has_deadband_symmetry_and_bounded_tail(self):
        cost = HANDOFF['moving_rate_cost'](
            torch.tensor([[.7, -.8], [1.8, -1.8], [99., -99.]]), .8)
        torch.testing.assert_close(cost, torch.tensor([0., 2., 18.]))
        self.assertEqual(tuple(cost.shape), (3,))

    def test_twenty_step_timer_does_not_need_contact_timer(self):
        count = RECOVERY['recovery_window_steps'](.20, .01)
        self.assertEqual(count, 20)
        timer = torch.zeros(2, dtype=torch.long)
        advance = RECOVERY['advance_recovery_window']
        timer = advance(timer, torch.tensor([True, False]), count)
        for _ in range(19):
            self.assertGreater(int(timer[0]), 0)
            timer = advance(timer, torch.tensor([False, False]), count)
        self.assertEqual(timer.tolist(), [1, 0])
        timer = advance(timer, torch.tensor([False, False]), count)
        self.assertEqual(timer.tolist(), [0, 0])

    def test_settings_change_only_task_shaping_and_manifest_roundtrip(self):
        cfg = SimpleNamespace()
        CONTRACT['apply_settings'](cfg, v3=True)
        old = vars(cfg).copy()
        HANDOFF['apply_settings'](cfg)
        for name, value in old.items():
            self.assertEqual(getattr(cfg, name), value)
        self.assertEqual(cfg.stage2_energy_penalty_scale, 0)
        self.assertFalse(cfg.use_raibert_hermite_guidance)
        manifest = {'protocol': CONTRACT['PROTOCOL_V3']}
        self.assertEqual(CONTRACT['manifest_flags'](manifest), ['--path-free-stage2-v3'])
        manifest['moving_handoff_v11'] = True
        self.assertEqual(CONTRACT['manifest_flags'](manifest),
                         ['--path-free-stage2-v3', '--moving-handoff-v11'])

    def test_legacy_config_off_and_wrench_keeps_long_only_gate(self):
        tree = ast.parse((ROOT / 'Quadhopper_Planner_Random/spatial_tracking_env.py').read_text())
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'SpatialTrackingEnvCfg')
        values = {n.targets[0].id: n.value for n in cfg.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertFalse(ast.literal_eval(values['moving_contact_recovery_all_hops']))
        self.assertEqual(ast.literal_eval(values['moving_launch_stability_window_s']), 0)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'SpatialTrackingEnv')
        reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reset_idx')
        self.assertIn('self._moving_launch_remaining[env_ids] = 0', ast.unparse(reset))
        events = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_update_cycle_events')
        text = ast.unparse(events)
        self.assertIn('recovery_touchdown | moving_long_touchdown', text)
        self.assertIn('self._moving_launch_remaining[self._touchdown_event] = 0', text)
        self.assertIn('self._moving_long_recovery_remaining[self._liftoff_event] = 0', text)

    def test_launchers_dry_run_keep_existing_training_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fake.pt'
            checkpoint.touch()
            commands = []
            for version, suffix in (('v10', 'contact_window'), ('v11', 'moving_handoff')):
                output = Path(directory) / version
                env = dict(os.environ, DRY_RUN='1', CHECKPOINT=str(checkpoint), OUTPUT=str(output),
                           ITERATIONS='900', NUM_ENVS='256', LEARNING_RATE='0.00000010')
                # The outer CLI uses system Python; do not inherit Kit's Python home.
                env.pop('PYTHONHOME', None)
                env.pop('PYTHONPATH', None)
                env['PATH'] = '/usr/bin:/bin:' + env.get('PATH', '')
                run = subprocess.run(
                    ['bash', str(ROOT / f'run_train_two_circle_path_free_stage2_{version}_{suffix}_075.sh')],
                    env=env, check=False, capture_output=True, text=True,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                command = shlex.split(run.stdout.strip())
                command[command.index('--output') + 1] = 'OUTPUT'
                commands.append(command)
                self.assertFalse(output.exists())
            self.assertIn('--moving-handoff-v11', commands[1])
            commands[1].remove('--moving-handoff-v11')
            self.assertEqual(commands[0], commands[1])

    def test_eval_and_visualization_forward_v11_without_isaac(self):
        import run_directional_eval
        import run_two_circle_eval
        manifest = dict(
            protocol=CONTRACT['PROTOCOL_V3'], moving_handoff_v11=True,
            stability_profile='two_circle_v18', observation_layout='history_foot_v19',
            two_circle_route=dict(short_hop_radius_m=[.12, .3], long_hop_radius_m=[.5, .75]),
            reverse_turn_halfwidth_deg=30, max_turn_angle_deg=180,
            zero_hop_probability=.12, reverse_turn_probability=.80,
            policy_target_root_z_m=1., target_tolerance_m=.1,
        )
        for module, extra in ((run_directional_eval, ['--scenario', 'all']),
                              (run_two_circle_eval, ['--visualize'])):
            with patch.object(sys, 'argv', ['test', '/tmp/fake-v11.pt', '--dry-run', *extra]), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'exists', return_value=False), \
                 patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                 patch.object(module.subprocess, 'run') as run:
                module.main()
                self.assertTrue(run.called)
                for call in run.call_args_list:
                    self.assertIn('--moving-handoff-v11', call.args[0])
                    self.assertIn('--path-free-stage2-v3', call.args[0])
                    self.assertIn('--dry-run', call.args[0])


if __name__ == '__main__':
    unittest.main()
