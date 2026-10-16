"""CPU checks for V30 event geometry and matched launcher contract."""

import ast
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_directional_eval


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'run_train_two_circle_path_free_stage2_v30_liftoff_cross_ab.sh'
term = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/liftoff_cross_track_v30.py'))[
    'liftoff_cross_track_term']


class V30LiftoffCrossTests(unittest.TestCase):
    def test_asset_preflight_blocks_isaac_without_remote_ground(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fake_curl = directory / 'curl'
            fake_curl.write_text(
                '#!/usr/bin/env bash\n'
                '[[ -z "${http_proxy-}${https_proxy-}${all_proxy-}${HTTP_PROXY-}${HTTPS_PROXY-}${ALL_PROXY-}" ]] || exit 99\n'
                'exit "${V30_FAKE_CURL_STATUS}"\n'
            )
            fake_curl.chmod(0o755)
            base_env = {
                **os.environ,
                'PATH': str(directory) + os.pathsep + os.environ['PATH'],
                'http_proxy': 'http://dummy.invalid',
                'https_proxy': 'http://dummy.invalid',
                'all_proxy': 'socks5://dummy.invalid',
                'HTTPS_PROXY': 'http://dummy.invalid',
                'ASSET_PREFLIGHT_ONLY': '1',
                'ISAAC_SIM_ROOT': str(directory / 'missing_isaac'),
                'OUTPUT': str(directory / 'uncreated_output'),
            }
            failed = subprocess.run(
                ['bash', str(SCRIPT)], cwd=ROOT,
                env={**base_env, 'V30_FAKE_CURL_STATUS': '7'},
                capture_output=True, text=True,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn('Direct download failed; Isaac training was not started',
                          failed.stderr)
            self.assertFalse((directory / 'uncreated_output').exists())
            passed = subprocess.run(
                ['bash', str(SCRIPT)], cwd=ROOT,
                env={**base_env, 'V30_FAKE_CURL_STATUS': '0'},
                capture_output=True, text=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn('Preflight-only mode; Isaac training was not started',
                          passed.stdout)

    def test_event_penalty_uses_actual_foot_to_p1_only(self):
        foot = torch.zeros(5, 2)
        target = torch.tensor([[.3, 0], [.3, 0], [.3, 0], [0., 0.], [0., .3]])
        velocity = torch.tensor([[.8, 0], [0, .05], [0, .2], [0, .4], [-.2, 0]])
        event = torch.tensor([True, True, True, True, False])
        reward, speed, valid = term(foot, target, velocity, event, -5)
        torch.testing.assert_close(reward, torch.tensor([0., 0., -5., 0., 0.]))
        torch.testing.assert_close(speed[:3], torch.tensor([0., .05, .2]))
        self.assertEqual(valid.tolist(), [True, True, True, False, False])

    def test_rotation_equivariance_and_cap(self):
        foot = torch.tensor([[.05, -.02]])
        target = torch.tensor([[.35, -.02]])
        velocity = torch.tensor([[.2, .8]])
        event = torch.ones(1, dtype=torch.bool)
        reward, speed, _ = term(foot, target, velocity, event, -5)
        rotation = torch.tensor([[0., -1.], [1., 0.]])
        turned, turned_speed, _ = term(foot @ rotation, target @ rotation,
                                      velocity @ rotation, event, -5)
        torch.testing.assert_close(reward, turned)
        torch.testing.assert_close(speed, turned_speed)
        self.assertEqual(reward.item(), -20.0)

    def test_v3_accounting_reward_adds_term_only_on_liftoff(self):
        source = ROOT / 'Quadhopper_Planner_Random/path_free_stage2_v3_accounting_env.py'
        tree = ast.parse(source.read_text())
        cls_ast = next(node for node in tree.body if isinstance(node, ast.ClassDef))

        class Parent:
            def _get_rewards(self):
                return torch.full((2,), 10.)

            def stage2_summary(self):
                return {}

        namespace = dict(
            torch=torch, PathFreeStage2V3Env=Parent,
            liftoff_cross_track_term=term,
            terminal_reward_correction=lambda death, z, quat, scale: (
                torch.zeros(2), torch.zeros(2, dtype=torch.bool)),
        )
        exec(compile(ast.Module(body=[cls_ast], type_ignores=[]), str(source), 'exec'), namespace)
        env = namespace['PathFreeStage2V3AccountingEnv']()
        env.cfg = SimpleNamespace(termination_penalty_scale=-300.,
                                  liftoff_cross_track_penalty_scale=-5.)
        env.device = 'cpu'
        env._robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[1., 0, 0, 0.]]).repeat(2, 1),
            root_lin_vel_w=torch.tensor([[0., .2, 0.], [0., .2, 0.]])))
        env._accounting_physical_death = torch.zeros(2, dtype=torch.bool)
        env._accounting_apex_totals = torch.zeros(3, dtype=torch.long)
        env._episode_sums = {'termination_p': torch.zeros(2)}
        env.extras = {'log': {}}
        env.commands = SimpleNamespace(lookahead=lambda: (torch.tensor([[.3, 0.], [.3, 0.]]), None))
        env.foot_position_w = lambda: torch.zeros(2, 3)
        env._liftoff_event = torch.tensor([True, False])
        env._initial_drop_pending = torch.zeros(2, dtype=torch.bool)
        torch.testing.assert_close(env._get_rewards(), torch.tensor([5., 10.]))
        self.assertEqual(env.stage2_summary()['v30_penalized_liftoff_events'], 1)
        self.assertEqual(env.stage2_summary()['v30_liftoff_cross_track_penalty_total'], -5.0)
        self.assertEqual(env.extras['log']['V30/liftoff_cross_track_penalty_total'], -5.0)

    def test_train_arms_differ_only_in_reward_scale(self):
        commands = {}
        for variant in ('control', 'cross5'):
            env = {**os.environ, 'DRY_RUN': '1', 'VARIANT': variant,
                   'NUM_ENVS': '1024', 'ITERATIONS': '100',
                   'OUTPUT': '/tmp/v30-liftoff-cross-contract'}
            result = subprocess.run(['bash', str(SCRIPT)], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            command = shlex.split(result.stdout.strip().splitlines()[-1])
            self.assertTrue(command[0].endswith('/python.sh'))
            self.assertEqual(command[command.index('--num_envs') + 1], '1024')
            self.assertEqual(command[command.index('--iterations') + 1], '100')
            self.assertIn('--v3-accounting-fix', command)
            self.assertEqual(command[command.index('--long-distance-max') + 1], '0.5')
            self.assertNotIn('--resume-runner-state', command)
            commands[variant] = command
        for command in commands.values():
            command[command.index('--liftoff-cross-track-penalty-scale') + 1] = '<scale>'
        self.assertEqual(commands['control'], commands['cross5'])

    def test_directional_eval_restores_experiment_reward(self):
        manifest = {
            'stability_profile': 'two_circle_v18',
            'protocol': 'physical_1m_history_foot_path_free_stage2_v3_strict_success',
            'v3_accounting_fix': True,
            'observation_layout': 'history_foot_v19',
            'two_circle_route': {
                'short_hop_radius_m': [0.12, 0.30],
                'long_hop_radius_m': [0.30, 0.50],
            },
            'reverse_turn_halfwidth_deg': 30.0,
            'policy_target_root_z_m': 1.0,
            'target_tolerance_m': 0.1,
            'height_progress_reward_scale': 3.0,
            'v30_liftoff_cross_track': {'penalty_scale': -5.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            checkpoint = source / 'model_99.pt'
            checkpoint.touch()
            (source / 'manifest.json').write_text(json.dumps(manifest))
            argv = ['run_directional_eval.py', str(checkpoint), '--dry-run',
                    '--root', str(source / 'eval'), '--scenario', 'uniform_turn180',
                    '--condition', 'robust']
            with patch.object(sys, 'argv', argv), \
                 patch.object(run_directional_eval.subprocess, 'run') as run:
                run_directional_eval.main()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--liftoff-cross-track-penalty-scale') + 1], '-5.0')
            self.assertIn('--v3-accounting-fix', command)


if __name__ == '__main__':
    unittest.main()
