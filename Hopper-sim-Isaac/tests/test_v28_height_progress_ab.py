"""Check the matched V28 launcher and reward-only evaluation contract."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_directional_eval


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'run_train_two_circle_path_free_stage2_v28_height_progress_ab.sh'


class V28HeightProgressABTests(unittest.TestCase):
    def test_train_arms_differ_only_in_existing_reward_coefficient(self):
        commands = {}
        for variant in ('control', 'progress30'):
            env = {**os.environ, 'DRY_RUN': '1', 'VARIANT': variant,
                   'NUM_ENVS': '1024', 'ITERATIONS': '100',
                   'OUTPUT': '/tmp/v28-height-progress-contract'}
            result = subprocess.run(['bash', str(SCRIPT)], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            command = shlex.split(result.stdout.strip().splitlines()[-1])
            self.assertTrue(command[0].endswith('/python.sh'))
            self.assertEqual(command[command.index('--num_envs') + 1], '1024')
            self.assertEqual(command[command.index('--iterations') + 1], '100')
            self.assertIn('--v3-accounting-fix', command)
            self.assertNotIn('--resume-runner-state', command)
            commands[variant] = command
        for command in commands.values():
            command[command.index('--height-progress-reward-scale') + 1] = '<scale>'
        self.assertEqual(commands['control'], commands['progress30'])

    def test_directional_eval_restores_checkpoint_reward(self):
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
            'height_progress_reward_scale': 30.0,
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
            self.assertEqual(command[command.index('--height-progress-reward-scale') + 1], '30.0')
            self.assertIn('--v3-accounting-fix', command)


if __name__ == '__main__':
    unittest.main()
