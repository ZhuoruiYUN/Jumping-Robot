"""Check that the v35 reward change and its restored evaluation contract stay paired."""
import os
import math
from pathlib import Path
import shlex
import subprocess
import unittest
from types import SimpleNamespace

import torch

from Quadhopper_Planner_Random.path_free_stage2 import (
    PROTOCOL_V3, SETTINGS_V3, endpoint_terms, manifest_flags,
)

ROOT = Path(__file__).resolve().parents[1]


class FootPrecisionABTests(unittest.TestCase):
    def test_precision_gain_only_pays_at_valid_touchdown(self):
        error = torch.tensor([0.0, 0.06, 0.0])
        valid = torch.tensor([True, True, False])
        baseline = SimpleNamespace(**SETTINGS_V3)
        precision = SimpleNamespace(**{**SETTINGS_V3, 'stage2_foot_precision_scale': 75.0})
        reward60 = endpoint_terms(error, valid, torch.zeros(3), 0.01, baseline)[0]
        reward75 = endpoint_terms(error, valid, torch.zeros(3), 0.01, precision)[0]
        torch.testing.assert_close(reward75 - reward60,
                                   torch.tensor([15.0, 15.0 / math.e, 0.0]))

    def test_manifest_restores_custom_precision_for_evaluation(self):
        manifest = {'protocol': PROTOCOL_V3, 'v3_accounting_fix': True,
                    'foot_pair_v33': {'foot_radius_m': 0.1},
                    'path_free_stage2': {'stage2_foot_precision_scale': 75.0}}
        flags = manifest_flags(manifest)
        self.assertEqual(flags[-2:], ['--stage2-foot-precision-scale', '75.0'])
        manifest['path_free_stage2']['stage2_foot_precision_scale'] = 60.0
        self.assertNotIn('--stage2-foot-precision-scale', manifest_flags(manifest))

    def test_training_arms_share_all_options_except_reward_and_output(self):
        commands = {}
        for arm in ('control', 'precision75'):
            env = {**os.environ, 'DRY_RUN': '1', 'VARIANT': arm,
                   'NUM_ENVS': '16', 'ITERATIONS': '2',
                   'OUTPUT': f'/tmp/v35_{arm}_contract_test'}
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_two_circle_path_free_stage2_v35_foot_precision_ab.sh')],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True,
            )
            command = shlex.split(result.stdout.strip().splitlines()[-1])
            self.assertTrue(command[0].endswith('/python.sh'))
            output_at = command.index('--output') + 1
            scale_at = command.index('--stage2-foot-precision-scale') + 1
            self.assertEqual(command[scale_at], '60.0' if arm == 'control' else '75.0')
            command[output_at] = '<output>'
            command[scale_at] = '<scale>'
            commands[arm] = command
        self.assertEqual(commands['control'], commands['precision75'])


if __name__ == '__main__':
    unittest.main()
