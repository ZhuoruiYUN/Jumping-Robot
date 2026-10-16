"""Contract checks for the matched V36 two-foot XY quality experiment."""
import os
from pathlib import Path
import shlex
import subprocess
import unittest

from Quadhopper_Planner_Random.path_free_stage2 import PROTOCOL_V3, manifest_flags


ROOT = Path(__file__).resolve().parents[1]


class PairXYABTests(unittest.TestCase):
    def test_manifest_restores_pair_xy_weight_only_for_experiment(self):
        manifest = {
            'protocol': PROTOCOL_V3,
            'v3_accounting_fix': True,
            'foot_pair_v33': {'foot_radius_m': 0.1},
            'path_free_stage2': {
                'stage2_foot_precision_scale': 60.0,
                'stage2_pair_xy_quality_weight': 0.5,
            },
        }
        flags = manifest_flags(manifest)
        self.assertEqual(flags[-2:], ['--pair-xy-quality-weight-v36', '0.5'])
        manifest['path_free_stage2']['stage2_pair_xy_quality_weight'] = 0.0
        self.assertNotIn('--pair-xy-quality-weight-v36', manifest_flags(manifest))

    def test_training_arms_match_except_pair_xy_weight_and_output(self):
        commands = {}
        for arm in ('control', 'xy50'):
            env = {**os.environ, 'DRY_RUN': '1', 'VARIANT': arm,
                   'NUM_ENVS': '16', 'ITERATIONS': '2',
                   'OUTPUT': f'/tmp/v36_{arm}_contract_test'}
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_two_circle_path_free_stage2_v36_pair_xy_ab.sh')],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True,
            )
            command = shlex.split(result.stdout.strip().splitlines()[-1])
            self.assertTrue(command[0].endswith('/python.sh'))
            self.assertEqual(command[command.index('--seed') + 1], '181')
            self.assertEqual(command[command.index('--stage2-foot-precision-scale') + 1], '60.0')
            output_at = command.index('--output') + 1
            weight_at = command.index('--pair-xy-quality-weight-v36') + 1
            self.assertEqual(command[weight_at], '0.0' if arm == 'control' else '0.5')
            command[output_at] = '<output>'
            command[weight_at] = '<weight>'
            commands[arm] = command
        self.assertEqual(commands['control'], commands['xy50'])


if __name__ == '__main__':
    unittest.main()
