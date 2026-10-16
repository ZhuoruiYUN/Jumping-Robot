"""Exercise the V3-only wrapper against the actual inherited event/death methods."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest
from unittest.mock import patch

import torch

from Quadhopper_Planner_Random.apex_event_fix_v15 import ApexEventState
from Quadhopper_Planner_Random.path_free_stage2 import PROTOCOL_V3, manifest_flags
from Quadhopper_Planner_Random.terminal_penalty_fix_v16 import terminal_reward_correction
from tests.test_apex_event_fix_v15 import load_classes, airborne_fixture
from tests.test_terminal_penalty_fix_v16 import load_actual_reward_and_dones, fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Quadhopper_Planner_Random/path_free_stage2_v3_accounting_env.py'


def load_v3_wrapper(parent):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    ns = dict(torch=torch, PathFreeStage2V3Env=parent,
              ApexEventState=ApexEventState,
              terminal_reward_correction=terminal_reward_correction)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns['PathFreeStage2V3AccountingEnv']


class V3AccountingFixTest(unittest.TestCase):
    def test_actual_cycle_detector_uses_owned_history_and_once_per_hop_event(self):
        parent, _ = load_classes()
        cls = load_v3_wrapper(parent)
        env = airborne_fixture(cls)
        vel = env._robot.data.root_lin_vel_w
        for vz, expected in ((.3, False), (-.1, True), (.2, False), (-.2, False)):
            vel[:, 2] = vz
            env._update_cycle_events()
            self.assertEqual(env._apex_event.tolist(), [expected, expected])
        self.assertEqual(env._accounting_apex_totals.tolist(), [2, 0, 2])
        before = vel.clone()
        env._reset_idx(torch.tensor([0]))
        torch.testing.assert_close(vel, before)
        self.assertEqual(env._apex_state().emitted.tolist(), [False, True])

    def test_actual_death_mask_replaces_only_legacy_terminal_reward(self):
        parent, _ = load_actual_reward_and_dones()
        cls = load_v3_wrapper(parent)
        env = fixture(cls)
        env._accounting_apex_totals = torch.zeros(3, dtype=torch.long)
        done, timeout = env._get_dones()
        self.assertEqual(done.tolist(), [False, True, True, True, True, False, True, False])
        self.assertEqual(timeout.tolist(), env.timeout.tolist())
        self.assertEqual(env._get_rewards().tolist(), [123, 93, 93, 93, 93, 123, 123, 123])
        self.assertEqual(env._episode_sums['termination_p'].tolist(), [0, -30, -30, -30, -30, 0, 0, 0])
        self.assertEqual(env._accounting_terminal_totals.tolist(), [4, 2, 1])

    def test_manifest_restores_only_v3_accounting_task(self):
        manifest = {'protocol': PROTOCOL_V3, 'v3_accounting_fix': True}
        self.assertEqual(manifest_flags(manifest), ['--path-free-stage2-v3', '--v3-accounting-fix'])
        with self.assertRaises(ValueError):
            manifest_flags({**manifest, 'landing_recovery_v12': True})
        import run_directional_eval
        manifest.update(stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route={'short_hop_radius_m': [.12, .30],
                                          'long_hop_radius_m': [.30, .50]},
                        reverse_turn_halfwidth_deg=30, policy_target_root_z_m=1.,
                        target_tolerance_m=.1)
        with patch('sys.argv', ['test', '/tmp/v3fix_model.pt', '--dry-run',
                                '--scenario', 'uniform_turn180']), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
             patch.object(run_directional_eval.subprocess, 'run') as run:
            run_directional_eval.main()
            self.assertTrue(run.called)
            for call in run.call_args_list:
                self.assertIn('--v3-accounting-fix', call.args[0])
                self.assertNotIn('--landing-recovery-v12', call.args[0])

    def test_launcher_differs_from_v20_control_only_by_fix_and_output(self):
        commands = {}
        scripts = {'control': 'run_train_two_circle_path_free_stage2_v20_episode_horizon.sh',
                   'fix': 'run_train_two_circle_path_free_stage2_v21_v3_accounting.sh'}
        for name, script in scripts.items():
            env = {**os.environ, 'DRY_RUN': '1', 'NUM_ENVS': '256', 'ITERATIONS': '150',
                   'OUTPUT': '/tmp/v3-accounting-command-check'}
            if name == 'control':
                env['HORIZON_VARIANT'] = 'control61'
            result = subprocess.run(['bash', str(ROOT / script)], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands[name] = shlex.split(result.stdout.strip().splitlines()[-1])
            self.assertTrue(commands[name][0].endswith('/python.sh'))
        commands['fix'].remove('--v3-accounting-fix')
        self.assertEqual(commands['fix'], commands['control'])


if __name__ == '__main__':
    unittest.main()
