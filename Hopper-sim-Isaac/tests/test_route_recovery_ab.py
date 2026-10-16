"""CPU checks for selective route retry and first/recovery accounting."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

from Quadhopper_Planner_Random.route_recovery_eval import RouteRecoveryMetrics
from tools.compare_v22_route_recovery_ab import compare

ROOT = Path(__file__).resolve().parents[1]


def load_route_env_with_stub():
    path = ROOT / 'Quadhopper_Planner_Random/path_free_stage2_v3_route_eval_env.py'
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef))

    class Parent:
        def _advance_route(self, ids):
            self.commands.route_index[ids] += 1

    ns = {'PathFreeStage2V3AccountingEnv': Parent,
          'RouteRecoveryMetrics': RouteRecoveryMetrics}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    return ns['PathFreeStage2V3RouteEvalEnv']


class RouteRecoveryABTest(unittest.TestCase):
    def test_retry_large_filters_only_severe_task_misses(self):
        cls = load_route_env_with_stub()
        for mode, expected in (('advance', [1, 1, 1]),
                               ('retry_large', [0, 1, 1])):
            env = object.__new__(cls)
            env.cfg = NS(route_recovery_mode=mode, route_recovery_severe_miss_m=.20)
            env._target_miss_event = torch.tensor([True, True, False])
            env.commands = NS(route_index=torch.zeros(3, dtype=torch.long),
                              lookahead=lambda ids: (torch.zeros(len(ids), 2), None))
            env.foot_position_w = lambda: torch.tensor([[.44, 0., 0.],
                                                       [.15, 0., 0.], [.45, 0., 0.]])
            env._advance_route(torch.arange(3))
            self.assertEqual(env.commands.route_index.tolist(), expected)

    def test_first_hit_retry_hit_and_next_target_after_miss_are_separate(self):
        state = RouteRecoveryMetrics(2, 'cpu', .20)
        ids = torch.tensor([0, 1])
        state.observe(ids, task_hit=torch.tensor([False, False]),
                      foot_error_m=torch.tensor([.44, .15]),
                      settled_apex_m=torch.tensor([1.01, .95]),
                      target_apex_m=torch.ones(2),
                      route_before=torch.tensor([0, 0]),
                      route_after=torch.tensor([0, 1]),
                      next_target_distance_m=torch.tensor([.44, .48]), step=10)
        state.observe(ids, task_hit=torch.tensor([True, True]),
                      foot_error_m=torch.tensor([.04, .06]),
                      settled_apex_m=torch.tensor([1.03, 1.02]),
                      target_apex_m=torch.ones(2),
                      route_before=torch.tensor([0, 1]),
                      route_after=torch.tensor([1, 2]),
                      next_target_distance_m=torch.tensor([.21, .32]), step=100)
        result = state.summary()
        self.assertEqual(result['first_attempts'], 3)
        self.assertEqual(result['first_task_hits'], 1)
        self.assertEqual(result['retry_attempts'], 1)
        self.assertEqual(result['retry_task_hits'], 1)
        self.assertEqual(result['after_miss_attempts'], 2)
        self.assertEqual(result['after_miss_task_hits'], 2)
        self.assertEqual(result['miss_route_advances'], 1)
        self.assertEqual(result['severe_miss_retries'], 1)
        self.assertEqual(result['first_miss_step_by_env'], [10, 10])
        state.reset_pending(torch.tensor([0]))
        self.assertFalse(state.retry_pending[0])
        self.assertEqual(state.summary()['first_attempts'], 3)

    def test_launcher_dry_run_uses_same_policy_and_protocol_in_both_arms(self):
        root = '/tmp/v22-route-ab-dry-run'
        process = subprocess.run(
            ['bash', str(ROOT / 'run_eval_two_circle_path_free_stage2_v22_route_ab.sh')],
            cwd=ROOT, env={**os.environ, 'DRY_RUN': '1', 'AB_ROOT': root},
            capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        commands = [shlex.split(line) for line in process.stdout.splitlines()
                    if line.startswith('/home/terry/Documents/isaacsim/')]
        self.assertEqual(len(commands), 2)
        values = []
        for command in commands:
            self.assertTrue(command[0].endswith('/python.sh'))
            self.assertIn('--v3-accounting-fix', command)
            self.assertNotIn('--landing-recovery-v12', command)
            # The outer launcher consumes --dry-run and only prints this
            # final Isaac command; it never executes the command here.
            self.assertNotIn('--dry-run', command)
            mode = command[command.index('--route-recovery-mode') + 1]
            output = command[command.index('--output') + 1]
            command[command.index('--route-recovery-mode') + 1] = '<MODE>'
            command[command.index('--output') + 1] = '<OUTPUT>'
            values.append((mode, output, command))
        self.assertEqual([v[0] for v in values], ['advance', 'retry_large'])
        self.assertEqual(values[0][2], values[1][2])

    def test_comparator_checks_first_miss_pairing_without_pair_rate_shortcut(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for mode, route_after in (('advance', 1), ('retry_large', 0)):
                state = RouteRecoveryMetrics(1, 'cpu', .20)
                state.observe(torch.tensor([0]), task_hit=torch.tensor([False]),
                              foot_error_m=torch.tensor([.44]),
                              settled_apex_m=torch.tensor([1.01]),
                              target_apex_m=torch.tensor([1.]),
                              route_before=torch.tensor([0]),
                              route_after=torch.tensor([route_after]),
                              next_target_distance_m=torch.tensor([.62]), step=100)
                metrics = state.summary()
                metrics['first_miss_step_by_env'] = [100, 300 if mode == 'advance' else 301]
                metrics['first_miss_foot_error_m_by_env'] = [.44, .15]
                result = {f'route_recovery_{k}': v for k, v in metrics.items()}
                result.update(num_envs=16, steps=30000, death_event_count=0,
                              checkpoint='/tmp/model_149.pt')
                summary = dict(checkpoint='/tmp/model_149.pt', route={'short': [.12, .30]},
                               checkpoint_route={'short': [.12, .30]},
                               seeds={'robust': 483}, steps=30000,
                               route_recovery_mode=mode,
                               route_recovery_severe_miss_m=.20,
                               scenarios={'uniform_turn180': {
                                   'definition': {'max_turn': 180},
                                   'results': {'robust': result}}})
                mode_root = Path(directory) / mode
                raw_result = mode_root / 'uniform_turn180' / 'robust' / 'result.json'
                raw_result.parent.mkdir(parents=True)
                raw_result.write_text(json.dumps({
                    **result,
                    'evaluation_contract': {'route_recovery_eval': {'mode': mode}},
                }))
                path = mode_root / 'summary.json'
                path.write_text(json.dumps(summary))
                paths.append(path)
            report = compare(*paths)
            self.assertTrue(report['global_first_miss_matches'])
            self.assertEqual(report['matched_first_miss_env_count'], 1)
            self.assertEqual(report['first_miss_mismatch_env_ids'], [1])
            self.assertEqual(report['counts_and_rates']['route_recovery_severe_miss_retries'],
                             {'advance': 0, 'retry_large': 1})


if __name__ == '__main__':
    unittest.main()
