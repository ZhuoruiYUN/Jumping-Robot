"""CPU regressions for the actual inherited detector and a reused PhysX buffer."""
import ast
import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import unittest
from unittest.mock import patch
import torch

from Quadhopper_Planner_Random.apex_event_fix_v15 import ApexEventState
from Quadhopper_Planner_Random.path_free_stage2 import PROTOCOL_V3, manifest_flags

ROOT = Path(__file__).resolve().parents[1]


def load_classes():
    tree = ast.parse((ROOT / 'Quadhopper_Planner_Circular/planner_circular_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PlannerCircularEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_update_cycle_events')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-inherited-detector>', 'exec'), ns)

    def reset(env, ids):
        ids = env._robot._ALL_INDICES if ids is None else ids
        env._previous_vz[ids] = 0
        env._apex_event[ids] = False

    legacy = type('Legacy', (), {'_update_cycle_events': ns['_update_cycle_events'],
                                 '_reset_idx': reset})
    tree = ast.parse((ROOT / 'Quadhopper_Planner_Random/path_free_stage2_v15_env.py').read_text())
    fixed_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    ns = {'torch': torch, 'PathFreeStage2V12Env': legacy, 'ApexEventState': ApexEventState}
    exec(compile(ast.Module(body=[fixed_cls], type_ignores=[]), '<actual-fixed-env>', 'exec'), ns)
    return legacy, ns['PathFreeStage2V15Env']


def airborne_fixture(cls):
    env = object.__new__(cls)
    env.num_envs, env.device, env.step_dt = 2, 'cpu', .01
    data = NS(joint_pos=torch.zeros(2, 1),
              root_pos_w=torch.tensor([[0., 0., 1.], [0., 0., .8]]),
              root_lin_vel_w=torch.zeros(2, 3))
    env._robot = NS(data=data, _ALL_INDICES=torch.arange(2))
    env._spring_joint_id = 0
    for name in ('previous_contact', 'initial_drop_pending', 'target_hit_event', 'target_miss_event',
                 'apex_event', 'liftoff_event', 'touchdown_event', 'valid_apex_touchdown_event',
                 'prepared_landing_event', 'setup_touchdown_event', 'final_touchdown_event',
                 'circle_complete_event', 'initial_drop_settled_event'):
        setattr(env, '_' + name, torch.zeros(2, dtype=torch.bool))
    env._contact_confirmed = torch.ones(2, dtype=torch.bool)
    env._cycle_active = torch.ones(2, dtype=torch.bool)
    for name in ('cycle_time', 'previous_vz', 'apex_error', 'touchdown_apex_shortfall'):
        setattr(env, '_' + name, torch.zeros(2))
    for name in ('cycle_max_z', 'previous_cycle_max_z', 'active_target_height', 'apex_target_height'):
        setattr(env, '_' + name, torch.ones(2))
    env._update_reference = lambda: None
    return env


class ApexEventFixTest(unittest.TestCase):
    def test_reproduces_live_buffer_alias_and_repairs_actual_inherited_detector(self):
        legacy_cls, fixed_cls = load_classes()
        for cls, expected in ((legacy_cls, [False, False]), (fixed_cls, [True, False])):
            env = airborne_fixture(cls)
            vel = env._robot.data.root_lin_vel_w
            vel[:, 2] = torch.tensor([.3, .2])
            env._update_cycle_events()
            # PhysX refreshes the same storage before the next reward call.
            vel[:, 2] = torch.tensor([-.1, .1])
            env._update_cycle_events()
            self.assertEqual(env._apex_event.tolist(), expected)
            if cls is fixed_cls:
                self.assertNotEqual(env._previous_vz.data_ptr(), vel[:, 2].data_ptr())

    def test_repeated_velocity_crossings_do_not_pay_twice(self):
        _, fixed_cls = load_classes()
        env = airborne_fixture(fixed_cls)
        for vz in (.3, -.1, .2, -.2):
            env._robot.data.root_lin_vel_w[:, 2] = vz
            env._update_cycle_events()
        self.assertFalse(env._apex_event.any())
        self.assertEqual(env._v15_apex_totals.tolist(), [2, 0, 2])

    def test_partial_reset_does_not_write_simulator_velocity_or_reset_other_env(self):
        _, fixed_cls = load_classes()
        env = airborne_fixture(fixed_cls)
        vel = env._robot.data.root_lin_vel_w
        vel[:, 2] = .3
        env._update_cycle_events()
        vel[:, 2] = -.1
        env._update_cycle_events()
        before = vel.clone()
        env._reset_idx(torch.tensor([0]))
        torch.testing.assert_close(vel, before)
        self.assertEqual(env._apex_state().emitted.tolist(), [False, True])
        self.assertEqual(env._previous_vz[0].item(), 0)
        env._reset_idx(None)
        self.assertFalse(env._apex_state().emitted.any())

    def test_touchdown_fallback_only_for_unsettled_hop_and_liftoff_rearms(self):
        state = ApexEventState(3, 'cpu')
        candidate = torch.tensor([True, False, True])
        zero = torch.zeros(3, dtype=torch.bool)
        initial = torch.tensor([False, False, True])
        event, fallback, _ = state.update(candidate, zero, zero, initial, torch.zeros(3))
        self.assertEqual(event.tolist(), [True, False, False])
        event, fallback, suppressed = state.update(
            torch.ones(3, dtype=torch.bool), zero, torch.ones(3, dtype=torch.bool),
            initial, torch.zeros(3))
        self.assertEqual(event.tolist(), [False, True, False])
        self.assertEqual(fallback.tolist(), [False, True, False])
        self.assertEqual(suppressed.tolist(), [True, False, False])
        event, _, _ = state.update(torch.ones(3, dtype=torch.bool),
            torch.ones(3, dtype=torch.bool), zero, zero, torch.zeros(3))
        self.assertTrue(event.all())

    def test_manifest_restores_isolated_v12_fix(self):
        flags = manifest_flags({'protocol': PROTOCOL_V3, 'apex_event_fix_v15': True})
        self.assertEqual(flags, ['--path-free-stage2-v3', '--moving-handoff-v11',
                                 '--landing-recovery-v12', '--apex-event-fix-v15'])
        self.assertNotIn('--apex-event-fix-v15', manifest_flags(
            {'protocol': PROTOCOL_V3, 'landing_recovery_v12': True}))
        with self.assertRaises(ValueError):
            manifest_flags({'protocol': PROTOCOL_V3, 'apex_event_fix_v15': True,
                            'contact_readiness_v13': True})

    def test_outer_cli_forwards_fix_and_rejects_wrong_task(self):
        checkpoint = ROOT / 'outputs/tracking_recovery/two_circle_path_free_stage2_v12_landing_recovery_075_20260927_101643/model_399.pt'
        isaac = '/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh'
        cmd = [isaac, str(ROOT / 'run_tracking_recovery.py'), 'train',
               '--checkpoint', str(checkpoint), '--root', '/tmp/v15-apex-test',
               '--stability-profile', 'two_circle_v18', '--observation-layout', 'history_foot_v19',
               '--path-free-stage2-v3', '--moving-handoff-v11', '--landing-recovery-v12',
               '--distance-min', '.12', '--distance-max', '.75',
               '--short-distance-min', '.12', '--short-distance-max', '.30',
               '--long-distance-min', '.50', '--long-distance-max', '.75',
               '--hard-sample-probability', '0',
               '--apex-event-fix-v15', '--dry-run']
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--apex-event-fix-v15', result.stdout)
        invalid = subprocess.run(cmd + ['--contact-readiness-v13'], cwd=ROOT,
                                 capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn('excludes v13/v14', invalid.stderr)

    def test_control_and_fix_launchers_differ_only_by_event_fix(self):
        commands = {}
        for variant in ('control', 'fix'):
            process = subprocess.run(
                ['bash', str(ROOT / 'run_train_two_circle_path_free_stage2_v15_apex_event_075.sh')],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, 'VARIANT': variant, 'DRY_RUN': '1',
                     'OUTPUT': '/tmp/v15-command-check', 'ITERATIONS': '100', 'NUM_ENVS': '256'})
            self.assertEqual(process.returncode, 0, process.stderr)
            commands[variant] = shlex.split(process.stdout.strip())
        self.assertTrue(commands['fix'][0].endswith('/python.sh'))
        self.assertIn('--apex-event-fix-v15', commands['fix'])
        commands['fix'].remove('--apex-event-fix-v15')
        self.assertEqual(commands['control'], commands['fix'])

    def test_directional_eval_restores_fix_from_manifest(self):
        import run_directional_eval
        manifest = dict(protocol=PROTOCOL_V3, apex_event_fix_v15=True,
                        stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12, .3],
                                              long_hop_radius_m=[.5, .75]),
                        reverse_turn_halfwidth_deg=30, max_turn_angle_deg=180,
                        zero_hop_probability=.12, reverse_turn_probability=.8,
                        policy_target_root_z_m=1., target_tolerance_m=.1)
        with patch('sys.argv', ['test', '/tmp/v15_model.pt', '--dry-run', '--scenario', 'reverse_only']), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'exists', return_value=False), \
             patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
             patch.object(run_directional_eval.subprocess, 'run') as run:
            run_directional_eval.main()
            self.assertTrue(run.called)
            for call in run.call_args_list:
                self.assertIn('--apex-event-fix-v15', call.args[0])


if __name__ == '__main__':
    unittest.main()
