"""v13 safety-shaping checks: CPU-only, no Isaac App or training."""
import ast
import json
import math
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

ROOT = Path(__file__).resolve().parents[1]
V13 = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/contact_readiness_v13.py'))
FLIGHT = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/flight_stability.py'))
CONTRACT = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/path_free_stage2.py'))


def roll_quat(degrees):
    r = math.radians(degrees) / 2
    return [math.cos(r), math.sin(r), 0., 0.]


class ContactReadinessTests(unittest.TestCase):
    def test_projection_is_yaw_invariant_and_rate_directional(self):
        q = torch.tensor([roll_quat(20.)] * 3)
        w = torch.tensor([[2., 0., 0.], [-2., 0., 0.], [2., 0., 0.]])
        c, vz = torch.full((3,), .20), torch.full((3,), -.5)
        tilt, pred, horizon = V13['predicted_touchdown_tilt'](q, w, c, vz)
        self.assertAlmostEqual(float(torch.rad2deg(tilt[0])), 20., places=3)
        self.assertGreater(float(pred[0]), float(tilt[0]))
        self.assertLess(float(pred[1]), float(tilt[1]))
        self.assertLessEqual(float(horizon.max()), .300001)
        # A world-Z yaw rotation changes neither total tilt nor its derivative.
        yaw_q = torch.stack((-q[:, 3], -q[:, 2], q[:, 1], q[:, 0]), -1)
        yaw_tilt, yaw_pred, _ = V13['predicted_touchdown_tilt'](yaw_q, w, c, vz)
        torch.testing.assert_close(yaw_tilt, tilt)
        torch.testing.assert_close(yaw_pred, pred)
        torch.testing.assert_close(V13['danger_cost'](tilt, pred),
                                   V13['danger_cost'](yaw_tilt, yaw_pred))

    def test_projection_respects_zero_clearance_and_clamped_horizon(self):
        q = torch.tensor([roll_quat(0.), roll_quat(30.), roll_quat(90.)])
        w = torch.tensor([[0., 0., 0.], [20., 0., 0.], [99., 0., 0.]])
        tilt, pred, horizon = V13['predicted_touchdown_tilt'](
            q, w, torch.tensor([-.01, .5, .5]), torch.tensor([-.5, -.1, -.1]))
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(float(horizon[0]), 0.)
        self.assertLessEqual(float(horizon.max()), .300001)
        self.assertAlmostEqual(float(pred[0]), float(tilt[0]), places=5)
        safe = torch.deg2rad(torch.tensor([20., 30., 70.]))
        torch.testing.assert_close(V13['danger_cost'](safe, safe), torch.tensor([0., 0., 16.]))
        with self.assertRaises(ValueError):
            V13['predicted_touchdown_tilt'](q, w[:, :2], horizon, horizon)

    def test_approach_excludes_previous_touchdown_window_and_zero_hops(self):
        n = 7
        mask = V13['approach_mask'](
            torch.tensor([1, 1, 0, 1, 1, 1, 1], dtype=torch.bool),
            torch.tensor([1, 1, 1, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
            torch.tensor([0, 5, 0, 0, 0, 0, 0]),
            torch.tensor([.1, .1, .1, .1, .1, .7, .1]),
            torch.tensor([-.5] * 6 + [.2]))
        self.assertEqual(mask.tolist(), [True] + [False] * 6)

    def test_low_altitude_only_fills_v12_gap_not_normal_ascent(self):
        n = 7
        mask = V13['low_altitude_recovery_mask'](
            torch.tensor([20, 20, 0, 20, 20, 20, 20]),
            torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.bool),
            torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.bool),
            torch.tensor([.2] * 6 + [.5]),
            torch.tensor([-.2] * 5 + [.5, -.2]),
            torch.tensor([.05] * n),
            torch.deg2rad(torch.tensor([40., 40., 40., 20., 40., 40., 40.])))
        self.assertEqual(mask.tolist(), [True] + [False] * 6)

    def test_touchdown_rate_cost_symmetric_bounded_and_ignores_yaw(self):
        r = V13['touchdown_rate_cost'](
            torch.tensor([[.5, -.5, 50.], [2.2, -2.2, 0.], [50., -50., 0.]]))
        torch.testing.assert_close(r, torch.tensor([0., 2., 18.]))

    def test_actual_reward_expression_once_and_distinct_phase_masks(self):
        class Parent:
            def _get_rewards(self):
                self.calls += 1
                return torch.ones(3)

        path = ROOT / 'Quadhopper_Planner_Random/path_free_stage2_v13_env.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        ns = dict(V13, torch=torch, PathFreeStage2V12Env=Parent,
                  touchdown_roll_pitch_cost=FLIGHT['touchdown_roll_pitch_cost'],
                  euler_xyz_from_quat=lambda q:(torch.zeros(3),) * 3)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
        env = ns['PathFreeStage2V13Env']()
        env.calls, env.step_dt = 0, .01
        env.cfg = SimpleNamespace(stationary_command_radius_m=.002,
                                  moving_long_recovery_angle_deadband_rad=math.radians(10),
                                  moving_long_recovery_rate_deadband_rad_s=.8,
                                  moving_long_recovery_angle_penalty_scale=-20.,
                                  moving_long_recovery_rate_penalty_scale=-8.)
        q = torch.tensor([roll_quat(20.), roll_quat(40.), roll_quat(45.)])
        w = torch.tensor([[2., 0., 0.], [2., 0., 0.], [2., 0., 0.]])
        data = SimpleNamespace(root_quat_w=q, root_ang_vel_b=w,
                               root_lin_vel_w=torch.tensor([[0., 0., -.5]] * 3),
                               root_pos_w=torch.tensor([[0., 0., .2]] * 3))
        env._robot = SimpleNamespace(data=data)
        env.foot_geometry = {'sphere_radius_m': .012}
        env.foot_position_w = lambda: torch.tensor([[0., 0., .112], [0., 0., .062], [0., 0., .012]])
        env.commands = SimpleNamespace(anchor_w=torch.zeros(3, 2),
                                       lookahead=lambda: (torch.ones(3, 2), None))
        env._near_ground_recovery = SimpleNamespace(remaining=torch.tensor([0, 20, 40]))
        env._cycle_active = torch.ones(3, dtype=torch.bool)
        env._touchdown_event = torch.tensor([False, False, True])
        env._completed_stationary_command = torch.zeros(3, dtype=torch.bool)
        env._initial_drop_pending = torch.zeros(3, dtype=torch.bool)
        env.extras = {'log': {}}
        result = env._get_rewards()
        self.assertEqual(env.calls, 1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(float(result[0]), 1.)  # forecast approach
        self.assertLess(float(result[1]), 1.)  # low-altitude recovery
        self.assertLess(float(result[2]), 1.)  # dangerous touchdown + rate
        self.assertAlmostEqual(float(env.extras['log']['V13/approach_active_rate']), 1/3)
        self.assertAlmostEqual(float(env.extras['log']['V13/low_altitude_recovery_active_rate']), 1/3)
        self.assertAlmostEqual(float(env.extras['log']['V13/touchdown_danger_rate']), 1/3)

    def test_launchers_dry_run_change_only_flag_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fake.pt'; checkpoint.touch()
            commands = []
            for name in ('v12_landing_recovery', 'v13_contact_readiness'):
                output = Path(directory) / name
                env = dict(os.environ, DRY_RUN='1', CHECKPOINT=str(checkpoint),
                           OUTPUT=str(output), ITERATIONS='400')
                env.pop('PYTHONHOME', None); env.pop('PYTHONPATH', None)
                env['PATH'] = '/usr/bin:/bin:' + env.get('PATH', '')
                run = subprocess.run(['bash', str(ROOT / (
                    f'run_train_two_circle_path_free_stage2_{name}_075.sh'))],
                    env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                command = shlex.split(run.stdout.strip())
                command[command.index('--output') + 1] = 'OUTPUT'
                commands.append(command)
                self.assertFalse(output.exists())
            commands[1].remove('--contact-readiness-v13')
            self.assertEqual(commands[0], commands[1])

    def test_manifest_roundtrip_in_eval_and_visualization(self):
        import run_directional_eval, run_two_circle_eval
        manifest = dict(protocol=CONTRACT['PROTOCOL_V3'], moving_handoff_v11=True,
                        landing_recovery_v12=True, contact_readiness_v13=True,
                        stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12, .3], long_hop_radius_m=[.5, .75]),
                        reverse_turn_halfwidth_deg=30, max_turn_angle_deg=180,
                        zero_hop_probability=.12, reverse_turn_probability=.80,
                        policy_target_root_z_m=1., target_tolerance_m=.1)
        for module, extra in ((run_directional_eval, ['--scenario', 'all']),
                              (run_two_circle_eval, ['--visualize'])):
            with patch.object(sys, 'argv', ['test', '/tmp/fake-v13.pt', '--dry-run', *extra]), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'exists', return_value=False), \
                 patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                 patch.object(module.subprocess, 'run') as run:
                module.main()
                self.assertTrue(run.called)
                for call in run.call_args_list:
                    for flag in ('--path-free-stage2-v3', '--moving-handoff-v11',
                                 '--landing-recovery-v12', '--contact-readiness-v13', '--dry-run'):
                        self.assertIn(flag, call.args[0])


if __name__ == '__main__':
    unittest.main()
