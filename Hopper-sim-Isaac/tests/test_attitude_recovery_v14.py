"""CPU-only numerical / reward / launcher checks. Never starts Isaac Sim."""
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
PACKAGE = ROOT / 'Quadhopper_Planner_Random'
V12 = runpy.run_path(str(PACKAGE / 'landing_recovery_v12.py'))
V13 = runpy.run_path(str(PACKAGE / 'contact_readiness_v13.py'))
V14 = runpy.run_path(str(PACKAGE / 'attitude_recovery_v14.py'))
FLIGHT = runpy.run_path(str(PACKAGE / 'flight_stability.py'))
CONTRACT = runpy.run_path(str(PACKAGE / 'path_free_stage2.py'))


def roll_quat(degrees):
    a = torch.deg2rad(torch.as_tensor(degrees, dtype=torch.float64)) / 2
    return torch.stack((a.cos(), a.sin(), torch.zeros_like(a), torch.zeros_like(a)), -1)


class AttitudeRecoveryTests(unittest.TestCase):
    def test_crossing_upright_detects_risk_missed_by_linear_predictor(self):
        q = roll_quat([-20.3, -9.0])
        rates = torch.tensor([[3.68, 0., 0.], [4.16, 0., 0.]], dtype=q.dtype)
        horizon = torch.tensor([.286, .204], dtype=q.dtype)
        vz = -torch.ones(2, dtype=q.dtype)
        old = V13['predicted_touchdown_tilt'](q, rates, horizon, vz)[1]
        _, forecast, actual_horizon = V14['predicted_touchdown_tilt'](q, rates, horizon, vz)
        torch.testing.assert_close(old, torch.zeros_like(old))
        expected = torch.tensor([-20.3, -9.0], dtype=q.dtype) + torch.rad2deg(rates[:, 0] * horizon)
        torch.testing.assert_close(torch.rad2deg(forecast), expected)
        torch.testing.assert_close(actual_horizon, horizon)
        self.assertTrue((V13['danger_cost'](forecast, forecast) > 0).all())

    def test_arbitrary_3d_rotation_matches_independent_matrix_exponential(self):
        generator = torch.Generator().manual_seed(14)
        q = torch.randn(12, 4, generator=generator, dtype=torch.float64)
        q /= q.norm(dim=-1, keepdim=True)
        rates = torch.randn(12, 3, generator=generator, dtype=q.dtype) * 3
        c, vz = torch.full((12,), .2, dtype=q.dtype), torch.full((12,), -.9, dtype=q.dtype)
        tilt, forecast, horizon = V14['predicted_touchdown_tilt'](q, rates, c, vz)
        w, x, y, z = q.unbind(-1)
        rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                                2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                                2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
        wx, wy, wz = rates.unbind(-1)
        zero = torch.zeros_like(wx)
        skew = torch.stack((zero, -wz, wy, wz, zero, -wx, -wy, wx, zero), -1).reshape(-1, 3, 3)
        expected = [tilt]
        for fraction in (.25, .5, .75, 1.):
            future = rotation @ torch.matrix_exp(skew * (horizon * fraction)[:, None, None])
            expected.append(torch.acos(future[:, 2, 2].clamp(-1, 1)))
        torch.testing.assert_close(forecast, torch.stack(expected).amax(0))
        # Rotate world axes in yaw; quaternion sign and normalization must not matter.
        yaw_q = torch.stack((-q[:, 3], -q[:, 2], q[:, 1], q[:, 0]), -1)
        for transformed in (yaw_q, -q, 2*q):
            result = V14['predicted_touchdown_tilt'](transformed, rates, c, vz)
            torch.testing.assert_close(result[0], tilt)
            torch.testing.assert_close(result[1], forecast)

    def test_zero_rates_horizon_shapes_and_intermediate_peak(self):
        q = roll_quat([0., 20., 90.])
        z = torch.zeros(3, dtype=q.dtype)
        tilt, forecast, horizon = V14['predicted_touchdown_tilt'](
            q, torch.zeros(3, 3, dtype=q.dtype), torch.tensor([-.01, .5, .5]), -z)
        torch.testing.assert_close(forecast, tilt)
        self.assertEqual(horizon[0].item(), 0.)
        self.assertLessEqual(horizon.max().item(), .300001)
        # A full roll returns to the starting orientation but passes upside down.
        _, peak, _ = V14['predicted_touchdown_tilt'](
            q[:1], torch.tensor([[2*math.pi/.3, 0., 0.]], dtype=q.dtype),
            torch.tensor([.3]), torch.tensor([-1.]))
        self.assertGreater(peak.item(), math.radians(179))
        with self.assertRaises(ValueError):
            V14['predicted_touchdown_tilt'](q, torch.zeros(3, 2), z, z)
        with self.assertRaises(ValueError):
            V14['predicted_touchdown_tilt'](q[:, :3], torch.zeros(3, 3), z, z)
        with self.assertRaises(ValueError):
            V14['predicted_touchdown_tilt'](q, torch.zeros(3, 3), z[:, None], z)

    def test_failed_trace_release_snapshots_stay_active_during_ascent(self):
        state = V14['AttitudeRecovery'](3, 'cpu', .01)
        yes, no = torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
        q = roll_quat([32.9, 32.6, 10.])
        rates = torch.tensor([[3.25, -4.96, 0.], [2.31, 0., 0.], [.2, .2, 0.]])
        clearance, vz = torch.tensor([.054, .078, .05]), torch.tensor([.23, 1.33, .5])
        state.update(yes, no, no, clearance*0, vz*0, q, rates)
        for _ in range(2):
            active, released = state.update(no, no, no, clearance, vz, q, rates)
        self.assertEqual(active.tolist(), [True, True, False])
        self.assertEqual(released.tolist(), [False, False, True])
        self.assertEqual(state.remaining.tolist(), [38, 38, 0])

    def test_safe_release_requires_both_attitude_rate_and_consecutive_samples(self):
        state = V14['AttitudeRecovery'](4, 'cpu', .01)
        yes, no = torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)
        q = roll_quat([10., 25., 10., 10.])
        rates = torch.tensor([[.5, .5, 0.], [.5, .5, 0.], [1., 1., 0.], [.5, .5, 99.]])
        c, vz = torch.full((4,), .1), torch.ones(4)
        state.update(yes, no, no, c*0, vz*0, q, rates)
        state.update(no, no, no, c, vz, q, rates)
        # Safe yaw spin alone is outside this roll/pitch recovery gate.
        self.assertEqual(state.release_count.tolist(), [1, 0, 0, 1])
        state.update(no, no, no, c, -vz, q, rates)
        self.assertEqual(state.release_count.tolist(), [0, 0, 0, 0])
        state.update(no, no, no, c, vz, q, rates)
        _, released = state.update(no, no, no, c, vz, q, rates)
        self.assertEqual(released.tolist(), [True, False, False, True])

    def test_timeout_recontacts_stationary_drop_and_partial_reset(self):
        state = V14['AttitudeRecovery'](3, 'cpu', .01)
        yes, no = torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
        q, rates, z = roll_quat([40.]*3), torch.ones(3, 3)*2, torch.zeros(3)
        state.update(yes, torch.tensor([False, True, False]),
                     torch.tensor([False, False, True]), z, z, q, rates)
        self.assertEqual(state.remaining.tolist(), [40, 0, 0])
        state.reset(slice(None))
        state.update(yes, no, no, z, z, q, rates)
        state.reset(torch.tensor([1]))
        self.assertEqual(state.active.tolist(), [True, False, True])
        state.update(yes, no, no, z, z, q, rates)
        self.assertEqual(state.remaining.tolist(), [39, 40, 39])
        for _ in range(40):
            state.update(no, no, no, z+.1, z+.5, q, rates)
        self.assertEqual(state.remaining.tolist(), [0]*3)
        self.assertFalse(state.active.any())
        self.assertFalse(state.release_count.any())
        with self.assertRaises(ValueError):
            V14['AttitudeRecovery'](1, 'cpu', 0)

    def test_actual_inherited_reward_chain_once_without_double_charge(self):
        class Parent:
            def __init__(self, cfg, **kwargs):
                self.cfg, self.num_envs, self.device, self.step_dt = cfg, 3, 'cpu', .01
            def _get_rewards(self):
                self.calls += 1
                return torch.ones(3, dtype=torch.float64)
            def _reset_idx(self, ids):
                self.parent_reset = ids

        base = Parent
        for version, helpers in ((12, V12), (13, V13), (14, V14)):
            path = PACKAGE / f'path_free_stage2_v{version}_env.py'
            cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
            ns = dict(helpers, torch=torch,
                      touchdown_roll_pitch_cost=FLIGHT['touchdown_roll_pitch_cost'],
                      euler_xyz_from_quat=lambda q: (2*torch.atan2(q[:, 1], q[:, 0]),
                                                     torch.zeros(3), torch.zeros(3)))
            ns[f'PathFreeStage2V{3 if version == 12 else version-1}Env'] = base
            exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
            base = ns[f'PathFreeStage2V{version}Env']
        cfg = SimpleNamespace(stationary_command_radius_m=.002,
                              moving_long_recovery_angle_deadband_rad=math.radians(10),
                              moving_long_recovery_rate_deadband_rad_s=.8,
                              moving_long_recovery_angle_penalty_scale=-20.,
                              moving_long_recovery_rate_penalty_scale=-8.)
        env = base(cfg)
        env.calls = 0
        data = SimpleNamespace(root_quat_w=roll_quat([40., 40., 10.]),
                               root_ang_vel_b=torch.tensor([[2., 0., 0.], [2., 0., 0.], [.2, 0., 0.]]),
                               root_pos_w=torch.tensor([[0., 0., .2]]*3),
                               root_lin_vel_w=torch.tensor([[0., 0., -.2]]*3),
                               joint_pos=torch.tensor([[.01], [0.], [0.]]))
        env._robot = SimpleNamespace(data=data, _ALL_INDICES=torch.arange(3))
        env._spring_joint_id = 0
        env.foot_geometry = {'sphere_radius_m': .012}
        env.foot_position_w = lambda: torch.tensor([[0., 0., .062]]*3)
        env.commands = SimpleNamespace(anchor_w=torch.zeros(3, 2),
                                       lookahead=lambda: (torch.ones(3, 2), None))
        env._near_ground_recovery.remaining.fill_(20)
        env._moving_long_recovery_remaining = torch.tensor([10, 0, 0])
        env._touchdown_event = torch.zeros(3, dtype=torch.bool)
        env._completed_stationary_command = torch.zeros(3, dtype=torch.bool)
        env._initial_drop_pending = torch.zeros(3, dtype=torch.bool)
        env._cycle_active = torch.ones(3, dtype=torch.bool)
        env.extras = {'log': {}}
        result = env._get_rewards()
        torch.testing.assert_close(result, torch.tensor([1., 1.-1.8-.1152, 1.], dtype=result.dtype))
        self.assertEqual(env.calls, 1)
        self.assertEqual(env.extras['log']['V13/low_altitude_recovery_active_rate'].item(), 0.)
        self.assertEqual(env.extras['log']['V13/approach_active_rate'].item(), 0.)
        data.root_lin_vel_w[:, 2] = .5  # airborne ascent still penalized if unsafe
        torch.testing.assert_close(env._get_rewards(), result)
        self.assertEqual(env.calls, 2)
        env._reset_idx(torch.tensor([1]))
        self.assertFalse(env._near_ground_recovery.active[1])
        self.assertEqual(env._near_ground_recovery.remaining[1].item(), 0)
        env._reset_idx(None)
        self.assertFalse(env._near_ground_recovery.remaining.any())

    def test_launcher_dry_run_only_adds_v14_flag_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fake.pt'; checkpoint.touch()
            commands = []
            for name in ('v13_contact_readiness', 'v14_attitude_recovery'):
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
            commands[1].remove('--attitude-recovery-v14')
            self.assertEqual(commands[0], commands[1])

    def test_eval_visualization_restore_v14_and_dependencies(self):
        import run_directional_eval, run_two_circle_eval
        manifest = dict(protocol=CONTRACT['PROTOCOL_V3'], attitude_recovery_v14=True,
                        stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12, .3], long_hop_radius_m=[.5, .75]),
                        reverse_turn_halfwidth_deg=30, max_turn_angle_deg=180,
                        zero_hop_probability=.12, reverse_turn_probability=.80,
                        policy_target_root_z_m=1., target_tolerance_m=.1)
        for module, extra in ((run_directional_eval, ['--scenario', 'all']),
                              (run_two_circle_eval, ['--visualize'])):
            with patch.object(sys, 'argv', ['test', '/tmp/fake-v14.pt', '--dry-run', *extra]), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'exists', return_value=False), \
                 patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                 patch.object(module.subprocess, 'run') as run:
                module.main()
                self.assertTrue(run.called)
                for call in run.call_args_list:
                    for flag in ('--path-free-stage2-v3', '--moving-handoff-v11',
                                 '--landing-recovery-v12', '--contact-readiness-v13',
                                 '--attitude-recovery-v14', '--dry-run'):
                        self.assertIn(flag, call.args[0])


if __name__ == '__main__':
    unittest.main()
