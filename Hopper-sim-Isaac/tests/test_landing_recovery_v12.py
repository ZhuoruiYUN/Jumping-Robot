"""CPU-only checks. Never starts Isaac Sim, training or evaluation."""
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
HELPERS = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/landing_recovery_v12.py'))
FLIGHT = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/flight_stability.py'))
CONTRACT = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/path_free_stage2.py'))
Recovery = HELPERS['NearGroundRecovery']


class LandingRecoveryTests(unittest.TestCase):
    def test_prelanding_gate_excludes_ascent_far_zero_drop_contact_and_recovery(self):
        n = 8
        active = HELPERS['prelanding_mask'](
            torch.tensor([.1, .1, .3, .1, .1, .1, .1, .1]),
            torch.tensor([-.3, .3, -.3, -.3, -.3, -.3, -.3, -.3]),
            torch.tensor([1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.bool),
            torch.tensor([1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.bool),
            torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.bool),
            torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.bool),
            torch.tensor([0, 0, 0, 0, 0, 0, 5, 0]))
        self.assertEqual(active.tolist(), [True] + [False] * (n - 1))

    def test_cost_deadbands_sign_symmetry_yaw_invariance_and_bounded_tail(self):
        tilt = torch.deg2rad(torch.tensor([10., 30., -30., 90.]))
        q = torch.stack((torch.cos(tilt/2), torch.sin(tilt/2),
                         torch.zeros(4), torch.zeros(4)), -1)
        rates = torch.tensor([[1., -1., 99.], [2.5, -2.5, 0.],
                              [-2.5, 2.5, 0.], [99., -99., 0.]])
        a, r = HELPERS['prelanding_costs'](q, rates)
        torch.testing.assert_close(a, torch.tensor([0., 1., 1., 9.]), atol=1e-5, rtol=0)
        torch.testing.assert_close(r, torch.tensor([0., 2., 2., 18.]))
        # Left-multiply by a 180-degree world-Z rotation; tilt is unchanged.
        yaw_rotated = torch.stack((-q[:, 3], -q[:, 2], q[:, 1], q[:, 0]), -1)
        torch.testing.assert_close(HELPERS['prelanding_costs'](yaw_rotated, rates)[0], a)
        torch.testing.assert_close(HELPERS['prelanding_costs'](-q, rates)[0], a)

    def test_false_spring_liftoff_does_not_end_near_ground_recovery(self):
        state = Recovery(16, 'cpu', .01)
        yes, no = torch.ones(16, dtype=torch.bool), torch.zeros(16, dtype=torch.bool)
        near, down = torch.full((16,), .0004), torch.full((16,), -.12)
        state.update(yes, no, no, near, down)
        for _ in range(20):
            active, released = state.update(no, no, no, near, down)
            self.assertTrue(bool(active.all()))
            self.assertFalse(bool(released.any()))
        self.assertEqual(state.remaining.tolist(), [20] * 16)

    def test_release_requires_two_consecutive_clear_and_upward_samples(self):
        state = Recovery(1, 'cpu', .01)
        yes, no = torch.tensor([True]), torch.tensor([False])
        t = lambda x: torch.tensor([x])
        state.update(yes, no, no, t(0.), t(-.2))
        state.update(no, no, no, t(.04), t(-.2))  # tipped leg, still falling
        self.assertEqual(state.release_count.item(), 0)
        state.update(no, no, no, t(.04), t(.2))
        self.assertGreater(state.remaining.item(), 0)
        state.update(no, no, no, t(.03), t(.2))  # interrupt confirmation
        self.assertEqual(state.release_count.item(), 0)
        state.update(no, no, no, t(.04), t(.2))
        _, released = state.update(no, no, no, t(.05), t(.2))
        self.assertTrue(released.item())
        self.assertEqual(state.remaining.item(), 0)

    def test_no_refresh_timeout_initial_drop_stationary_and_partial_reset(self):
        state = Recovery(3, 'cpu', .01)
        yes, no = torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)
        z = torch.zeros(3)
        active, _ = state.update(yes, torch.tensor([0, 1, 0], dtype=torch.bool),
                                 torch.tensor([0, 0, 1], dtype=torch.bool), z, z)
        self.assertEqual(active.tolist(), [True, False, False])
        state.reset(slice(None))
        state.update(yes, no, no, z, z)
        state.reset(torch.tensor([1]))
        self.assertEqual(state.remaining.tolist(), [40, 0, 40])
        state.update(yes, no, no, z, z)
        self.assertEqual(state.remaining.tolist(), [39, 40, 39])
        for _ in range(40):
            state.update(no, no, no, z, z)
        self.assertEqual(state.remaining.tolist(), [0, 0, 0])
        self.assertEqual(state.release_count.tolist(), [0, 0, 0])

    def test_actual_environment_reward_method_no_double_count_and_reset(self):
        class Parent:
            def __init__(self, cfg, **kwargs):
                self.cfg, self.num_envs, self.device, self.step_dt = cfg, 3, 'cpu', .01
            def _get_rewards(self):
                self.calls += 1
                return torch.ones(3)
            def _reset_idx(self, ids):
                self.parent_reset = ids

        path = ROOT / 'Quadhopper_Planner_Random/path_free_stage2_v12_env.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        namespace = dict(HELPERS, torch=torch, PathFreeStage2V3Env=Parent,
                         touchdown_roll_pitch_cost=FLIGHT['touchdown_roll_pitch_cost'],
                         euler_xyz_from_quat=lambda q: (torch.zeros(3),) * 3)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        cfg = SimpleNamespace(moving_long_recovery_angle_deadband_rad=math.radians(10),
                              moving_long_recovery_rate_deadband_rad_s=.8,
                              moving_long_recovery_angle_penalty_scale=-20.,
                              moving_long_recovery_rate_penalty_scale=-8.,
                              stationary_command_radius_m=.002)
        env = namespace['PathFreeStage2V12Env'](cfg)
        data = SimpleNamespace(root_lin_vel_w=torch.tensor([[0., 0., -.2]] * 3),
                               joint_pos=torch.tensor([[.01], [0.], [0.]]),
                               root_quat_w=torch.tensor([[1., 0., 0., 0.]] * 3),
                               root_ang_vel_b=torch.tensor([[1.8, 0., 0.]] * 3))
        env._robot = SimpleNamespace(data=data, _ALL_INDICES=torch.arange(3))
        env._spring_joint_id = 0
        env.foot_geometry = {'sphere_radius_m': .012}
        env.foot_position_w = lambda: torch.tensor([[0., 0., .012]] * 3)
        env.commands = SimpleNamespace(anchor_w=torch.zeros(3, 2),
                                       lookahead=lambda: (torch.ones(3, 2), None))
        env._touchdown_event = torch.ones(3, dtype=torch.bool)
        env._completed_stationary_command = torch.tensor([False, False, True])
        env._initial_drop_pending = torch.zeros(3, dtype=torch.bool)
        env._cycle_active = torch.zeros(3, dtype=torch.bool)
        env._moving_long_recovery_remaining = torch.tensor([12, 0, 0])
        env.extras, env.calls = {'log': {}}, 0
        result = env._get_rewards()
        # Env 0: parent contact penalty already counted. Env 1: extension only.
        # Env 2: stationary touchdown has no new penalty.
        torch.testing.assert_close(result, torch.tensor([1., .92, 1.]))
        self.assertEqual(env.calls, 1)
        self.assertTrue(torch.isfinite(result).all())
        env._reset_idx(torch.tensor([1]))
        self.assertEqual(env._near_ground_recovery.remaining.tolist(), [40, 0, 0])
        env._reset_idx(None)
        self.assertEqual(env._near_ground_recovery.remaining.tolist(), [0, 0, 0])
        # Execute the actual prelanding reward arithmetic as well (30-degree tilt).
        env._touchdown_event.zero_()
        env._cycle_active.fill_(True)
        env._moving_long_recovery_remaining.zero_()
        data.joint_pos.zero_()
        data.root_quat_w[:] = torch.tensor([math.cos(math.pi/12), math.sin(math.pi/12), 0., 0.])
        env.foot_position_w = lambda: torch.tensor([[0., 0., .112]] * 3)
        env.commands.lookahead = lambda: (torch.tensor([[1., 0.], [1., 0.], [0., 0.]]), None)
        torch.testing.assert_close(env._get_rewards(), torch.tensor([.9382, .9382, 1.]))
        self.assertEqual(env.calls, 2)

    def test_launcher_dry_run_changes_only_flag_output_and_default_source_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fake.pt'
            checkpoint.touch()
            commands = []
            for suffix in ('v11_moving_handoff', 'v12_landing_recovery'):
                output = Path(directory) / suffix
                env = dict(os.environ, DRY_RUN='1', CHECKPOINT=str(checkpoint),
                           OUTPUT=str(output), ITERATIONS='400')
                env.pop('PYTHONHOME', None)
                env.pop('PYTHONPATH', None)
                env['PATH'] = '/usr/bin:/bin:' + env.get('PATH', '')
                run = subprocess.run(['bash', str(ROOT / (
                    f'run_train_two_circle_path_free_stage2_{suffix}_075.sh'))],
                    env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                command = shlex.split(run.stdout.strip())
                command[command.index('--output') + 1] = 'OUTPUT'
                commands.append(command)
                self.assertFalse(output.exists())
            commands[1].remove('--landing-recovery-v12')
            self.assertEqual(commands[0], commands[1])

    def test_eval_and_visualization_restore_v12_manifest_flags(self):
        import run_directional_eval
        import run_two_circle_eval
        manifest = dict(protocol=CONTRACT['PROTOCOL_V3'], moving_handoff_v11=True,
                        landing_recovery_v12=True, stability_profile='two_circle_v18',
                        observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12, .3], long_hop_radius_m=[.5, .75]),
                        reverse_turn_halfwidth_deg=30, max_turn_angle_deg=180,
                        zero_hop_probability=.12, reverse_turn_probability=.80,
                        policy_target_root_z_m=1., target_tolerance_m=.1)
        for module, extra in ((run_directional_eval, ['--scenario', 'all']),
                              (run_two_circle_eval, ['--visualize'])):
            with patch.object(sys, 'argv', ['test', '/tmp/fake-v12.pt', '--dry-run', *extra]), \
                 patch.object(Path, 'is_file', return_value=True), \
                 patch.object(Path, 'exists', return_value=False), \
                 patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
                 patch.object(module.subprocess, 'run') as run:
                module.main()
                self.assertTrue(run.called)
                for call in run.call_args_list:
                    for flag in ('--path-free-stage2-v3', '--moving-handoff-v11',
                                 '--landing-recovery-v12', '--dry-run'):
                        self.assertIn(flag, call.args[0])


if __name__ == '__main__':
    unittest.main()
