"""CPU regression checks for reward boundaries, curricula and nominal resets."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock
import torch
from Quadhopper_Planner_Random.path_free_stage2 import (
    PROTOCOL, PROTOCOL_V2, PROTOCOL_V3, SETTINGS, SETTINGS_V2, SETTINGS_V3,
    curriculum_values, apply_settings,
)
from Quadhopper_Planner_Random.tracking_progress import LandingXYProgress

ROOT = Path(__file__).resolve().parents[1]


def extract_class(filename, classname, parent, *, methods=None, **namespace):
    """Run actual environment methods on CPU with only the Isaac parent stubbed."""
    tree = ast.parse((ROOT / 'Quadhopper_Planner_Random' / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    parent_name = cls.bases[0].id
    if methods is not None:
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    scope = {parent_name: parent, 'torch': torch, **namespace}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), filename, 'exec'), scope)
    return scope[classname]


class V2Tests(unittest.TestCase):
    def cfg(self):
        return NS(**SETTINGS_V2, stage2_curriculum_training=True,
                  stage2_training_iterations=600, curriculum_steps_per_iteration=256,
                  hardware_randomization_probability=.45,
                  hardware_randomization_curriculum_iterations=0.,
                  sysid_20261011_plant=False)

    def test_schedule_boundaries_and_eval_bypass(self):
        cfg = self.cfg()
        for iteration, p, energy in [(0,0,0),(120,0,0),(260,.225,0),
                                     (400,.45,0),(450,.45,0),
                                     (525,.45,-.025),(600,.45,-.05),(900,.45,-.05)]:
            actual = curriculum_values(cfg, iteration * 256)
            self.assertAlmostEqual(actual[0], p)
            self.assertAlmostEqual(actual[1], energy)
        cfg.stage2_curriculum_training = False
        for probability in (0., 1.):
            cfg.hardware_randomization_probability = probability
            self.assertEqual(curriculum_values(cfg, 0), (probability, -.05))
        old = NS()
        apply_settings(old)
        self.assertEqual(vars(old), SETTINGS)
        self.assertEqual(SETTINGS_V2['spatial_xy_progress_reward_scale'], 0.)
        strict = NS()
        apply_settings(strict, v3=True)
        self.assertEqual(vars(strict), SETTINGS_V3)
        self.assertEqual(strict.stage2_energy_penalty_scale, 0.)
        self.assertTrue(strict.require_apex_at_or_above_target_for_hit)

    def test_explicit_hardware_ramp_controls_stage2_reset_probability(self):
        class Parent:
            def _current_hardware_randomization_probability(self):
                duration = self.cfg.hardware_randomization_curriculum_iterations
                fraction = min(1.0, self.common_step_counter /
                               (self.cfg.curriculum_steps_per_iteration * duration))
                return (self.cfg.hardware_randomization_probability_start
                        + fraction * (self.cfg.hardware_randomization_probability
                                      - self.cfg.hardware_randomization_probability_start))

        cls = extract_class('path_free_stage2_v2_env.py', 'PathFreeStage2V2Env', Parent,
                            methods={'_curriculum', '_current_hardware_randomization_probability'},
                            curriculum_values=curriculum_values)
        env = object.__new__(cls)
        env.cfg = NS(**SETTINGS_V3, curriculum_steps_per_iteration=256,
                     hardware_randomization_probability=.45,
                     hardware_randomization_probability_start=0.,
                     hardware_randomization_curriculum_iterations=100.)
        env._stage2_final_energy_scale = 0.
        for iteration, expected in ((0, 0.), (50, .225), (100, .45), (150, .45)):
            env.common_step_counter = iteration * 256
            self.assertAlmostEqual(env._current_hardware_randomization_probability(),
                                   expected)
            self.assertAlmostEqual(env._curriculum()[0], expected)

    def test_actual_v2_reward_progress_cancels_and_boundaries_do_not_pay(self):
        class Parent:
            def _get_rewards(self):
                return torch.zeros(2)
            def _replan(self, ids):
                pass
            def _reset_idx(self, ids):
                pass
        cls = extract_class('path_free_stage2_v2_env.py', 'PathFreeStage2V2Env', Parent,
                            curriculum_values=curriculum_values,
                            LandingXYProgress=LandingXYProgress)
        env = object.__new__(cls)
        env.cfg = self.cfg()
        env._stage2_final_energy_scale = -.05
        env._foot_progress = LandingXYProgress(2, 'cpu')
        env._foot_progress_totals = torch.zeros(3, dtype=torch.float64)
        env._cycle_active = torch.ones(2, dtype=torch.bool)
        env._touchdown_event = torch.zeros(2, dtype=torch.bool)
        env._initial_drop_pending = torch.zeros(2, dtype=torch.bool)
        env._hardware_hard_mask = torch.zeros(2, dtype=torch.bool)
        env.num_envs = 2
        env.extras = {'log': {}}
        foot = torch.tensor([[.30, 0., .20], [.30, 0., .20]])
        target = torch.zeros(2, 2)
        env.foot_position_w = lambda: foot
        env.commands = NS(lookahead=lambda ids=None: (target if ids is None else target[ids], None))
        env._robot = NS(_ALL_INDICES=torch.arange(2))
        torch.testing.assert_close(env._get_rewards(), torch.zeros(2))
        foot[:, 0] = .20
        approach = env._get_rewards()
        torch.testing.assert_close(approach, torch.full((2,), 2.))
        foot[:, 0] = .30
        torch.testing.assert_close(env._get_rewards(), -approach)
        target[0, 0] = 3.0  # Queue changes only for environment 0.
        env._replan(torch.tensor([0]))
        foot[1, 0] = .25
        torch.testing.assert_close(env._get_rewards(), torch.tensor([0., 1.]))
        env._reset_idx(torch.tensor([0]))
        foot[0, 0] = 2.0
        torch.testing.assert_close(env._get_rewards(), torch.zeros(2))
        env._initial_drop_pending[:] = True
        foot[:, 0] = 0.
        torch.testing.assert_close(env._get_rewards(), torch.zeros(2))
        env._initial_drop_pending[:] = False
        torch.testing.assert_close(env._get_rewards(), torch.zeros(2))
        env._touchdown_event[:] = True
        foot[:, 0] = 1.
        torch.testing.assert_close(env._get_rewards(), torch.zeros(2))
        env.common_step_counter = 600 * 256
        env._get_rewards()
        self.assertAlmostEqual(env.cfg.stage2_energy_penalty_scale, -.05)
        env._get_rewards()  # Schedule must not repeatedly scale its live value.
        self.assertAlmostEqual(env.cfg.stage2_energy_penalty_scale, -.05)

    def test_zero_probability_restores_parent_randomized_dynamics(self):
        class Parent:
            def _reset_idx(self, ids):
                for name in ('dr_mass_multi','dr_inertia_multi','dr_tm',
                             'dr_thrust_scale','dr_thrust_curve_shape'):
                    getattr(self, name)[ids] = 9.0
        cls = extract_class('spatial_tracking_env.py', 'SpatialTrackingEnv', Parent,
                            methods={'_reset_idx'})
        env = object.__new__(cls)
        env.num_envs, env.device = 2, 'cpu'
        env.cfg = NS(randomize_dynamics=True, comprehensive_hardware_randomization=True,
                     play_motor_time_constant=.0674, sysid_20261011_plant=False)
        env._robot = NS(_ALL_INDICES=torch.arange(2),
                        write_joint_stiffness_to_sim=Mock(), write_joint_damping_to_sim=Mock())
        env._write_body_com_offsets = Mock()
        env._spring_joint_id = 0
        env._default_spring_stiffness = torch.ones(2,1)
        env._default_spring_damping = torch.ones(2,1)
        env._hardware_rp_effectiveness = torch.zeros(2,2,2)
        env._hardware_body_com_offset_b = torch.zeros(2,3)
        env._hardware_hard_mask = torch.zeros(2,dtype=torch.bool)
        for name in ('_hardware_global_thrust_scale','_hardware_yaw_effectiveness',
                     '_hardware_yaw_bias_torque_nm','_hardware_yaw_reaction_scale',
                     '_hardware_spring_stiffness_scale','_hardware_spring_damping_scale',
                     'dr_mass_multi','dr_inertia_multi','dr_tm','dr_thrust_scale',
                     'dr_thrust_curve_shape'):
            setattr(env, name, torch.zeros(2))
        env._current_hardware_randomization_probability = lambda: 0.
        env._reset_idx(torch.tensor([0]))
        for name in ('dr_mass_multi','dr_inertia_multi','dr_thrust_scale','dr_thrust_curve_shape'):
            torch.testing.assert_close(getattr(env, name), torch.tensor([1.,0.]))
        torch.testing.assert_close(env.dr_tm, torch.tensor([.0674,0.]))
        self.assertFalse(env._hardware_hard_mask.any())
        env._robot.write_joint_stiffness_to_sim.assert_called_once()
        env._write_body_com_offsets.assert_called_once()

    def test_both_eval_entrypoints_select_v1_and_v2_without_simulation_imports(self):
        import run_directional_eval
        import run_two_circle_eval
        manifest = dict(stability_profile='two_circle_v18', observation_layout='history_foot_v19',
                        two_circle_route=dict(short_hop_radius_m=[.12,.3],long_hop_radius_m=[.3,.5]),
                        reverse_turn_halfwidth_deg=30, max_turn_angle_deg=180,
                        zero_hop_probability=.12, reverse_turn_probability=.65,
                        policy_target_root_z_m=1., target_tolerance_m=.1)
        for protocol, flag in ((PROTOCOL, '--path-free-stage2'),
                               (PROTOCOL_V2, '--path-free-stage2-v2'),
                               (PROTOCOL_V3, '--path-free-stage2-v3')):
            manifest['protocol'] = protocol
            for module, extra, count in ((run_directional_eval, ['--scenario','all'], 6),
                                         (run_two_circle_eval, ['--visualize'], 1)):
                with patch('sys.argv', ['test','/tmp/fake.pt','--dry-run'] + extra), \
                     patch.object(Path,'is_file',return_value=True), \
                     patch.object(Path,'exists',return_value=False), \
                     patch.object(Path,'read_text',return_value=json.dumps(manifest)), \
                     patch.object(module.subprocess,'run') as run:
                    module.main()
                    self.assertEqual(run.call_count, count)
                    for call in run.call_args_list:
                        self.assertIn(flag, call.args[0])
                        self.assertNotIn('--raibert-hermite-guidance', call.args[0])


if __name__ == '__main__':
    unittest.main()
