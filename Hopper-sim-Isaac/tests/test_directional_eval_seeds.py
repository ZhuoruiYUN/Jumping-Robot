"""Seed forwarding checks; these tests never launch Isaac or an evaluation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_directional_eval


class DirectionalEvalSeedTests(unittest.TestCase):
    def test_v38_two_hop_eval_recreates_pair_task_from_continuous_checkpoint(self):
        manifest = {
            'protocol': 'physical_1m_history_foot_path_free_stage2_v3_strict_success',
            'stability_profile': 'two_circle_v18',
            'observation_layout': 'history_foot_v19',
            'two_circle_route': {
                'short_hop_radius_m': [0.12, 0.30],
                'long_hop_radius_m': [0.30, 0.50],
            },
            'reverse_turn_halfwidth_deg': 30.0,
            'policy_target_root_z_m': 1.0,
            'target_tolerance_m': 0.1,
            'v3_accounting_fix': True,
            'foot_pair_v33': {'event': 'second_hop_touchdown'},
            'sysid_20261011_plant': True,
            'comprehensive_hardware_randomization': True,
            'cli': {
                'hardware_randomization_probability': .45,
                'observation_noise_std': .002,
                'initial_yaw_perturb_deg': 30,
                'initial_angular_velocity_perturb_rad_s': .15,
                'comprehensive_global_thrust_scale_min': .95,
                'comprehensive_global_thrust_scale_max': 1.05,
                'comprehensive_com_offset_x_max_mm': 2,
                'comprehensive_com_offset_y_max_mm': 2,
                'comprehensive_com_offset_z_max_mm': 1,
                'comprehensive_yaw_effectiveness_min': 1,
                'comprehensive_yaw_effectiveness_max': 1,
                'comprehensive_yaw_bias_torque_max_mnm': 0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model_199.pt'
            checkpoint.touch()
            (checkpoint.parent / 'manifest.json').write_text(json.dumps(manifest))
            argv = [
                'run_directional_eval.py', str(checkpoint), '--root',
                str(Path(directory) / 'eval'), '--episode-mode', 'two_hop',
                '--condition', 'robust', '--robust-profile', 'checkpoint',
                '--robust-seed', '483', '--dry-run',
            ]
            with patch.object(sys, 'argv', argv), \
                 patch.object(run_directional_eval.subprocess, 'run') as run:
                run_directional_eval.main()
                command = run.call_args.args[0]
                self.assertIn('--two-hop-episode-v38', command)
                self.assertIn('--allow-compatible-checkpoint', command)
                self.assertEqual(command[command.index('--episode-length-s') + 1], '12.0')
                self.assertEqual(command[command.index('--seed') + 1], '483')
                self.assertEqual(command[command.index('--hardware-randomization-probability') + 1], '0.45')
                self.assertIn('two_hop_mixture', command[command.index('--root') + 1])

    def test_v38_weights_use_explicit_continuous_eval_compatibility(self):
        manifest = {
            'protocol': 'physical_1m_history_foot_path_free_stage2_v3_strict_success',
            'stability_profile': 'two_circle_v18',
            'observation_layout': 'history_foot_v19',
            'two_circle_route': {
                'short_hop_radius_m': [0.12, 0.30],
                'long_hop_radius_m': [0.30, 0.50],
            },
            'reverse_turn_halfwidth_deg': 30.0,
            'policy_target_root_z_m': 1.0,
            'target_tolerance_m': 0.1,
            'v3_accounting_fix': True,
            'foot_pair_v33': {'event': 'second_hop_touchdown'},
            'sysid_20261011_plant': True,
            'two_hop_episode_v38': {'hops': 'two_attempted_touchdowns_after_initial_drop'},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model_99.pt'
            checkpoint.touch()
            (checkpoint.parent / 'manifest.json').write_text(json.dumps(manifest))
            argv = [
                'run_directional_eval.py', str(checkpoint), '--root',
                str(Path(directory) / 'eval'), '--scenario', 'uniform_turn180',
                '--condition', 'nominal', '--dry-run',
            ]
            with patch.object(sys, 'argv', argv), \
                 patch.object(run_directional_eval.subprocess, 'run') as run:
                run_directional_eval.main()
                command = run.call_args.args[0]
                self.assertIn('--allow-compatible-checkpoint', command)
                self.assertIn('--sysid-20261011-plant', command)
                self.assertNotIn('--two-hop-episode-v38', command)

    def test_robust_seed_override_and_default_are_forwarded(self):
        manifest = {
            'stability_profile': 'two_circle_v18',
            'observation_layout': 'history_foot_v19',
            'two_circle_route': {
                'short_hop_radius_m': [0.12, 0.30],
                'long_hop_radius_m': [0.50, 0.75],
            },
            'reverse_turn_halfwidth_deg': 30.0,
            'policy_target_root_z_m': 1.0,
            'target_tolerance_m': 0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model_899.pt'
            checkpoint.touch()
            (checkpoint.parent / 'manifest.json').write_text(json.dumps(manifest))
            for extra, expected_seed in (([], '183'), (['--robust-seed', '287'], '287')):
                output = Path(directory) / f'eval_{expected_seed}'
                argv = [
                    'run_directional_eval.py', str(checkpoint), '--root', str(output),
                    '--scenario', 'reverse_only', '--condition', 'robust', '--dry-run',
                    *extra,
                ]
                with patch.object(sys, 'argv', argv), \
                     patch.object(run_directional_eval.subprocess, 'run') as run:
                    run_directional_eval.main()
                    run.assert_called_once()
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index('--seed') + 1], expected_seed)
                    self.assertIn('--dry-run', command)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
