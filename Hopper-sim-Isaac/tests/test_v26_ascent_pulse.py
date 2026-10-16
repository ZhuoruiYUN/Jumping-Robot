"""Offline checks for the short additive ascent probe."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_directional_eval
from tools.compare_v24_contact_floor_ab import METRICS
from tools.compare_v26_ascent_pulse_ab import compare_pair


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ascent_pulse_v26', ROOT / 'Quadhopper_Planner_Random/ascent_pulse_v26.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AscentPulseTests(unittest.TestCase):
    def test_pulse_is_small_and_limited_to_first_20_control_steps(self):
        target = torch.full((5, 4), 0.40)
        result = module.apply_ascent_pulse(
            target, spring_q=torch.tensor([0, 0.003, 0, 0, 0]),
            cycle_active=torch.tensor([True] * 5),
            initial_drop=torch.tensor([False, False, False, True, False]),
            root_vz=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.05]),
            cycle_time_s=torch.tensor([0.20, 0.05, 0.21, 0.05, 0.05]),
            bias_pwm=50,
        )
        torch.testing.assert_close(result[0], torch.full((4,), 0.45))
        torch.testing.assert_close(result[1:], target[1:])
        self.assertIs(module.apply_ascent_pulse(
            target, spring_q=torch.zeros(5), cycle_active=torch.ones(5, dtype=torch.bool),
            initial_drop=torch.zeros(5, dtype=torch.bool), root_vz=torch.ones(5),
            cycle_time_s=torch.zeros(5), bias_pwm=0), target)

    def test_contract_rejects_other_changes(self):
        base = dict(seed=483, diagnostic_ascent_pulse={
            'pwm': 0, 'duration_s': 0.20,
            'gate': 'airborne_positive_vz_and_cycle_time_since_liftoff'},
            hardware_randomization_probability=1.0)
        a = dict(evaluation_contract=base, death_env_count=2,
                 **{key: 0.5 for key in METRICS})
        b = dict(evaluation_contract=dict(base, diagnostic_ascent_pulse={
            'pwm': 50, 'duration_s': 0.20,
            'gate': 'airborne_positive_vz_and_cycle_time_since_liftoff'}),
                 death_env_count=1, **{key: 0.6 for key in METRICS})
        self.assertIn('pulse50', compare_pair(a, b, 483)['metrics']['death_event_count'])
        b['evaluation_contract']['hardware_randomization_probability'] = 0.5
        with self.assertRaisesRegex(ValueError, 'other evaluation contract differences'):
            compare_pair(a, b, 483)

    def test_launcher_forwards_only_pulse_pwm(self):
        manifest = {
            'stability_profile': 'two_circle_v18',
            'observation_layout': 'history_foot_v19',
            'two_circle_route': {'short_hop_radius_m': [0.12, 0.30],
                                 'long_hop_radius_m': [0.30, 0.50]},
            'reverse_turn_halfwidth_deg': 30.0,
            'policy_target_root_z_m': 1.0,
            'target_tolerance_m': 0.1,
            'v3_accounting_fix': True,
        }
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model_120.pt'
            model.touch()
            (model.parent / 'manifest.json').write_text(json.dumps(manifest))
            commands = []
            for pwm in (0, 50):
                argv = ['run_directional_eval.py', str(model), '--root',
                        str(Path(directory) / f'pulse{pwm}'),
                        '--scenario', 'uniform_turn180', '--condition', 'robust',
                        '--diagnostic-ascent-pulse-pwm', str(pwm), '--dry-run']
                with patch.object(sys, 'argv', argv), \
                     patch.object(run_directional_eval.subprocess, 'run') as run:
                    run_directional_eval.main()
                    commands.append(run.call_args.args[0])
            for command in commands:
                self.assertNotIn('--simulation-only-phase-collective-baseline', command)
            def normalized(command):
                items = command.copy()
                for flag in ('--diagnostic-ascent-pulse-pwm', '--root'):
                    items[items.index(flag) + 1] = '<variable>'
                return items
            self.assertEqual(normalized(commands[0]), normalized(commands[1]))


if __name__ == '__main__':
    unittest.main()
