"""Offline checks for the frozen-policy one-variable evaluation contract."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_directional_eval
from tools.compare_v24_contact_floor_ab import METRICS, compare_pair


class ContactFloorABTests(unittest.TestCase):
    def test_launcher_forwards_only_contact_floor(self):
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
            for floor in (0, 600):
                argv = ['run_directional_eval.py', str(model), '--root',
                        str(Path(directory) / f'floor{floor}'),
                        '--scenario', 'uniform_turn180', '--condition', 'robust',
                        '--robust-seed', '483', '--steps', '30000',
                        '--diagnostic-contact-floor-pwm', str(floor), '--dry-run']
                with patch.object(sys, 'argv', argv), \
                     patch.object(run_directional_eval.subprocess, 'run') as run:
                    run_directional_eval.main()
                    commands.append(run.call_args.args[0])
            for command in commands:
                self.assertIn('--simulation-only-phase-collective-baseline', command)
                self.assertEqual(command[command.index('--phase-ascent-collective-floor-pwm') + 1], '0')
            def normalized(command):
                items = command.copy()
                for flag in ('--phase-contact-collective-floor-pwm', '--root'):
                    items[items.index(flag) + 1] = '<variable>'
                return items
            self.assertEqual(normalized(commands[0]), normalized(commands[1]))

    def test_comparator_rejects_any_other_contract_difference(self):
        contract = dict(seed=483, simulation_only_phase_collective_baseline=True,
                        phase_contact_collective_floor_pwm=0.0,
                        phase_ascent_collective_floor_pwm=0.0,
                        hardware_randomization_probability=1.0)
        control = {'evaluation_contract': contract, 'steps': 30000, 'num_envs': 16,
                   **{key: 0.5 for key in METRICS}}
        assisted_contract = dict(contract, phase_contact_collective_floor_pwm=600.0)
        assisted = {'evaluation_contract': assisted_contract, 'steps': 30000, 'num_envs': 16,
                    **{key: 0.6 for key in METRICS}}
        report = compare_pair(control, assisted, seed=483)
        self.assertAlmostEqual(report['metrics']['foot_xy_error_m']['delta'], 0.1)
        assisted_contract['hardware_randomization_probability'] = 0.5
        with self.assertRaisesRegex(ValueError, 'other evaluation contract differences'):
            compare_pair(control, assisted, seed=483)

    def test_ascent_arm_keeps_contact_floor_zero(self):
        contract = dict(seed=583, simulation_only_phase_collective_baseline=True,
                        phase_contact_collective_floor_pwm=0.0,
                        phase_ascent_collective_floor_pwm=0.0)
        control = {'evaluation_contract': contract, 'steps': 30000, 'num_envs': 16,
                   **{key: 0.5 for key in METRICS}}
        assisted = {'evaluation_contract': dict(contract, phase_ascent_collective_floor_pwm=550.0),
                    'steps': 30000, 'num_envs': 16,
                    **{key: 0.6 for key in METRICS}}
        report = compare_pair(control, assisted, seed=583, phase='ascent', floor_pwm=550,
                              assisted_arm='ascent550')
        self.assertIn('ascent550', report['metrics']['death_event_count'])
        assisted['evaluation_contract']['phase_contact_collective_floor_pwm'] = 600.0
        with self.assertRaisesRegex(ValueError, 'must stay zero'):
            compare_pair(control, assisted, seed=583, phase='ascent', floor_pwm=550,
                         assisted_arm='ascent550')


if __name__ == '__main__':
    unittest.main()
