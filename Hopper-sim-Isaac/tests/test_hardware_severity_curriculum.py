"""Offline checks for the V3 hardware-strength training comparison."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch

from Quadhopper_Planner_Random.hardware_severity_curriculum import (
    scale_hard_reset, severity_at_step,
)

ROOT = Path(__file__).resolve().parents[1]
ISAAC = '/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh'


class _Robot:
    def __init__(self):
        self.writes = []

    def write_joint_stiffness_to_sim(self, values, *, joint_ids, env_ids):
        self.writes.append(('stiffness', values.clone(), env_ids.clone()))

    def write_joint_damping_to_sim(self, values, *, joint_ids, env_ids):
        self.writes.append(('damping', values.clone(), env_ids.clone()))


class HardwareSeverityTest(unittest.TestCase):
    def test_schedule_and_reset_strength_include_motor_and_physx_factors(self):
        self.assertEqual([severity_at_step(i * 256, 256, 30)
                          for i in (0, 29, 30, 59, 60, 149)],
                         [.5, .5, .75, .75, 1.0, 1.0])
        env = SimpleNamespace(
            cfg=SimpleNamespace(play_motor_time_constant=.0674,
                                play_thrust_scale=1., play_thrust_curve_shape=1.,
                                comprehensive_hardware_randomization=True),
            _hardware_hard_mask=torch.tensor([True, False]),
            dr_mass_multi=torch.tensor([[1.06], [1.]]),
            dr_inertia_multi=torch.tensor([[1.10] * 3, [1.] * 3]),
            dr_tm=torch.tensor([[.09] * 4, [.0674] * 4]),
            dr_thrust_scale=torch.tensor([[.88] * 4, [1.] * 4]),
            dr_thrust_curve_shape=torch.tensor([[.94] * 4, [1.] * 4]),
            _hardware_global_thrust_scale=torch.tensor([[.90], [1.]]),
            _hardware_rp_effectiveness=torch.tensor([[[.9, .03], [-.02, 1.1]],
                                                     [[1., 0.], [0., 1.]]]),
            _hardware_yaw_effectiveness=torch.tensor([.8, 1.]),
            _hardware_yaw_bias_torque_nm=torch.tensor([.004, 0.]),
            _hardware_yaw_reaction_scale=torch.tensor([[.9, 1.1, .9, 1.1], [1.] * 4]),
            _hardware_spring_stiffness_scale=torch.tensor([[.92], [1.]]),
            _hardware_spring_damping_scale=torch.tensor([[1.1], [1.]]),
            _hardware_body_com_offset_b=torch.tensor([[.002, -.002, .001], [0.] * 3]),
            _default_spring_stiffness=torch.tensor([[604.], [604.]]),
            _default_spring_damping=torch.tensor([[2.], [2.]]),
            _spring_joint_id=0,
            _robot=_Robot(),
        )
        com_writes = []
        env._write_body_com_offsets = lambda ids, values: com_writes.append(values.clone())
        scale_hard_reset(env, torch.tensor([0, 1]), .5)
        self.assertAlmostEqual(env.dr_mass_multi[0, 0].item(), 1.03, places=5)
        self.assertAlmostEqual(env.dr_tm[0, 0].item(), .0787, places=5)
        self.assertAlmostEqual(env.dr_thrust_scale[0, 0].item(), .94, places=5)
        self.assertAlmostEqual(env._hardware_global_thrust_scale[0, 0].item(), .95, places=5)
        self.assertAlmostEqual(env._hardware_rp_effectiveness[0, 0, 1].item(), .015, places=5)
        self.assertAlmostEqual(env._hardware_yaw_bias_torque_nm[0].item(), .002, places=5)
        self.assertAlmostEqual(env._hardware_spring_stiffness_scale[0, 0].item(), .96, places=5)
        self.assertAlmostEqual(com_writes[0][0, 0].item(), .001, places=5)
        self.assertAlmostEqual(env._robot.writes[0][1][0, 0].item(), 604 * .96, places=4)
        self.assertAlmostEqual(env._robot.writes[1][1][0, 0].item(), 2 * 1.05, places=5)
        self.assertTrue(torch.equal(env.dr_thrust_scale[1], torch.ones(4)))
        self.assertEqual(len(env._robot.writes), 2)
        scale_hard_reset(env, torch.tensor([0, 1]), 1.0)
        self.assertEqual(len(env._robot.writes), 2)

    def test_two_arm_dry_run_changes_only_strength_flag_and_output(self):
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            for variant in ('control', 'curriculum'):
                proc = subprocess.run(
                    ['bash', str(ROOT / 'run_train_two_circle_path_free_stage2_v23_severity_ab.sh')],
                    cwd=ROOT, env={**os.environ, 'DRY_RUN': '1', 'VARIANT': variant,
                                   'NUM_ENVS': '16', 'ITERATIONS': '2',
                                   'OUTPUT': str(Path(directory) / variant)},
                    capture_output=True, text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                command = shlex.split(next(line for line in proc.stdout.splitlines()
                                            if line.startswith(ISAAC)))
                self.assertIn('--v3-accounting-fix', command)
                self.assertEqual(command[command.index('--hardware-randomization-probability') + 1],
                                 '0.45')
                command[command.index('--output') + 1] = '<OUTPUT>'
                if variant == 'curriculum':
                    index = command.index('--hardware-severity-curriculum')
                    self.assertEqual(command[index + 2], '30')
                    del command[index:index + 3]
                commands.append(command)
        self.assertEqual(commands[0], commands[1])

    def test_distance_ladder_preserves_accounting_fix_from_accepted_checkpoint(self):
        anchor = ROOT / 'outputs/tracking_recovery/two_circle_path_free_stage2_v3_20260923_085412'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'accepted'
            source.mkdir()
            manifest = json.loads((anchor / 'manifest.json').read_text())
            manifest['v3_accounting_fix'] = True
            (source / 'manifest.json').write_text(json.dumps(manifest))
            checkpoint = source / 'model_99.pt'
            checkpoint.touch()
            proc = subprocess.run(
                ['bash', str(ROOT / 'run_train_two_circle_path_free_stage2_v19_distance_ladder.sh')],
                cwd=ROOT, env={**os.environ, 'DRY_RUN': '1', 'STAGE': '1',
                               'CHECKPOINT': str(checkpoint),
                               'OUTPUT': str(Path(directory) / 'stage1')},
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            command = shlex.split(next(line for line in proc.stdout.splitlines()
                                        if line.startswith(ISAAC)))
            self.assertIn('--v3-accounting-fix', command)

    def test_evaluation_launcher_accepts_both_contracts_without_starting_isaac(self):
        anchor = ROOT / 'outputs/tracking_recovery/two_circle_path_free_stage2_v3_20260923_085412'
        with tempfile.TemporaryDirectory() as directory:
            models = {}
            for mode in ('control', 'curriculum'):
                run_root = Path(directory) / mode
                run_root.mkdir()
                manifest = json.loads((anchor / 'manifest.json').read_text())
                manifest.update(v3_accounting_fix=True, source_sha256='same-anchor-sha',
                                hardware_randomization_probability=.45)
                if mode == 'curriculum':
                    manifest['hardware_severity_curriculum'] = {
                        'severity_stages': [.5, .75, 1.0]}
                (run_root / 'manifest.json').write_text(json.dumps(manifest))
                models[mode] = run_root / 'model_149.pt'
                models[mode].touch()
            proc = subprocess.run(
                ['bash', str(ROOT / 'run_eval_two_circle_path_free_stage2_v23_severity_ab.sh')],
                cwd=ROOT, env={**os.environ, 'DRY_RUN': '1',
                               'CONTROL_MODEL': str(models['control']),
                               'CURRICULUM_MODEL': str(models['curriculum']),
                               'AB_ROOT': str(Path(directory) / 'evaluation')},
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.count('[SEVERITY-EVAL] mode='), 2)
            self.assertIn('--v3-accounting-fix', proc.stdout)


if __name__ == '__main__':
    unittest.main()
