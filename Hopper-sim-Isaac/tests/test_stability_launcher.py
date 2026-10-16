"""Dry-run tests: never start Isaac Sim or a training update."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StabilityLauncherTests(unittest.TestCase):
    def test_v9_weights_only_curriculum_command(self):
        env = dict(os.environ, DRY_RUN='1')
        # Do not inherit interactive training overrides into this test.
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v9_dry_run')
            result = subprocess.run(['bash', str(ROOT / 'run_train_40v1_stability_v9.sh')],
                                    env=env, capture_output=True, text=True, check=True)
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'takeoff_v9')
        self.assertEqual(value('--mode'), 'train')
        self.assertEqual(value('--iterations'), '360')
        self.assertEqual(value('--hardware-randomization-probability'), '0.4')
        self.assertEqual(value('--height-curriculum-start'), '1.0')
        self.assertEqual(value('--target-tolerance'), '0.15')
        self.assertTrue(value('--checkpoint').endswith('model_759.pt'))
        self.assertNotIn('--resume-runner-state', cmd)

    def test_incompatible_motor_contract_rejected_before_launch(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'run_tracking_recovery.py'), '--dry-run',
            '--stability-profile', 'takeoff_v9', '--actuator-mode', 'baseline',
        ], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('require parity/direct_motor', result.stderr)

    def test_v10_launches_from_v8_with_v12_yaw_envelope(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v10_dry_run')
            result = subprocess.run(['bash', str(ROOT / 'run_train_40v1_stability_v10.sh')],
                                    env=env, capture_output=True, text=True, check=True)
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'launch_robust_v10')
        self.assertTrue(value('--checkpoint').endswith('height_attitude_yaw_40v1_v8_yaw_finetune/model_759.pt'))
        self.assertEqual(value('--initial-yaw-perturb-deg'), '25.0')
        self.assertEqual(value('--comprehensive-yaw-bias-torque-max-mnm'), '4.0')
        self.assertEqual(value('--hardware-randomization-probability'), '0.45')
        self.assertNotIn('--resume-runner-state', cmd)

    def test_v10_rejects_unshaped_actuator(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'run_tracking_recovery.py'), '--dry-run',
            '--stability-profile', 'launch_robust_v10', '--actuator-mode', 'baseline',
        ], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('require parity/direct_motor', result.stderr)

    def test_v11_refines_v10_with_event_yaw_profile(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v11_dry_run')
            result = subprocess.run(['bash', str(ROOT / 'run_train_40v1_stability_v11_yaw.sh')],
                                    env=env, capture_output=True, text=True, check=True)
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'yaw_event_v11')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v10_launch_robust/model_359.pt'))
        self.assertEqual(value('--initial-yaw-perturb-deg'), '30.0')
        self.assertEqual(value('--comprehensive-yaw-bias-torque-max-mnm'), '5.0')
        self.assertEqual(value('--hardware-randomization-probability'), '0.45')
        self.assertEqual(value('--learning-rate'), '5e-07')

    def test_v11_rejects_unshaped_actuator(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'run_tracking_recovery.py'), '--dry-run',
            '--stability-profile', 'yaw_event_v11', '--actuator-mode', 'baseline',
        ], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('require parity/direct_motor', result.stderr)

    def test_v12_starts_from_v8_weights_only(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v12_dry_run')
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_40v1_stability_v12_touchdown.sh')],
                env=env, capture_output=True, text=True, check=True,
            )
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'touchdown_robust_v12')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v8_yaw_finetune/model_759.pt'))
        self.assertEqual(value('--iterations'), '360')
        self.assertEqual(value('--hardware-randomization-probability'), '0.45')
        self.assertEqual(value('--learning-rate'), '5e-07')
        self.assertNotIn('--resume-runner-state', cmd)

    def test_v13_refines_v12_with_more_hard_samples(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v13_dry_run')
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_40v1_stability_v13_height.sh')],
                env=env, capture_output=True, text=True, check=True,
            )
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'height_robust_v13')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v12_touchdown_robust/model_359.pt'))
        self.assertEqual(value('--iterations'), '240')
        self.assertEqual(value('--hardware-randomization-probability-start'), '0.45')
        self.assertEqual(value('--hardware-randomization-probability'), '0.65')
        self.assertEqual(value('--learning-rate'), '3e-07')
        self.assertNotIn('--resume-runner-state', cmd)

    def test_v14_uses_v13_weights_and_continuous_episode(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v14_dry_run')
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_40v1_stability_v14_cross_hop.sh')],
                env=env, capture_output=True, text=True, check=True,
            )
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'cross_hop_v14')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v13_height_robust/model_239.pt'))
        self.assertEqual(value('--episode-length-s'), '61.0')
        self.assertEqual(value('--successful-hops-per-episode'), '10000')
        self.assertEqual(value('--iterations'), '360')

    def test_v15_refines_v14_with_the_recovery_profile(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v15_dry_run')
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_40v1_stability_v15_cross_hop_recovery.sh')],
                env=env, capture_output=True, text=True, check=True,
            )
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'cross_hop_v15')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v14_cross_hop_robust/model_359.pt'))
        self.assertEqual(value('--episode-length-s'), '61.0')
        self.assertEqual(value('--successful-hops-per-episode'), '10000')
        self.assertEqual(value('--iterations'), '360')
        self.assertEqual(value('--comprehensive-yaw-effectiveness-min'), '0.7')
        self.assertEqual(value('--comprehensive-yaw-effectiveness-max'), '1.3')
        self.assertEqual(value('--comprehensive-yaw-bias-torque-max-mnm'), '6.0')
        self.assertEqual(value('--learning-rate'), '2e-07')

    def test_v15_1_restarts_from_v14_with_v14_yaw_range(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v15_1_dry_run')
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_40v1_stability_v15_1_cross_hop_recovery.sh')],
                env=env, capture_output=True, text=True, check=True,
            )
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'cross_hop_v15_1')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v14_cross_hop_robust/model_359.pt'))
        self.assertEqual(value('--comprehensive-yaw-effectiveness-min'), '0.8')
        self.assertEqual(value('--comprehensive-yaw-effectiveness-max'), '1.2')
        self.assertEqual(value('--comprehensive-yaw-bias-torque-max-mnm'), '4.0')
        self.assertEqual(value('--learning-rate'), '3e-07')

    def test_v16_refines_v15_1_with_nominal_hard_mixture(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS', 'LEARNING_RATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'v16_dry_run')
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_40v1_stability_v16_persistent_hw.sh')],
                env=env, capture_output=True, text=True, check=True,
            )
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'cross_hop_v16')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v15_1_cross_hop_recovery/model_359.pt'))
        self.assertEqual(value('--hardware-randomization-probability-start'), '0.35')
        self.assertEqual(value('--hardware-randomization-probability'), '0.65')
        self.assertEqual(value('--learning-rate'), '2e-07')
        self.assertNotIn('--resume-runner-state', cmd)

    def test_two_circle_stage_b_starts_from_deployed_v15_1_weights_only(self):
        env = dict(os.environ, DRY_RUN='1')
        for key in ('CHECKPOINT', 'OUTPUT', 'ITERATIONS', 'NUM_ENVS', 'STEPS',
                    'LEARNING_RATE', 'RESUME_RUNNER_STATE'):
            env.pop(key, None)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temp_dir:
            env['OUTPUT'] = str(Path(temp_dir) / 'two_circle_stage_b_dry_run')
            result = subprocess.run(
                ['bash', str(ROOT / 'run_train_40v1_two_circle_v18_stage_b.sh')],
                env=env, capture_output=True, text=True, check=True,
            )
        cmd = shlex.split(result.stdout)
        value = lambda key: cmd[cmd.index(key) + 1]
        self.assertEqual(value('--stability-profile'), 'two_circle_v18')
        self.assertTrue(value('--checkpoint').endswith(
            'height_attitude_yaw_40v1_v15_1_cross_hop_recovery/model_359.pt'))
        self.assertEqual(value('--short-distance-min'), '0.06')
        self.assertEqual(value('--short-distance-max'), '0.12')
        self.assertEqual(value('--long-distance-min'), '0.06')
        self.assertEqual(value('--long-distance-max'), '0.12')
        self.assertEqual(value('--hard-sample-probability'), '0.0')
        self.assertEqual(value('--episode-length-s'), '61.0')
        self.assertEqual(value('--successful-hops-per-episode'), '10000')
        self.assertNotIn('--resume-runner-state', cmd)


if __name__ == '__main__':
    unittest.main()
