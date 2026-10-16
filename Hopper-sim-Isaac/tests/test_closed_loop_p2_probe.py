"""CPU-only checks for the matched-prefix P2 branch."""

from types import SimpleNamespace
import csv
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import torch

from Quadhopper_Planner_Random import closed_loop_p2_probe as branch_module

compare = runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / 'tools/compare_closed_loop_p2_branch.py'))['compare']


class FakeCommands:
    def __init__(self):
        self.targets_w = torch.tensor([[[.2, 0.], [.3, .1]]])
        self.anchor_w = torch.zeros(1, 2)
        self.route_index = torch.zeros(1, dtype=torch.long)
        self.cycle_index = torch.zeros(1, dtype=torch.long)

    def advance(self, ids):
        self.anchor_w[ids] = self.targets_w[ids, 0]
        self.targets_w[ids, 0] = self.targets_w[ids, 1]
        self.targets_w[ids, 1] = self.targets_w[ids, 0] + torch.tensor([.25, 0.])
        self.route_index[ids] += 1
        self.cycle_index[ids] += 1


def fake_probe(mode, expected=None):
    env = SimpleNamespace(num_envs=1, common_step_counter=12, commands=FakeCommands())
    policy = SimpleNamespace(memory_a=SimpleNamespace(
        hidden_states=(torch.zeros(1, 1, 3), torch.ones(1, 1, 3))))
    probe = branch_module.ClosedLoopP2Probe(
        env, policy, 0, 12, mode, expected_sha256=expected,
        episode_getter=lambda: 0)
    probe.install()
    probe.begin_step(torch.zeros(1, 336), torch.zeros(1, 4))
    env.commands.advance(torch.tensor([0]))
    post_obs = torch.zeros(1, 336)
    post_obs[0, 39:41] = env.commands.targets_w[0, 1] - env.commands.targets_w[0, 0]
    probe.after_step(post_obs, torch.tensor([False]))
    return probe


class ClosedLoopP2ProbeTests(unittest.TestCase):
    def test_rotation_preserves_radius_and_common_state(self):
        with patch.object(branch_module, 'plant_values', return_value={
                'root_x': torch.tensor([1.])}), patch.object(
                branch_module.torch.cuda, 'is_available', return_value=False):
            rng = torch.get_rng_state().clone()
            original = fake_probe('original')
            changed = fake_probe('left90', original.branch['state_sha256'])
            self.assertEqual(original.branch['state'], changed.branch['state'])
            self.assertEqual(original.branch['state_sha256'], changed.branch['state_sha256'])
            self.assertEqual(original.branch['original_p1_xy_w'], changed.branch['original_p1_xy_w'])
            self.assertAlmostEqual(float(torch.linalg.vector_norm(
                changed.env.commands.targets_w[0, 1]
                - changed.env.commands.targets_w[0, 0])), .25, places=6)
            self.assertAlmostEqual(float(changed.env.commands.targets_w[0, 1, 1]), .35, places=6)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_expected_hash_rejects_unmatched_prefix(self):
        with patch.object(branch_module, 'plant_values', return_value={
                'root_x': torch.tensor([1.])}), patch.object(
                branch_module.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'state mismatch'):
                fake_probe('reverse', '0' * 64)

    def test_finish_checks_first_p1_and_changed_p2(self):
        with patch.object(branch_module, 'plant_values', return_value={
                'root_x': torch.tensor([1.])}), patch.object(
                branch_module.torch.cuda, 'is_available', return_value=False):
            probe = fake_probe('left90')
        branch = probe.branch
        p1, p2 = branch['original_p1_xy_w'], branch['executed_p2_xy_w']
        rows = [dict(env_id=0, episode_id=0, liftoff_step=2, touchdown_step=12),
                dict(env_id=0, episode_id=0, liftoff_step=18, touchdown_step=95,
                     target_x_m=p1[0], target_y_m=p1[1],
                     next_target_x_m=p2[0], next_target_y_m=p2[1])]
        result = probe.finish(rows)
        self.assertTrue(result['first_hop_completed'])
        self.assertFalse(result['second_hop_completed'])
        rows[1]['next_target_x_m'] += .1
        with self.assertRaisesRegex(RuntimeError, 'next-target input'):
            probe.finish(rows)

    def test_comparator_requires_matching_prefix(self):
        with patch.object(branch_module, 'plant_values', return_value={
                'root_x': torch.tensor([1.])}), patch.object(
                branch_module.torch.cuda, 'is_available', return_value=False):
            original = fake_probe('original')
            changed = fake_probe('left90', original.branch['state_sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for name, probe in (('original', original), ('left90', changed)):
                root = Path(tmp) / name / 'uniform_turn180' / 'robust'
                audit = root / 'takeoff_velocity_audit'
                audit.mkdir(parents=True)
                branch = probe.branch
                p1, p2 = branch['original_p1_xy_w'], branch['executed_p2_xy_w']
                rows = [dict(env_id=0, episode_id=0, liftoff_step=2,
                             touchdown_step=12, touchdown_foot_error_m=.12),
                        dict(env_id=0, episode_id=0, liftoff_step=18,
                             touchdown_step=95, touchdown_foot_error_m=.05,
                             target_x_m=p1[0], target_y_m=p1[1],
                             next_target_x_m=p2[0], next_target_y_m=p2[1])]
                report = probe.finish(rows)
                (root / 'closed_loop_p2_probe.json').write_text(json.dumps(report))
                (root / 'result.json').write_text(json.dumps(dict(
                    checkpoint='same.pt', seed=483, num_envs=16)))
                with (audit / 'hops.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[1])
                    writer.writeheader()
                    writer.writerows(rows)
                roots.append(root.parents[1])
            result = compare(*roots)
            self.assertEqual(result['matched_completed_prefix_hops'], 1)
            self.assertEqual(result['first_hop_foot_error_delta_m'], 0)
            self.assertEqual(result['state_sha256'], original.branch['state_sha256'])
            self.assertEqual(result['changed_observation_indices'], [39, 40])
            branch_path = roots[1] / 'uniform_turn180' / 'robust' / 'closed_loop_p2_probe.json'
            altered_report = json.loads(branch_path.read_text())
            altered_report['branch']['observation_after_branch'][0] = 1.0
            branch_path.write_text(json.dumps(altered_report))
            with self.assertRaisesRegex(ValueError, 'outside P2 dimensions'):
                compare(*roots)


if __name__ == '__main__':
    unittest.main()
