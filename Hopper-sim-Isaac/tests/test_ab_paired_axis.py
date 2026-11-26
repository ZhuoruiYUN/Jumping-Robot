import json
import unittest
from pathlib import Path

import torch

from benchmark37.stationary_retention import RetentionRoute


class PairedAxisTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'training_config_v51' / 'ab_paired_axis' / 'course.json'
        course = json.loads(source.read_text())
        self.route = RetentionRoute(16, 'cpu', course['omni'], 42, training_stationary_fraction=0.)
        self.ids = torch.arange(16)
        self.route.reset(self.ids)
        self.paired = self.route.ab_pairs.paired_axes

    def test_async_resets_reuse_only_this_pair_and_reverse_direction(self):
        scenario = self.ids[:, None].float()
        self.paired.match(self.ids, 'example', scenario)
        previous_directions = self.paired.direction.clone()
        reset_ids = torch.tensor([8, 3, 0, 1])
        self.route.reset(reset_ids)
        got = self.paired.match(reset_ids, 'example', scenario[reset_ids]+20)
        torch.testing.assert_close(got[:, 0], torch.tensor([8., 23., 0., 1.]))
        torch.testing.assert_close(self.paired.direction[[8,0,1]], previous_directions[[8,0,1]] ^ 1)
        self.assertEqual(self.paired.direction[4], previous_directions[4])
        self.route.reset(torch.tensor([0]))
        got = self.paired.match(torch.tensor([0]), 'example', torch.tensor([[44.]]))
        torch.testing.assert_close(got, torch.tensor([[44.]]))

    def test_only_valid_a_advance_issues_axis_and_keeps_terminal_preview(self):
        release = torch.tensor([.03, -.01]).expand(16, 2)
        self.paired.anchor(self.route, self.ids, release)
        ab = self.route.ab_pairs
        paired_ids = self.ids[self.paired.lanes]
        torch.testing.assert_close(self.route.points[paired_ids, 1], self.route.points[paired_ids, 2])
        valid = self.paired.lanes.clone()
        strict = valid.clone()
        strict[1] = False
        ab.recovery.start(valid & ~strict)
        advanced = self.route.advance(valid, strict, strict)
        terminal = torch.zeros_like(valid)
        terminal[0] = True
        self.paired.publish_departure(self.route, advanced, terminal)
        self.assertEqual(int(self.paired.issued.sum()), len(advanced) - 1)
        self.assertEqual(self.route.index[1], 1)
        torch.testing.assert_close(self.route.points[1, 1], self.route.points[1, 2])
        for lane in advanced:
            direction = int(self.paired.direction[lane])
            step = torch.tensor([[.5, 0.], [-.5, 0.], [0., .5], [0., -.5]])[direction]
            torch.testing.assert_close(self.route.points[lane, 2] - release[lane], step)
            torch.testing.assert_close(self.route.points[lane, 2:5], self.route.points[lane, 2].expand(3,2))


if __name__ == '__main__':
    unittest.main()
