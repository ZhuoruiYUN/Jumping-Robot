import unittest

import torch

from benchmark37.precision_release import OpeningReversalRelease


class OpeningReleaseTest(unittest.TestCase):
    def test_only_first_free_ascent_after_drop_contact_scores(self):
        tracker = OpeningReversalRelease(4, 'cpu')
        edge = torch.tensor([[.5, 0.]] * 4)
        preview = torch.tensor([[-.5, 0.], [.5, 0.], [-.5, 0.], [-.5, 0.]])
        foot = torch.zeros_like(edge)
        velocity = torch.tensor([[0., 0.], [0., 0.], [.5, 0.], [0., 0.]])
        flight = torch.ones(4)
        all_lanes = torch.ones(4, dtype=torch.bool)
        config = dict(allowance_m_s=.17, weight=1.)

        def step(contact, ascent):
            return tracker.step(contact, ascent, foot, edge, edge, preview, velocity, flight, 1.2, config)

        self.assertFalse(step(~all_lanes, all_lanes).any())
        self.assertFalse(step(all_lanes, ~all_lanes).any())
        tracker.reset(torch.tensor([3]))
        foot[:, 0] = .45  # velocity reference must come from the contact sample
        self.assertEqual((step(~all_lanes, all_lanes) < 0).tolist(), [True, False, False, False])
        self.assertFalse(step(~all_lanes, all_lanes).any())
        self.assertEqual(tracker.counts.tolist(), [2, 2, 1])
