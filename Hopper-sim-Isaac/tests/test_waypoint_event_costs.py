import unittest

import torch

from benchmark37.waypoint_event_costs import StraightFirstApex, WaypointOutcome, retry_rebound_growth


class WaypointEventTest(unittest.TestCase):
    def test_only_added_high_error_surviving_to_release_is_charged(self):
        release = torch.tensor([.255, .55, .55, .55, .47, .9])
        entry = torch.tensor([.51, .58, .50, .25, .05, .1])
        active = torch.tensor([True, True, True, True, True, False])
        result = retry_rebound_growth(release, entry, torch.ones_like(entry), active, .47, .05, 4.)
        excess = torch.tensor([0., 0., .05, .08, 0., 0.])
        torch.testing.assert_close(result, 4 * torch.expm1(-.5 * (excess / .05).square()))

    def test_once_at_apex_rotation_disk_and_partial_reset(self):
        track = StraightFirstApex(5, 'cpu')
        ones = torch.ones(5, dtype=torch.bool)
        zeros = ~ones
        edge = torch.tensor([[.5,0.], [0.,.5], [-.5,0.], [.5,0.], [.5,0.]])
        root = torch.tensor([[0.,.08], [-.08,0.], [-.03,-.035], [.08,0.], [0.,.08]])
        foot = root.clone()
        velocity = torch.zeros(5, 3)
        eligible = torch.tensor([True, True, True, True, False])
        def step(t, normal, opening, z):
            velocity[:, 2] = z
            return track.step(normal, opening, eligible, ones, foot, root, velocity,
                              torch.zeros_like(root), edge, torch.ones(5), t, .01, .05)
        cost, active = step(2, zeros, ones, 2.)
        self.assertFalse(active.any())
        cost, active = step(12, zeros, zeros, -.1)
        torch.testing.assert_close(active, eligible)
        torch.testing.assert_close(cost[:1], cost[1:2])
        self.assertAlmostEqual(float(cost[0]), .03, places=6)
        self.assertFalse(cost[2:].any())
        self.assertFalse(step(13, zeros, ones, 0.)[1].any())
        track.reset(torch.tensor([0]))
        self.assertEqual(track.release_step.tolist(), [-1, 2, 2, 2, -1])
        step(20, torch.tensor([True,False,False,False,False]), zeros, 2.)
        self.assertEqual(step(21, zeros, zeros, -.1)[1].tolist(), [True,False,False,False,False])

    def test_settle_only_persistent_misses_and_physical_failure(self):
        track = WaypointOutcome(6, 'cpu')
        edge = torch.tensor([[1.,0.]] * 6)
        yes = torch.ones(6, dtype=torch.bool)
        no = ~yes
        signed = torch.full((6,), .03)
        budget = torch.full((6,), -2.)
        errors = torch.tensor([[.01,.01], [.01,.09], [-.1,-.08], [.1,0.], [0.,0.], [.1,.07]])
        dead = torch.tensor([False,False,False,False,True,False])
        touch = ~dead
        for cost in track.step(yes, signed, yes, budget, edge, no, no, errors, edge, .05, 2.):
            self.assertFalse(cost.any())
        track.reset(torch.tensor([5]))
        c,g = track.step(no, signed, no, budget, edge, touch, dead, errors, edge, .05, 2.)
        self.assertEqual((c<0).tolist(), [False,True,False,False,True,False])
        self.assertEqual((g<0).tolist(), [False,True,False,True,True,False])
        for cost in track.step(no, signed, no, budget, edge, touch, dead, errors, edge, .05, 2.):
            self.assertFalse(cost.any())
