"""Saved-touchdown diagnostic semantics, without running an Isaac scene."""
import unittest

import torch

from Quadhopper_Planner_Random.next_hop_distance_audit import NextHopDistanceAudit


class NextHopDistanceAuditTest(unittest.TestCase):
    def test_following_apex_is_attributed_to_the_previous_actual_distance(self):
        audit = NextHopDistanceAudit(2, 'cpu', long_radius_max_m=.75)
        ids = torch.tensor([0, 1])
        audit.observe(
            ids=ids,
            old_target_xy=torch.zeros(2, 2),
            next_target_xy=torch.tensor([[.622, 0.], [.650, 0.]]),
            foot_xy=torch.tensor([[-.249, 0.], [0., 0.]]),
            next_is_short=torch.tensor([False, False]),
            completed_apex_m=torch.tensor([1.011, 1.020]),
            completed_target_m=torch.ones(2),
        )
        audit.observe(
            ids=ids,
            old_target_xy=torch.tensor([[.622, 0.], [.650, 0.]]),
            next_target_xy=torch.tensor([[.8, 0.], [.8, 0.]]),
            foot_xy=torch.tensor([[.622, 0.], [.650, 0.]]),
            next_is_short=torch.tensor([True, True]),
            completed_apex_m=torch.tensor([.908, 1.050]),
            completed_target_m=torch.ones(2),
        )
        summary = audit.summary()
        self.assertEqual(summary['next_long_count'], 2)
        self.assertEqual(summary['next_long_over_cap_count'], 1)
        self.assertAlmostEqual(summary['next_long_over_cap_actual_mean_m'], .871, places=5)
        self.assertEqual(summary['next_long_over_cap_followup_count'], 1)
        self.assertEqual(summary['next_long_within_cap_followup_count'], 1)
        self.assertEqual(summary['next_long_over_cap_followup_low_apex_rate'], 1.)
        self.assertEqual(summary['next_long_within_cap_followup_low_apex_rate'], 0.)
        self.assertAlmostEqual(summary['next_long_over_cap_followup_apex_mean_m'], .908, places=5)

    def test_reset_censors_pending_followup_instead_of_pairing_a_new_episode(self):
        audit = NextHopDistanceAudit(1, 'cpu', long_radius_max_m=.75)
        ids = torch.tensor([0])
        audit.observe(ids=ids, old_target_xy=torch.zeros(1, 2),
                      next_target_xy=torch.tensor([[.622, 0.]]),
                      foot_xy=torch.tensor([[-.249, 0.]]),
                      next_is_short=torch.tensor([False]),
                      completed_apex_m=torch.tensor([1.011]),
                      completed_target_m=torch.ones(1))
        audit.reset(ids)
        audit.observe(ids=ids, old_target_xy=torch.zeros(1, 2),
                      next_target_xy=torch.tensor([[.20, 0.]]),
                      foot_xy=torch.zeros(1, 2),
                      next_is_short=torch.tensor([True]),
                      completed_apex_m=torch.tensor([.5]),
                      completed_target_m=torch.ones(1))
        summary = audit.summary()
        self.assertEqual(summary['next_long_over_cap_followup_count'], 0)
        self.assertEqual(summary['next_long_over_cap_reset_before_followup_count'], 1)


if __name__ == '__main__':
    unittest.main()
