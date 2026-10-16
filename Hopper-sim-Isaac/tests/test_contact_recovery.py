import unittest

import torch

from Quadhopper_Planner_Random.contact_recovery import (
    advance_recovery_window,
    differential_recovery_costs,
    recovery_window_steps,
)


class ContactRecoveryTests(unittest.TestCase):
    def test_window_is_touchdown_inclusive_and_counts_down(self):
        remaining = torch.tensor([0, 2, 1])
        touchdown = torch.tensor([True, False, False])
        got = advance_recovery_window(remaining, touchdown, 8)
        self.assertTrue(torch.equal(got, torch.tensor([8, 1, 0])))
        self.assertEqual(recovery_window_steps(0.08, 0.01), 8)
        self.assertEqual(recovery_window_steps(0.0, 0.01), 0)

    def test_collective_change_has_no_differential_cost(self):
        previous = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
        current = previous + 0.25
        spread_cost, delta_cost, spread, delta = differential_recovery_costs(
            current, previous, 0.20, 0.10
        )
        self.assertGreater(float(spread), 0.20)
        self.assertAlmostEqual(float(delta), 0.0, places=6)
        self.assertAlmostEqual(float(delta_cost), 0.0, places=6)
        self.assertGreater(float(spread_cost), 0.0)

    def test_differential_step_is_penalized(self):
        previous = torch.full((1, 4), 0.5)
        current = torch.tensor([[0.9, 0.1, 0.5, 0.5]])
        spread_cost, delta_cost, spread, delta = differential_recovery_costs(
            current, previous, 0.30, 0.20
        )
        self.assertGreater(float(spread), 0.30)
        self.assertGreater(float(delta), 0.20)
        self.assertGreater(float(spread_cost), 0.0)
        self.assertGreater(float(delta_cost), 0.0)


if __name__ == '__main__':
    unittest.main()
