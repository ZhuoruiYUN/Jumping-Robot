import unittest

import torch

from benchmark37.ab_failure_settlement import ABFailureSettlement


class TestFailureSettlement(unittest.TestCase):
    def test_discount_and_terminal_frame(self):
        settlement = ABFailureSettlement(dict(window_s=.04, slot_decay=.9, missing_tick_cost=2), .01, 'cpu')
        age = torch.tensor([1, 2, 3, 4, 5, 1])
        failed = torch.tensor([True, True, True, True, True, False])
        expected = -2 * torch.tensor([.9 + .81 + .729, .81 + .729, .729, 0., 0., 0.])
        torch.testing.assert_close(settlement.charge(age, failed), expected)
        torch.testing.assert_close(settlement.charge(age, ~failed)[-1], expected[0])

    def test_ab_completion_timeout_and_later_fall(self):
        settlement = ABFailureSettlement(dict(window_s=.02, slot_decay=1., missing_tick_cost=1.), .01, 'cpu')
        torch.testing.assert_close(settlement.charge(torch.tensor([1,1,1,3]), torch.tensor([False,False,True,True])),
                                   torch.tensor([0.,0.,-1.,0.]))
