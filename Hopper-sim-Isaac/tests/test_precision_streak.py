import unittest

import torch

from benchmark37.precision_streak import streak_step


class PrecisionStreakTest(unittest.TestCase):
    def test_first_touch_retry_skip_and_noncontact(self):
        prior=torch.tensor([2,2,2,2,2,2,-1,9])
        target=torch.tensor([3,2,4,3,3,3,0,10])
        streak=torch.tensor([3,0,3,3,3,0,0,10])
        touch=torch.tensor([1,1,1,1,0,1,1,1],dtype=torch.bool)
        strict=torch.tensor([1,1,1,0,0,1,1,1],dtype=torch.bool)
        updated,reward=streak_step(streak,prior,target,touch,strict,2.)
        self.assertEqual(updated.tolist(),[4,0,0,0,3,1,1,10])
        torch.testing.assert_close(reward,torch.tensor([1.1,0.,0.,0.,0.,.65,.65,2.]))
