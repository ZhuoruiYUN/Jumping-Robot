import unittest
from types import SimpleNamespace

import torch

from benchmark37.precision_distance import DistancePractice


class DistanceTest(unittest.TestCase):
    def test_only_new_training_routes_scale(self):
        practice=DistancePractice(16,'cpu')
        ids=torch.arange(16)
        points=torch.zeros(16,12,2)
        points[:,:,0]=torch.arange(12)*.5
        route=SimpleNamespace(index=torch.zeros(16,dtype=torch.long),points=points.clone())
        practice.reset(route,ids)
        torch.testing.assert_close(route.points[~practice.mask],points[~practice.mask])
        torch.testing.assert_close(route.points[3],points[3]*.7)
        for index in range(1,7):
            route.index.fill_(index)
            practice.observe(route,ids==3,ids==3,ids<0)
            practice.observe(route,ids==3,ids<0,ids<0)
        self.assertEqual(practice.attempts[3].item(),6)
        self.assertEqual(practice.hits[3].item(),6)
        self.assertEqual(practice.level[3].item(),0)
        route.points.copy_(points)
        practice.reset(route,ids)
        torch.testing.assert_close(route.points[3],points[3]*.85)
        self.assertEqual(practice.last[3].item(),-1)
        practice.observe(route,ids<0,ids<0,ids==3)
        route.points.copy_(points)
        practice.reset(route,ids)
        torch.testing.assert_close(route.points[3],points[3]*.7)
