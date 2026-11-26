import json
import unittest
from pathlib import Path

import torch

from benchmark37.metrics import same_target_recovery_metrics
from benchmark37.stationary_retention import RetentionRoute


class ABRecoveryTest(unittest.TestCase):
    def test_recovery_uses_the_second_touch_at_index_one_and_resets_per_lane(self):
        folder = Path(__file__).resolve().parents[1] / 'training_config_v51'
        config = json.loads((folder/'ab_recovery/course.json').read_text())['omni']
        prior = json.loads((folder/'ab_yaw_peak/course.json').read_text())['omni']
        def build(cfg):
            return RetentionRoute(1024, 'cpu', cfg, 42, training_stationary_fraction=0.)
        route, control = build(config), build(prior)
        ids = torch.arange(1024)
        route.reset(ids)
        control.reset(ids)
        torch.testing.assert_close(route.points, control.points)
        ab = route.ab_pairs
        tracker = ab.recovery
        original, preview = route.pair(ids)
        valid = torch.ones_like(ab.seen_a)
        # Every lane misses its first goal. Among these, sample all three paths.
        failed = ~valid
        self.assertFalse(ab.touchdown(route.index, valid, failed).any())
        hold = tracker.start(valid)
        retry = tracker.active == tracker.RETRY
        ordinary = tracker.active == tracker.CONTINUE
        self.assertTrue(all(mask.any() for mask in (ordinary, hold, retry)))
        foot = original + .1
        route.advance(valid, failed, ordinary)
        tracker.publish_hold(route, ids[hold], foot)
        curr, nxt = route.pair(ids)
        torch.testing.assert_close(curr[ordinary], preview[ordinary])
        torch.testing.assert_close(curr[retry], original[retry])
        torch.testing.assert_close(nxt[retry], preview[retry])
        torch.testing.assert_close(curr[hold], foot[hold])
        torch.testing.assert_close(nxt[hold], foot[hold])
        self.assertFalse(ab.touchdown(route.index, failed, failed).any())
        self.assertTrue(ab.touchdown(route.index, valid, valid).all())
        self.assertEqual(ab.totals.tolist()[1:5], [1024, 0, 1024, 1024])
        active = tracker.active.clone()
        route.reset(ids[::2])
        control.reset(ids[::2])
        self.assertFalse(tracker.active[::2].any())
        torch.testing.assert_close(tracker.active[1::2], active[1::2])
        torch.testing.assert_close(route.points[::2], control.points[::2])

    def test_retry_metrics_count_unfinished_attempts_and_terminal_release(self):
        miss = dict(moving=True, navigation_advanced=False, target_index=4, foot_error_m=.12)
        hit = dict(miss, navigation_advanced=True, foot_error_m=.04,
                   release_velocity_xy_m_s=[.3,.4], desired_release_velocity_xy_m_s=[0,0])
        rows = [dict(events=[miss, miss, hit], physical_death=False),
                dict(events=[miss],physical_death=True,terminal_flight=dict(target_index=4,
                     confirmed_release=True,release_velocity_xy_m_s=[1,0],desired_release_velocity_xy_m_s=[0,0])),
                dict(events=[miss],physical_death=False)]
        result = same_target_recovery_metrics(rows,.05)
        self.assertEqual([result[k] for k in ('starts','next_landings','within_radius','physical_failures','unfinished_at_timeout')], [4,2,1,1,1])
        self.assertAlmostEqual(result['release_xy_error_m_s']['mean'],.75)
