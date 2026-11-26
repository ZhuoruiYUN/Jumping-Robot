import json
from pathlib import Path
import unittest

import numpy as np
import torch

from benchmark37.checks.ab_only_check import check_routes
from benchmark37.precision_continuous import AB_COURSE, START, proposal
from benchmark37.stationary_retention import RetentionRoute
from benchmark37.waypoint_stream import straight_half_metre
from deployment.obs73_stream import StreamingObservation73


class ABStreamTest(unittest.TestCase):
    def setUp(self):
        self.course = json.loads(AB_COURSE.read_text())

    def test_spawn_resampling_preserves_other_lanes_and_rng(self):
        from benchmark37.ab_spawn import ABSpawn
        c = self.course['ab_spawn']
        ids = torch.arange(1024)
        spawn = ABSpawn(len(ids), 'cpu', 42, c)
        cpu_rng = torch.random.get_rng_state()
        mask, xy, angles, velocity = spawn.sample(ids)
        self.assertTrue(.40 < mask.float().mean() < .60)
        self.assertTrue((xy.norm(dim=1) <= c['xy_radius_m']).all())
        limits = torch.tensor([c['roll_pitch_deg'], c['roll_pitch_deg'], c['yaw_deg']])
        self.assertTrue((angles.rad2deg().abs() <= limits).all())
        self.assertTrue((velocity[:, :2].norm(dim=1) <= c['xy_speed_m_s']).all())
        replay = ABSpawn(len(ids), 'cpu', 42, c).sample(ids)
        for got, expected in zip(replay, (mask, xy, angles, velocity)):
            torch.testing.assert_close(got, expected)
        spawn.sample(ids[::2])
        torch.testing.assert_close(spawn.perturbed[1::2], mask[1::2])
        torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng)

    def test_training_and_worker(self):
        feedback = dict(first_failed_events=[], attitude=False)
        config = proposal(START, feedback, 1, None)
        self.assertTrue(config['omni']['ab_pair_only'])
        check_routes(config)
        stationary = RetentionRoute(8, 'cpu', config['omni'], 42, evaluation=True, stationary_only=True,
                                    training_stationary_fraction=0.)
        stationary.reset(torch.arange(8))
        self.assertIsNone(stationary.ab_pairs)
        self.assertFalse(stationary.points.any())

    def test_stream_exceeds_initial_storage_and_retries_per_lane(self):
        cfg = dict(self.course['omni'], route_points=32)
        route = RetentionRoute(2, 'cpu', cfg, 42, evaluation=True, training_stationary_fraction=0.)
        route.waypoint_source = straight_half_metre
        ids = torch.arange(2)
        route.reset(ids)
        for i in range(110):
            route.advance(torch.tensor([True, True]), torch.tensor([True, False]))
            self.assertEqual(route.index.tolist(), [i+1, 0])
            current, preview = route.pair(ids)
            self.assertEqual(current[0].tolist(), [.5*(i+1), 0.])
            self.assertEqual(preview[1].tolist(), [.5, 0.])
        route.reset(ids[1:])
        self.assertEqual(route.pair(ids)[0][0].tolist(), [55., 0.])

    def test_repeated_holds_keep_ab_events_and_isolate_weak_response_sampling(self):
        from benchmark37.obs73_response_robustness import ResponseEnvelope

        config = Path(__file__).resolve().parents[1] / 'training_config_v51/ab_ws_first_precision/course.json'
        course = json.loads(config.read_text())
        ids = torch.arange(4096)
        original = RetentionRoute(len(ids), 'cpu', course['omni'], 42, training_stationary_fraction=0.)
        revised = RetentionRoute(len(ids), 'cpu', dict(course['omni'], ab_repeated_hold_fraction=.5), 42,
                                 training_stationary_fraction=0.)
        original.reset(ids)
        revised.reset(ids)
        hold = revised.ab_pairs.repeated_hold
        self.assertTrue(.035 < hold.float().mean() < .065)
        self.assertFalse(revised.stationary_mask.any())
        self.assertTrue(revised.ab_pairs.mask.all())
        self.assertFalse(revised.points[hold, :4].any())
        self.assertFalse(original.points[hold, :2].any())
        self.assertTrue(original.points[hold, 2].ne(0.).any(dim=-1).all())
        torch.testing.assert_close(original.points[~hold], revised.points[~hold], rtol=0., atol=0.)
        torch.testing.assert_close(original.index, revised.index, rtol=0., atol=0.)
        torch.testing.assert_close(original.ab_pairs.rng.get_state(), revised.ab_pairs.rng.get_state())
        self.assertFalse(revised.ab_pairs.touchdown(revised.index, hold, hold).any())
        revised.advance(hold, hold)
        self.assertTrue(torch.equal(revised.ab_pairs.touchdown(revised.index, hold, hold), hold))

        response = course['vertical_pretraining']['response_robustness']
        a = ResponseEnvelope(len(ids), 'cpu', response, 42)
        b = ResponseEnvelope(len(ids), 'cpu', response, 42)
        a.reset(ids, torch.ones(len(ids), 3))
        b.reset(ids, torch.ones(len(ids), 3), hold)
        unchanged = ~hold | (a.gain == 1.)
        torch.testing.assert_close(a.gain[unchanged], b.gain[unchanged], rtol=0., atol=0.)
        self.assertTrue((b.gain[hold] < .8).any())
        torch.testing.assert_close(a.rng.get_state(), b.rng.get_state())

    def test_historical_mixed_course(self):
        from benchmark37.guarded_keys import GuardedKeys
        cfg = json.loads((AB_COURSE.parents[1]/'ab/course.json').read_text())
        route = RetentionRoute(32, 'cpu', cfg['omni'], 42)
        route.reset(torch.arange(32))
        keys = GuardedKeys(route, .01, False)
        self.assertEqual(int(route.stationary_mask.sum()), 16)
        self.assertEqual(int(keys.mask.sum()), 4)
        self.assertEqual(int(route.ab_pairs.mask.sum()), 4)
        self.assertFalse((route.ab_pairs.mask & (route.stationary_mask | keys.mask)).any())

    def test_deployment_contact_tick_observation_and_memory(self):
        observer = StreamingObservation73(self.course, straight_half_metre)
        action = np.array([.2, .1, -.1, -.2], dtype=np.float32)

        def tick(x, z, vz, spring):
            obs = observer.observe([x, 0., z], [1., 0., 0., 0.], [0., 0., vz],
                                   [0., 0., 0.], spring)
            observer.accept(action)
            return obs

        def hop(x, peak=1.02):
            for z, vz in [(.4, 1.), (.7, 1.), (peak, .01), (.7, -1.), (.6, -1.), (.4, -1.)]:
                tick(x, z, vz, 0.)
            return tick(x, .27, -1., .005)

        tick(0., .27, -1., .005)  # initial release contact
        self.assertIsNone(observer.last_landing)
        hop(0., peak=.9)
        self.assertEqual(observer.index, 0)
        observer.set_next((.25, 0.))
        with self.assertRaises(ValueError):
            observer.set_next((float('nan'), 0.))
        obs = hop(0.)
        np.testing.assert_allclose(obs[:6], [.25, 0., .75, 0., .75, 0.])
        np.testing.assert_allclose(obs[45:57], np.tile(action, 3))
        self.assertEqual(observer.last_landing['target_index'], 0)
        hop(.31)
        self.assertFalse(observer.last_landing['navigation_advanced'])
        hop(.25, peak=.9)
        self.assertFalse(observer.last_landing['strict'])
        self.assertEqual(observer.index, 2)
        for _ in range(100):
            hop(observer.current_xy[0])
        self.assertEqual(observer.index, 102)
        observer.reset()
        self.assertEqual(observer.current_xy, (0., 0.))
        self.assertFalse(observer.recent_action.any())

    def test_deployment_observation_matches_simulator(self):
        from benchmark37.crossing_obs72 import ObservationMemory
        from benchmark37.crossing_obs73_quat import assemble
        observer = StreamingObservation73(self.course)
        memory = ObservationMemory(1, 'cpu')
        zero = torch.zeros(1)
        inactive = torch.zeros(1, 3)
        identity = torch.tensor([[1., 0., 0., 0.]])
        q = torch.tensor([[-.93, .08, .01, -.24]])
        q = q / q.norm()
        for i in range(10):
            observer.set_next((i/30, -.11))
            pos = torch.tensor([[.02*i, .1, .7]])
            velocity = torch.tensor([[.2, .03*i, -.1]])
            omega = torch.tensor([[.02*i, .1, -.2]])
            result = observer.observe(pos[0], q[0], velocity[0], omega[0], 0.)
            targets = torch.tensor([[observer.current_xy, observer.preview_xy, observer.preview_xy]])
            expected, snapshot = assemble(pos, q, velocity, omega, targets, zero, zero+1, memory,
                                          inactive, identity, inactive, inactive, zero)
            np.testing.assert_allclose(result, expected[0].numpy(), atol=1e-6)
            action = torch.tensor([[.05*i, -.03*i, .2, -.4]])
            observer.accept(action[0])
            memory.commit(snapshot)
            memory.recent_action.copy_(action)
