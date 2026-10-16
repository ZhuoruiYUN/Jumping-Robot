import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from Quadhopper_Planner_Random.waypoint_command import TwoHopRandomCommand


class RandomTwoHopCommandTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.num_envs = 4096
        self.origins = torch.zeros(self.num_envs, 3)
        self.command = TwoHopRandomCommand(
            self.num_envs, "cpu", 0.5, 0.8, 0.8, 1.0, 58
        )
        self.env_ids = torch.arange(self.num_envs)
        self.command.reset(self.env_ids, self.origins, random_phase=False)

    def assert_distance_range(self, start, end, lower, upper):
        distance = torch.linalg.norm(end - start, dim=1)
        self.assertTrue(torch.all(distance >= lower - 1.0e-6))
        self.assertTrue(torch.all(distance <= upper + 1.0e-6))

    def test_reset_samples_short_then_long(self):
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(self.command.current_hop_is_short(self.env_ids)))
        self.assert_distance_range(self.origins[:, :2], p_t, 0.5, 0.8)
        self.assert_distance_range(p_t, p_t1, 0.8, 1.0)

    def test_two_hop_scenario_buckets_cover_reversal_and_zero_transitions(self):
        count = 1024
        ids = torch.arange(count)
        origins = torch.zeros(count, 3)
        command = TwoHopRandomCommand(
            count, 'cpu', .12, .30, .30, .50, 10000,
            math.pi, zero_hop_probability=.12,
            reverse_turn_probability=.65,
            reverse_turn_halfwidth_rad=math.pi / 6,
            pair_scenario_groups=True,
        )
        command.reset(ids, origins, random_phase=False)
        self.assertEqual(torch.bincount(command.pair_scenario).tolist(), [512, 256, 256])
        first, second = command.lookahead()
        first_leg = first - origins[:, :2]
        second_leg = second - first
        self.assertTrue(torch.all(first_leg[:768].norm(dim=1) >= .12 - 1e-6))
        self.assertTrue(torch.all(second_leg[:768].norm(dim=1) >= .30 - 1e-6))
        reverse = slice(512, 768)
        signed_turn = torch.atan2(
            first_leg[reverse, 0] * second_leg[reverse, 1]
            - first_leg[reverse, 1] * second_leg[reverse, 0],
            (first_leg[reverse] * second_leg[reverse]).sum(dim=1),
        ).abs()
        self.assertTrue(torch.all(signed_turn >= 5 * math.pi / 6 - 1e-5))
        pattern = command.pair_zero_pattern[768:].clone()
        self.assertEqual(torch.bincount(pattern, minlength=3).tolist(), [86, 85, 85])
        self.assertTrue(torch.all(first_leg[768:][pattern != 2] == 0))
        self.assertTrue(torch.all(second_leg[768:][pattern != 1] == 0))
        command.reset(ids, origins, random_phase=False)
        self.assertTrue(torch.equal(
            command.pair_zero_pattern[768:], (pattern + 1) % 3
        ))

    def test_advance_keeps_two_point_lookahead_and_alternates_range(self):
        # Vectorized height training is allowed to randomize its phase without
        # changing the short/long geometric period.
        self.command.cycle_index[:] = torch.randint(0, 2, (self.num_envs,))
        _, original_second = self.command.lookahead()
        original_second = original_second.clone()
        self.command.advance(self.env_ids)
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(~self.command.current_hop_is_short(self.env_ids)))
        torch.testing.assert_close(p_t, original_second)
        self.assert_distance_range(p_t, p_t1, 0.5, 0.8)

        self.command.advance(self.env_ids)
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(self.command.current_hop_is_short(self.env_ids)))
        self.assert_distance_range(p_t, p_t1, 0.8, 1.0)

    def test_headings_cover_all_quadrants(self):
        p_t, _ = self.command.lookahead()
        heading = torch.atan2(p_t[:, 1], p_t[:, 0])
        quadrant = torch.floor((heading + math.pi) / (0.5 * math.pi)).long()
        counts = torch.bincount(quadrant, minlength=4)
        self.assertTrue(torch.all(counts > 800), counts)

    def test_vectorized_reset_balances_short_and_long_start_phases(self):
        self.command.reset(self.env_ids, self.origins, random_phase=True)
        p_t, p_t1 = self.command.lookahead()
        starts_short = self.command.current_hop_is_short(self.env_ids)
        short_count = torch.sum(starts_short).item()
        self.assertGreater(short_count, 1800)
        self.assertLess(short_count, 2300)
        self.assert_distance_range(
            self.origins[starts_short, :2], p_t[starts_short], 0.5, 0.8
        )
        self.assert_distance_range(p_t[starts_short], p_t1[starts_short], 0.8, 1.0)
        self.assert_distance_range(
            self.origins[~starts_short, :2], p_t[~starts_short], 0.8, 1.0
        )
        self.assert_distance_range(p_t[~starts_short], p_t1[~starts_short], 0.5, 0.8)

    def test_invalid_radius_range_is_rejected(self):
        with self.assertRaises(ValueError):
            TwoHopRandomCommand(1, "cpu", 0.8, 0.5, 0.8, 1.0, 58)

    def test_hard_replay_sampler_applies_to_both_queue_phases(self):
        command = TwoHopRandomCommand(
            self.num_envs,
            "cpu",
            0.20,
            0.30,
            0.20,
            0.30,
            58,
            math.pi,
            0.25,
            0.75,
        )
        command.reset(self.env_ids, self.origins, random_phase=False)
        first, second = command.lookahead()
        first_radius = torch.linalg.norm(first - self.origins[:, :2], dim=1)
        second_radius = torch.linalg.norm(second - first, dim=1)
        for radius in (first_radius, second_radius):
            self.assertTrue(torch.all(radius >= 0.20 - 1.0e-6))
            self.assertTrue(torch.all(radius <= 0.30 + 1.0e-6))
            hard_fraction = (radius >= 0.25).float().mean().item()
            self.assertGreater(hard_fraction, 0.72)
            self.assertLess(hard_fraction, 0.78)

        command.advance(self.env_ids)
        current, following = command.lookahead()
        appended_radius = torch.linalg.norm(following - current, dim=1)
        hard_fraction = (appended_radius >= 0.25).float().mean().item()
        self.assertGreater(hard_fraction, 0.72)
        self.assertLess(hard_fraction, 0.78)

    def test_zero_hard_probability_keeps_uniform_evaluation_sampling(self):
        command = TwoHopRandomCommand(
            self.num_envs,
            "cpu",
            0.20,
            0.30,
            0.20,
            0.30,
            58,
            math.pi,
            0.25,
            0.0,
        )
        command.reset(self.env_ids, self.origins, random_phase=False)
        first, second = command.lookahead()
        for radius in (
            torch.linalg.norm(first - self.origins[:, :2], dim=1),
            torch.linalg.norm(second - first, dim=1),
        ):
            upper_half_fraction = (radius >= 0.25).float().mean().item()
            self.assertGreater(upper_half_fraction, 0.47)
            self.assertLess(upper_half_fraction, 0.53)

    def test_invalid_hard_replay_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TwoHopRandomCommand(1, "cpu", .2, .3, .2, .3, 58, math.pi, .25, 1.1)
        with self.assertRaises(ValueError):
            TwoHopRandomCommand(1, "cpu", .2, .3, .2, .3, 58, math.pi, .30, .75)
        with self.assertRaises(ValueError):
            TwoHopRandomCommand(1, "cpu", .2, .3, .2, .3, 58, math.pi, None, .75)

    def test_restart_pair_uses_measured_position_and_short_long_order(self):
        self.command.advance(self.env_ids)
        self.command.max_turn_angle_rad = math.pi / 6.0
        previous_offset = (
            self.command.targets_w[:, 0] - self.command.anchor_w
        ).clone()
        measured = torch.randn(self.num_envs, 2)
        self.command.restart_pair(self.env_ids, measured)
        p_t, p_t1 = self.command.lookahead()
        self.assertTrue(torch.all(self.command.current_hop_is_short(self.env_ids)))
        torch.testing.assert_close(self.command.anchor_w, measured)
        self.assert_distance_range(measured, p_t, 0.5, 0.8)
        self.assert_distance_range(p_t, p_t1, 0.8, 1.0)
        first_offset = p_t - measured
        cross = (
            previous_offset[:, 0] * first_offset[:, 1]
            - previous_offset[:, 1] * first_offset[:, 0]
        )
        dot = torch.sum(previous_offset * first_offset, dim=1)
        turn = torch.abs(torch.atan2(cross, dot))
        self.assertTrue(
            torch.all(turn <= self.command.max_turn_angle_rad + 1.0e-6)
        )

    def test_turn_curriculum_bounds_consecutive_heading_change(self):
        max_turn = math.pi / 4.0
        command = TwoHopRandomCommand(
            self.num_envs, "cpu", 0.5, 0.8, 0.8, 1.0, 58, max_turn
        )
        command.reset(self.env_ids, self.origins, random_phase=False)

        def assert_bounded(first, second):
            cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
            dot = torch.sum(first * second, dim=1)
            turn = torch.abs(torch.atan2(cross, dot))
            self.assertTrue(torch.all(turn <= max_turn + 1.0e-6))

        p_t, p_t1 = command.lookahead()
        assert_bounded(p_t - self.origins[:, :2], p_t1 - p_t)
        command.advance(self.env_ids)
        p_t, p_t1 = command.lookahead()
        assert_bounded(p_t - command.anchor_w, p_t1 - p_t)

    def test_zero_width_reverse_band_produces_exact_ab_return(self):
        command = TwoHopRandomCommand(
            128, "cpu", .25, .25, .25, .25, 58,
            math.pi, zero_hop_probability=0.0,
            reverse_turn_probability=1.0, reverse_turn_halfwidth_rad=0.0,
        )
        env_ids = torch.arange(128)
        origins = torch.zeros(128, 3)
        command.reset(env_ids, origins, random_phase=False)
        point_b, point_a = command.lookahead()
        point_b = point_b.clone()
        torch.testing.assert_close(point_a, origins[:, :2], atol=1e-6, rtol=0)
        command.advance(env_ids)
        point_a, next_b = command.lookahead()
        torch.testing.assert_close(point_a, origins[:, :2], atol=1e-6, rtol=0)
        torch.testing.assert_close(next_b, point_b, atol=1e-6, rtol=0)

    def test_fixed_initial_heading_places_ab_return_on_world_x_axis(self):
        command = TwoHopRandomCommand(
            8, 'cpu', .5, .5, .5, .5, 58, math.pi,
            reverse_turn_probability=1.0, reverse_turn_halfwidth_rad=0.0,
            fixed_initial_heading_rad=0.0,
        )
        env_ids = torch.arange(8)
        origins = torch.zeros(8, 3)
        command.reset(env_ids, origins, random_phase=False)
        point_b, point_a = command.lookahead()
        torch.testing.assert_close(point_b[:, 0], torch.full((8,), .5))
        torch.testing.assert_close(point_b[:, 1], torch.zeros(8), atol=1e-6, rtol=0)
        torch.testing.assert_close(point_a, origins[:, :2], atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
