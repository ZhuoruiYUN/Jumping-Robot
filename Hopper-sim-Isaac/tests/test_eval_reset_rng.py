import unittest

import torch

from benchmark37.eval_reset_rng import isolate_post_quota_resets


class ResettingEnv:
    device = "cpu"

    def __init__(self):
        self.reset_draws = []

    def _reset_idx(self, ids):
        self.reset_draws.append(torch.rand(len(ids)))


class EvalResetRngTests(unittest.TestCase):
    def test_reset_schedule_does_not_shift_rollout_rng(self):
        def replay(schedule):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7001)
                env = ResettingEnv()
                audit = isolate_post_quota_resets(env, 7001)
                noise = []
                for step in range(8):
                    if step in schedule:
                        env._reset_idx(torch.arange(schedule[step]))
                    noise.append(torch.randn(4, 73))
                return torch.stack(noise), env.reset_draws, audit["calls"]

        before, resets, before_count = replay({1: 2, 3: 2})
        after, _, after_count = replay({4: 1})
        self.assertTrue(torch.equal(before, after))
        self.assertEqual((before_count, after_count), (2, 1))
        self.assertFalse(torch.equal(resets[0], resets[1]))


if __name__ == "__main__":
    unittest.main()
