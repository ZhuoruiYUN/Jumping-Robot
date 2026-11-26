"""Keep first-episode simulator RNG independent of resets in finished lanes."""

import torch

from .fixed_case_rewards import reset_rng_context


PROTOCOL = "first_episode_dedicated_reset_rng_v1"


def isolate_post_quota_resets(env, seed):
    device = torch.device(env.device)
    cpu = torch.Generator(device="cpu").manual_seed(seed + 22310449)
    cuda = (torch.Generator(device=device).manual_seed(seed + 37718049)
            if device.type == "cuda" else None)
    reset = env._reset_idx
    audit = {"calls": 0}

    def reset_with_independent_rng(ids):
        with reset_rng_context(device):
            torch.set_rng_state(cpu.get_state())
            if cuda is not None:
                torch.cuda.set_rng_state(cuda.get_state(), device)
            reset(ids)
            cpu.set_state(torch.get_rng_state())
            if cuda is not None:
                cuda.set_state(torch.cuda.get_rng_state(device))
        audit["calls"] += 1

    env._reset_idx = reset_with_independent_rng
    return audit
