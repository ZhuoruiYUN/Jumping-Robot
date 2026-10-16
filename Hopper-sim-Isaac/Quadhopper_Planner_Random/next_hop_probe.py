"""Read-only counterfactual next-hop probe for a frozen recurrent actor."""

import torch


VARIANTS = ("actual", "left90", "reverse", "stop")


def changed_segment(segment: torch.Tensor, variant: str) -> torch.Tensor:
    """Rotate the current P2-P1 body-frame segment, keeping its length."""
    if variant == "actual":
        return segment
    if variant == "left90":
        return torch.stack((-segment[..., 1], segment[..., 0]), dim=-1)
    if variant == "reverse":
        return -segment
    if variant == "stop":
        return torch.zeros_like(segment)
    raise ValueError(variant)


def probe_actor(policy, obs: torch.Tensor, hidden, env_ids: list[int],
                live_actions: torch.Tensor) -> list[dict]:
    """Evaluate alternatives from the same LSTM state without advancing it."""
    if obs.shape[-1] != 336 or hidden is None:
        raise ValueError("probe requires history_foot_v19 observations and a live LSTM state")
    index = torch.as_tensor(env_ids, device=obs.device, dtype=torch.long)
    # Keep the live vector batch intact. A one-environment cuDNN LSTM uses
    # different kernels/rounding from the 16-environment policy forward.
    baseline_segment = obs[index, 39:41].clone()
    baseline = live_actions[index]
    rows = []
    for variant in VARIANTS:
        trial = obs.clone()
        trial[index, 39:41] = changed_segment(baseline_segment, variant)
        normalized = policy.actor_obs_normalizer(trial)
        recurrent, _ = policy.memory_a.rnn(normalized.unsqueeze(0), hidden)
        predicted = policy.actor(recurrent.squeeze(0))[index]
        if variant == "actual" and not torch.allclose(predicted, baseline, atol=1e-6, rtol=1e-6):
            largest = float((predicted - baseline).abs().max().detach().cpu())
            raise RuntimeError(f"probe baseline differs from the executed recurrent actor: max_abs={largest:.3g}")
        for row_index, env_id in enumerate(env_ids):
            action = predicted[row_index]
            delta = action - baseline[row_index]
            rows.append(dict(
                env_id=env_id, variant=variant,
                segment_body_normalized=trial[env_id, 39:41].detach().cpu().tolist(),
                action=action.detach().cpu().tolist(),
                action_delta_l2=float(torch.linalg.vector_norm(delta).detach().cpu()),
                collective_delta=float(delta.mean().detach().cpu()),
                differential_delta_l2=float(torch.linalg.vector_norm(delta - delta.mean()).detach().cpu()),
            ))
    return rows
