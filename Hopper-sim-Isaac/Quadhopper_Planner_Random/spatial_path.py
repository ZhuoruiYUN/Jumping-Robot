"""Geometry-only hop guidance. No clock, duration or desired speed input."""
import torch


def hop_curve(start, landing, following, apex_height, nodes=33):
    """C1 Hermite arch with an outgoing XY tangent informed by the next hop."""
    if nodes < 9 or nodes % 2 == 0:
        raise ValueError('nodes must be odd and >= 9')
    delta = landing - start
    outgoing = following - landing
    outgoing[:, 2] = 0
    outgoing = outgoing / outgoing.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    incoming = delta.clone()
    incoming[:, 2] = 0
    length = incoming.norm(dim=-1, keepdim=True)
    incoming = incoming / length.clamp_min(1e-6)
    landing_direction = 0.75 * incoming + 0.25 * outgoing
    landing_direction /= landing_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    peak = (start + landing) * 0.5
    peak[:, 2] = apex_height
    t0 = 0.25 * length * incoming
    t0[:, 2] = 2 * (apex_height - start[:, 2]).clamp_min(0)
    tm = 0.65 * length * incoming
    t1 = 0.25 * length * landing_direction
    t1[:, 2] = -2 * (apex_height - landing[:, 2]).clamp_min(0)
    u = torch.linspace(0, 1, nodes // 2 + 1, device=start.device)[None, :, None]

    def segment(a, b, ta, tb):
        return ((2*u**3 - 3*u**2 + 1)*a[:, None] + (u**3 - 2*u**2 + u)*ta[:, None]
                + (-2*u**3 + 3*u**2)*b[:, None] + (u**3 - u**2)*tb[:, None])

    return torch.cat((segment(start, peak, t0, tm), segment(peak, landing, tm, t1)[:, 1:]), 1)


def project_path(position, curve, descending, lookahead_m=0.10, minimum_fraction=None):
    """Project geometrically and return a non-timed normalized arc coordinate."""
    a = curve[:, :-1]
    segment = curve[:, 1:] - a
    length = segment.norm(dim=-1).clamp_min(1e-8)
    fraction = (((position[:, None] - a) * segment).sum(-1)
                / length.square()).clamp(0, 1)
    projected = a + fraction[..., None] * segment
    distance_sq = (position[:, None] - projected).square().sum(-1)
    index = torch.arange(segment.shape[1], device=position.device)[None]
    mid = segment.shape[1] // 2
    allowed = torch.where(descending[:, None], index >= mid, index < mid)
    chosen = distance_sq.masked_fill(~allowed, float('inf')).argmin(-1)
    batch = torch.arange(len(position), device=position.device)
    starts = torch.cat((length.new_zeros(len(position), 1), length.cumsum(-1)[:, :-1]), -1)
    total = length.sum(-1)
    raw_arc = starts[batch, chosen] + fraction[batch, chosen] * length[batch, chosen]
    arc = raw_arc
    if minimum_fraction is not None:
        arc = torch.maximum(arc, minimum_fraction.clamp(0, 1) * total)
    forward_arc = (arc + lookahead_m).minimum(total)
    forward_index = (length.cumsum(-1) < forward_arc[:, None]).sum(-1).clamp_max(segment.shape[1]-1)
    forward_fraction = ((forward_arc - starts[batch, forward_index])
                        / length[batch, forward_index]).clamp(0, 1)
    carrot = a[batch, forward_index] + forward_fraction[:, None] * segment[batch, forward_index]
    tangent = segment[batch, forward_index] / length[batch, forward_index, None]
    return carrot, tangent, distance_sq[batch, chosen].sqrt(), raw_arc / total.clamp_min(1e-8)


def expand_policy_inputs(state, width=52):
    """Preserve all learned inputs; initialize only appended guidance columns to zero."""
    result = {k: v.clone() for k, v in state.items()}
    for key in ('memory_a.rnn.weight_ih_l0', 'memory_c.rnn.weight_ih_l0'):
        old = result[key]
        if old.shape[1] != 43:
            raise ValueError('Expected the 43-D preview warm-start checkpoint')
        result[key] = old.new_zeros(old.shape[0], width)
        result[key][:, :43] = old
    return result
