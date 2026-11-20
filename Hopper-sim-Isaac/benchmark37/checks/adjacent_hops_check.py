"""Exercise the v45 route sampler through the same reset path used by PPO."""
import json
import math
from pathlib import Path

import torch

from ..stationary_retention import RetentionRoute


def check_long_pairs(route, course):
    from ..guarded_keys import GuardedKeys, RECOVER
    from ..moving_phase_reward import long_pair_landing_cost
    from ..obs73_omni import DIRECT_REVERSAL_KIND, PAUSED_REVERSAL_KIND

    options = course['omni']['training_reversals']['paired_long_steps']
    minimum = options['min_spacing_m']
    for kind in (DIRECT_REVERSAL_KIND, PAUSED_REVERSAL_KIND):
        points = route.points[route.sequence_kind==kind]
        edges = points.diff(dim=1)
        lengths = edges.norm(dim=-1)
        first = (lengths>0).long().argmax(1)
        row = torch.arange(len(first))
        assert len(row)>0 and (first>=1).all()
        first_edge = edges[row,first]
        assert ((first_edge.norm(dim=-1)>=minimum-1e-6)&(first_edge.norm(dim=-1)<=.500001)).all()
        assert (first_edge.norm(dim=-1)<.49).any() and (first_edge.norm(dim=-1)>.499999).any()
        torch.testing.assert_close(edges[row,first+1],(-1 if kind==DIRECT_REVERSAL_KIND else 1)*first_edge)
        assert not lengths[row,first+2].any()
        assert points.norm(dim=-1).max()<=1.000001

    edge = torch.tensor([[0.,-.5]]).repeat(6,1)
    inbound = edge.clone()
    inbound[1] *= -1
    inbound[4] = 0.
    offset = torch.tensor([[0.,.05],[0.,-.05],[.05,0.],[0.,0.],[0.,.05],[0.,.05]])
    cost = long_pair_landing_cost(inbound,edge,offset,torch.arange(6)!=5,minimum,.05,2.)
    assert cost[0]==cost[1]<0 and not cost[2:].any()

    # Drive two complete request windows. On one lane the first pair is
    # ignored in recovery; its successful second pair must count only once.
    guard = GuardedKeys(route,.01,False,training_pair_gap_s=course['guarded_keys']['training_pair_gap_s'])
    guard.mask[:] = False
    guard.mask[:2] = True
    guard.start = torch.full_like(guard.ticks,1)
    guard.reset(guard.ids)
    guard.state[1]=RECOVER
    zeros = torch.zeros_like(guard.rp)
    for tick in (1,26,101,201,251,301,901,926,1001,1101,1151,1201):
        guard.ticks.fill_(tick-1)
        hit = guard.mask & (tick%100==1) & (tick not in (1,901))
        if tick in (251,1151):
            guard.observe_free_liftoff(guard.mask)
        guard.update(hit,hit,hit,guard.xy,zeros,zeros,zeros,zeros,guard.mask,guard.mask,~guard.mask)
    guard.reset(guard.ids)
    assert guard.pair_totals.tolist()==[4,3,3]
    assert guard.second_results.sum(0).tolist()==[3,3]
    # An episode can end without a touchdown after the second liftoff.
    guard.second_in_flight[1]=True
    guard.step_dir[1]=3
    guard.reset(guard.ids)
    assert guard.second_results[3].tolist()==[1,0]


def check_second_release(course):
    from ..flight_plane import travel_peak_cost
    from ..moving_phase_reward import pair_release_cost

    axis = torch.tensor([[0.,-1.]]).repeat(4,1)
    target = .54*axis
    speed = torch.tensor([.23,.31,.23,.23])[:,None]*axis
    speed[2,0] = .45
    cost,_,_ = pair_release_cost(speed,target,axis,torch.ones(4),torch.tensor([True,True,True,False]),
                                 .05,course['omni_rebound_handoff']['south_pair_release'])
    assert cost[0]==cost[2]<0 and cost[1]==cost[3]==0
    cfg = course['flight_plane_control']
    start = math.radians(cfg['peak_start_deg'])
    peak = torch.zeros(2)
    total = torch.zeros_like(peak)
    for deg in (7.,11.,9.,14.,14.):
        angle = torch.full_like(peak,math.radians(deg))
        delta,peak = travel_peak_cost(angle,torch.tensor([True,False]),peak,start,cfg['plane_peak'])
        total += delta
    torch.testing.assert_close(total,torch.tensor([-cfg['plane_peak']*(math.radians(14.)-start)**2,0.]))


def check(course):
    config = course['omni']
    route = RetentionRoute(768,'cpu',config,42)
    ids = torch.arange(768)
    route.reset(ids)
    ordinary = ~route.stationary_mask & (ids%8!=1) & (route.sequence_kind==0)
    points = route.points[ordinary]
    assert points.norm(dim=-1).max() <= config['workspace_radius_m']+1e-6
    assert not route.points[route.stationary_mask].any()
    edges = points.diff(dim=1)
    lengths = edges.norm(dim=-1)
    assert lengths.max() <= .500001 and ((lengths>0)&(lengths<.1)).any()
    zero = (lengths==0).float().mean().item()
    full = ((lengths-.5).abs()<1e-6).float().mean().item()
    assert abs(zero-config['zero_probability'])<.02
    assert abs(full-config['training_adjacent_hops']['full_step_probability'])<.02
    # Inside this disk all headings fit, so collision with the workspace
    # boundary cannot cause a sampled turn to be replaced.
    mask = (points[:,1:-1].norm(dim=-1)<1.49)&(lengths[:,1:]>.0001)&(lengths[:,:-1]>.0001)
    incoming, outgoing = edges[:,:-1][mask], edges[:,1:][mask]
    cross = incoming[:,0]*outgoing[:,1]-incoming[:,1]*outgoing[:,0]
    angle = torch.atan2(cross,(incoming*outgoing).sum(1)).rad2deg()
    fractions = [((angle.abs()-a).abs()<.05).float().mean().item() for a in (0,90,180)]
    assert all(.22<f<.28 for f in fractions)
    assert .10 < ((angle-90).abs()<.05).float().mean().item() < .15
    first_moves = edges[:,0][lengths[:,0]>0]
    headings = (torch.atan2(first_moves[:,1],first_moves[:,0])*4/math.pi+.5).floor().long()%8
    assert torch.bincount(headings,minlength=8).min()>0
    if config.get('training_reversals',{}).get('paired_long_steps'):
        check_long_pairs(route,course)
    if course.get('omni_rebound_handoff',{}).get('south_pair_release'):
        check_second_release(course)
    untouched = route.points[1::2].clone()
    route.reset(ids[::2])
    assert torch.equal(untouched,route.points[1::2])
    parent = json.loads(Path(config['initial_checkpoint']).with_name('ring_config.json').read_text())
    old = RetentionRoute(128,'cpu',parent['omni'],7001,evaluation=True)
    new = RetentionRoute(128,'cpu',config,7001,evaluation=True)
    for r in (old,new):
        r.reset(torch.arange(128))
    assert torch.equal(old.points,new.points)
    # A missed reversal followed by a successful retry counts as one failed
    # first attempt. Multiple holds after that movement score individually.
    route.points[1,:5] = torch.tensor([[0.,0.],[.5,0.],[0.,0.],[0.,0.],[0.,0.]])
    valid = ids==1
    error = torch.zeros(768)
    for index,miss in ((1,False),(2,True),(2,False),(3,False),(4,False)):
        route.index[1]=index
        error[1]=.11 if miss else .01
        route.adjacent_metrics.observe(route,valid,error)
    assert route.adjacent_metrics.count[8:].tolist()==[[1,0],[2,2]]
    return dict(free_route_edges=lengths.numel(),zero_fraction=zero,full_fraction=full,
                straight_right_angle_reverse_fractions=fractions,baseline_evaluation_identical=True)
