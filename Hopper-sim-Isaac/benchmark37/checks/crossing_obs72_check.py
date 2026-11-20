"""CPU contract/geometry/real checkpoint checks, no simulation or learning."""
import json
import math
from pathlib import Path

import torch

from .. import crossing_obs72 as new
from .. import ring_obs73 as old
from ..crossing_transfer import migrate, KEEP
from ..t_frame import (DEFAULT_ENVELOPE, vertices, aperture_center_in_A,
                      target_A_position, projected_clearance, quaternion_from_rpy)

ROOT = Path(__file__).resolve().parents[2]


def run():
    p=torch.tensor([[.2,-.1,.8],[1.,2.,.5]])
    q=quaternion_from_rpy(torch.tensor([[.1,-.2,.5],[-.2,.1,-.3]]))
    v=torch.tensor([[.1,.2,.3],[-.2,.1,-.5]]);w=v*.3
    targets=torch.tensor([[[.3,.4],[.7,.8],[.7,.8]],[[1.1,2.2],[1.4,2.5],[1.4,2.5]]])
    ground=torch.zeros(2);apex=torch.ones(2)
    nm=new.ObservationMemory(2,'cpu',new.VERSION);om=old.ObservationMemory(2,'cpu')
    nm.states.copy_(torch.arange(52).reshape(2,2,13)*.01);om.states.copy_(nm.states)
    nm.actions.fill_(.3);om.actions.copy_(nm.actions)
    nm.recent_action.fill_(.2);om.recent_action.copy_(nm.recent_action)
    target=p+torch.tensor([.1,.2,.3]);tq=q.clone();tv=v*.5;tw=w*.5
    state=torch.tensor([.5,0.])
    a,s=new.assemble(p,q,v,w,targets,ground,apex,nm,target,tq,tv,tw,state)
    b,bs=old.assemble(p,q,v,w,targets,ground,apex,om,target,tv,.4,torch.zeros(2,dtype=torch.bool))
    torch.testing.assert_close(a[:,:59],b[:,KEEP]);torch.testing.assert_close(s,bs)
    torch.testing.assert_close(a[0,59:62],old.inverse_rotate(q,target-p)[0])
    torch.testing.assert_close(a[0,62:65],torch.zeros(3),atol=1e-6,rtol=0)
    torch.testing.assert_close(a[0,65:68],old.inverse_rotate(q,tv)[0])
    torch.testing.assert_close(a[0,68:71],old.inverse_rotate(q,tw)[0])
    assert not a[1,59:].any() and a.shape==(2,72)
    assert not hasattr(nm,'apices') and not hasattr(nm,'landings')
    # Desired yaw versus current yaw: wrap to small relative error across +/-pi.
    yaw=quaternion_from_rpy(torch.tensor([[0.,0.,math.pi-.01],[0.,0.,0.]]))
    desired=quaternion_from_rpy(torch.tensor([[0.,0.,-math.pi+.01],[0.,0.,0.]]))
    c,_=new.assemble(p,yaw,v,w,targets,ground,apex,nm,target,desired,tv,tw,state)
    torch.testing.assert_close(c[0,64],torch.tensor(.02),atol=1e-6,rtol=0)
    negative,_=new.assemble(p,-yaw,v,w,targets,ground,apex,nm,target,-desired,tv,tw,state)
    torch.testing.assert_close(c,negative)
    before=nm.states[1].clone();nm.commit(s);nm.reset(torch.tensor([0]))
    assert not nm.states[0].any() and not nm.actions[0].any()
    torch.testing.assert_close(nm.states[1,0],before[1]);torch.testing.assert_close(nm.states[1,1],s[1])
    # Full box dimensions and placement, with convex disk/rectangle containment.
    local=vertices();dims=local.amax(0)-local.amin(0)
    torch.testing.assert_close(dims,torch.tensor([.38,.38,.30]))
    center=torch.zeros(1,3);identity=torch.tensor([[1.,0.,0.,0.]])
    circle=dict(shape='circle',radius_m=.25,margin_m=.02)
    pa=target_A_position(center,identity,'circle')
    clearance=projected_clearance(pa,identity,center,identity,circle)
    assert clearance.item()>0
    too_small=dict(circle,radius_m=.20)
    assert projected_clearance(pa,identity,center,identity,too_small).item()<0
    rect=dict(shape='rectangle',width_m=.42,height_m=.34,margin_m=.02)
    pr=target_A_position(center,identity,'rectangle')
    assert abs(projected_clearance(pr,identity,center,identity,rect).item())<1e-6
    shifted=pr+torch.tensor([[0.,.01,0.]])
    assert projected_clearance(shifted,identity,center,identity,rect).item()<0
    tilted=quaternion_from_rpy(torch.tensor([[0.,0.,math.pi/4]]))
    assert projected_clearance(pr,tilted,center,identity,rect).item()<0  # 3D depth matters
    # Rigidly rotate/translate the entire aperture and commanded robot: invariant fit.
    from ..ring_geometry import rotate
    rq=quaternion_from_rpy(torch.tensor([[.2,-.1,1.2]]));translation=torch.tensor([[2.,-3.,1.]])
    rotated=rotate(rq,pa)+translation
    torch.testing.assert_close(projected_clearance(rotated,rq,translation,rq,circle),clearance)
    # Actual v23 state shapes and unchanged recurrent/head/std tensors.
    course=json.loads((ROOT/'benchmark37/configs/obs72_omni_course_v1.json').read_text())
    source=ROOT/course['omni']['initial_checkpoint']
    previous=json.loads((source.parent/'ring_config.json').read_text())
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
    migrated,folded=migrate(checkpoint,previous,course)
    for key,value in checkpoint.items():
        changed=key in folded or ('obs_normalizer.' in key and value.shape==(1,73))
        if not changed:assert torch.equal(value,migrated[key]),key
    for side in ('a','c'):
        weight=migrated[f'memory_{side}.rnn.weight_ih_l0']
        assert weight.shape==(1024,72) and not weight[:,59:].any() and not weight[:,2:6].any()
        torch.testing.assert_close(weight[:,6:59],checkpoint[f'memory_{side}.rnn.weight_ih_l0'][:,KEEP[6:]])
        torch.nn.LSTM(72,256).load_state_dict({k.split('.rnn.')[1]:v for k,v in migrated.items() if k.startswith(f'memory_{side}.rnn.')})
    for side in ('actor','critic'):
        for suffix in ('_mean','_std','_var'):
            key=f'{side}_obs_normalizer.{suffix}'
            torch.testing.assert_close(migrated[key][:,:59],checkpoint[key][:,KEEP])
            expected=torch.zeros(1,13) if suffix=='_mean' else torch.ones(1,13)
            torch.testing.assert_close(migrated[key][:,59:],expected)
    assert course['vertical_pretraining']==previous['vertical_pretraining']
    assert course['stationary_pretraining']==previous['stationary_pretraining']
    return dict(observation_dimension=72,retained59_match=True,history_removed=True,
                relative_pose_velocity_frames_checked=True,partial_reset_checked=True,
                shape_fit_checks=True,orientation_depth_checked=True,
                real_v23_migration_checked=True,behavior_equivalence_claimed=False,
                reward_physics_config_preserved=True,
                circle_center_in_A_m=aperture_center_in_A('circle'),
                rectangle_center_in_A_m=aperture_center_in_A('rectangle'),
                minimum_circle_diameter_without_margin_m=2*(.25-.02-clearance.item()),
                envelope=DEFAULT_ENVELOPE,optimizer_updates=0)


if __name__=='__main__': print(json.dumps(run(),indent=2))
