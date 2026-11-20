"""Small regression at the real5ms contact seam; no simulator or policy run."""
import json
import argparse
from pathlib import Path
import torch
from .ring_geometry import RingEvents
from .ring_geometry_v2 import RingEventsV2, VERSION, make_ring_events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();device=args.device
    torch.set_default_device(device)
    fixture=json.loads(Path(__file__).with_name('ring_contact_fixtures.json').read_text())
    course=fixture['course']
    def tensors(s):return torch.tensor([s['position_local_m']]),torch.tensor([s['quaternion_wxyz']]),torch.tensor([s['spring_q_m']])
    for row in fixture['contacts']:
        old,new=tensors(row['previous']),tensors(row['current'])
        for cls,expected in ((RingEvents,True),(RingEventsV2,False)):
            event=cls(1,device,course);event.reset(torch.tensor([0]),*old,torch.zeros(1,3))
            event.sample(*new,torch.ones(1,dtype=torch.bool))
            assert bool(event.collision[0])==expected,(row['case_id'],cls.__name__)
    q=torch.tensor([[1.,0.,0.,0.]]);spring=torch.zeros(1);ids=torch.tensor([0]);enabled=torch.ones(1,dtype=torch.bool)
    def static(z):
        p=torch.tensor([[course['center_local_m'][0],0.,z]])
        event=RingEventsV2(1,device,course);event.reset(ids,p,q,spring,torch.zeros_like(p));event.sample(p,q,spring,enabled)
        return bool(event.collision[0])
    assert static(.78),'real lower-rim/foot envelope contact must remain terminal'
    assert static(1.17),'real upper-rim/rotor envelope contact must remain terminal'
    assert not static(.25),'external lower bypass must not be classified as ring contact'
    event=RingEventsV2(1,device,course)
    p=torch.tensor([[-.1,0.,.94]]);event.reset(ids,p,q,spring,torch.zeros_like(p))
    for x in torch.linspace(-.1,.7,81):event.sample(torch.tensor([[float(x),0.,.94]]),q,spring,enabled)
    assert bool(event.passed[0]) and not bool(event.collision[0]),'full forward crossing'
    event.event.zero_();event.sample(torch.tensor([[.8,0.,.94]]),q,spring,enabled)
    assert not bool(event.event[0]),'no duplicate event reward'
    # Keep conservative continuous coverage for a fast step through the foot rim.
    p=torch.tensor([[-.3,0.,.78]]);event=RingEventsV2(1,device,course);event.reset(ids,p,q,spring,torch.zeros_like(p))
    event.sample(torch.tensor([[.8,0.,.78]]),q,spring,enabled)
    assert bool(event.collision[0]),'fast real rim crossing must not tunnel'
    # A partial reset must not clear the other environment's latched state.
    event=RingEventsV2(2,device,course);p=torch.tensor([[-.1,0.,.94],[-.1,0.,.94]])
    qq=q.repeat(2,1);ss=spring.repeat(2);event.reset(torch.arange(2),p,qq,ss,torch.zeros_like(p));event.passed[1]=True
    event.reset(ids,p,qq,ss,torch.zeros_like(p));assert bool(event.passed[1])
    assert type(make_ring_events(1,device,course)) is RingEvents,'V1 artifact semantics must remain unchanged'
    p=torch.tensor([[.27,0.,z] for z in (.78,1.17,.25,.94)]);qq=q.repeat(4,1);ss=spring.repeat(4)
    event=RingEventsV2(4,device,course);event.reset(torch.arange(4),p,qq,ss,torch.zeros_like(p))
    event.sample(p,qq,ss,torch.ones(4,dtype=torch.bool))
    assert event.collision.cpu().tolist()==[True,True,False,False],'batched narrow phase/refinement'
    print(f'RING_CONTACT_OFFLINE=passed device={device}:2 real failure fixtures, true foot/rotor contact, external bypass, full passage, once-only, high-speed sweep, partial reset, V1 compatibility, batched narrow phase')


if __name__=='__main__':main()
