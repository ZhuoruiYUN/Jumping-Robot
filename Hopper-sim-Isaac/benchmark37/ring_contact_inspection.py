"""Inspect saved V2 first-contact segments only. Never starts a simulator."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import torch

from .ring_contact_diagnostic import PART_NAMES, clipped_points, radial_range, hull
from .ring_geometry_v2 import FullEnvelopeV2, RingEventsV2


def inspect(trace):
    course=trace['course'];shape=FullEnvelopeV2('cpu')
    center=course['center_local_m'];radius=course['inner_radius_m']
    half=course['axial_thickness_m']/2;delta=course['delta_m']
    outer=radius+course['tube_radial_width_m'];edges=shape.edges.tolist()

    def state(s):return torch.tensor([s['position_local_m']]),torch.tensor([s['quaternion_wxyz']]),torch.tensor([s['spring_q_m']])
    def interpolation(a,b,t):
        p,q,s=a;bp,bq,bs=b
        bq=torch.where(((q*bq).sum(-1)<0)[:,None],-bq,bq)
        return p+(bp-p)*t,torch.nn.functional.normalize(q+(bq-q)*t,dim=-1),s+(bs-s)*t
    def bound(a,b):
        p,q,s=a;bp,bq,bs=b
        angle=2*math.acos(min(1.,abs(float((q*bq).sum()))))
        return float(torch.linalg.vector_norm(bp-p))/2+2*shape.radius*math.sin(angle/4)+abs(float(bs-s))/2+delta
    def details(s,margin):
        vertices=shape.vertices(*s)[0].tolist();out=[]
        for i,part in enumerate(vertices):
            points=clipped_points(part,edges[i],center,half+margin)
            if not points:continue
            lo,hi=radial_range(points)
            if hi+margin<radius or lo-margin>outer:continue
            far=max(points,key=lambda p:p[0]**2+p[1]**2)
            polygon=hull(points)
            if lo<1e-12:near=(0.,0.)
            elif len(polygon)==1:near=polygon[0]
            else:
                projections=[]
                for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                    dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
                    t=max(0.,min(1.,-(a[0]*dx+a[1]*dy)/den)) if den else 0.
                    projections.append((a[0]+t*dx,a[1]+t*dy))
                near=min(projections,key=lambda p:p[0]**2+p[1]**2)
            # A point in the convex projection nearest the inner annulus,
            # rather than a distant extreme vertex of the same touching part.
            target=max(lo,min(radius,hi));dx,dy=far[0]-near[0],far[1]-near[1]
            aa=dx*dx+dy*dy;bb=2*(near[0]*dx+near[1]*dy);cc=lo*lo-target*target
            t=max(0.,min(1.,(-bb+math.sqrt(max(0.,bb*bb-4*aa*cc)))/(2*aa))) if aa else 0.
            y,z=near[0]+t*dx,near[1]+t*dy
            side=('lower' if z<0 else 'upper') if abs(z)>=abs(y) else ('yminus' if y<0 else 'yplus')
            name=PART_NAMES[i].removesuffix('_box')+'_disk_prism' if i>=3 else PART_NAMES[i]
            out.append(dict(part=name,sector=side,support_yz_local_m=[center[1]+y,center[2]+z],
                            signed_inner_clearance_with_margin_m=radius-hi-margin,
                            nominal_support_inner_clearance_m=radius-hi,margin_m=margin,
                            minimum_radial_m=lo,maximum_radial_m=hi))
        return out

    records=[]
    for row in trace['contacts']:
        center=row.get('ring_center_local_m',course['center_local_m'])
        row_course=dict(course,center_local_m=center)
        a,b=state(row['previous']),state(row['current'])
        e=RingEventsV2(1,'cpu',row_course);e.reset(torch.tensor([0]),*a,torch.zeros(1,3))
        e.sample(*b,torch.ones(1,dtype=torch.bool))
        coarse=[];fine=[];nominal=[];margin_static=[]
        # V2 uses8 swept subintervals.32 is an offline bound refinement on the
        # SAME5ms segment, not another policy rollout or another ring candidate.
        for count,dst in ((8,coarse),(32,fine)):
            for j in range(count):
                left,right=interpolation(a,b,j/count),interpolation(a,b,(j+1)/count)
                mid=interpolation(left,right,.5)
                dst.extend(details(mid,bound(left,right)))
                if count==8:
                    nominal.extend(details(mid,0.))
                    margin_static.extend(details(mid,delta))
        for endpoint in (a,b):
            nominal.extend(details(endpoint,0.));margin_static.extend(details(endpoint,delta))
        current=row['current'];p=current['position_local_m'];q=current['quaternion_wxyz']
        velocity=current.get('velocity_world_m_s',row['root_link_velocity_world_m_s'])
        group=row.get('group','unknown')
        actual_group=group
        if group in ('first_entry','first_exit'):actual_group='first_entry' if p[0]<center[0] else 'first_exit'
        hits=fine or coarse
        best=min(hits,key=lambda d:d['signed_inner_clearance_with_margin_m']) if hits else None
        records.append(dict(case_id=row['case_id'],group=actual_group,original_group=group,
                            ring_center_local_m=center,
                            control_step=row['control_step'],position_local_m=p,velocity_world_m_s=velocity,
                            tilt_deg=math.degrees(math.acos(max(-1.,min(1.,1-2*(q[1]**2+q[2]**2))))),
                            attempt_index=row['attempt_index'],v2_replayed_contact=bool(e.collision[0]),
                            swept8_contact_parts=sorted({d['part'] for d in coarse}),
                            swept32_contact_parts=sorted({d['part'] for d in fine}),
                            sectors=sorted({d['sector'] for d in hits}),dominant_support=best,
                            nominal_mesh_contact_in_sampled_segment=bool(nominal),
                            static_2mm_envelope_contact=bool(margin_static),
                            refined32_contact=bool(fine)))
    summaries={}
    for group in ('first_entry','first_exit','later'):
        selected=[r for r in records if r['group']==group]
        if not selected:continue
        def med(key,i=None):return statistics.median(r[key][i] if i is not None else r[key] for r in selected)
        summaries[group]=dict(count=len(selected),
            median_root_local_m=[med('position_local_m',i) for i in range(3)],
            median_velocity_world_m_s=[med('velocity_world_m_s',i) for i in range(3)],median_tilt_deg=med('tilt_deg'),
            ascending=sum(r['velocity_world_m_s'][2]>0 for r in selected),
            descending=sum(r['velocity_world_m_s'][2]<0 for r in selected),
            moving_backward_x=sum(r['velocity_world_m_s'][0]<0 for r in selected),
            part_counts_overlapping=dict(Counter(p for r in selected for p in r['swept32_contact_parts'])),
            sector_counts_overlapping=dict(Counter(s for r in selected for s in r['sectors'])),
            v2_replayed_contacts=sum(r['v2_replayed_contact'] for r in selected),
            refined32_contacts=sum(r['refined32_contact'] for r in selected),
            nominal_mesh_contacts=sum(r['nominal_mesh_contact_in_sampled_segment'] for r in selected),
            static_2mm_envelope_contacts=sum(r['static_2mm_envelope_contact'] for r in selected))
    centers={tuple(r['ring_center_local_m']) for r in records}
    fixed_center=next(iter(centers)) if len(centers)==1 else course['center_local_m'] if not centers else None
    return dict(scope='saved first-contact5ms segments only; no new simulation, training, or policy evaluation',
                geometry_version=course['envelope']['version'],source_episodes=trace.get('source_episodes'),
                ring_inner_lower_z_m=fixed_center[2]-radius if fixed_center else None,
                ring_inner_upper_z_m=fixed_center[2]+radius if fixed_center else None,
                safety_margin_m=delta,groups=summaries,
                original_root_flag_vs_position_mismatch=sum(r['group']!=r['original_group'] for r in records),
                interpretation='All contacts include the engineering2mm clearance margin. A nominal-mesh miss at the first safety-trigger step is not by itself a detector bug or proof of a safe full trajectory.',
                contacts=records)


def plot(result,course,path):
    centers={tuple(r['ring_center_local_m']) for r in result['contacts']}
    if len(centers)>1:raise ValueError('Different ring heights: inspect per-case contacts instead of overlaying one fixed ring')
    if centers:course=dict(course,center_local_m=list(next(iter(centers))))
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/benchmark37_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Rectangle
    colors={'first_entry':'#dc6b24','first_exit':'#287bb5','later':'#8b56a6'}
    labels={'first_entry':'First hop: before root plane','first_exit':'First hop: after root plane','later':'Later hops'}
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    center=course['center_local_m'];radius=course['inner_radius_m'];half=course['axial_thickness_m']/2;w=course['tube_radial_width_m']
    for low in (center[2]-radius-w,center[2]+radius):a.add_patch(Rectangle((center[0]-half,low),2*half,w,fc='#555555'))
    a.plot([center[0]]*2,[center[2]-radius,center[2]+radius],ls='--',color='#46936d',lw=1.4)
    b.add_patch(Circle((center[1],center[2]),radius+w,fc='#dddddd',ec='#888888'))
    b.add_patch(Circle((center[1],center[2]),radius,fc='white',ec='#46936d'))
    for group,color in colors.items():
        rows=[r for r in result['contacts'] if r['group']==group]
        a.scatter([r['position_local_m'][0] for r in rows],[r['position_local_m'][2] for r in rows],s=24,c=color,alpha=.8,label=labels[group])
        supports=[r['dominant_support']['support_yz_local_m'] for r in rows if r['dominant_support']]
        b.scatter([p[0] for p in supports],[p[1] for p in supports],s=23,c=color,alpha=.8)
    a.set(xlabel='Root x (m)',ylabel='Root z (m)',title='Root positions at first contact; ring plane dashed')
    b.set(xlabel='y (m)',ylabel='z (m)',title='Nearest annulus witnesses; 2mm margin retained')
    a.legend(fontsize=8);b.set_aspect('equal');a.grid(alpha=.15);b.grid(alpha=.15)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();trace=json.loads(args.trace.read_text());result=inspect(trace)
    target=args.trace.parent/'inspection.json';target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('groups','original_root_flag_vs_position_mismatch')},indent=2))
    if args.plot:plot(result,trace['course'],args.trace.parent/'contact_geometry.png')
    print('CONTACT_INSPECTION='+str(target))


if __name__=='__main__':main()
