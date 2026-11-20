"""Explicit opt-in correction of v1 contact false positives.

Retains the100mm engineering rotor radius,2mm margin, full-body event state
machine, course dimensions and physics. V1 artifacts keep using V1 geometry.
"""
import itertools
import math
import torch
from .ring_geometry import BOXES, FullEnvelope, RingEvents, rotate

VERSION='full_assembly_disk_prisms_swept_v2'


def convex_radial_minimum(points):
    """Closest distance to origin in a convex projection, not a loose bound."""
    points=sorted(set(points))
    if len(points)==1:return math.hypot(*points[0])
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def chain(sequence):
        result=[]
        for p in sequence:
            while len(result)>=2 and cross(result[-2],result[-1],p)<=0:result.pop()
            result.append(p)
        return result
    hull=chain(points)[:-1]+chain(reversed(points))[:-1]
    inside=len(hull)>=3;closest=math.inf
    for a,b in zip(hull,hull[1:]+hull[:1]):
        if a[0]*b[1]-a[1]*b[0] < -1e-12:inside=False
        dx,dy=b[0]-a[0],b[1]-a[1];den=dx*dx+dy*dy
        t=max(0.,min(1.,-(a[0]*dx+a[1]*dy)/den)) if den else 0.
        closest=min(closest,math.hypot(a[0]+t*dx,a[1]+t*dy))
    return 0. if inside else closest


class FullEnvelopeV2(FullEnvelope):
    def __init__(self,device):
        bits=list(itertools.product((0,1),repeat=3));locals_=[];edges_=[]
        box_edges=[(i,j) for i,a in enumerate(bits) for j,b in enumerate(bits)
                   if i<j and sum(x!=y for x,y in zip(a,b))==1]
        for b in BOXES[:3]:
            locals_.append([[b[bit][i] for i,bit in enumerate(v)] for v in bits])
            edges_.append(box_edges)
        # Circumscribed32-gon: includes the complete disk and has tangencies at
        # cardinal directions. Max excess radius <0.5mm, no fictitious square corners.
        count=32;outer=.1/math.cos(math.pi/count)
        for x,y in itertools.product((-.0813,.0813),repeat=2):
            locals_.append([[x+outer*math.cos((2*i+1)*math.pi/count),
                             y+outer*math.sin((2*i+1)*math.pi/count),z]
                            for z in (-.008,.008) for i in range(count)])
            edges_.append([(i,(i+1)%count) for i in range(count)]
                         +[(count+i,count+(i+1)%count) for i in range(count)]
                         +[(i,count+i) for i in range(count)])
        self.local=torch.tensor([v+[v[-1]]*(64-len(v)) for v in locals_],device=device)
        self.edges=torch.tensor([e+[(0,0)]*(96-len(e)) for e in edges_],device=device)
        self.moving=torch.tensor([b[2] for b in BOXES],device=device)
        self.radius=float(torch.linalg.vector_norm(self.local,dim=-1).max())+.087
        self.sweep_context=None
        self.tube_width=.025

    def vertices(self,p,q,spring):
        v=self.local[None].expand(len(p),-1,-1,-1).clone()
        v[...,2]+=spring[:,None,None]*self.moving[None,:,None]
        return rotate(q[:,None,None,:].expand(-1,len(BOXES),64,-1),v)+p[:,None,None,:]

    def spatial_slab(self,vertices,center,normal,radius,half,inflate):
        # V2 currently supports the frozen R1 vertical x-normal ring only.
        v=vertices-center[:,None,None,:];s=v[...,0];h=half+inflate[:,None,None]
        parts=torch.arange(len(BOXES),device=v.device)[:,None]
        a,b=v[:,parts,self.edges[...,0]],v[:,parts,self.edges[...,1]]
        sa,sb=a[...,0],b[...,0];den=sb-sa
        points=[v];valid=[s.abs()<=h]
        for sign in (-1,1):
            alpha=(sign*h-sa)/torch.where(den.abs()>1e-10,den,torch.ones_like(den))
            points.append(a+alpha[...,None]*(b-a))
            valid.append((alpha>=0)&(alpha<=1)&(den.abs()>1e-10))
        points=torch.cat(points,2);valid=torch.cat(valid,2)
        radial=torch.linalg.vector_norm(points[...,1:],dim=-1)
        upper=torch.where(valid,radial,-torch.inf).amax(-1)+inflate[:,None]
        intersect=valid.any(-1)
        outside=(intersect&(upper>=radius)).any(-1)
        clearance=torch.where(intersect,radius-upper,torch.inf).amin(-1)
        # A bounding sphere is only a broad-phase rejection bound. Confirm its
        # candidates against the clipped convex projection before declaring contact.
        c=v.mean(2)
        lower=torch.linalg.vector_norm(c[...,1:],dim=-1)-torch.linalg.vector_norm(v-c[:,:,None,:],dim=-1).amax(-1)-inflate[:,None]
        candidates=intersect&(upper>=radius)&(lower<=radius+self.tube_width)
        ids=torch.nonzero(candidates)
        hit=torch.zeros(len(v),dtype=torch.bool,device=v.device)
        if len(ids):
            selected=points[ids[:,0],ids[:,1],:,1:].detach().cpu().tolist()
            masks=valid[ids[:,0],ids[:,1]].cpu().tolist()
            margins=inflate[ids[:,0]].cpu().tolist();environments=ids[:,0].cpu().tolist()
            hit_ids=[]
            for env,polygon,mask,margin in zip(environments,selected,masks,margins):
                radial_min=convex_radial_minimum([tuple(p) for p,keep in zip(polygon,mask) if keep])
                if radial_min-margin<=radius+self.tube_width:hit_ids.append(env)
            if hit_ids:hit[torch.tensor(hit_ids,device=v.device)]=True
        return clearance,hit,outside

    def slab(self,vertices,center,normal,radius,half,inflate):
        clearance,hit,outside=self.spatial_slab(vertices,center,normal,radius,half,inflate)
        # Refine only ambiguous outside-aperture segments. Every subinterval
        # retains a motion bound and the full2mm margin: no point-only tunnelling.
        ids=torch.nonzero(outside).flatten()
        if not len(ids) or self.sweep_context is None:return clearance,hit,outside
        old,new,delta=self.sweep_context
        p,q,s=(x[ids] for x in old);bp,bq,bs=(x[ids] for x in new)
        bq=torch.where(((q*bq).sum(-1)<0)[:,None],-bq,bq)
        count=8;t=torch.arange(count+1,device=p.device,dtype=p.dtype)/count
        ps=p[:,None,:]+(bp-p)[:,None,:]*t[None,:,None]
        qs=torch.nn.functional.normalize(q[:,None,:]+(bq-q)[:,None,:]*t[None,:,None],dim=-1)
        ss=s[:,None]+(bs-s)[:,None]*t[None,:]
        pm=(ps[:,:-1]+ps[:,1:])/2;qm=torch.nn.functional.normalize(qs[:,:-1]+qs[:,1:],dim=-1)
        sm=(ss[:,:-1]+ss[:,1:])/2
        angle=2*torch.acos((qs[:,:-1]*qs[:,1:]).sum(-1).abs().clamp(max=1))
        margin=torch.linalg.vector_norm(ps[:,1:]-ps[:,:-1],dim=-1)/2+2*self.radius*torch.sin(angle/4)+(ss[:,1:]-ss[:,:-1]).abs()/2+delta
        refined=self.vertices(pm.flatten(0,1),qm.flatten(0,1),sm.flatten(0,1))
        cc,hh,oo=self.spatial_slab(refined,center[ids].repeat_interleave(count,0),normal[ids].repeat_interleave(count,0),radius,half,margin.flatten())
        clearance[ids]=cc.reshape(-1,count).amin(-1);hit[ids]=hh.reshape(-1,count).any(-1);outside[ids]=oo.reshape(-1,count).any(-1)
        return clearance,hit,outside


class RingEventsV2(RingEvents):
    def __init__(self,n,device,course):
        if course['normal_world']!=[1.,0.,0.]:raise ValueError('V2 fixed R1 geometry requires x-normal')
        super().__init__(n,device,course)
        self.shape=FullEnvelopeV2(device)
        self.shape.tube_width=course['tube_radial_width_m']

    def sample(self,p,q,spring,eligible):
        self.shape.sweep_context=(self.previous,(p,q,spring),self.course['delta_m'])
        return super().sample(p,q,spring,eligible)


def make_ring_events(n,device,course):
    version=course['envelope']['version']
    if version==VERSION:return RingEventsV2(n,device,course)
    if version=='full_assembly_boxes_swept_v1':return RingEvents(n,device,course)
    raise ValueError(f'Unknown frozen envelope version: {version}')
