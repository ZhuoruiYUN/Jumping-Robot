"""Conservative full-assembly boxes, finite annulus and 5ms swept-volume events.

Each CAD/collision primitive is contained by these boxes. Rotor disks are
enclosed by squares: conservative, including their interior, not point samples.
R1 is a virtual ring; collision here means envelope/annulus intersection.
"""
import itertools
import math
import torch

VERSION = 'full_assembly_boxes_swept_v1'
# Composed calibrated USD bounds in Body/A axes, q=0. Springs lie inside base.
BOXES = [
    ([-.09630001, -.09630001, -.00600001], [.09630001, .09630001, .00600001], False),
    ([-.02674208, -.01001132, -.14400001], [.02670056, .00998869, -.00599998], False),
    ([-.02725001, -.01250001, -.28500001], [.02725001, .01250001, -.11899999], True),
]
# No rotor sweep collider or measured blade diameter exists in the source USD.
# 100mm radius / +/-8mm Z is an explicitly provisional conservative engineering
# allowance, NOT a claimed hardware measurement; freeze it with every course.
for x, y in itertools.product((-.0813, .0813), repeat=2):
    BOXES.append(([x-.100, y-.100, -.008], [x+.100, y+.100, .008], False))


def envelope_report():
    lo = [min(b[0][i] for b in BOXES) for i in range(3)]
    hi = [max(b[1][i] + (.087 if b[2] and i == 2 else 0) for b in BOXES) for i in range(3)]
    dims = [h-l for l,h in zip(lo,hi)]
    diameter = math.sqrt(sum(v*v for v in dims))
    compressed_lo = [min(b[0][i]+(.087 if b[2] and i==2 else 0) for b in BOXES) for i in range(3)]
    return dict(version=VERSION, bounds_q0_min_m=lo, bounds_all_q_max_m=hi,
                dimensions_q0_m=dims, dimensions_q087_m=[h-l for h,l in zip(hi,compressed_lo)],
                sphere_center_body_m=[(l+h)/2 for l,h in zip(lo,hi)], D_env_m=diameter,
                sphere_construction='half diagonal of union AABB over q=0..0.087m',
                D0_m=1.15*diameter, rotor_radius_m=.100, rotor_half_height_m=.008,
                rotor_source='provisional engineering allowance; measured blade sweep unavailable',
                asset_audit='composed HopperCalibrated20261028.usda: body/arms/motors/base/leg/rod/foot collision present; rotor sweep absent',
                delta_m=.002, delta_source='simulation numerical engineering margin only; not a hardware error budget')


def rotate(q, v):
    t = 2*torch.cross(q[...,1:], v, dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)


class FullEnvelope:
    def __init__(self, device):
        bits = list(itertools.product((0,1),repeat=3))
        self.local = torch.tensor([[[b[bit][i] for i,bit in enumerate(v)] for v in bits] for b in BOXES], device=device)
        self.moving = torch.tensor([b[2] for b in BOXES], device=device)
        self.edges = torch.tensor([(i,j) for i,a in enumerate(bits) for j,b in enumerate(bits)
                                   if i<j and sum(x!=y for x,y in zip(a,b))==1],device=device)
        self.radius = float(torch.linalg.vector_norm(self.local,dim=-1).max())+.087

    def vertices(self,p,q,spring):
        v=self.local[None].expand(len(p),-1,-1,-1).clone()
        v[:,:,:,2] += spring[:,None,None]*self.moving[None,:,None]
        return rotate(q[:,None,None,:].expand(-1,len(BOXES),8,-1),v)+p[:,None,None,:]

    def slab(self,vertices,center,normal,radius,half,inflate):
        v=vertices-center[:,None,None,:]
        s=(v*normal[:,None,None,:]).sum(-1)
        h=half+inflate[:,None,None]
        a,b=v[:,:,self.edges[:,0]],v[:,:,self.edges[:,1]]
        sa,sb=s[:,:,self.edges[:,0]],s[:,:,self.edges[:,1]]
        points=[v]; valid=[s.abs()<=h]
        for sign in (-1,1):
            alpha=(sign*h-sa)/torch.where((sb-sa).abs()>1e-10,sb-sa,torch.ones_like(sa))
            points.append(a+alpha[...,None]*(b-a))
            valid.append((alpha>=0)&(alpha<=1)&((sb-sa).abs()>1e-10))
        points=torch.cat(points,2); valid=torch.cat(valid,2)
        axial=(points*normal[:,None,None,:]).sum(-1)
        radial=torch.linalg.vector_norm(points-axial[...,None]*normal[:,None,None,:],dim=-1)
        upper=torch.where(valid,radial,-torch.inf).amax(-1)+inflate[:,None]
        intersect=valid.any(-1)
        clearance=torch.where(intersect,radius-upper,torch.inf).amin(-1)
        # Lower bound for radial distance of every point in each box. This
        # distinguishes clearly external bypasses from conservative tube hits.
        c=v.mean(2); cs=(c*normal[:,None,:]).sum(-1)
        lower=torch.linalg.vector_norm(c-cs[...,None]*normal[:,None,:],dim=-1)
        lower-=torch.linalg.vector_norm(v-c[:,:,None,:],dim=-1).amax(-1)+inflate[:,None]
        hit=(intersect&(upper>=radius)&(lower<=radius+.025)).any(-1)
        outside=(intersect&(upper>=radius)).any(-1)
        return clearance,hit,outside


class RingEvents:
    def __init__(self,n,device,course):
        self.shape=FullEnvelope(device); self.course=course
        self.center=torch.zeros(n,3,device=device);self.normal=torch.zeros_like(self.center);self.normal[:,0]=1
        self.started=torch.zeros(n,dtype=torch.bool,device=device)
        self.passed=self.started.clone();self.failed=self.started.clone();self.collision=self.started.clone()
        self.root_crossed=self.started.clone();self.reverse=self.started.clone();self.bypass=self.started.clone()
        self.min_clearance=torch.full((n,),torch.inf,device=device)
        self.previous=None
        self.event=self.started.clone()

    def reset(self,ids,p,q,spring,origin):
        self.center[ids]=origin[ids]+p.new_tensor(self.course['center_local_m'])
        for b in (self.started,self.passed,self.failed,self.collision,self.root_crossed,self.reverse,self.bypass,self.event):b[ids]=False
        self.min_clearance[ids]=torch.inf
        if self.previous is None:self.previous=(p.clone(),q.clone(),spring.clone())
        else:
            for dst,src in zip(self.previous,(p,q,spring)):dst[ids]=src[ids]

    def sample(self,p,q,spring,eligible):
        old,oq,os=self.previous
        r=self.course['inner_radius_m'];half=self.course['axial_thickness_m']/2
        delta=self.course['delta_m']
        # Hemisphere-correct midpoint orientation and a bound covering the
        # entire swept box between the two physical samples. Large steps become
        # conservatively blocked instead of silently tunnelling through a tube.
        dot=(oq*q).sum(-1);nq=torch.where((dot<0)[:,None],-q,q)
        mq=torch.nn.functional.normalize(oq+nq,dim=-1)
        angle=2*torch.acos(dot.abs().clamp(max=1))
        inflation=torch.linalg.vector_norm(p-old,dim=-1)/2+2*self.shape.radius*torch.sin(angle/4)+(spring-os).abs()/2+delta
        mid=self.shape.vertices((p+old)/2,mq,(spring+os)/2)
        clearance,hit,outside=self.shape.slab(mid,self.center,self.normal,r,half,inflation)
        v=self.shape.vertices(p,q,spring)-self.center[:,None,None,:]
        s=(v*self.normal[:,None,None,:]).sum(-1)
        low=s.amin((1,2));high=s.amax((1,2))
        oldv=self.shape.vertices(old,oq,os)-self.center[:,None,None,:]
        oldhigh=(oldv*self.normal[:,None,None,:]).sum(-1).amax((1,2))
        active=eligible&~self.failed&~self.passed
        self.started |= active&(oldhigh < -half-delta)
        oldroot=((old-self.center)*self.normal).sum(-1)
        root=((p-self.center)*self.normal).sum(-1)
        self.root_crossed |= active&(oldroot<0)&(root>=0)
        self.reverse |= active&self.started&(oldroot>=0)&(root<0)
        self.bypass |= active&self.started&outside&~hit
        # Contact remains terminal even outside the designated first flight,
        # including a later approach after an earlier miss or completed pass.
        self.collision |= hit
        self.failed |= active&(hit|outside|self.reverse)
        window=active&self.started
        self.min_clearance=torch.where(window,torch.minimum(self.min_clearance,clearance),self.min_clearance)
        success=window&~self.failed&(low>half+delta)
        self.passed |= success; self.event |= success
        self.previous=(p.clone(),q.clone(),spring.clone())
        return success
