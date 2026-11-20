"""Shape-independent crossing command geometry, in metres and body/A axes.

The measured side silhouette is extruded into two conservative boxes. This
is a task envelope, never a change to mass, inertia, USD or rotor dynamics.
Static fit is a necessary planning check, not a dynamic traversal certificate.
"""
import itertools
import math

import torch

DEFAULT_ENVELOPE = dict(width_m=.38, height_m=.30, top_z_A_m=.015,
                        bar_height_m=.030, stem_width_m=.060,
                        bar_depth_m=.38, stem_depth_m=.060,
                        source='width/height: user; thickness/depth/offset: provisional model enclosure')


def quaternion_from_rpy(rpy):
    """Active body->world Rz(yaw) Ry(pitch) Rx(roll), wxyz, radians."""
    r, p, y = (rpy / 2).unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                        cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy), -1)


def quaternion_product(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def rpy_from_quaternion(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                        torch.asin((2*(w*y-z*x)).clamp(-1, 1)),
                        torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))), -1)


def boxes(config=None):
    c = DEFAULT_ENVELOPE if config is None else config
    w, h, top = c['width_m'], c['height_m'], c['top_z_A_m']
    t, s, d, sd = (c[k] for k in ('bar_height_m', 'stem_width_m', 'bar_depth_m', 'stem_depth_m'))
    if not all(math.isfinite(v) for v in (w,h,top,t,s,d,sd)) or not (0 < t < h and 0 < s <= w and d > 0 and sd > 0):
        raise ValueError('Invalid T envelope dimensions')
    return [([-d/2, -w/2, top-t], [d/2, w/2, top]),
            ([-sd/2, -s/2, top-h], [sd/2, s/2, top-t])]


def vertices(config=None, *, device='cpu', dtype=torch.float32):
    return torch.tensor([[b[bit][i] for i, bit in enumerate(bits)]
                         for b in boxes(config) for bits in itertools.product((0, 1), repeat=3)],
                        device=device, dtype=dtype)


def aperture_center_in_A(shape, config=None):
    """Upright, aligned placement; optimal circle centre or rectangle midpoint.

    T plane is body YZ, +X is forward. The circle solution equalizes top
    outer and bottom stem corners; clamping covers unusually thick bars.
    """
    c = DEFAULT_ENVELOPE if config is None else config
    boxes(c)
    top, bottom = c['top_z_A_m'], c['top_z_A_m']-c['height_m']
    if shape == 'rectangle':
        z = (top+bottom)/2
    elif shape == 'circle':
        w, s = c['width_m']/2, c['stem_width_m']/2
        equal = (w*w-s*s+top*top-bottom*bottom)/(2*(top-bottom))
        z = max((top+bottom)/2, min(top-c['bar_height_m']/2, equal))
    else:
        raise ValueError('Supported apertures: circle, rectangle')
    return [0., 0., z]


def target_A_position(center_world, target_q, shape, config=None):
    from .ring_geometry import rotate
    offset = center_world.new_tensor(aperture_center_in_A(shape, config)).expand_as(center_world)
    return center_world-rotate(target_q, offset)


def projected_clearance(position_A, q, aperture_center, aperture_q, aperture, config=None):
    """Conservative *whole-volume projection* clearance for convex openings.

    Aperture frame: X normal/forward, Y horizontal, Z up. Checking all box
    vertices suffices because disk and rectangle openings are convex. This
    deliberately checks the entire projection, not just a zero-thickness slice.
    Caller must additionally check the trajectory and finite frame thickness.
    """
    from .ring_geometry import rotate
    from .ring_obs73 import inverse_rotate
    local = vertices(config, device=position_A.device, dtype=position_A.dtype)
    world = rotate(q[:, None].expand(-1,len(local),-1), local[None].expand(len(q),-1,-1))
    rel = world+position_A[:, None]-aperture_center[:, None]
    rel = inverse_rotate(aperture_q[:, None].expand(-1,len(local),-1), rel)
    margin = float(aperture.get('margin_m', .02))
    if margin < 0 or not math.isfinite(margin):
        raise ValueError('Nonnegative finite margin required')
    if aperture['shape'] == 'circle':
        radius = float(aperture['radius_m'])
        if radius <= margin: raise ValueError('Circle radius must exceed margin')
        return radius-rel[..., 1:].norm(dim=-1).amax(-1)-margin
    if aperture['shape'] == 'rectangle':
        half = rel.new_tensor([aperture['width_m']/2, aperture['height_m']/2])
        if not bool((half > margin).all()): raise ValueError('Rectangle must exceed margin')
        return (half-rel[..., 1:].abs()).amin(dim=(-1,-2))-margin
    raise ValueError('Unsupported aperture shape')
