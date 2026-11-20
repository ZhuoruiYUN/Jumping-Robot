"""Motion perpendicular to a commanded world-frame hop, and episode rigid-body DR."""
import torch


def flight_plane_errors(up, omega, foot_velocity, command, moving):
    """Constrain both horizontal components in a hold and the cross-track tilt in a step.

    Horizontal angular velocity rotates the vertical axis by 90 degrees. A
    rotation about the travel axis therefore generates unwanted cross-track lean.
    """
    along = torch.nn.functional.normalize(command, dim=-1)
    cross = torch.stack((-along[:,1], along[:,0]), dim=-1)
    def magnitude(vector, axis):
        return torch.where(moving, (vector*axis).sum(-1).abs(), vector.norm(dim=-1))
    lean = torch.atan2(magnitude(up[:,:2], cross), up[:,2])
    return lean, magnitude(omega[:,:2], along), magnitude(foot_velocity[:,:2], cross)


def travel_peak_cost(lean, airborne_travel, previous, threshold, weight):
    """Charge only the increase in squared excess for this physical hop."""
    excess = (lean - threshold).clamp_min(0.).square()
    peak = torch.where(airborne_travel, torch.maximum(previous, excess), previous)
    return -weight * (peak - previous), peak


def set_rigid_scales(env, ids, scales):
    """Apply two independent per-robot factors to nominal physical masses and inertias."""
    view = env._robot.root_physx_view
    ids = ids.cpu()
    scales = scales.cpu()
    mass = view.get_masses()
    inertia = view.get_inertias()
    mass[ids] = env._robot.data.default_mass[ids] * scales[:,0,None]
    inertia[ids] = env._robot.data.default_inertia[ids] * scales[:,1,None,None]
    view.set_masses(mass, ids)
    view.set_inertias(inertia, ids)
    env.rigid_scales[ids] = scales


def restore_rigid_scales(env, ids, cases):
    if not hasattr(env,'rigid_scales'):
        return
    # Legacy case files describe force/torque gains only and used nominal rigid bodies.
    scales = torch.tensor([c.get('physical_mass_inertia_scale',[1.,1.]) for c in cases])
    set_rigid_scales(env, ids, scales)


def audit_rigid_scales(env):
    scales = env.rigid_scales
    lo,hi = env.course['rigid_body_randomization']['factor_range']
    assert torch.isfinite(scales).all() and (scales>=lo).all() and (scales<=hi).all()
    data = env._robot.data
    view = env._robot.root_physx_view
    torch.testing.assert_close(view.get_masses(), data.default_mass*scales[:,0,None])
    torch.testing.assert_close(view.get_inertias(), data.default_inertia*scales[:,1,None,None],atol=1e-10,rtol=2e-5)
    return dict(mass_factor_minmax=[float(scales[:,0].min()),float(scales[:,0].max())],
                inertia_factor_minmax=[float(scales[:,1].min()),float(scales[:,1].max())],
                bodies=list(env._robot.body_names), verified_physx_readback=True)


def flight_plane_report(rows, limits, modes):
    import math
    report = {}
    for col,mode in enumerate(('hold','travel')):
        if mode not in modes:
            continue
        totals = torch.tensor([r['flight_plane_totals'][col] for r in rows],dtype=torch.float64)
        n,angle,rate,speed = totals[:,:4].sum(0).tolist()
        values = dict(lean_rms_deg=math.degrees(math.sqrt(angle/n)) if n else None,
                      lean_peak_deg=math.degrees(totals[:,4].max().item()) if n else None,
                      rate_rms_rad_s=math.sqrt(rate/n) if n else None,
                      leg_speed_rms_m_s=math.sqrt(speed/n) if n else None)
        report[mode] = dict(samples=int(n), **values,
                            passed=bool(n and all(values[k]<=v for k,v in limits.items())))
    return report
