"""Motion reward and optional flight-control extensions for the OBS73 hopper."""
import math

import torch
from .all_phase_yaw import yaw_envelope_cost

PHASE_COSTS=(
    'attitude_penalty','vertical_tilt','vertical_roll_pitch_rate',
    'vertical_leg_swing','vertical_rotation_work','vertical_outward_rotation',
    'vertical_near_upright_braking')
REMOVED_COSTS=('lateral_vel_p','vertical_flight_quality_discount','vertical_settled_pitch_cost')
REWEIGHTED=PHASE_COSTS+REMOVED_COSTS


def late_weight(phase,spring,clearance,vz,window,landing_time_s=None,gravity=9.81):
    # Stance and true touchdown retain original stabilization. During flight,
    # use only descent/actual foot clearance, never an external clock/command.
    stance=(phase==1)|(spring>.002)
    descent=(vz<0)&((phase==2)|(phase==3))
    ramp=(1-clearance/window).clamp(0.,1.)
    if landing_time_s is not None:
        # Rationalized descending ballistic clock avoids cancellation near
        # contact. This only schedules reward costs; it is not a controller
        # or an accurate impact predictor under rotor thrust/body rotation.
        height=clearance.clamp(min=0.)
        speed=vz.abs()
        time=2*height/(torch.sqrt(speed.square()+2*gravity*height)+speed).clamp(min=1e-9)
        ramp=(1-time/landing_time_s).clamp(0.,1.)
    return torch.where(stance,1.,torch.where(descent,ramp,0.))


def outward_weight(weight,tilt,config):
    c=config.get('transport_outward_braking')
    if not c:return weight
    ramp=((torch.rad2deg(tilt)-c['start_tilt_deg'])
        /(c['full_tilt_deg']-c['start_tilt_deg'])).clamp(0.,1.)
    return torch.maximum(weight,ramp)


def reward_deltas(ledger,moving,weight,tilt,dt,config,terminal_hold=None):
    result={}
    for name,value in ledger.items():
        factor=weight if name in PHASE_COSTS else torch.zeros_like(weight)
        if name=='vertical_outward_rotation':factor=outward_weight(weight,tilt,config)
        result[name]=torch.where(moving,(factor-1.)*value,0.)
    # Prevent unlimited early-flight leaning without restoring a penalty for
    # every useful degree of tilt. This is reward shaping, not a termination.
    threshold=math.radians(config['free_tilt_deg'])
    if terminal_hold is not None:
        threshold=torch.where(terminal_hold,tilt.new_tensor(math.radians(config['terminal_hold_tilt_deg'])),threshold)
    excess=(tilt-threshold).clamp(min=0.)
    result['moving_excess_tilt']=torch.where(moving,
        -config['excess_tilt_weight']*excess.square()*dt,0.)
    return result


def landing_guidance(foot_xy,clearance,velocity,root_height,target_xy,target_height,
                     phase,moving,valid,gravity,dt,config,remaining_time_floor=None,
                     projection_xy=None,blend_mask=None,straight_precision_mask=None,
                     powered_clock=None, first_clock_cap=None):
    """Reward-only approximate ballistic guidance, never a motor/pose command.

    The clock is recomputed from current state, including missed-hop retries.
    Rotor thrust, drag, spring launch and body rotation make this an approximate
    learning reference, not a predicted collision time or reachability proof.
    """
    vz=velocity[:,2]
    rise=(target_height+config['apex_margin_m']-root_height).clamp(min=0.)
    stance_time=2*torch.sqrt(2*rise/gravity)
    flight_time=(vz+torch.sqrt(vz.square()+2*gravity*clearance.clamp(min=0.)))/gravity
    remaining=torch.where(phase==1,stance_time,flight_time).clamp(
        min=config['minimum_time_s'],max=config['maximum_time_s'])
    if remaining_time_floor is not None:
        remaining=torch.maximum(remaining,remaining_time_floor).clamp(max=config['maximum_time_s'])
    if powered_clock is not None:
        from .imu_link_flight import bounded_clock
        estimated_time, blend_weight, limit = powered_clock
        remaining = bounded_clock(remaining, estimated_time, blend_weight,
                                  config['minimum_time_s'], config['maximum_time_s'], limit)
    if first_clock_cap is not None:
        cap, strength = first_clock_cap
        remaining = torch.lerp(remaining, torch.minimum(remaining, cap.clamp_min(config['minimum_time_s'])),
                               strength)
    physical_clock_active=torch.zeros_like(phase,dtype=torch.bool)
    if config.get('projection_clock_version')=='zero_at_contact_separate_divisor_v2':
        # The150ms velocity denominator guard is not physical flight time.
        # On descent use a cancellation-safe clock which reaches zero at
        # contact. A past long flight must not invent time after this impact.
        descending=(vz<=0)&((phase==2)|(phase==3))
        physical_clock_active=descending
        h=clearance.clamp(min=0.)
        impact=2*h/(torch.sqrt(vz.square()+2*gravity*h)+vz.abs()).clamp(min=1e-9)
        remaining=torch.where(descending,impact.clamp(max=config['maximum_time_s']),remaining)
    origin=foot_xy if projection_xy is None else projection_xy
    blend_time=config.get('measured_foot_blend_s',0.)
    if blend_time:
        # The intended leg is a placement reference early in flight. Close
        # to contact, the measured foot offset better predicts the landing.
        blend=(1.-(remaining/blend_time).square()).clamp(0.,1.)
        blend=torch.where(physical_clock_active & moving,blend,0.)
        if blend_mask is not None:
            blend=torch.where(blend_mask,blend,0.)
        origin=torch.lerp(origin,foot_xy,blend[:,None])
    divisor=remaining.clamp(min=config['minimum_time_s'])
    desired=(target_xy-origin)/divisor[:,None]
    speed=desired.norm(dim=-1,keepdim=True).clamp(min=1e-9)
    desired=desired*torch.clamp(config['speed_limit_m_s']/speed,max=1.)
    descending=(vz<0)&((phase==2)|(phase==3))
    weight=torch.where(descending,(clearance/config['landing_clearance_m']).clamp(0.,1.),1.)
    velocity_cost=-config['velocity_weight']*(velocity[:,:2]-desired).square().sum(-1)*weight*dt
    # A task-only alternative to velocity tracking: give the same forecast
    # position miss the same cost early and late (before the contact fade).
    # Constant horizontal velocity plus the existing approximate clock is a
    # learning reference, not a thrust/rotation-aware touchdown predictor.
    projected_xy=origin+velocity[:,:2]*remaining[:,None]
    projected_error=(projected_xy-target_xy).norm(dim=-1)
    if config.get('objective')=='projected_touchdown_error_v1':
        width=config['projected_error_width_m']
        if straight_precision_mask is not None:
            # Increase sensitivity only where the measured touchdown forecast is reliable.
            peak=(1.-(remaining-.25).abs()/.10).clamp(0.,1.)
            blend=torch.where(straight_precision_mask & descending,peak,0.)
            width=width+(config['landing_width_m']-width)*blend
        if config.get('precision_to_contact'):
            # As the measured foot takes over, tighten the spatial tolerance and
            # retain guidance through descent. Score physical contact separately.
            arrival = torch.where(descending & (clearance > 0.) & ~valid, blend, 0.)
            width = width + (config['landing_width_m'] - width) * arrival
            weight = torch.lerp(weight, torch.ones_like(weight), arrival)
        scaled=projected_error/width
        # Stable smooth-L1 form, including tiny errors: sqrt(1+x²)-1.
        shape=scaled.square()/(torch.sqrt(1+scaled.square())+1)
        velocity_cost=-config['velocity_weight']*shape*weight*dt
    error=(target_xy-foot_xy).norm(dim=-1)
    # Smooth L1-like error grows approximately linearly outside the tolerance;
    # it has no10cm gate and no Gaussian tail that vanishes on a large miss.
    landing_cost=-config['landing_weight']*(torch.sqrt(1+(error/config['landing_width_m']).square())-1)
    terms=dict(moving_velocity_guidance=torch.where(moving,velocity_cost,0.),
        moving_landing_error=torch.where(moving&valid,landing_cost,0.))
    return terms,dict(desired_velocity_xy=desired,remaining_time_s=remaining,
        division_time_s=divisor,projection_clock_active=physical_clock_active,
        flight_weight=weight,foot_error_m=error,projected_foot_xy=projected_xy,
        projected_error_m=projected_error,projection_origin_xy=origin)


def touchdown_rate_cost(omega_body,moving,valid,config):
    """Once at impact; retain useful yaw and a small roll/pitch rate allowance."""
    excess=(omega_body[:,:2].norm(dim=-1)-config['free_rate_rad_s']).clamp(min=0.)
    cost=-config['weight']*(torch.sqrt(1+(excess/config['rate_width_rad_s']).square())-1)
    return torch.where(moving&valid,cost,0.)


def transport_body_rate_cost(omega_body,phase,spring,spring_rate,clearance,
                             moving,landing_weight,dt,config):
    """Soft speed envelope in the gap left by phase-reweighted rate costs.

    The complement of landing weight avoids adding a second full rate cost
    near touchdown. No pose reference: steady lean and body yaw are free here.
    This shapes future training; it never modifies policy motor commands.
    """
    excess=(omega_body[:,:2].norm(dim=-1)-config['free_rate_rad_s']).clamp(min=0.)
    active=(moving&((phase==2)|(phase==3))&(clearance>0)
        &(spring<=config['free_spring_q_max_m'])
        &(spring_rate.abs()<=config['free_spring_qdot_max_m_s']))
    if config.get('version')=='free_air_roll_pitch_excess_rate_v2':
        # Internal airborne spring motion must not switch off the body-rate
        # objective. Foot clearance excludes stance/impact without this gap.
        active=moving&((phase==2)|(phase==3))&(clearance>config['minimum_clearance_m'])
    cost=-config['weight']*(1-landing_weight)*excess.square()*dt
    return torch.where(active,cost,0.),dict(active=active,excess_rate=excess)


def launch_rotation_cost(omega_body,launch,config):
    """Charge the existing once-only free-liftoff event, without a dt factor."""
    excess=(omega_body[:,:2].norm(dim=-1)-config['free_rate_rad_s']).clamp(min=0.)
    return torch.where(launch,-config['weight']*excess.square(),0.)


def outward_tilt_work_cost(quaternion, omega_body, moment_body, tilt, phase, spring,
                           clearance, moving, landing_weight, dt, native_config, config):
    """Charge motor work along the direction that increases the tilt cone."""
    w, x, y, z = quaternion.unbind(-1)
    # Gradient of 1-Rzz in body roll/pitch. Normalizing makes the
    # projected angular velocity the tilt rate in radians per second.
    gradient = torch.stack((2*(y*z + w*x), 2*(w*y - x*z)), dim=-1)
    direction = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    tilt_rate = (direction * omega_body[:, :2]).sum(dim=-1)
    outward_moment = (direction * moment_body[:, :2]).sum(dim=-1)
    excess_rate = (tilt_rate - config['free_outward_rate_rad_s']).clamp_min(0.)
    tilt_weight = ((torch.rad2deg(tilt) - config['start_tilt_deg'])
        / (config['full_tilt_deg'] - config['start_tilt_deg'])).clamp(0., 1.)
    native_weight = torch.where(spring <= native_config['free_spring_q_max_m'],
                                landing_weight, 0.)
    active = moving & ((phase == 2) | (phase == 3)) & (clearance > config['minimum_clearance_m'])
    work = excess_rate * outward_moment.clamp_min(0.) * dt
    cost = -native_config['weight'] * tilt_weight * (1 - native_weight) * work
    return torch.where(active, cost, 0.), active


def launch_velocity_cost(velocity_xy,target_velocity_xy,flight_time,launch,weight,config=None):
    """Once at physical release, using the destination latched at contact.

    Optional endpoint units expose the landing consequence of launch velocity
    error. This constant-speed/previous-flight proxy is a learning signal,
    not a prediction of actual powered flight or an action correction.
    """
    squared=(velocity_xy-target_velocity_xy).square().sum(-1)
    if config is not None:
        scaled_squared=squared*(flight_time/config['width_m']).square()
        # Stable smooth-L1 expression; zero has finite gradients and the
        # large-error tail is linear instead of an unbounded quadratic gain.
        squared=scaled_squared/(torch.sqrt(1+scaled_squared)+1)
    return torch.where(launch,-weight*squared,0.)


def long_pair_landing_cost(incoming, edge, foot_minus_goal, eligible, min_length, radius, weight):
    """Once on the second long landing, penalize both overshoot and undershoot along its direction."""
    direction = torch.nn.functional.normalize(edge,dim=-1)
    along = (foot_minus_goal*direction).sum(-1)
    active = eligible & (incoming.norm(dim=-1)>=min_length-1e-6) & (edge.norm(dim=-1)>=min_length-1e-6)
    return torch.where(active, weight*torch.expm1(-.5*(along/radius).square()), 0.)


def pair_release_cost(velocity, target, axis, flight, enabled, radius, config):
    along = (velocity*axis).sum(-1)
    requested = (target*axis).sum(-1)
    active = enabled & (requested>0.)
    deficit = (config['speed_fraction']*requested - along).clamp_min(0.)
    squared = (deficit*flight/radius).square()
    cost = -config['weight']*squared/(torch.sqrt(1.+squared)+1.)
    return torch.where(active,cost,0.), active, along


def launch_direction_cost(velocity_xy,target_velocity_xy,launch,config):
    """Once at release: penalize backwards and cross-track rebound velocity.

    The destination is the actual advance/retry goal latched at touchdown.
    Stable forward travel and near-zero hold references receive no cost.
    """
    speed=target_velocity_xy.norm(dim=-1)
    direction=target_velocity_xy/speed.clamp(min=1e-9)[:,None]
    along=(velocity_xy*direction).sum(-1)
    cross=velocity_xy-along[:,None]*direction
    error=(-along).clamp(min=0).square()+cross.square().sum(-1)
    return torch.where(launch&(speed>=config['minimum_target_speed_m_s']),-config['weight']*error,0.)


def launch_velocity_window_cost(velocity_xy,target_velocity_xy,flight_time,launch_step,now,
                                moving,phase,clearance,dt,weight,config):
    """Soft release-state learning over delayed motor response, not control.

    Normalize by the integer number of control samples: a constant error
    over the complete window costs exactly one endpoint-distance loss.
    The target is latched by the preceding real advance/retry at contact.
    """
    ticks=math.ceil(config['window_s']/dt-1e-9)
    elapsed=now-launch_step
    active=(moving&(launch_step>=0)&(elapsed>=0)&(elapsed<ticks)
        &((phase==2)|(phase==3))&(clearance>config['minimum_clearance_m']))
    error_m=(velocity_xy-target_velocity_xy).norm(dim=-1)*flight_time
    scaled_squared=(error_m/config['width_m']).square()
    cost=-weight*scaled_squared/(torch.sqrt(1+scaled_squared)+1)/ticks
    return torch.where(active,cost,0.),dict(active=active,endpoint_error_m=error_m)


def impact_alignment_cost(root_position,foot_position,velocity,phase,spring,spring_rate,
                          clearance,moving,weight,dt,config):
    """Precontact approach proxy, not an exact articulated impact model.

    Motion along the leg may include horizontal speed and body lean. Penalize
    only transverse root-A approach speed during the existing landing window;
    contact impulses, compression and spring release receive no new cost.
    """
    axis=root_position-foot_position
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp(min=1e-6)
    transverse=velocity-(velocity*axis).sum(-1,keepdim=True)*axis
    speed_squared=transverse.square().sum(-1)
    active=(moving&((phase==2)|(phase==3))&(clearance>0)&(velocity[:,2]<0)
        &(spring<=config['free_spring_q_max_m'])
        &(spring_rate.abs()<=config['free_spring_qdot_max_m_s']))
    cost=torch.where(active,-config['weight']*speed_squared*weight*dt,0.)
    return cost,dict(active=active,transverse_speed_m_s=torch.sqrt(speed_squared))


def heading_motion(q,omega):
    """World heading and its actual derivative, not body wz at nonzero tilt."""
    from .ring_geometry import rotate
    x=q.new_zeros(q.shape[0],3);x[:,0]=1
    forward=rotate(q,x)
    derivative=rotate(q,torch.linalg.cross(omega,x))
    heading=torch.atan2(forward[:,1],forward[:,0])
    rate=(forward[:,0]*derivative[:,1]-forward[:,1]*derivative[:,0])/forward[:,:2].square().sum(-1).clamp(min=.01)
    return heading,rate


def yaw_contact_jump_cost(previous_spin, spin, previously_loaded, loaded, config):
    """Charge once at spring engagement for a jump in spin about world vertical."""
    excess = ((spin - previous_spin).abs() - config['free_rate_jump_rad_s']).clamp_min(0.)
    quadratic = excess.clamp(max=1.)
    cost = -config['weight'] * (quadratic.square() + 2. * (excess - quadratic))
    return torch.where(loaded & ~previously_loaded, cost, 0.)


def yaw_peak_increment(error,rate,previous,moving,horizon,limit_deg,weight):
    """Nonpositive increments of predicted peak excess within a bounce cycle.

    The unwrapped constant-rate endpoint cannot hide a full turn. Include
    the present heading so approaching the reference at high rate does not
    erase an existing excursion. No reward is paid for reducing the peak.
    """
    predicted=torch.maximum(error.abs(),(error+horizon*rate).abs())
    excess=(predicted-math.radians(limit_deg)).clamp(min=0.).square()
    peak=torch.where(moving,torch.maximum(previous,excess),previous)
    return -weight*(peak-previous),peak


def predicted_max_tilt(q,omega,horizon):
    """Maximum cone angle over constant-body-rate rotation, including extrema.

    Vertical projection has form A+B*cos(theta)+C*sin(theta). Its minimum
    over [0,|omega|*horizon] gives the maximum tilt, including full revolutions.
    This predicts unchanged angular velocity, not a controlled future pose.
    """
    w,x,y,z=q.unbind(-1)
    up=torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)
    speed=omega.norm(dim=-1)
    axis=omega/speed.clamp(min=1e-9)[:,None]
    constant=(up*axis).sum(-1)*axis[:,2]
    cosine=up[:,2]-constant
    sine=up[:,0]*axis[:,1]-up[:,1]*axis[:,0]
    angle=speed*horizon
    end=constant+cosine*angle.cos()+sine*angle.sin()
    minimum=torch.minimum(up[:,2],end)
    extremum=torch.remainder(torch.atan2(sine,cosine)+math.pi,2*math.pi)
    internal=constant-torch.sqrt(cosine.square()+sine.square())
    minimum=torch.where(angle>=extremum,torch.minimum(minimum,internal),minimum)
    return torch.acos(minimum.clamp(-1.,1.))


def predicted_rotation_cost(q,omega,phase,spring,spring_rate,clearance,moving,
                            free_tilt_deg,dt,config):
    current=torch.acos((1-2*q[:,1:3].square().sum(-1)).clamp(-1.,1.))
    predicted=predicted_max_tilt(q,omega,config['response_horizon_s'])
    limit=math.radians(free_tilt_deg)
    # Current cone excess is already charged by moving_excess_tilt.
    extra=((predicted-limit).clamp(min=0).square()-(current-limit).clamp(min=0).square()).clamp(min=0)
    active=(moving&((phase==2)|(phase==3))&(clearance>0)
        &(spring<=config['free_spring_q_max_m'])&(spring_rate.abs()<=config['free_spring_qdot_max_m_s']))
    return torch.where(active,-config['weight']*extra*dt,0.),dict(
        active=active,current_tilt=current,predicted_max_tilt=predicted,incremental_excess_squared=extra)


def rebound_approach(root,foot,velocity,outgoing,stance_time,config):
    """Raibert-inspired neutral foot placement plus velocity correction.

    Reference only: fixed-body/leg rotor hopper is not an actuated-hip SLIP.
    Do not reverse the entire incoming velocity as a radial-only impact goal.
    """
    length=(root-foot).norm(dim=-1).clamp(min=.05)
    lead=.5*stance_time[:,None]*velocity[:,:2]+config['velocity_gain_s']*(velocity[:,:2]-outgoing)
    limit=length*math.sin(math.radians(config['touchdown_tilt_limit_deg']))
    lead=lead*(limit/lead.norm(dim=-1).clamp(min=1e-9)).clamp(max=1.)[:,None]
    horizontal=-lead/length[:,None]
    normal=torch.cat((horizontal,torch.sqrt((1-horizontal.square().sum(-1)).clamp(min=0))[:,None]),-1)
    return normal,lead


def impulse_approach(velocity,outgoing,vertical_change,limit_deg):
    """Leg axis for the requested velocity change across spring contact."""
    horizontal=outgoing-velocity[:,:2]
    magnitude=horizontal.norm(dim=-1,keepdim=True)
    limit=vertical_change*math.tan(math.radians(limit_deg))
    horizontal=horizontal*(limit/magnitude.clamp_min(1.e-8)).clamp(max=1.)
    return torch.nn.functional.normalize(torch.cat((horizontal,torch.full_like(magnitude,vertical_change)),dim=-1),dim=-1)


def consistent_outgoing_velocity(foot_xy,clearance,velocity,current,next_goal,
                                 flight_time,tilt,gravity,speed_limit,config,*,blend_after_gate=False,foot_limit_m=.1):
    """Reward-only preview of the same retain/advance rule used at impact.

    Constant translation of the current foot offset is only a local preview;
    actual measured touchdown still decides navigation. Smooth confidence
    avoids a discontinuous desired normal near the10cm/15deg boundary.
    """
    h=clearance.clamp(min=0.)
    speed=velocity[:,2].abs()
    time=2*h/(torch.sqrt(speed.square()+2*gravity*h)+speed).clamp(min=1e-9)
    predicted=foot_xy+velocity[:,:2]*time[:,None]
    distance=(predicted-current).norm(dim=-1)
    offset = 1. if blend_after_gate else 0.
    position=(offset+(foot_limit_m-distance)/config['position_blend_width_m']).clamp(0.,1.)
    near = config.get('near_touchdown_preview') if blend_after_gate else None
    if near:
        # The far preview is optimistic just outside the acceptance disk.
        # As contact approaches, blend toward its measured short-horizon reliability.
        close = (1. + (foot_limit_m - distance) / near['position_blend_width_m']).clamp(0., 1.)
        blend = ((near['start_within_s'] - time) / (near['start_within_s'] - near['full_within_s'])).clamp(0., 1.)
        position = torch.lerp(position, close, blend)
    orientation=(offset+(15.-torch.rad2deg(tilt))/config['tilt_blend_width_deg']).clamp(0.,1.)
    confidence=position*orientation
    destination=current+confidence[:,None]*(next_goal-current)
    outgoing=(destination-predicted)/flight_time[:,None].clamp(min=.1)
    outgoing*=torch.clamp(speed_limit/outgoing.norm(dim=-1).clamp(min=1e-9),max=1.)[:,None]
    return outgoing,confidence,predicted


def contact_handoff_velocity_cost(plan, velocity, phase, spring_rate, clearance, moving,
                                  step, dt, launch_weight, launch_config, handoff_config):
    """Teach the outgoing velocity during unloading, before the free-flight cost starts."""
    age = step - plan.contact_step
    contact_ticks = math.ceil(handoff_config['contact_window_s'] / dt - 1e-9)
    unloading = (moving & plan.pending & (phase == 1) & (velocity[:, 2] > 0)
                 & ((spring_rate < 0) | (clearance > 0))
                 & (age > 0) & (age < contact_ticks))
    turn = unloading & plan.handoff_turn
    retry = unloading & plan.handoff_near_retry
    direction = plan.velocity / plan.velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    error = velocity[:, :2] - plan.velocity
    transverse = error - (error * direction).sum(dim=-1, keepdim=True) * direction
    residual = torch.where(retry[:, None], error, transverse)
    ticks = math.ceil(launch_config['window_s'] / dt - 1e-9)
    cost = launch_velocity_cost(residual, torch.zeros_like(residual), plan.flight,
                                turn | retry, launch_weight / ticks, launch_config)
    retry_weight = handoff_config.get('near_retry_weight', 1.)
    if retry_weight != 1.:
        cost = torch.where(retry, retry_weight * cost, cost)
    return cost, turn, retry


def retry_midflight_reference(foot_proxy_xy, velocity_xy, target_xy, elapsed, duration,
                              eligible, radius, config):
    """Acceleration reference from a live XY endpoint estimate, in world axes."""
    remaining = (duration - elapsed).clamp_min(.01)
    miss = target_xy - (foot_proxy_xy + velocity_xy * remaining[:, None])
    distance = miss.norm(dim=-1)
    # Aim for the acceptance disk. Limit requested acceleration at short time-to-go.
    magnitude = (2. * (distance - radius).clamp_min(0.) / remaining.square()).clamp(
        max=config['max_acceleration_m_s2'])
    fraction = elapsed / duration.clamp_min(.01)
    active = (eligible & (fraction >= config['start_fraction'])
              & (fraction < config['end_fraction']) & (distance > radius))
    acceleration = miss * (magnitude / distance.clamp_min(1.e-6))[:, None]
    return torch.where(active[:, None], acceleration, 0.), active


def safe_retry_cruise_residual(actual, reference, release_ok, q, omega, config, yaw_config):
    """Discount only existing thrust along the correction while attitude permits it."""
    horizon = yaw_config['response_horizon_s']
    tilt = predicted_max_tilt(q, omega, horizon)
    heading, rate = heading_motion(q, omega)
    difference = heading - yaw_config['target_heading_rad']
    error = torch.atan2(difference.sin(), difference.cos())
    yaw = torch.maximum(error.abs(), (error + horizon * rate).abs())

    def margin(angle, warning_deg, limit_deg):
        return ((math.radians(limit_deg) - angle) / math.radians(limit_deg - warning_deg)).clamp(0., 1.)

    tilt_margin = margin(tilt, config['tilt_warning_deg'], config['tilt_limit_deg'])
    yaw_margin = margin(yaw, yaw_config['all_phase_envelope']['warning_deg'], yaw_config['peak_heading_goal_deg'])
    headroom = torch.minimum(tilt_margin, yaw_margin)
    magnitude = reference.norm(dim=-1)
    direction = reference / magnitude.clamp_min(1.e-8)[:, None]
    along = (actual * direction).sum(-1).clamp_min(0.)
    discount = torch.minimum(along, magnitude) * headroom * release_ok
    return actual - discount[:, None] * direction, discount, headroom


def retry_capture_cost(velocity_xy, incoming_xy, target_xy_velocity, stance_age_s,
                       stance_duration_s, released, active, q, omega, dt, config):
    """Track the velocity transition while leaving room for a controlled landing lean."""
    fraction = (stance_age_s / stance_duration_s.clamp_min(dt)).clamp(0., 1.)
    fraction = torch.where(released, 1., fraction)
    reference = torch.lerp(incoming_xy, target_xy_velocity, fraction[:, None])
    speed_error = (velocity_xy - reference).norm(dim=-1) / config['velocity_width_m_s']
    tilt = predicted_max_tilt(q, omega, config['tilt_horizon_s'])
    tilt_excess = (tilt - math.radians(config['tilt_free_deg'])).clamp_min(0.)
    angle_error = tilt_excess / math.radians(config['tilt_width_deg'])
    # Smooth linear tails keep rare tumbles from dominating the value target.
    def penalty(error, weight):
        return torch.where(active, -dt * weight * (torch.hypot(torch.ones_like(error), error) - 1.), 0.)
    return penalty(speed_error, config['velocity_weight']), penalty(angle_error, config['tilt_weight'])


def touchdown_projection_origin(root,foot,spring,normal):
    """Translate A using A velocity, then subtract the touchdown leg offset.

    The current airborne foot offset rotates. Holding that offset fixed over
    the rest of the flight mixes body translation with a transient leg pose.
    Use the existing desired touchdown axis instead. This is a reward-only
    constant-velocity approximation; the real touchdown scorer uses the foot.
    """
    rest_length=(root-foot).norm(dim=-1)+spring.clamp(min=0.)
    return root[:,:2]-rest_length[:,None]*normal[:,:2]


class ReboundPlan:
    def __init__(self,n,device,seed,config):
        self.config=config;self.rng=torch.Generator(device=device).manual_seed(int(seed)+911773)
        self.flight=torch.full((n,),config['initial_flight_time_s'],device=device)
        self.stance=torch.full((n,),config['initial_stance_time_s'],device=device)
        self.horizon=torch.empty(n,device=device)
        self.pending=torch.zeros(n,dtype=torch.bool,device=device)
        self.velocity=torch.zeros(n,2,device=device)
        self.contact_step=torch.zeros(n,dtype=torch.long,device=device)
        self.launch_step=torch.full((n,),-1,dtype=torch.long,device=device)
        self.contact_normal=torch.zeros(n,3,device=device);self.contact_normal[:,2]=1.
        self.prepared_normal=self.contact_normal.clone()
        self.incoming_velocity=torch.zeros(n,3,device=device)
        self.incoming_valid=torch.zeros(n,dtype=torch.bool,device=device)
        self.contact_reference_valid=torch.zeros(n,dtype=torch.bool,device=device)
        self.continuation_axis=torch.zeros_like(self.velocity)  # zero means inactive
        self.handoff_turn = torch.zeros_like(self.pending)
        self.handoff_near_retry = torch.zeros_like(self.pending)
        self.yaw_peak_squared=torch.zeros(n,device=device)
        # air_dt,air_horizontal_impulse,launch_error_squared,launch_count,
        # heading_error_squared_dt,heading_rate_squared_dt,active_dt,heading_peak,prep_dt
        self.totals=torch.zeros(n,9,device=device)
        self.reset(torch.arange(n,device=device),False)

    def sample_horizon(self,ids,evaluation):
        low,high=self.config['preparation_horizon_range_s']
        self.horizon[ids]=(low+high)/2 if evaluation else low+(high-low)*torch.rand(len(ids),device=self.flight.device,generator=self.rng)

    def reset(self,ids,evaluation):
        self.flight[ids]=self.config['initial_flight_time_s'];self.stance[ids]=self.config['initial_stance_time_s']
        self.pending[ids]=False;self.velocity[ids]=0;self.contact_step[ids]=0;self.totals[ids]=0
        self.launch_step[ids]=-1
        self.contact_normal[ids]=0.;self.contact_normal[ids,2]=1.
        self.prepared_normal[ids]=0.;self.prepared_normal[ids,2]=1.
        self.incoming_velocity[ids]=0.;self.incoming_valid[ids]=False
        self.contact_reference_valid[ids]=False;self.yaw_peak_squared[ids]=0.
        self.continuation_axis[ids]=0.
        self.handoff_turn[ids] = False
        self.handoff_near_retry[ids] = False
        self.sample_horizon(ids,evaluation)


def wrap_env(base):
    class MovingPhaseEnv(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.plane_cost = self.course.get('flight_plane_control')
            self.directional_landing = self.course.get('directional_landing')
            self.precision_release = self.course.get('precision_pre_reversal_release')
            if self.precision_release:
                self._episode_sums['precision_pre_reversal_release'] = torch.zeros(self.num_envs,device=self.device)
                if self.precision_release.get('ab_opening_release'):
                    if not self.course['omni'].get('ab_pair_only'):
                        raise ValueError('Opening release requires the AB-only course')
                    from .precision_release import OpeningReversalRelease
                    self.opening_release = OpeningReversalRelease(self.num_envs, self.device)
                if 'ascent_forecast_bias_m' in self.precision_release:
                    from .precision_release import ReversalShortfall
                    bias=self.precision_release['ascent_forecast_bias_m']
                    if not math.isfinite(bias) or abs(bias)>.10:
                        raise ValueError('Invalid signed landing forecast bias')
                    self.reversal_shortfall=ReversalShortfall(self.num_envs,self.device,bias)
                    self._episode_sums['reversal_ascent_shortfall']=torch.zeros(self.num_envs,device=self.device)
                if 'first_apex_allowance_m_s' in self.precision_release:
                    from .precision_release import FirstFlightApex
                    allowance=self.precision_release['first_apex_allowance_m_s']
                    if not math.isfinite(allowance) or allowance<=0:
                        raise ValueError('Expected positive pre-reversal apex speed allowance')
                    self.reversal_apex=FirstFlightApex(self.num_envs,self.device)
                    self.apex_release_config=dict(self.precision_release,allowance_m_s=allowance)
                    self._episode_sums['precision_pre_reversal_apex']=torch.zeros(self.num_envs,device=self.device)
            if self.directional_landing:
                self._episode_sums['directional_landing'] = torch.zeros(self.num_envs,device=self.device)
            if self.plane_cost:
                for name in ('plane_lean','plane_rate','plane_leg_speed','hold_motor_balance'):
                    self._episode_sums[name] = torch.zeros(self.num_envs,device=self.device)
                self.hold_attitude_config = self.plane_cost.get('hold_attitude')
                if self.hold_attitude_config is not None:
                    from .flight_plane import validate_hold_attitude
                    validate_hold_attitude(self.hold_attitude_config, self.course['flight_plane_limits']['lean_peak_deg'])
                    self._episode_sums['hold_attitude_excess'] = torch.zeros(self.num_envs, device=self.device)
                # hold/move x (air samples, squared forbidden lean/rate/foot speed, max lean)
                self.plane_totals = torch.zeros(self.num_envs,2,5,device=self.device)
                if 'plane_peak' in self.plane_cost:
                    weight = self.plane_cost['plane_peak']
                    start = self.plane_cost['peak_start_deg']
                    if not (math.isfinite(weight) and weight > 0 and
                            0 < start <= self.course['flight_plane_limits']['lean_peak_deg']):
                        raise ValueError('Invalid travel peak lean objective')
                    self.plane_peak_start = math.radians(start)
                    self.hop_plane_excess = torch.zeros(self.num_envs,device=self.device)
                    self.rollout_travel_peak = torch.zeros((),device=self.device)
                    self._episode_sums['plane_peak'] = torch.zeros_like(self.hop_plane_excess)
                    if self.plane_cost.get('include_hold_peak'):
                        self.hold_plane_excess = torch.zeros_like(self.hop_plane_excess)
                        self._episode_sums['hold_plane_peak'] = torch.zeros_like(self.hop_plane_excess)
            self.moving_phase_config=self.course['moving_phase_reward']
            c=self.moving_phase_config
            if (c['version'] not in ('transport_then_landing_reward_v1','transport_then_landing_reward_v2')
                    or not 0<c['landing_clearance_m']<=.5
                    or not 15<c['free_tilt_deg']<45 or c['excess_tilt_weight']<=0):
                raise ValueError('Invalid phase-aware moving reward configuration')
            if c['version']=='transport_then_landing_reward_v2':
                horizon=c.get('landing_time_s',float('nan'))
                if not math.isfinite(horizon) or not .1<=horizon<=.3:
                    raise ValueError('Invalid moving landing anticipation time')
            elif 'landing_time_s' in c:
                raise ValueError('Time-based landing anticipation requires phase reward v2')
            if not isinstance(c.get('include_route_holds',False),bool):
                raise ValueError('include_route_holds must be boolean')
            if 'terminal_hold_tilt_deg' in c and not (self.course.get('guarded_keys') and
                    c.get('include_route_holds') and 0 < c['terminal_hold_tilt_deg'] < c['free_tilt_deg']):
                raise ValueError('Terminal hold tilt requires guarded holds and a tighter positive cone')
            braking=c.get('transport_outward_braking')
            if braking and (not all(math.isfinite(braking[k]) for k in ('start_tilt_deg','full_tilt_deg'))
                    or not 0<braking['start_tilt_deg']<braking['full_tilt_deg']<=c['free_tilt_deg']):
                raise ValueError('Invalid transport outward braking ramp')
            if not all(name in self._episode_sums for name in REWEIGHTED):
                raise ValueError('Phase ablation requires the audited v8 reward ledger')
            self._episode_sums['moving_excess_tilt']=torch.zeros(self.num_envs,device=self.device)
            self.stop_config = self.course.get('zero_step_recovery')
            if self.stop_config:
                self._episode_sums['moving_zero_step_speed'] = torch.zeros(self.num_envs,device=self.device)
                self.stop_after_move = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
                self.stop_impulse_pending = torch.zeros_like(self.stop_after_move)
                self.stop_rng = torch.Generator(device=self.device).manual_seed(self.cfg.seed+44003)
            self._moving_phase_last_audit=None
            self.rotation_work_extension_config=self.course.get('moving_rotation_work_extension')
            if self.rotation_work_extension_config:
                r=self.rotation_work_extension_config
                if (r.get('version')!='excess_rate_positive_motor_work_v1'
                        or not isinstance(r.get('enabled'),bool)
                        or any(not math.isfinite(r.get(k,float('nan'))) for k in
                            ('free_rate_rad_s','full_rate_rad_s','free_spring_q_max_m','free_spring_qdot_max_m_s'))
                        or not 0<r['free_rate_rad_s']<r['full_rate_rad_s']
                        or not 0<r['free_spring_q_max_m']<=.002
                        or not 0<r['free_spring_qdot_max_m_s']<=.1
                        or not self.course.get('vertical_pretraining',{}).get('rotation_work_penalty')
                        or self.course.get('contact_attitude_continuity') or self.course.get('transition_repair')):
                    raise ValueError('Invalid isolated positive motor-work extension')
            self.body_rate_config=self.course.get('moving_body_rate')
            if self.body_rate_config:
                r=self.body_rate_config
                if (r.get('version') not in ('transport_roll_pitch_excess_rate_v1','free_air_roll_pitch_excess_rate_v2')
                        or any(not math.isfinite(r[k]) or r[k]<=0 for k in
                            ('weight','free_rate_rad_s','free_spring_q_max_m','free_spring_qdot_max_m_s'))
                        or r['free_spring_q_max_m']>.002 or r['free_spring_qdot_max_m_s']>.1):
                    raise ValueError('Invalid transport body-rate configuration')
                if r['version']=='free_air_roll_pitch_excess_rate_v2' and not .025<=r.get('minimum_clearance_m',0.)<=.15:
                    raise ValueError('Invalid free-air rate clearance')
                self._episode_sums['moving_transport_body_rate']=torch.zeros(self.num_envs,device=self.device)
            self.outward_tilt_work_config = self.course.get('moving_outward_tilt_work')
            if self.outward_tilt_work_config:
                work = self.outward_tilt_work_config
                self.native_tilt_work_config = self.course['vertical_pretraining'].get('rotation_work_penalty')
                if (work.get('version') != 'outward_tilt_motor_work_v1'
                        or not self.native_tilt_work_config or self.rotation_work_extension_config
                        or not 0 < work['start_tilt_deg'] < work['full_tilt_deg'] <= c['free_tilt_deg']
                        or not 0 < work['free_outward_rate_rad_s'] <= 2.
                        or not .025 <= work['minimum_clearance_m'] <= .15):
                    raise ValueError('Invalid moving outward tilt work configuration')
                self._episode_sums['moving_outward_tilt_work'] = torch.zeros(self.num_envs, device=self.device)
            self.rotation_prediction_config=self.course.get('moving_rotation_prediction')
            if self.rotation_prediction_config:
                r=self.rotation_prediction_config
                if (r.get('version')!='constant_body_rate_max_cone_v1'
                        or any(not math.isfinite(r[k]) or r[k]<=0 for k in
                            ('response_horizon_s','weight','free_spring_q_max_m','free_spring_qdot_max_m_s'))
                        or r['response_horizon_s']>.3 or r['free_spring_q_max_m']>.002
                        or r['free_spring_qdot_max_m_s']>.1):
                    raise ValueError('Invalid moving rotation prediction configuration')
                self._episode_sums['moving_predicted_rotation']=torch.zeros(self.num_envs,device=self.device)
            self.rebound_config=self.course.get('rebound_transition')
            self.rebound_plan=None
            self.transition_repair=self.course.get('transition_repair')
            if self.transition_repair:
                repair=self.transition_repair
                if (not self.rebound_config or repair.get('version')!='clock_contact_reference_retry_and_yaw_peak_v1'
                        or not isinstance(repair.get('rebound_consistency'),bool)
                        or not isinstance(repair.get('yaw_peak_guard'),bool)
                        or not .1<=repair.get('contact_reference_blend_s',0.)<=.3
                        or not 0<repair.get('position_blend_width_m',0.)<.1
                        or not 0<repair.get('tilt_blend_width_deg',0.)<15.):
                    raise ValueError('Invalid transition repair configuration')
            self.contact_continuity_config=self.course.get('contact_attitude_continuity')
            if self.contact_continuity_config:
                if (not self.rebound_config or self.transition_repair
                        or self.contact_continuity_config.get('version')!='last_descent_reference_through_stance_v1'):
                    raise ValueError('Contact-only continuity requires rebound guidance without bundled transition repair')
            self.launch_endpoint_config=self.course.get('launch_endpoint_refinement')
            if self.launch_endpoint_config:
                endpoint=self.launch_endpoint_config
                if (not self.rebound_config
                        or endpoint.get('version')!='liftoff_velocity_error_as_endpoint_distance_v1'
                        or not math.isfinite(endpoint.get('width_m',float('nan')))
                        or endpoint['width_m']!=.1):
                    raise ValueError('Launch endpoint objective requires rebound guidance and the10cm distance scale')
            self.launch_rotation_config=self.course.get('launch_rotation_refinement')
            self.launch_velocity_window_config=self.course.get('launch_velocity_window')
            self.window_along_discount=(self.launch_velocity_window_config or {}).get('first_long_along_discount_m_s',0.)
            if not 0.<=self.window_along_discount<=.15:
                raise ValueError('Invalid early-flight along-track reference adjustment')
            if self.launch_velocity_window_config:
                r=self.launch_velocity_window_config
                if 'release_deficit_fade_m_s' in r:
                    low,high=r['release_deficit_fade_m_s']
                    if not (self.window_along_discount and 0<low<high and math.isfinite(high)):
                        raise ValueError('Invalid release deficit thresholds for the adjusted velocity reference')
                    self.release_discount_scale=torch.zeros(self.num_envs,device=self.device)
                    self.release_discount_counts=torch.zeros(2,dtype=torch.long,device=self.device)
                if (not self.rebound_config or self.launch_endpoint_config
                        or r.get('version')!='free_liftoff_endpoint_response_window_v1'
                        or not math.isfinite(r.get('window_s',float('nan')))
                        or not .1<=r['window_s']<=.3
                        or r.get('width_m')!=.1 or r.get('minimum_clearance_m')!=.05):
                    raise ValueError('Invalid isolated launch velocity response window')
            self.launch_direction_config=self.course.get('launch_direction_refinement')
            self.omni_handoff_config = self.course.get('omni_rebound_handoff')
            self.retry_midflight = self.course.get('retry_midflight')
            self.safe_assist = (self.retry_midflight or {}).get('safe_assist')
            if self.safe_assist:
                cfg = self.safe_assist
                if not (0 < cfg['tilt_warning_deg'] < cfg['tilt_limit_deg'] <= 90
                        and 0 < cfg['max_release_error_m_s'] < math.inf
                        and self.course.get('moving_yaw_prediction', {}).get('all_phase_envelope')):
                    raise ValueError('Safe retry assist requires a valid release and attitude envelope')
                self.retry_release_ok = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.compression_config = self.course.get('retry_compression')
            if self.compression_config is not None:
                from .retry_compression import RetryCompression
                c = self.compression_config
                if not (self.course['omni'].get('ab_pair_only') and self.course['omni'].get('ab_miss_recovery')
                        and self.omni_handoff_config and self.launch_velocity_window_config
                        and 0 < c['sweep_horizon_s'] <= self.omni_handoff_config['contact_window_s']
                        and 1 <= c['vertical_impulse_ratio'] <= 2 and 0 < c['threshold_m_s'] < 1):
                    raise ValueError('Invalid AB compression-braking course')
                self.retry_compression = RetryCompression(self.num_envs, self.device, c['threshold_m_s'],
                    self.launch_velocity_window_config['width_m'], self.rebound_config['launch_weight'])
                self._episode_sums['retry_compression'] = torch.zeros(self.num_envs, device=self.device)
            self.support_config = self.course.get('support_preview')
            if self.support_config is not None:
                from .support_preview import SupportPreview
                cfg = self.support_config
                if not (self.omni_handoff_config and 'continuation_max_angle_deg' in self.omni_handoff_config
                        and self.launch_velocity_window_config and self.course['omni'].get('ab_miss_recovery')
                        and 0 < cfg['start_s'] < cfg['end_s'] <= .2
                        and 0 < cfg['support_time_s'] < .1 and 1 <= cfg['impulse_ratio'] <= 2
                        and 0 < cfg['release_threshold_m_s'] < 1):
                    raise ValueError('Invalid support preview configuration')
                self.support_preview = SupportPreview(self.num_envs, self.device)
                self._episode_sums['support_preview_release'] = torch.zeros(self.num_envs, device=self.device)
            self.waypoint_events = self.course.get('waypoint_event_costs')
            if self.waypoint_events is not None:
                from .waypoint_event_costs import StraightFirstApex, WaypointOutcome
                if not (self.safe_assist and self.directional_landing
                        and 0 < self.waypoint_events['minimum_straight_step_m'] <= .5):
                    raise ValueError('Waypoint event costs require straight landing and confirmed retry tracking')
                self.straight_first_apex = StraightFirstApex(self.num_envs, self.device)
                self.waypoint_outcome = WaypointOutcome(self.num_envs, self.device)
                self.retry_entry_error = torch.zeros(self.num_envs, device=self.device)
                for name in ('straight_first_apex_cross', 'retry_rebound_growth'):
                    self._episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self.retry_capture = self.course.get('retry_capture')
            if self.retry_midflight:
                cfg = self.retry_midflight
                if not (self.omni_handoff_config and self.course['omni'].get('ab_miss_recovery')
                        and 0 < cfg['start_fraction'] < cfg['end_fraction'] < 1
                        and 0 < cfg['max_acceleration_m_s2'] < abs(self.sim.cfg.gravity[2])):
                    raise ValueError('Retry midflight feedback requires AB recovery and bounded acceleration')
                self.retry_foot_offset = torch.zeros(self.num_envs, 2, device=self.device)
            if self.retry_capture or self.retry_midflight:
                self.retry_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            if self.retry_capture:
                if not self.omni_handoff_config or not self.course['omni'].get('ab_miss_recovery'):
                    raise ValueError('Retry capture requires confirmed waypoint recovery and rebound handoff')
                for key in ('flight_window_s', 'tilt_horizon_s', 'tilt_free_deg', 'tilt_width_deg',
                            'velocity_width_m_s', 'velocity_weight', 'tilt_weight'):
                    if not math.isfinite(self.retry_capture[key]) or self.retry_capture[key] <= 0:
                        raise ValueError(f'Invalid retry capture parameter {key}')
                if not isinstance(self.retry_capture.get('require_free_ascent', False), bool):
                    raise ValueError('Retry capture ascent gate must be boolean')
                self.retry_contact_velocity = torch.zeros(self.num_envs, 2, device=self.device)
                for name in ('retry_capture_velocity', 'retry_capture_tilt'):
                    self._episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self.adjust_continuation_approach = False
            if self.omni_handoff_config:
                handoff = self.omni_handoff_config
                near = handoff.get('near_touchdown_preview')
                if near and not (0 < near['position_blend_width_m'] < handoff['position_blend_width_m']
                                 and 0 < near['full_within_s'] < near['start_within_s']):
                    raise ValueError('Invalid near-touchdown handoff preview')
                self.continuation_foot_offset_fraction = handoff.get('continuation_foot_offset_fraction', 0.)
                if 'long_continuation_impulse_m_s' in handoff:
                    speed=handoff['long_continuation_impulse_m_s']
                    if not (math.isfinite(speed) and speed>0 and self.continuation_foot_offset_fraction>0):
                        raise ValueError('Contact impulse reference requires a positive velocity change and foot preview')
                if not 0. <= self.continuation_foot_offset_fraction <= 1. or (
                        self.continuation_foot_offset_fraction and 'continuation_max_angle_deg' not in handoff):
                    raise ValueError('Continuation foot-offset reward needs a straight cone and a fraction in [0,1]')
                if (handoff.get('version') != 'descent_preview_and_unloading_velocity_v1'
                        or not self.rebound_config or not self.launch_velocity_window_config
                        or not self.course.get('moving_navigation')
                        or self.transition_repair or self.contact_continuity_config
                        or not 0 < handoff['turn_min_deg'] < 180
                        or not 0 < handoff['position_blend_width_m'] < .1
                        or not 0 < handoff['tilt_blend_width_deg'] < 15.
                        or not 0 < handoff['near_retry_speed_max_m_s'] <= .5
                        or not 0. < handoff.get('near_retry_weight', 1.) <= 4.
                        or not .06 <= handoff['contact_window_s'] <= .15):
                    raise ValueError('Invalid omnidirectional rebound handoff')
                if 'continuation_max_angle_deg' in handoff:
                    angle=handoff['continuation_max_angle_deg']
                    if not 0<angle<handoff['turn_min_deg']:raise ValueError('Invalid straight handoff cone')
                    self.continuation_cosine=math.cos(math.radians(angle))
                    self.continuation_lead_scale=handoff.get('continuation_lead_scale',1.)
                    if not 0<=self.continuation_lead_scale<=1:
                        raise ValueError('Invalid neutral foot lead scale')
                    self.adjust_continuation_approach = (
                        self.continuation_lead_scale != 1. or self.continuation_foot_offset_fraction > 0.)
                    self._episode_sums['moving_continuation_cross_velocity']=torch.zeros(self.num_envs,device=self.device)
                    self.continuation_cross_weight=handoff.get('continuation_cross_weight',1.)
                    if not 0.<self.continuation_cross_weight<=4.:
                        raise ValueError('Invalid continuation cross-track reward weight')
                    self.approach_cross_weight=handoff.get('approach_cross_weight',0.)
                    if not math.isfinite(self.approach_cross_weight) or self.approach_cross_weight<0.:
                        raise ValueError('Invalid approach cross-track reward weight')
                    if self.approach_cross_weight:
                        self._episode_sums['moving_approach_cross_velocity']=torch.zeros(self.num_envs,device=self.device)
                    self.continuation_shortfall_weight=handoff.get('continuation_shortfall_weight',0.)
                    self.south_release = handoff.get('south_pair_release')
                    if self.south_release:
                        cfg = self.south_release
                        if not (0 < cfg['speed_fraction'] <= 1 and 0 < cfg['weight'] <= 4
                                and 0 < cfg['heading_half_width_deg'] < 90):
                            raise ValueError('Invalid southbound long-pair release objective')
                        self.south_cosine = math.cos(math.radians(cfg['heading_half_width_deg']))
                        self.south_min_edge = self.course['omni']['training_reversals']['paired_long_steps']['min_spacing_m']
                        self.south_pair = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
                        self.south_release_stats = torch.zeros(3,device=self.device)
                        self._episode_sums['moving_south_pair_release'] = torch.zeros(self.num_envs,device=self.device)
                    self.release_carry_weight=handoff.get('release_carry_weight',0.)
                    if not 0. <= self.release_carry_weight <= 4.:
                        raise ValueError('Invalid continuation release carry weight')
                    if self.release_carry_weight:
                        self.carry_sample_step=torch.full((self.num_envs,),-2,dtype=torch.long,device=self.device)
                        self.carry_velocity=torch.zeros(self.num_envs,2,device=self.device)
                        self._episode_sums['moving_release_carry']=torch.zeros(self.num_envs,device=self.device)
                    self.shortfall_before_hold=handoff.get('shortfall_before_hold',False)
                    if type(self.shortfall_before_hold) is not bool or (self.shortfall_before_hold and not self.continuation_shortfall_weight):
                        raise ValueError('End-of-segment shortfall requires an enabled shortfall reward')
                    if not 0 <= self.continuation_shortfall_weight <= 4:
                        raise ValueError('Invalid continuation shortfall weight')
                    if self.continuation_shortfall_weight:
                        self._episode_sums['moving_continuation_shortfall']=torch.zeros(self.num_envs,device=self.device)
                self.handoff_turn_cosine = math.cos(math.radians(handoff['turn_min_deg']))
                self.retry_launch_tail_weight = handoff.get('retry_launch_tail_weight', 0.)
                if not 0. <= self.retry_launch_tail_weight <= 10.:
                    raise ValueError('Invalid near-retry launch velocity tail weight')
                if self.retry_launch_tail_weight:
                    self._episode_sums['moving_retry_launch_tail'] = torch.zeros(
                        self.num_envs, device=self.device)
                self._episode_sums['moving_contact_handoff_velocity'] = torch.zeros(
                    self.num_envs, device=self.device)
            if self.launch_direction_config:
                r=self.launch_direction_config
                if (not self.rebound_config or r.get('version')!='backward_and_cross_track_liftoff_v1'
                        or not 0<r.get('weight',0.)<=20 or not 0<r.get('minimum_target_speed_m_s',0.)<=.2):
                    raise ValueError('Invalid launch direction refinement')
                self._episode_sums['moving_launch_direction']=torch.zeros(self.num_envs,device=self.device)
            if self.launch_rotation_config:
                r=self.launch_rotation_config
                if (not self.rebound_config or r.get('version') not in (
                        'once_free_liftoff_roll_pitch_rate_v1','free_liftoff_response_window_rate_v2')
                        or any(not math.isfinite(r[k]) or r[k]<=0 for k in ('weight','free_rate_rad_s'))):
                    raise ValueError('Invalid launch rotation refinement')
                if r['version']=='free_liftoff_response_window_rate_v2':
                    if not math.isfinite(r.get('window_s',float('nan'))) or not .1<=r['window_s']<=.3:
                        raise ValueError('Invalid launch response window')
                elif 'window_s' in r:raise ValueError('Launch response window requires the v2 objective')
                self._episode_sums['moving_launch_rotation']=torch.zeros(self.num_envs,device=self.device)
                self.release_rate_tail_weight = r.get('release_rate_tail_weight', 0.)
                if not 0. <= self.release_rate_tail_weight <= 10.:
                    raise ValueError('Invalid free-liftoff body-rate tail weight')
                if self.release_rate_tail_weight:
                    self._episode_sums['moving_release_rate_tail'] = torch.zeros(self.num_envs,device=self.device)
            if self.rebound_config:
                r=self.rebound_config
                keys=('initial_flight_time_s','initial_stance_time_s','velocity_gain_s',
                    'approach_weight','launch_weight','cruise_acceleration_weight','nominal_mass_kg')
                if (any(not math.isfinite(r[k]) or r[k]<=0 for k in keys)
                        or not .1<=r['preparation_horizon_range_s'][0]<r['preparation_horizon_range_s'][1]<=.6
                        or not 0<r['touchdown_tilt_limit_deg']<=15 or not 0<r['duration_ema']<=1
                        or not .6<=r['initial_flight_time_s']<=1.2 or not .03<=r['initial_stance_time_s']<=.12
                        or c.get('impact_alignment')):
                    raise ValueError('Invalid rebound transition configuration or conflicting radial alignment')
                self.rebound_plan=ReboundPlan(self.num_envs,self.device,
                    getattr(getattr(self,'cfg',None),'seed',42),r)
                for name in ('moving_rebound_approach','moving_launch_velocity','moving_cruise_acceleration'):
                    self._episode_sums[name]=torch.zeros(self.num_envs,device=self.device)
                self.body_rate_totals=torch.zeros(self.num_envs,9,device=self.device)
            self.yaw_prediction_config=self.course.get('moving_yaw_prediction')
            self.all_phase_yaw = (self.yaw_prediction_config or {}).get('all_phase_envelope')
            if self.all_phase_yaw is not None:
                cfg = self.all_phase_yaw
                if (not 0 < cfg['warning_deg'] < self.yaw_prediction_config['peak_heading_goal_deg']
                        or not all(math.isfinite(cfg[k]) and cfg[k] > 0 for k in
                                   ('heading_weight', 'forecast_excess_weight', 'body_z_rate_weight'))):
                    raise ValueError('Invalid all-phase yaw envelope')
                band = cfg.get('control_band')
                if band is not None and not (
                        0 < band['warning_deg'] < cfg['warning_deg']
                        and math.isfinite(band['weight']) and band['weight'] > 0):
                    raise ValueError('Invalid all-phase yaw control band')
                self._episode_sums['all_phase_yaw'] = torch.zeros(self.num_envs, device=self.device)
            if self.yaw_prediction_config:
                y=self.yaw_prediction_config
                if (not 0<y['response_horizon_s']<=.3
                        or y['target_heading_rad']!=self.course['vertical_pretraining']['yaw_refinement']['target_heading_rad']
                        or any(not math.isfinite(y[k]) or y[k]<=0 for k in ('rms_heading_goal_deg','peak_heading_goal_deg'))
                        or not all(k in self._episode_sums for k in ('vertical_yaw_heading','vertical_yaw_rate'))):
                    raise ValueError('Yaw prediction requires the preserved heading/rate ledger')
            legacy_yaw_peak=bool(self.transition_repair and self.transition_repair['yaw_peak_guard'])
            self.air_yaw_peak_weight=(self.yaw_prediction_config or {}).get('free_air_peak_weight',0.)
            if not 0. <= self.air_yaw_peak_weight <= 80. or (
                    self.air_yaw_peak_weight and (legacy_yaw_peak or self.rebound_plan is None)):
                raise ValueError('Free-flight yaw peak reward requires one rebound peak objective')
            self.score_yaw_peak=legacy_yaw_peak or bool(self.air_yaw_peak_weight)
            if self.score_yaw_peak:
                if not self.yaw_prediction_config:
                    raise ValueError('Yaw peak guard requires world heading prediction')
                self._episode_sums['moving_yaw_peak']=torch.zeros(self.num_envs,device=self.device)
            self.impact_alignment_config=c.get('impact_alignment')
            if self.impact_alignment_config:
                a=self.impact_alignment_config
                if (any(not math.isfinite(a[k]) or a[k]<=0 for k in
                        ('weight','free_spring_q_max_m','free_spring_qdot_max_m_s'))
                        or a['free_spring_q_max_m']>.002 or a['free_spring_qdot_max_m_s']>.1):
                    raise ValueError('Invalid precontact impact alignment configuration')
                self._episode_sums['moving_impact_alignment']=torch.zeros(self.num_envs,device=self.device)
            self.landing_guidance_config=self.course.get('moving_landing_guidance')
            self.first_landing_cross_config = self.course.get('first_landing_cross_feedback')
            self.signed_retry_rebound_config = self.course.get('signed_retry_rebound')
            self.retry_rebound_tracking_config = self.course.get('retry_rebound_tracking')
            if self.first_landing_cross_config is not None:
                from .first_landing_cross_feedback import validate as validate_first_landing_cross
                validate_first_landing_cross(self.first_landing_cross_config)
                if not (self.landing_guidance_config and self.rebound_plan is not None
                        and self.course['omni'].get('ab_pair_only')) or self.retry_midflight:
                    raise ValueError('First-landing cross feedback requires AB rebound guidance without a cruise override')
                self.first_cross_release_step = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device)
                self.first_cross_foot_offset = torch.zeros(self.num_envs, 2, device=self.device)
                self.first_cross_pending_cost = torch.zeros_like(self.first_cross_foot_offset)
                self._episode_sums['first_landing_cross'] = torch.zeros(self.num_envs, device=self.device)
            if self.signed_retry_rebound_config is not None:
                from .signed_retry_rebound import validate_signed_retry_rebound
                validate_signed_retry_rebound(self.signed_retry_rebound_config)
                if not self.omni_handoff_config or self.rebound_plan is None:
                    raise ValueError('Signed retry rebound requires the landing handoff preview')
            if self.retry_rebound_tracking_config is not None:
                from .retry_rebound_tracking import validate as validate_retry_rebound_tracking
                validate_retry_rebound_tracking(self.retry_rebound_tracking_config)
                if not (self.signed_retry_rebound_config and self.course.get('imu_link_plant')
                        and self.course['omni'].get('ab_miss_recovery')):
                    raise ValueError('Rebound tracking requires signed AB recovery and the calibrated attitude plant')
                self.rebound_tracking_axis = torch.zeros(self.num_envs, 2, device=self.device)
                self.rebound_tracking_along = torch.zeros(self.num_envs, device=self.device)
                self.rebound_tracking_prepared_step = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device)
                self.rebound_tracking_target = torch.full_like(self.rebound_tracking_prepared_step, -1)
                self.rebound_tracking_in_support = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device)
            self.imu_link_flight_guidance = self.course.get('imu_link_flight_guidance')
            if self.imu_link_flight_guidance is not None:
                from .imu_link_flight import validate
                validate(self.imu_link_flight_guidance)
                if not (self.landing_guidance_config and self.course.get('imu_link_plant')
                        and self.course.get('yaw_heading_servo') and self.course.get('rigid_body_randomization')
                        and self.course['omni'].get('ab_pair_only')):
                    raise ValueError('imu_link teaching requires the yaw-servo AB plant and landing guide')
                self._episode_sums['imu_link_collective_guidance'] = torch.zeros(self.num_envs, device=self.device)
            self.cross_rebound_control = self.course.get('cross_rebound_control')
            self.apex_governor = self.course.get('apex_governor')
            self.y_rebound_teaching = self.course.get('y_rebound_teaching')
            self.first_ascent_teaching = self.course.get('first_ascent_teaching')
            self.ws_transfer_teaching = self.course.get('ws_transfer_teaching')
            self.ws_continuation_forecast = None
            self.continuation_cruise = (self.ws_transfer_teaching or {}).get('continuation_cruise')
            if self.ws_transfer_teaching is not None:
                from .v81_ws_transfer import terms, validate as validate_ws_transfer
                if (not self.imu_link_flight_guidance or self.first_landing_cross_config is None
                        or self.cross_rebound_control or self.apex_governor):
                    raise ValueError('WS references require the calibrated AB plant and unassisted actions')
                validate_ws_transfer(self.ws_transfer_teaching)
                for name in terms(self.ws_transfer_teaching):
                    self._episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
            if self.cross_rebound_control is not None or self.apex_governor is not None:
                if not self.imu_link_flight_guidance:
                    raise ValueError('Flight-control constraints require the calibrated AB plant')
            if self.cross_rebound_control is not None:
                if self.rebound_plan is None:
                    raise ValueError('Cross rebound control requires the measured rebound flight duration')
                from .cross_rebound_control import validate as validate_cross_control
                validate_cross_control(self.cross_rebound_control)
            if self.y_rebound_teaching is not None:
                if self.cross_rebound_control is None:
                    raise ValueError('Y preimpact control requires the rebound controller')
                if not 0. < self.y_rebound_teaching['cost_per_s'] <= 10.:
                    raise ValueError('Invalid transverse rebound teaching cost')
                self._episode_sums['y_rebound_prediction'] = torch.zeros(self.num_envs, device=self.device)
            if self.apex_governor is not None:
                from .apex_governor import validate as validate_apex_governor
                validate_apex_governor(self.apex_governor)
            if self.first_ascent_teaching is not None:
                ascent = self.first_ascent_teaching
                if (not self.apex_governor or self.rebound_plan is None
                        or not .95 <= ascent['reference_flight_s'] <= 1.15
                        or not 0. < ascent['cost_per_s'] <= 12.):
                    raise ValueError('Invalid first landing ascent teaching configuration')
                self._episode_sums['first_ascent_work'] = torch.zeros(self.num_envs, device=self.device)
            self.overheight_band = self.course.get('overheight_band')
            if (self.ws_transfer_teaching is not None and self.ws_transfer_teaching.get('retained_ascent_pulse')
                    and self.overheight_band is None):
                raise ValueError('Retained ascent pulse teaching requires the measured height band')
            if self.overheight_band is not None:
                band = self.overheight_band
                if (not 1.03 <= band['upper_m'] <= 1.15 or not .005 <= band['soft_band_m'] <= .05
                        or not 0. < band['cost_per_m'] <= 50.):
                    raise ValueError('Invalid upper apex training band')
                self._episode_sums['moving_overheight'] = torch.zeros(self.num_envs, device=self.device)
            repeatability = self.course.get('hop_repeatability')
            if repeatability is not None:
                if not self.course['omni'].get('ab_pair_only') or not self.overheight_band:
                    raise ValueError('Hop repeatability requires AB-only training and an apex band')
                from .hop_repeatability import HopRepeatability
                self.hop_repeatability = HopRepeatability(self.num_envs, self.device, repeatability)
                self._episode_sums['ab_hop_repeatability'] = torch.zeros(self.num_envs, device=self.device)
            if self.landing_guidance_config:
                c=self.landing_guidance_config
                self.first_arrival_bonus=c.get('first_arrival_bonus',0.)
                if not math.isfinite(self.first_arrival_bonus) or self.first_arrival_bonus<0.:
                    raise ValueError('Moving first-arrival reward must be finite and nonnegative')
                self.center_arrival_bonus=c.get('center_arrival_bonus',0.)
                if not math.isfinite(self.center_arrival_bonus) or self.center_arrival_bonus<0.:
                    raise ValueError('Invalid moving center-arrival reward')
                self.long_pair_weight=c.get('long_pair_along_weight',0.)
                self.precision_streak_weight=c.get('precision_streak_weight',0.)
                self.adjacent_landing_weight=c.get('adjacent_first_landing_weight',0.)
                if self.adjacent_landing_weight:
                    if not 0.<self.adjacent_landing_weight<=4. or self.precision_streak_weight:
                        raise ValueError('Invalid adjacent first-landing reward configuration')
                    self.previous_landing_quality=torch.zeros(self.num_envs,device=self.device)
                    self._episode_sums['adjacent_first_landing']=torch.zeros_like(self.previous_landing_quality)
                if self.precision_streak_weight:
                    if not 0.<self.precision_streak_weight<=4.:
                        raise ValueError('Invalid first-attempt streak weight')
                    self.precision_streak=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
                    self._episode_sums['precision_streak']=torch.zeros(self.num_envs,device=self.device)
                if self.long_pair_weight:
                    if not 0 < self.long_pair_weight <= 4.:
                        raise ValueError('Invalid long pair landing weight')
                    self.long_pair_min=self.course['omni']['training_reversals']['paired_long_steps']['min_spacing_m']
                    self._episode_sums['moving_long_pair_along']=torch.zeros(self.num_envs,device=self.device)
                if (self.first_arrival_bonus or self.center_arrival_bonus or self.long_pair_weight
                        or self.precision_streak_weight or self.adjacent_landing_weight or self.window_along_discount or c.get('straight_descent_precision')
                        or self.imu_link_flight_guidance is not None or self.first_landing_cross_config is not None
                        or 'long_continuation_impulse_m_s' in (self.omni_handoff_config or {}) or hasattr(self,'reversal_apex')
                        or hasattr(self,'reversal_shortfall') or self.waypoint_events is not None):
                    self.last_attempted_target=torch.full((self.num_envs,),-1,dtype=torch.long,device=self.device)
                if self.center_arrival_bonus:
                    self._episode_sums['moving_center_arrival']=torch.zeros(self.num_envs,device=self.device)
                if self.first_arrival_bonus:
                    self._episode_sums['moving_first_arrival']=torch.zeros(self.num_envs,device=self.device)
                self.touchdown_cone_weight=c.get('touchdown_cone_weight',0.)
                if not math.isfinite(self.touchdown_cone_weight) or self.touchdown_cone_weight<0.:
                    raise ValueError('Invalid moving touchdown tilt cost')
                if self.touchdown_cone_weight:
                    if not self.rebound_config or not 0.<self.rebound_config['touchdown_tilt_limit_deg']<15.:
                        raise ValueError('Touchdown tilt shaping requires a rebound cone inside the landing tolerance')
                    self._episode_sums['moving_touchdown_cone']=torch.zeros(self.num_envs,device=self.device)
                if ('projection_origin' in c and
                        (c['projection_origin']!='root_A_with_reference_touchdown_leg_v1'
                         or c.get('objective')!='projected_touchdown_error_v1'
                         or self.rebound_plan is None)):
                    raise ValueError('Reference touchdown projection requires rebound guidance')
                blend_time=c.get('measured_foot_blend_s',0.)
                if c.get('precision_to_contact') and (not blend_time or
                        c.get('objective') != 'projected_touchdown_error_v1'):
                    raise ValueError('Precision to contact requires measured-foot projected landing guidance')
                if c.get('long_turn_blend_only') and not (blend_time and self.omni_handoff_config and self.long_pair_weight):
                    raise ValueError('Turn-only blending requires impact blending, handoff and first-attempt tracking')
                if (not 0.<=blend_time<=c['maximum_time_s'] or
                        (blend_time and (not c.get('projection_origin') or
                         c.get('projection_clock_version')!='zero_at_contact_separate_divisor_v2'))):
                    raise ValueError('Measured foot blending requires the descending impact clock and reference leg')
                if (c['version']!='remaining_time_velocity_and_touchdown_error_v1'
                        or not 0<c['minimum_time_s']<c['maximum_time_s']
                        or not 0<c['speed_limit_m_s']<=2.
                        or any(not math.isfinite(c[k]) or c[k]<=0 for k in
                            ('velocity_weight','landing_weight','landing_width_m','landing_clearance_m',
                             'minimum_time_s','maximum_time_s','speed_limit_m_s'))
                        or not math.isfinite(c['apex_margin_m']) or c['apex_margin_m']<0):
                    raise ValueError('Invalid moving landing guidance configuration')
                objective=c.get('objective','velocity_tracking_v1')
                if objective not in ('velocity_tracking_v1','projected_touchdown_error_v1'):
                    raise ValueError('Unknown moving landing guidance objective')
                if 'projection_clock_version' in c and (
                        c['projection_clock_version']!='zero_at_contact_separate_divisor_v2'
                        or objective!='projected_touchdown_error_v1'):
                    raise ValueError('Physical projection clock requires the projected endpoint objective')
                if objective=='projected_touchdown_error_v1':
                    width=c.get('projected_error_width_m',float('nan'))
                    if not math.isfinite(width) or not 0<width<=1.:
                        raise ValueError('Invalid projected landing error width')
                elif 'projected_error_width_m' in c:
                    raise ValueError('Projected error width requires projected landing objective')
                for name in ('moving_velocity_guidance','moving_landing_error'):
                    self._episode_sums[name]=torch.zeros(self.num_envs,device=self.device)
            self.touchdown_stability_config=self.course.get('moving_touchdown_stability')
            if self.touchdown_stability_config:
                c=self.touchdown_stability_config
                if (c.get('version')!='moving_touchdown_rate_v1'
                        or any(not math.isfinite(c[k]) or c[k]<=0 for k in
                            ('weight','free_rate_rad_s','rate_width_rad_s'))):
                    raise ValueError('Invalid moving touchdown stability configuration')
                self._episode_sums['moving_touchdown_rate']=torch.zeros(self.num_envs,device=self.device)
            self.yaw_contact_jump_config = self.course.get('yaw_contact_jump')
            if self.yaw_contact_jump_config is not None:
                if any(not math.isfinite(self.yaw_contact_jump_config[key]) or self.yaw_contact_jump_config[key] <= 0
                       for key in ('free_rate_jump_rad_s', 'weight')):
                    raise ValueError('Invalid touchdown yaw-rate jump objective')
                self.yaw_precontact_spin = torch.zeros(self.num_envs, device=self.device)
                self.yaw_previously_loaded = torch.zeros_like(self.yaw_precontact_spin, dtype=torch.bool)
                self._episode_sums['yaw_contact_jump'] = torch.zeros_like(self.yaw_precontact_spin)

        def _reset_idx(self,ids):
            if hasattr(self, 'hop_repeatability'):
                self.hop_repeatability.reset(slice(None) if ids is None else ids)
            super()._reset_idx(ids)
            if hasattr(self,'plane_totals'):
                self.plane_totals[slice(None) if ids is None else ids] = 0.
            if hasattr(self,'hop_plane_excess'):
                self.hop_plane_excess[slice(None) if ids is None else ids] = 0.
                if hasattr(self,'hold_plane_excess'):
                    self.hold_plane_excess[slice(None) if ids is None else ids] = 0.
            if hasattr(self,'release_discount_scale'):
                self.release_discount_scale[slice(None) if ids is None else ids]=0.
            if hasattr(self,'reversal_apex'):
                self.reversal_apex.reset(slice(None) if ids is None else ids)
            if hasattr(self,'reversal_shortfall'):
                self.reversal_shortfall.reset(slice(None) if ids is None else ids)
            if hasattr(self, 'first_cross_release_step'):
                reset = slice(None) if ids is None else ids
                self.first_cross_release_step[reset] = -1
                self.first_cross_pending_cost[reset] = 0.
            if hasattr(self, 'rebound_tracking_prepared_step'):
                reset = slice(None) if ids is None else ids
                self.rebound_tracking_prepared_step[reset] = -1
                self.rebound_tracking_target[reset] = -1
                self.rebound_tracking_in_support[reset] = False
            if hasattr(self, 'opening_release'):
                self.opening_release.reset(slice(None) if ids is None else ids)
            if hasattr(self,'south_pair'):
                self.south_pair[slice(None) if ids is None else ids] = False
            if hasattr(self,'stop_after_move'):
                reset = slice(None) if ids is None else ids
                self.stop_after_move[reset] = False
                self.stop_impulse_pending[reset] = False
            if hasattr(self,'last_attempted_target'):
                if hasattr(self,'previous_landing_quality'):
                    self.previous_landing_quality[slice(None) if ids is None else ids]=0.
                if hasattr(self,'precision_streak'):
                    self.precision_streak[slice(None) if ids is None else ids]=0
                if ids is None:
                    self.last_attempted_target.fill_(-1)
                else:
                    self.last_attempted_target[ids]=-1
            forecast = (getattr(self, 'ws_transfer_teaching', None) or {}).get('continuation_forecast')
            if forecast is not None and not self.omni_route.evaluation:
                if self.ws_continuation_forecast is None:
                    from .v81_ws_transfer import ContinuationForecast
                    self.ws_continuation_forecast = ContinuationForecast(self.num_envs, self.device, forecast)
                else:
                    self.ws_continuation_forecast.reset(slice(None) if ids is None else ids)
            if getattr(self,'rebound_plan',None) is not None:
                ids=self._robot._ALL_INDICES if ids is None else ids
                if hasattr(self, 'retry_compression'):
                    self.retry_compression.reset(ids)
                if hasattr(self, 'support_preview'):
                    self.support_preview.reset(ids)
                self.rebound_plan.reset(ids,self.omni_route.evaluation)
                if hasattr(self, 'retry_latched'):
                    self.retry_latched[ids] = False
                if hasattr(self, 'straight_first_apex'):
                    self.straight_first_apex.reset(ids)
                    self.waypoint_outcome.reset(ids)
                    self.retry_entry_error[ids] = 0.
                if hasattr(self, 'retry_release_ok'):
                    self.retry_release_ok[ids] = False
                if hasattr(self, 'retry_contact_velocity'):
                    self.retry_contact_velocity[ids] = 0.
                if hasattr(self, 'retry_foot_offset'):
                    self.retry_foot_offset[ids] = 0.
                if hasattr(self,'carry_sample_step'):
                    self.carry_sample_step[ids]=-2
                self.body_rate_totals[ids]=0.

        def _current_route_edge(self):
            route = self.omni_route
            rows, index = self._robot._ALL_INDICES, route.index
            return route.points[rows, index] - route.points[rows, (index - 1).clamp_min(0)]

        def _ab_repeatability_cost(self, valid, physical, tilt):
            from .hop_repeatability import straight_pair

            data, route, track = self._robot.data, self.omni_route, self._foot_event
            ab = route.ab_pairs
            rows, a = self._robot._ALL_INDICES, ab.a_index
            start, first, following = (route.points[rows, a + offset] for offset in (-1, 0, 1))
            a_edge, b_edge = first - start, following - first
            straight = ab.mask & straight_pair(a_edge, b_edge)
            record = straight & (route.index == a) & ~ab.seen_a
            compare = straight & (route.index == a + 1) & ab.hit_a
            if ab.recovery is not None:
                compare &= ab.recovery.active == ab.recovery.CONTINUE
            spring = data.joint_pos[:, self._spring_joint_id]
            vertical_speed = data.root_com_lin_vel_w[:, 2]
            launch = (track.phase == track.GROUND) & (spring <= .001) & (vertical_speed > 0.) & ~physical
            in_air = (track.phase == track.CANDIDATE) | (track.phase == track.FLIGHT)
            height = data.root_link_pos_w[:, 2] - self._terrain.env_origins[:, 2]
            apex = torch.maximum(track.peak - self._terrain.env_origins[:, 2], height)
            error = (data.body_link_pos_w[:, self._foot_body_id, :2] - self._two_ground_pt_xy_w).norm(dim=-1)
            clean_a = (record & valid & (error <= self.landing_radius_m) & (tilt <= math.radians(15.))
                       & (apex >= 1.) & (apex <= self.overheight_band['upper_m']))
            return self.hop_repeatability.step(
                height, track.air_steps * self.step_dt, data.root_quat_w, data.root_ang_vel_b,
                data.root_link_lin_vel_w[:, :2], torch.where(record[:, None], a_edge, b_edge),
                launch, in_air & ~physical & (spring <= .002), (spring > .002) | physical, clean_a, record, compare)

        def _apply_cross_rebound_control(self, actions, clearance, up):
            from .cross_rebound_control import correct_cross_rebound

            route, data = self.omni_route, self._robot.data
            goal, following = self._two_ground_pt_xy_w, self._two_ground_pt1_xy_w
            edge = self._current_route_edge()
            next_edge = following - goal
            axis = torch.nn.functional.normalize(edge, dim=-1)
            cross = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
            next_distance = next_edge.norm(dim=-1)
            collinear = (next_edge * axis).sum(-1).abs() >= .97 * next_distance
            eligible = ~route.stationary_mask & (route.index > 0) & (edge.norm(dim=-1) > .10) & collinear
            guard = getattr(self, 'key_guard', None)
            if guard is not None:
                eligible &= ~guard.mask

            foot = data.body_link_pos_w[:, self._foot_body_id, :2]
            velocity = data.root_link_lin_vel_w
            gravity = abs(self.sim.cfg.gravity[2])
            landing_speed = (velocity[:, 2].square() + 2. * gravity * clearance.clamp_min(0.)).sqrt()
            time_to_touch = 2. * clearance.clamp_min(0.) / (landing_speed + velocity[:, 2].abs()).clamp_min(1.e-6)
            # Collinear next targets have the same transverse coordinate, whether
            # the real touchdown advances the queue or retains the current target.
            predicted_foot = foot + velocity[:, :2] * time_to_touch[:, None]
            outgoing_cross = ((goal - predicted_foot) * cross).sum(-1) / self.rebound_plan.flight.clamp_min(.5)
            incoming_cross = (velocity[:, :2] * cross).sum(-1)
            vertical_rebound = landing_speed + 2.7
            desired_lean = torch.atan2(outgoing_cross.clamp(-.25, .25) - incoming_cross,
                                      vertical_rebound).clamp(-math.radians(7.), math.radians(7.))
            phase = self._foot_event.phase
            free_descent = ((phase == 2) | (phase == 3)) & (velocity[:, 2] < 0.) & (clearance > 0.)
            timing = self._imu_link_plant.parameters[self._imu_link_plant.profile]
            lead = timing[:, 7] * self.step_dt + 2. * timing[:, 6]
            ramp = ((lead + .12 - time_to_touch) / .12).clamp(0., 1.)
            upright = ((up - math.cos(math.radians(24.))) / .06).clamp(0., 1.)
            gate = ramp * upright * eligible * free_descent
            if self.y_rebound_teaching is not None and not route.evaluation:
                y_handoff = ((axis[:, 1].abs() >= .966) & (next_distance >= .25)
                             & ((next_edge * axis).sum(-1) >= .99 * next_distance))
                from .ring_geometry import rotate
                unit_z = torch.zeros_like(data.root_ang_vel_b)
                unit_z[:, 2] = 1.
                body_up = rotate(data.root_quat_w, unit_z)
                cross_sine = (body_up[:, :2] * cross).sum(-1).clamp(-.99, .99)
                up_dot = torch.cross(data.root_ang_vel_w, body_up, dim=-1)
                cross_rate = (up_dot[:, :2] * cross).sum(-1) / (1. - cross_sine.square()).sqrt()
                impact_lean = (cross_sine.asin() + time_to_touch.clamp_max(.10) * cross_rate).clamp(-.35, .35)
                release_error = incoming_cross + vertical_rebound * impact_lean.tan() - outgoing_cross.clamp(-.25, .25)
                residual = ((release_error.abs() - .07) / .22).clamp(0., 1.)
                self._y_rebound_prediction_cost = (-self.y_rebound_teaching['cost_per_s'] * self.step_dt
                                                    * gate * y_handoff * residual.square())
            actions, cross_lean, force = correct_cross_rebound(
                actions, data.root_quat_w, data.root_ang_vel_b, cross, desired_lean, gate,
                self.cross_rebound_control)
            self._cross_rebound_control_audit = (gate, cross_lean, desired_lean, force)
            return actions

        def _record_first_ascent_cost(self, actions, height, vz, up, weight, drag):
            from .ascent_coast_preview import queued_coast
            from .imu_link_flight import NOMINAL_MASS_KG

            plant = self._imu_link_plant
            first = (self.last_attempted_target != self.omni_route.index) & (self._current_route_edge().norm(dim=-1) >= .35)
            ids = (first & (weight > 0.) & (height >= .72) & (vz > .28) & (up > .955)).nonzero().flatten()
            self._first_ascent_work_cost = torch.zeros_like(weight)
            if not len(ids):
                return
            timing = plant.parameters[plant.profile[ids], :2]
            args = (height[ids], vz[ids], up[ids], NOMINAL_MASS_KG * plant.mass_scale[ids],
                    drag[ids], plant.collective_scale[ids], plant.voltage_gain[ids], plant.collective_u[ids],
                    torch.stack(tuple(plant.history), dim=1)[ids], timing[:, 0], timing[:, 1], self.step_dt)
            floor, _ = queued_coast(*args)
            predicted_peak, _ = queued_coast(*args, candidate_duty=.5 * (actions[ids] + 1.))
            # Debit only the height increment attributable to new powered
            # ascent. Stored spring and already queued motor work set the floor.
            excess = (predicted_peak - floor.clamp_min(1.10)).clamp_min(0.)
            reach = ((floor - 1.005) / .025).clamp(0., 1.)
            self._first_ascent_work_cost[ids] = (-self.first_ascent_teaching['cost_per_s'] * self.step_dt
                * reach * (excess / .05).clamp_max(1.).square() * weight[ids])

        def _pre_physics_step(self,actions):
            if getattr(self, 'imu_link_flight_guidance', None) is not None:
                from .imu_link_flight import NOMINAL_MASS_KG, collective_reference, sample_ascent
                config = self.imu_link_flight_guidance
                # Entry points without a runner manifest install the same plant
                # at their first control tick, after reset and case restoration.
                if not hasattr(self, '_imu_link_plant'):
                    self.install_imu_link_plant()
                plant = self._imu_link_plant
                data = self._robot.data
                clearance = data.body_link_pos_w[:, self._foot_body_id, 2] - self._terrain.env_origins[:, 2] - .012
                height, up, thrust, apex, _, landing_time, weight = sample_ascent(self, config, clearance)
                if self.ws_transfer_teaching is not None and not self.omni_route.evaluation:
                    from .v81_ws_transfer import references
                    self._ws_teaching_costs, self._ws_teaching_audit = references(
                        self, actions, clearance, height, up, landing_time)
                if self.cross_rebound_control or self.apex_governor:
                    self._flight_control_input = actions
                if self.cross_rebound_control:
                    actions = self._apply_cross_rebound_control(actions, clearance, up)
                if self.apex_governor:
                    from .apex_governor import limit_ascent_collective, projected_landing_margin
                    from .calibration_20261028 import PARAMS
                    vz = data.root_link_lin_vel_w[:, 2]
                    drag = (PARAMS['air']['ascending_quadratic_Ns2_m2_whole']
                            * plant.parameters[plant.profile, 3])
                    landing_margin = None
                    if self.apex_governor.get('protect_landing'):
                        gravity = abs(self.sim.cfg.gravity[2])
                        ballistic_time = (vz + (vz.square() + 2. * gravity * clearance.clamp_min(0.)).sqrt()) / gravity
                        foot_offset = data.body_link_pos_w[:, self._foot_body_id, :2] - self._two_ground_pt_xy_w
                        landing_margin = projected_landing_margin(
                            self._current_route_edge(), foot_offset, data.root_link_lin_vel_w[:, :2],
                            torch.minimum(landing_time, ballistic_time))
                    actions, coast_apex, trimmed = limit_ascent_collective(
                        actions, height, vz, up, weight, NOMINAL_MASS_KG * plant.mass_scale,
                        plant.collective_scale, plant.voltage_gain, self.apex_governor,
                        ascent_drag_n_s2_m2=drag, landing_margin=landing_margin, applied_thrust=thrust)
                    if self.first_ascent_teaching is not None and not self.omni_route.evaluation:
                        self._record_first_ascent_cost(actions, height, vz, up, weight, drag)
                    self._apex_governor_audit = (weight, coast_apex, trimmed)
                target = self._desired_pos_w[:, 2] - self._terrain.env_origins[:, 2]
                yaw, rate = heading_motion(data.root_quat_w, data.root_ang_vel_b)
                y = self.yaw_prediction_config
                yaw -= y['target_heading_rad']
                forecast = yaw + y['response_horizon_s'] * rate
                self._imu_link_collective_reference = collective_reference(
                    apex, target, height, weight, up, torch.atan2(yaw.sin(), yaw.cos()),
                    torch.atan2(forecast.sin(), forecast.cos()), .5 * (self._actions + 1.),
                    plant.collective_u, thrust, plant.collective_scale, plant.voltage_gain, config)
            if getattr(self, 'yaw_contact_jump_config', None) is not None:
                from .ring_geometry import rotate
                d = self._robot.data
                self.yaw_precontact_spin.copy_(rotate(d.root_quat_w, d.root_ang_vel_b)[:, 2])
                self.yaw_previously_loaded.copy_(d.joint_pos[:, self._spring_joint_id] > .002)
            if getattr(self,'stop_config',None) and not self.omni_route.evaluation:
                ids = self.stop_impulse_pending.nonzero().flatten()
                if len(ids):
                    # Unannounced rebound residual, once per selected liftoff.
                    vel = self._robot.data.root_link_vel_w[ids].clone()
                    draws = torch.rand((len(ids),2),device=self.device,generator=self.stop_rng)
                    angle = 2*math.pi*draws[:,0]
                    axis = torch.stack((angle.cos(),angle.sin()),dim=-1)
                    speed = vel[:,:2].norm(dim=-1,keepdim=True)
                    axis = torch.where(speed>.05,vel[:,:2]/speed.clamp_min(1e-6),axis)
                    lo, hi = self.stop_config['extra_speed_m_s']
                    vel[:,:2] += (lo+(hi-lo)*draws[:,1:2])*axis
                    self._robot.write_root_link_velocity_to_sim(vel,env_ids=ids)
                    self.stop_impulse_pending[ids] = False
            super()._pre_physics_step(actions)

        def _get_rewards(self):
            d=self._robot.data;f=self._foot_event;route=self.omni_route
            # Freeze the old target/phase BEFORE the parent advances the queue.
            # Pure stationary episodes and the first hop retain their rewards.
            # Legacy courses also exclude holds; opted-in routes retain the
            # same phase/landing objective through moving-to-hold transitions.
            i=route.index
            previous=route.points[torch.arange(self.num_envs,device=self.device),(i-1).clamp(min=0)]
            current=route.points[torch.arange(self.num_envs,device=self.device),i]
            physical=(d.root_link_pos_w[:,2]<.05)|(d.root_quat_w[:,1:3].square().sum(-1)>.5)
            nonzero=(current!=previous).any(dim=-1)
            guard=getattr(self,'key_guard',None)
            if guard is not None:
                from .guarded_keys import RECOVER, STEP
                nonzero=torch.where(guard.mask,guard.state==STEP,nonzero)
            active_route=(~route.stationary_mask)&(i>0)&(f.phase!=f.DROP)&~physical
            route_hold=active_route&~nonzero
            moving=active_route&(nonzero|self.moving_phase_config.get('include_route_holds',False))
            clearance=d.body_link_pos_w[:,self._foot_body_id,2]-self._terrain.env_origins[:,2]-.012
            weight=late_weight(f.phase,d.joint_pos[:,self._spring_joint_id],clearance,
                d.root_link_lin_vel_w[:,2],self.moving_phase_config['landing_clearance_m'],
                self.moving_phase_config.get('landing_time_s'),abs(self.sim.cfg.gravity[2]))
            legacy_weight=late_weight(f.phase,d.joint_pos[:,self._spring_joint_id],clearance,
                d.root_link_lin_vel_w[:,2],self.moving_phase_config['landing_clearance_m'])
            work_extension_audit=None
            if self.rotation_work_extension_config:
                r=self.rotation_work_extension_config
                # Freeze eligibility BEFORE the parent scores touchdown and
                # changes phase/target. The original native work kernel owns
                # actual delayed/lagged torque, per-axis positive work and
                # recovery relief; only its previously missing phase weight
                # is restored. No force,action or upright reference changes.
                rate=d.root_ang_vel_b[:,:2].norm(dim=-1)
                gate=((rate-r['free_rate_rad_s'])/(r['full_rate_rad_s']-r['free_rate_rad_s'])).clamp(0.,1.)
                eligible=(moving&((f.phase==2)|(f.phase==3))&(clearance>0)
                    &(d.joint_pos[:,self._spring_joint_id]<=r['free_spring_q_max_m'])
                    &(d.joint_vel[:,self._spring_joint_id].abs()<=r['free_spring_qdot_max_m_s']))
                extra=torch.where(eligible&r['enabled'],(1-weight)*gate,0.)
                work_extension_audit=dict(eligible=eligible,rate=rate,gate=gate,extra_weight=extra)
            tilt=torch.acos((1-2*d.root_quat_w[:,1:3].square().sum(-1)).clamp(-1,1))
            guidance={};guidance_audit=None
            if not route.evaluation:
                if self.y_rebound_teaching is not None:
                    guidance['y_rebound_prediction'] = self._y_rebound_prediction_cost
                if self.first_ascent_teaching is not None:
                    guidance['first_ascent_work'] = self._first_ascent_work_cost
                if self.ws_transfer_teaching is not None:
                    guidance.update(self._ws_teaching_costs)
            if self.plane_cost:
                from .flight_plane import flight_plane_errors
                edge = current - previous
                if guard is not None:
                    edge = torch.where(guard.mask[:,None], guard.dirs[guard.step_dir], edge)
                w,x,y,z = d.root_quat_w.unbind(-1)
                up = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)),dim=-1)
                foot_speed = d.body_link_lin_vel_w[:,self._foot_body_id] - d.root_link_lin_vel_w
                lean, rate, swing = flight_plane_errors(up, d.root_ang_vel_w, foot_speed, edge, nonzero)
                air = ((f.phase==f.CANDIDATE)|(f.phase==f.FLIGHT)) & (clearance>0) & ~physical
                cfg = self.plane_cost
                if hasattr(self,'hop_plane_excess'):
                    from .flight_plane import travel_peak_cost
                    travel = air & nonzero
                    grounded = (f.phase!=f.CANDIDATE)&(f.phase!=f.FLIGHT)
                    self.hop_plane_excess.masked_fill_(grounded,0.)
                    guidance['plane_peak'], excess = travel_peak_cost(
                        lean,travel,self.hop_plane_excess,self.plane_peak_start,cfg['plane_peak'])
                    self.hop_plane_excess.copy_(excess)
                    if hasattr(self,'hold_plane_excess'):
                        self.hold_plane_excess.masked_fill_(grounded,0.)
                        guidance['hold_plane_peak'], hold_excess = travel_peak_cost(
                            lean,air & ~nonzero,self.hold_plane_excess,self.plane_peak_start,cfg['plane_peak'])
                        self.hold_plane_excess.copy_(hold_excess)
                    torch.maximum(self.rollout_travel_peak,torch.where(travel,lean,0.).max(),out=self.rollout_travel_peak)
                for name,value in (('plane_lean',lean),('plane_rate',rate),('plane_leg_speed',swing)):
                    guidance[name] = -cfg[name]*torch.where(air,value.square(),0.)*self.step_dt
                hold = ~nonzero & (f.phase!=f.DROP) & ~physical
                motor = self._motor_u
                imbalance = (motor - motor.mean(dim=1,keepdim=True)).square().mean(1)
                guidance['hold_motor_balance'] = -cfg['hold_motor_balance']*torch.where(hold,imbalance,0.)*self.step_dt
                if self.hold_attitude_config is not None:
                    from .flight_plane import hold_attitude_cost
                    cost, predicted = hold_attitude_cost(
                        up, d.root_ang_vel_w, lean, hold, air, self.step_dt, self.hold_attitude_config)
                    guidance['hold_attitude_excess'] = cost
                    hold_air, hold_support = hold & air, hold & ~air
                    self.extras.setdefault('log',{}).update({
                        'Hold/attitude_active_fraction':hold.float().mean(),
                        'Hold/attitude_air_active_fraction':hold_air.float().mean(),
                        'Hold/attitude_support_active_fraction':hold_support.float().mean(),
                        'Hold/attitude_excess_cost':cost.mean(),
                        'Hold/air_tilt_peak_deg':torch.where(hold_air, lean, 0.).max().rad2deg(),
                        'Hold/support_tilt_peak_deg':torch.where(hold_support, lean, 0.).max().rad2deg(),
                        'Hold/predicted_tilt_peak_deg':predicted.max().rad2deg(),
                    })
                for column,mask in enumerate((air & ~nonzero, air & nonzero)):
                    totals = self.plane_totals[:,column]
                    for j,val in enumerate((torch.ones_like(lean),lean.square(),rate.square(),swing.square())):
                        totals[:,j] += torch.where(mask,val,0.)
                    totals[:,4] = torch.maximum(totals[:,4],torch.where(mask,lean,0.))
                    label = ('Hold','Travel')[column]
                    self.extras.setdefault('log',{})[label+'/forbidden_lean_rms_deg'] = (
                        (lean.square()*mask).sum()/mask.sum().clamp_min(1)).sqrt().rad2deg()
            outward_tilt_work_active = None
            if self.outward_tilt_work_config:
                guidance['moving_outward_tilt_work'], outward_tilt_work_active = outward_tilt_work_cost(
                    d.root_quat_w, d.root_ang_vel_b, self._moment[:, 0], tilt, f.phase,
                    d.joint_pos[:, self._spring_joint_id], clearance, moving, weight,
                    self.step_dt, self.native_tilt_work_config, self.outward_tilt_work_config)
            body_rate_audit=None
            if self.body_rate_config:
                guidance['moving_transport_body_rate'],body_rate_audit=transport_body_rate_cost(
                    d.root_ang_vel_b,f.phase,d.joint_pos[:,self._spring_joint_id],
                    d.joint_vel[:,self._spring_joint_id],clearance,moving,weight,
                    self.step_dt,self.body_rate_config)
            rotation_audit=None
            if self.rotation_prediction_config:
                guidance['moving_predicted_rotation'],rotation_audit=predicted_rotation_cost(
                    d.root_quat_w,d.root_ang_vel_b,f.phase,d.joint_pos[:,self._spring_joint_id],
                    d.joint_vel[:,self._spring_joint_id],clearance,moving,
                    self.moving_phase_config['free_tilt_deg'],self.step_dt,self.rotation_prediction_config)
            rebound_audit=None;time_floor=None;launch=None;old_index=i.clone()
            first_cross_audit = retry_rebound_audit = tracking_audit = None
            if self.rebound_plan is not None:
                p=self.rebound_plan;r=self.rebound_config;spring=d.joint_pos[:,self._spring_joint_id]
                free=((f.phase==2)|(f.phase==3))&(spring<=.001)&(d.joint_vel[:,self._spring_joint_id].abs()<=.05)&(clearance>0)&~physical
                prep=late_weight(f.phase,spring,clearance,d.root_link_lin_vel_w[:,2],.3,p.horizon,abs(self.sim.cfg.gravity[2]))
                prep_active=moving&free&(d.root_link_lin_vel_w[:,2]<0)
                prep=torch.where(prep_active,prep,0.)
                outgoing=(self._two_ground_pt1_xy_w-self._two_ground_pt_xy_w)/p.flight[:,None]
                speed=outgoing.norm(dim=-1).clamp(min=1e-9)
                outgoing*=torch.clamp(self.landing_guidance_config['speed_limit_m_s']/speed,max=1.)[:,None]
                advance_confidence=None;predicted_landing=None
                if self.transition_repair and self.transition_repair['rebound_consistency']:
                    preview,advance_confidence,predicted_landing=consistent_outgoing_velocity(
                        d.body_link_pos_w[:,self._foot_body_id,:2],clearance,d.root_link_lin_vel_w,
                        self._two_ground_pt_xy_w,self._two_ground_pt1_xy_w,p.flight,tilt,
                        abs(self.sim.cfg.gravity[2]),self.landing_guidance_config['speed_limit_m_s'],self.transition_repair, foot_limit_m=self.landing_radius_m)
                    # Only descending preparation needs retain/advance preview;
                    # ascent preserves the established projection reference.
                    outgoing=torch.where(prep_active[:,None],preview,outgoing)
                    p.incoming_velocity[prep_active]=d.root_link_lin_vel_w[prep_active]
                    p.incoming_valid[prep_active]=True
                normal,_=rebound_approach(d.root_link_pos_w,d.body_link_pos_w[:,self._foot_body_id],
                    d.root_link_lin_vel_w,outgoing,p.stance,r)
                approach_normal = normal
                actual = d.root_link_pos_w - d.body_link_pos_w[:, self._foot_body_id]
                actual = actual / actual.norm(dim=-1, keepdim=True).clamp_min(1.e-6)
                if self.omni_handoff_config:
                    handoff = self.omni_handoff_config
                    current_edge = current - previous
                    next_edge = self._two_ground_pt1_xy_w - self._two_ground_pt_xy_w
                    edge_norm = next_edge.norm(dim=-1)
                    route_turn = (nonzero & (edge_norm > .1)
                                  & ((current_edge * next_edge).sum(dim=-1)
                                     <= self.handoff_turn_cosine * current_edge.norm(dim=-1) * edge_norm))
                    handoff_outgoing, confidence, forecast = consistent_outgoing_velocity(
                        d.body_link_pos_w[:, self._foot_body_id, :2], clearance, d.root_link_lin_vel_w,
                        self._two_ground_pt_xy_w, self._two_ground_pt1_xy_w, p.flight, tilt,
                        abs(self.sim.cfg.gravity[2]), self.landing_guidance_config['speed_limit_m_s'],
                        handoff, blend_after_gate=True,foot_limit_m=self.landing_radius_m)
                    handoff_preparing = prep_active & (route_turn | (confidence < 1.))
                    handoff_normal, _ = rebound_approach(
                        d.root_link_pos_w, d.body_link_pos_w[:, self._foot_body_id],
                        d.root_link_lin_vel_w, handoff_outgoing, p.stance, r)
                    approach_normal = torch.where(handoff_preparing[:, None], handoff_normal, normal)
                    if self.adjust_continuation_approach or getattr(self,'approach_cross_weight',0.):
                        continuing=(prep_active & nonzero & (edge_norm>.1) & (confidence>=1.)
                            & ((current_edge*next_edge).sum(-1)>=self.continuation_cosine
                               *current_edge.norm(dim=-1)*edge_norm))
                    if self.adjust_continuation_approach:
                        # On a confident advance, the remaining distance for the
                        # next hop starts at the forecast foot, including any
                        # within-tolerance miss of the current waypoint.
                        approach_velocity = outgoing
                        if self.continuation_foot_offset_fraction:
                            approach_velocity = torch.lerp(outgoing, handoff_outgoing,
                                                          self.continuation_foot_offset_fraction)
                        ref,_=rebound_approach(d.root_link_pos_w,d.body_link_pos_w[:,self._foot_body_id],
                            d.root_link_lin_vel_w,approach_velocity,p.stance*self.continuation_lead_scale,r)
                        vertical_change=handoff.get('long_continuation_impulse_m_s')
                        if vertical_change is not None:
                            selected=(self.last_attempted_target!=old_index) & (current_edge.norm(dim=-1)>=.40) & (edge_norm>=.40)
                            if guard is not None:
                                selected &= ~guard.mask
                            impulse_normal=impulse_approach(d.root_link_lin_vel_w,approach_velocity,vertical_change,
                                                           r['touchdown_tilt_limit_deg'])
                            ref=torch.where(selected[:,None],impulse_normal,ref)
                        previous_approach=approach_normal
                        approach_normal=torch.where(continuing[:,None],ref,approach_normal)
                    if self.signed_retry_rebound_config is not None:
                        from .signed_retry_rebound import signed_retry_rebound
                        eligible = prep_active & nonzero
                        ab = getattr(route, 'ab_pairs', None)
                        if ab is not None and not route.evaluation:
                            # A scheduled retry can retain a missed A. On the
                            # second jump only same-point retries are eligible,
                            # to practice their approach without adding a third hop.
                            recovery = ab.recovery
                            if recovery is None:
                                eligible = torch.zeros_like(eligible)
                            else:
                                retry_a = ~ab.seen_a & (recovery.assignment == recovery.RETRY)
                                retry_b = ab.seen_a & (recovery.active == recovery.RETRY)
                                eligible &= (old_index == ab.a_index) & (retry_a | retry_b)
                        if guard is not None:
                            eligible &= ~guard.mask
                        nominal_approach = approach_normal
                        forecast_error = forecast - self._two_ground_pt_xy_w
                        approach_normal, retry_rebound_audit = signed_retry_rebound(
                            nominal_approach, current_edge, forecast_error,
                            d.root_link_lin_vel_w, clearance, p.flight, eligible, self.landing_radius_m,
                            abs(self.sim.cfg.gravity[2]), r['touchdown_tilt_limit_deg'],
                            self.signed_retry_rebound_config)
                        if self.retry_rebound_tracking_config is not None:
                            from .retry_rebound_tracking import tracked_rebound_reference
                            from .ring_geometry import rotate
                            plant = self._imu_link_plant
                            timing = plant.parameters[plant.profile, 6:8]
                            tracking_eligible = eligible
                            if route.evaluation:
                                tracking_eligible = eligible & (self.last_attempted_target == old_index)
                            approach_normal, tracking_audit = tracked_rebound_reference(
                                nominal_approach, approach_normal, current_edge, forecast_error,
                                d.root_link_lin_vel_w, clearance, p.flight, tracking_eligible, actual,
                                rotate(d.root_quat_w, d.root_ang_vel_b), self.step_dt,
                                timing[:, 0], timing[:, 1], self.landing_radius_m,
                                abs(self.sim.cfg.gravity[2]), r['touchdown_tilt_limit_deg'],
                                self.signed_retry_rebound_config, self.retry_rebound_tracking_config)
                            prepared = tracking_audit['active']
                            self.rebound_tracking_axis[prepared] = tracking_audit['axis'][prepared]
                            self.rebound_tracking_along[prepared] = tracking_audit['reference_along'][prepared]
                            self.rebound_tracking_prepared_step[prepared] = self.common_step_counter
                            self.rebound_tracking_target[prepared] = old_index[prepared]
                    handoff_cost, turn_unloading, retry_unloading = contact_handoff_velocity_cost(
                        p, d.root_link_lin_vel_w, f.phase, d.joint_vel[:, self._spring_joint_id],
                        clearance, moving, self.common_step_counter, self.step_dt,
                        r['launch_weight'], self.launch_velocity_window_config, handoff)
                    guidance['moving_contact_handoff_velocity'] = handoff_cost
                    if hasattr(self,'continuation_cosine'):
                        age=self.common_step_counter-p.contact_step
                        axis=p.continuation_axis
                        active=(moving & p.pending & (f.phase==1) & (d.root_link_lin_vel_w[:,2]>0)
                                & ((d.joint_vel[:,self._spring_joint_id]<0)|(clearance>0))
                                & (age>0)&(age<math.ceil(handoff['contact_window_s']/self.step_dt-1e-9))
                                & (axis.square().sum(-1)>.5))
                        # Subtract the speed needed to correct the previous foot offset:
                        # the residual is relative to the confirmed waypoint, not world zero.
                        err=d.root_link_lin_vel_w[:,:2]-p.velocity
                        cross=err-(err*axis).sum(-1,keepdim=True)*axis
                        window=self.launch_velocity_window_config
                        ticks=math.ceil(window['window_s']/self.step_dt-1e-9)
                        guidance['moving_continuation_cross_velocity']=launch_velocity_cost(
                            cross,torch.zeros_like(cross),p.flight,active,
                            self.continuation_cross_weight*r['launch_weight']/ticks,window)
                        if self.continuation_shortfall_weight:
                            # Along the confirmed outgoing edge only; preserve the signed
                            # correction already included in the latched foot-to-goal speed.
                            shortfall=(-(err*axis).sum(-1,keepdim=True)).clamp_min(0)
                            shortfall_active=active
                            if self.shortfall_before_hold:
                                # During unloading the current point is the destination of
                                # this jump. A repeated preview announces the subsequent hold.
                                # Only read already-published points, including on keyboard lanes.
                                hold_next=(self._two_ground_pt1_xy_w==self._two_ground_pt_xy_w).all(dim=-1)
                                shortfall_active=active & nonzero & hold_next
                            guidance['moving_continuation_shortfall']=launch_velocity_cost(
                                shortfall,torch.zeros_like(shortfall),p.flight,shortfall_active,
                                self.continuation_shortfall_weight*r['launch_weight']/ticks,window)

                if self.contact_continuity_config:
                    # Preserve the existing v29 free-descent reference. The
                    # first compressed sample may already contain the impact
                    # impulse, so never use it as the incoming free velocity.
                    p.prepared_normal[prep_active]=normal[prep_active]
                    p.incoming_velocity[prep_active]=d.root_link_lin_vel_w[prep_active]
                    p.incoming_valid[prep_active]=True
                if getattr(self,'approach_cross_weight',0.):
                    # Approximate release velocity if the current leg orientation
                    # persists until impact. Only its transverse component matched
                    # the measured rebound across all four cardinal directions.
                    velocity=d.root_link_lin_vel_w
                    gravity=abs(self.sim.cfg.gravity[2])
                    down=(velocity[:,2].square()+2*gravity*clearance.clamp_min(0.)).sqrt()
                    impulse=2*(down*actual[:,2]-(velocity[:,:2]*actual[:,:2]).sum(-1))
                    predicted=velocity[:,:2]+impulse.clamp_min(0.)[:,None]*actual[:,:2]
                    axis=next_edge/edge_norm.clamp_min(1e-6)[:,None]
                    error=predicted-handoff_outgoing
                    cross=(axis[:,0]*error[:,1]-axis[:,1]*error[:,0])[:,None]
                    ramp=late_weight(f.phase,spring,clearance,velocity[:,2],.3,handoff['contact_window_s'],gravity)
                    guidance['moving_approach_cross_velocity']=launch_velocity_cost(
                        cross,torch.zeros_like(cross),p.flight,continuing,self.approach_cross_weight,
                        self.launch_velocity_window_config)*ramp*self.step_dt
                approach_error = (actual - approach_normal).square().sum(-1)
                if tracking_audit is not None:
                    measured_along = (actual[:, :2] * tracking_audit['axis']).sum(-1)
                    reference_along = tracking_audit['reference_along']
                    present_error = (measured_along - reference_along).clamp_min(0.).square()
                    support_error = (tracking_audit['projected_support_along'] - reference_along).clamp_min(0.).square()
                    # Charge the additional outward rotation that a delayed
                    # command would otherwise carry into loaded contact.
                    approach_error += tracking_audit['weight'] * (support_error - present_error).clamp_min(0.)
                guidance['moving_rebound_approach'] = -r['approach_weight'] * approach_error * prep * self.step_dt
                launch=moving&p.pending&free&(d.root_link_lin_vel_w[:,2]>0)
                if self.support_config is not None:
                    from .support_preview import predict_support_velocity
                    support = self.support_config
                    velocity = d.root_link_lin_vel_w
                    leg = d.root_link_pos_w - d.body_link_pos_w[:, self._foot_body_id]
                    predicted, clock = predict_support_velocity(
                        leg, d.root_quat_w, d.root_ang_vel_b, velocity, clearance,
                        abs(self.sim.cfg.gravity[2]), support['impulse_ratio'], support['support_time_s'])
                    preview_sample = (prep_active & (clock >= support['start_s']) & (clock <= support['end_s'])
                                      & (tilt <= math.radians(15.)))
                    self.support_preview.sample(preview_sample, predicted)
                    preview_budget = self.support_preview.release(
                        launch, velocity[:, :2], p.velocity, p.flight, support['release_threshold_m_s'],
                        self.launch_velocity_window_config['width_m'], r['launch_weight'])
                if self.precision_release:
                    from .precision_release import pre_reversal_loss
                    eligible=launch if guard is None else launch & ~guard.mask
                    if hasattr(self,'reversal_shortfall'):
                        remaining=p.flight-(self.common_step_counter-p.launch_step)*self.step_dt
                        guidance['reversal_ascent_shortfall']=self.reversal_shortfall.step(
                            eligible,self.last_attempted_target!=old_index,free,moving & ((p.launch_step>=0)|launch),
                            d.root_link_lin_vel_w,d.root_link_pos_w[:,:2],d.body_link_pos_w[:,self._foot_body_id,:2],
                            self._two_ground_pt_xy_w,current-previous,self._two_ground_pt1_xy_w-self._two_ground_pt_xy_w,
                            remaining,self.landing_radius_m,self.precision_release['weight'])
                    if hasattr(self,'reversal_apex'):
                        first=self.last_attempted_target!=old_index
                        apex=self.reversal_apex.step(eligible & first,moving & ((p.launch_step>=0)|launch),
                                                     free,d.root_link_lin_vel_w[:,2])
                        guidance['precision_pre_reversal_apex']=pre_reversal_loss(
                            current-previous,self._two_ground_pt1_xy_w-self._two_ground_pt_xy_w,
                            d.root_link_lin_vel_w[:,:2],p.velocity,p.flight,apex,self.apex_release_config)
                        eligible=eligible & ~first
                    guidance['precision_pre_reversal_release']=pre_reversal_loss(
                        current-previous,self._two_ground_pt1_xy_w-self._two_ground_pt_xy_w,
                        d.root_link_lin_vel_w[:,:2],p.velocity,p.flight,eligible,self.precision_release)
                    if hasattr(self, 'opening_release') and not route.evaluation:
                        at_a = old_index == route.ab_pairs.a_index
                        drop_contact = at_a & (f.phase == f.DROP) & (spring > .002) & ~physical
                        ascent = at_a & moving & free & (d.root_link_lin_vel_w[:, 2] > 0)
                        guidance['precision_pre_reversal_release'] += self.opening_release.step(
                            drop_contact, ascent, d.body_link_pos_w[:, self._foot_body_id, :2],
                            self._two_ground_pt_xy_w, current - previous,
                            self._two_ground_pt1_xy_w - self._two_ground_pt_xy_w, d.root_link_lin_vel_w[:, :2],
                            p.flight, self.landing_guidance_config['speed_limit_m_s'], self.precision_release)
                if guard is not None and guard.pair_gap:
                    guard.observe_free_liftoff(launch)
                if getattr(self,'south_release',None):
                    # Apply once at free liftoff into the final long step. The
                    # foot-to-goal reference and travel axis were latched at touchdown.
                    axis = p.continuation_axis
                    terminal_step = (self._two_ground_pt_xy_w==self._two_ground_pt1_xy_w).all(-1)
                    cost, active, along = pair_release_cost(d.root_link_lin_vel_w[:,:2],p.velocity,axis,p.flight,
                        launch & self.south_pair & terminal_step,self.landing_radius_m,self.south_release)
                    guidance['moving_south_pair_release'] = cost
                    self.south_release_stats += torch.stack((active.sum(),(cost<0.).sum(),torch.where(active,along,0.).sum()))
                if hasattr(self,'carry_sample_step'):
                    self.carry_velocity[prep_active]=d.root_link_lin_vel_w[prep_active,:2]
                    self.carry_sample_step[prep_active]=self.common_step_counter
                    axis=p.continuation_axis
                    eligible=(launch & (axis.square().sum(-1)>.5)
                              & (self.carry_sample_step==p.contact_step-1))
                    incoming=(self.carry_velocity*axis).sum(-1)
                    requested=(p.velocity*axis).sum(-1)
                    actual_speed=(d.root_link_lin_vel_w[:,:2]*axis).sum(-1)
                    floor=torch.minimum(requested,incoming-.15)
                    deficit=(floor-actual_speed).clamp_min(0.)
                    guidance['moving_release_carry']=launch_velocity_cost(
                        deficit[:,None],torch.zeros_like(deficit[:,None]),p.flight,
                        eligible & (floor>0.),self.release_carry_weight,self.launch_velocity_window_config)
                p.launch_step[launch]=self.common_step_counter
                if self.retry_capture:
                    stance_age = (self.common_step_counter - p.contact_step) * self.step_dt
                    release_age = (self.common_step_counter - p.launch_step) * self.step_dt
                    released = p.launch_step >= 0
                    stance = p.pending & ~released & (stance_age > 0.) & (
                        stance_age <= self.omni_handoff_config['contact_window_s'])
                    early_air = released & (release_age < self.retry_capture['flight_window_s'])
                    if self.retry_capture.get('require_free_ascent', False):
                        early_air &= free & (d.root_link_lin_vel_w[:, 2] > 0.)
                    capture_active = self.retry_latched & moving & (stance | early_air)
                    guidance['retry_capture_velocity'], guidance['retry_capture_tilt'] = retry_capture_cost(
                        d.root_link_lin_vel_w[:, :2], self.retry_contact_velocity, p.velocity,
                        stance_age, p.stance, released, capture_active, d.root_quat_w,
                        d.root_ang_vel_b, self.step_dt, self.retry_capture)
                error=(d.root_link_lin_vel_w[:,:2]-p.velocity).square().sum(-1)
                ab = getattr(route, 'ab_pairs', None)
                if ab is not None and ab.recovery is not None:
                    ab.recovery.record_release(launch, error.sqrt())
                if self.omni_handoff_config and self.retry_launch_tail_weight:
                    # 0.35 m/s was the 90th percentile of release error on low-speed
                    # recoveries that landed within 10 cm in the matched control.
                    excess = (error.sqrt() - .35).clamp_min(0.)
                    width = self.launch_velocity_window_config['width_m']
                    loss = -torch.expm1(-.5 * (excess * p.flight / width).square())
                    retry_launch = launch & p.handoff_near_retry
                    if guard is not None:
                        # A keyboard hold also has no new command at touchdown.
                        # Only RECOVER means the guarded route actually missed.
                        retry_launch &= ~guard.mask | (guard.state == RECOVER)
                    guidance['moving_retry_launch_tail'] = torch.where(
                        retry_launch, -self.retry_launch_tail_weight * loss, 0.)
                guidance['moving_launch_velocity']=launch_velocity_cost(d.root_link_lin_vel_w[:,:2],
                    p.velocity,p.flight,launch,r['launch_weight'],self.launch_endpoint_config)
                velocity_window_audit=None
                if self.launch_velocity_window_config:
                    window_reference=p.velocity
                    if self.window_along_discount:
                        from .precision_release import long_step_window_reference
                        first=self.last_attempted_target!=i
                        if guard is not None:
                            first=first & ~guard.mask
                        scale=getattr(self,'release_discount_scale',None)
                        if scale is not None:
                            from .precision_release import launch_discount_scale
                            edge=current-previous
                            at_release=launch_discount_scale(p.velocity,d.root_link_lin_vel_w[:,:2],edge,
                                self.launch_velocity_window_config['release_deficit_fade_m_s'])
                            scale.copy_(torch.where(launch,at_release,scale))
                            scored=launch & first & (edge.norm(dim=-1)>=.40)
                            self.release_discount_counts[0]+=scored.sum()
                            self.release_discount_counts[1]+=(scored & (scale==0)).sum()
                        window_reference=long_step_window_reference(
                            p.velocity,current-previous,first,self.window_along_discount,scale)
                    guidance['moving_launch_velocity'],velocity_window_audit=launch_velocity_window_cost(
                        d.root_link_lin_vel_w[:,:2],window_reference,p.flight,p.launch_step,self.common_step_counter,
                        moving,f.phase,clearance,self.step_dt,r['launch_weight'],self.launch_velocity_window_config)
                    if self.stop_config:
                        # A repeated absolute target is a zero command. Permit the small
                        # return velocity needed to recenter a landing inside the disk.
                        hold = route_hold & self.stop_after_move
                        if guard is not None:
                            from .guarded_keys import HOLD
                            hold &= ~guard.mask | (guard.state==HOLD)
                        active = hold & velocity_window_audit['active']
                        factor = self.stop_config['extra_velocity_weight']/r['launch_weight']
                        guidance['moving_zero_step_speed'] = torch.where(
                            active,guidance['moving_launch_velocity']*factor,0.)
                        kicks = hold & launch
                        if not route.evaluation:
                            sample = torch.rand(self.num_envs,device=self.device,generator=self.stop_rng)
                            self.stop_impulse_pending |= kicks & (sample < self.stop_config['probability'])
                        self.extras.setdefault('log',{}).update({
                            'Moving/zero_step_window_fraction':active.float().mean(),
                            'Moving/zero_step_disturbed_fraction':self.stop_impulse_pending.float().mean()})
                if self.launch_direction_config:
                    guidance['moving_launch_direction']=launch_direction_cost(d.root_link_lin_vel_w[:,:2],
                        p.velocity,launch,self.launch_direction_config)
                rotation_active=launch
                if self.launch_rotation_config:
                    angular=self.launch_rotation_config;scale=1.
                    if angular['version']=='free_liftoff_response_window_rate_v2':
                        elapsed=self.common_step_counter-p.launch_step
                        # Integer control ticks keep the0.18s end exclusive,
                        # avoiding float32 roundoff at exactly18 ticks.
                        ticks=math.ceil(angular['window_s']/self.step_dt-1e-9)
                        rotation_active=(moving&free&(p.launch_step>=0)&(elapsed>=0)&(elapsed<ticks))
                        scale=self.step_dt/angular['window_s']
                    guidance['moving_launch_rotation']=scale*launch_rotation_cost(d.root_ang_vel_b,rotation_active,angular)
                    if guard is not None:
                        focus=guard.mask & (route_hold | p.handoff_turn)
                        guidance['moving_launch_rotation']*=torch.where(focus,self.course['guarded_keys']['launch_multiplier'],1.)
                    if self.release_rate_tail_weight:
                        # The existing cost averages over the motor response window.
                        # Score a fast roll/pitch release at the physical event as well.
                        allowed = angular['free_rate_rad_s']
                        rp = d.root_ang_vel_b[:,:2].norm(dim=-1)
                        excess = ((rp - 2. * allowed) / allowed).clamp_min(0.)
                        loss = -torch.expm1(-.5 * excess.square())
                        guidance['moving_release_rate_tail'] = torch.where(
                            launch, -self.release_rate_tail_weight * loss, 0.)
                # Penalize powered lateral transport in cruise, while allowing
                # the prelanding rotation and preserving vertical replenishment.
                from .ring_geometry import rotate
                force=rotate(d.root_quat_w,self._thrust[:,0])[:,:2]
                air=moving&free
                rates=d.root_ang_vel_b;xy=rates[:,:2].norm(dim=-1);yaw=rates[:,2].abs()
                t=self.body_rate_totals
                t[:,0]+=air.float()*self.step_dt
                t[:,1]+=torch.where(air,xy.square()*self.step_dt,0.)
                t[:,2]=torch.maximum(t[:,2],torch.where(air,xy,0.))
                t[:,3]+=torch.where(air,yaw.square()*self.step_dt,0.)
                t[:,4]=torch.maximum(t[:,4],torch.where(air,yaw,0.))
                t[:,5]+=(air&(xy>.6)).float()*self.step_dt
                t[:,6]+=launch.float();t[:,7]+=torch.where(launch,xy.square(),0.)
                t[:,8]=torch.maximum(t[:,8],torch.where(launch,xy,0.))
                acceleration = force / r['nominal_mass_kg']
                if self.retry_midflight:
                    foot_xy = d.body_link_pos_w[:, self._foot_body_id, :2]
                    root_xy = d.root_link_pos_w[:, :2]
                    self.retry_foot_offset[launch] = foot_xy[launch] - root_xy[launch]
                    ref_acceleration, correcting = retry_midflight_reference(
                        root_xy + self.retry_foot_offset, d.root_link_lin_vel_w[:, :2],
                        self._two_ground_pt_xy_w, (self.common_step_counter-p.launch_step)*self.step_dt,
                        p.flight, air & self.retry_latched & (p.launch_step >= 0), self.landing_radius_m,
                        self.retry_midflight)
                    if self.safe_assist:
                        self.retry_release_ok[launch] = error[launch] <= self.safe_assist['max_release_error_m_s'] ** 2
                        acceleration, discount, headroom = safe_retry_cruise_residual(
                            acceleration, ref_acceleration, self.retry_release_ok, d.root_quat_w,
                            d.root_ang_vel_b, self.safe_assist, self.yaw_prediction_config)
                    else:
                        acceleration = acceleration - ref_acceleration
                guidance['moving_cruise_acceleration']=torch.where(air,
                    -r['cruise_acceleration_weight']*acceleration.square().sum(-1)*(1-prep)*self.step_dt,0.)
                if self.first_landing_cross_config is not None:
                    from .first_landing_cross_feedback import first_landing_cross_feedback
                    edge = current - previous
                    first_air = (air & nonzero & (self.last_attempted_target != old_index)
                                 & (edge.norm(dim=-1) >= self.first_landing_cross_config['minimum_step_m'] - 1.e-6))
                    if guard is not None:
                        first_air &= ~guard.mask
                    # Direct A can launch from the opening drop without a scored
                    # prior contact to arm the rebound plan.
                    velocity = d.root_link_lin_vel_w
                    start = first_air & (launch | ((self.first_cross_release_step < 0) & (velocity[:, 2] > 0.)))
                    self.first_cross_release_step[start] = self.common_step_counter
                    self.first_cross_pending_cost.masked_fill_(start[:, None], 0.)
                    root_xy = d.root_link_pos_w[:, :2]
                    foot_offset = d.body_link_pos_w[:, self._foot_body_id, :2] - root_xy
                    self.first_cross_foot_offset[start] = foot_offset[start]
                    elapsed = (self.common_step_counter - self.first_cross_release_step) * self.step_dt
                    remaining = (p.flight - elapsed).clamp_min(0.)
                    forecast_cost, credit, first_cross_audit = first_landing_cross_feedback(
                        root_xy + self.first_cross_foot_offset, velocity, self._two_ground_pt_xy_w,
                        edge, elapsed, remaining, first_air & (self.first_cross_release_step >= 0),
                        acceleration, self.landing_radius_m, self.step_dt, self.first_landing_cross_config)
                    side = first_cross_audit['projected_cross_error_m'] > 0.
                    self.first_cross_pending_cost += torch.stack(
                        (torch.where(side, forecast_cost, 0.), torch.where(~side, forecast_cost, 0.)), dim=-1)
                    guidance['moving_cruise_acceleration'] += (
                        r['cruise_acceleration_weight'] * credit * (1 - prep) * self.step_dt)
                cruise_audit = None
                if self.continuation_cruise is not None and not route.evaluation:
                    from .continuation_cruise import cruise_squared_relief
                    teaching = self._ws_teaching_audit
                    allowed = (air & ~physical & (self.last_attempted_target != old_index)
                               & (teaching['continuation_cruise_target'] == old_index)
                               & (teaching['continuation_cruise_step'] == self.common_step_counter))
                    allowance = torch.where(allowed, teaching['continuation_cruise_allowance_m_s2'], 0.)
                    reduction = cruise_squared_relief(acceleration, current - previous, allowance)
                    label = 'moving_cruise_acceleration'
                    base = guidance[label]
                    relief = r['cruise_acceleration_weight'] * reduction * (1 - prep) * self.step_dt
                    guidance[label] = base + relief
                    cruise_audit = dict(allowance_m_s2=allowance, baseline_cost=base, relief=relief)
                remaining=torch.where(f.phase==1,p.flight,(p.flight-f.air_steps*self.step_dt).clamp(min=0))
                time_floor=torch.where(moving,remaining,0.)
                rebound_audit=dict(preparation_weight=prep,preparation_active=prep_active,normal=normal,
                    outgoing_velocity=outgoing,launch=launch,launch_error_squared=error,
                    launch_rotation_active=rotation_active,
                    actual_launch_velocity=d.root_link_lin_vel_w[:,:2].clone(),target_launch_velocity=p.velocity.clone(),
                    time_floor=time_floor,air=air,horizontal_force=force.norm(dim=-1))
                if first_cross_audit is not None:
                    rebound_audit['first_landing_cross'] = first_cross_audit
                if cruise_audit is not None:
                    rebound_audit['continuation_cruise'] = cruise_audit
                if retry_rebound_audit is not None:
                    rebound_audit['signed_retry_rebound'] = retry_rebound_audit
                if tracking_audit is not None:
                    rebound_audit['retry_rebound_tracking'] = tracking_audit
                if self.support_config is not None:
                    rebound_audit.update(support_preview_sample=preview_sample, support_preview_budget=preview_budget)
                if self.retry_midflight:
                    rebound_audit.update(retry_midflight_active=correcting,
                                         retry_reference_acceleration=ref_acceleration)
                if self.safe_assist:
                    rebound_audit.update(retry_assist_discount=discount, retry_assist_headroom=headroom,
                                         retry_assist_release_ok=self.retry_release_ok.clone(),
                                         cruise_residual=acceleration)
                if self.retry_capture:
                    rebound_audit.update(retry_capture_stance=capture_active & stance,
                                         retry_capture_flight=capture_active & early_air)
                if self.adjust_continuation_approach:
                    rebound_audit.update(lead_changed=continuing,lead_delta=(approach_normal-previous_approach).norm(dim=-1))
                    if self.continuation_foot_offset_fraction:
                        rebound_audit['continuation_velocity_shift_m_s'] = torch.where(
                            continuing, (approach_velocity-outgoing).norm(dim=-1), 0.)
                if velocity_window_audit is not None:
                    rebound_audit.update(launch_velocity_active=velocity_window_audit['active'],
                        launch_velocity_endpoint_error_m=velocity_window_audit['endpoint_error_m'])
                if advance_confidence is not None:
                    rebound_audit.update(advance_confidence=advance_confidence,predicted_landing_xy=predicted_landing)
                if self.omni_handoff_config:
                    rebound_audit.update(handoff_preparation_active=handoff_preparing,
                        handoff_outgoing_velocity=handoff_outgoing, handoff_predicted_landing_xy=forecast,
                        turn_unloading_active=turn_unloading, retry_unloading_active=retry_unloading)
                    if route.evaluation or handoff.get('near_touchdown_preview'):
                        rebound_audit['handoff_confidence'] = confidence
                    if route.evaluation:
                        if self.adjust_continuation_approach:
                            rebound_audit['continuation_reference_velocity_xy']=approach_velocity
                        rebound_audit['desired_approach_normal_xy']=approach_normal[:,:2]
                        rebound_audit['actual_approach_normal_xy']=actual[:,:2]
                        rebound_audit['approach_velocity_z']=d.root_link_lin_vel_w[:,2].clone()
                duration=(self.common_step_counter-p.contact_step).float()*self.step_dt
                p.stance[launch]=(1-r['duration_ema'])*p.stance[launch]+r['duration_ema']*duration[launch].clamp(.03,.12)
                p.pending[launch]=False
                p.totals[:,0]+=air.float()*self.step_dt
                p.totals[:,1]+=torch.where(air,force.norm(dim=-1)*self.step_dt,0.)
                p.totals[:,2]+=torch.where(launch,error,0.);p.totals[:,3]+=launch.float()
                p.totals[:,8]+=prep*self.step_dt
            impact_audit=None
            if self.impact_alignment_config:
                guidance['moving_impact_alignment'],impact_audit=impact_alignment_cost(
                    d.root_link_pos_w,d.body_link_pos_w[:,self._foot_body_id],d.root_link_lin_vel_w,
                    f.phase,d.joint_pos[:,self._spring_joint_id],d.joint_vel[:,self._spring_joint_id],
                    clearance,moving,weight,self.step_dt,self.impact_alignment_config)
            if self.landing_guidance_config or self.touchdown_stability_config or hasattr(self, 'hop_repeatability'):
                valid=((f.phase==f.FLIGHT)&(f.air_steps>=5)
                    &(d.joint_pos[:,self._spring_joint_id]>.002)
                    &(f.apex_seen|(d.root_com_lin_vel_w[:,2]<=0))&~physical)
            if self.overheight_band is not None:
                from .apex_governor import overheight_debit
                origin_z = self._terrain.env_origins[:, 2]
                airborne = (f.phase == f.CANDIDATE) | (f.phase == f.FLIGHT)
                guidance['moving_overheight'] = overheight_debit(
                    d.root_link_pos_w[:, 2] - origin_z, f.peak - origin_z, airborne, self.overheight_band)
            repeatability_samples = None
            repeatability_active = hasattr(self, 'hop_repeatability') and getattr(route, 'ab_pairs', None) is not None
            if repeatability_active:
                guidance['ab_hop_repeatability'], repeatability_samples = self._ab_repeatability_cost(
                    valid, physical, tilt)
            if first_cross_audit is not None:
                # Settle on measured same-side lateral misses. Already accrued
                # cost is also settled if the hop ends before touchdown.
                axis = torch.nn.functional.normalize(current - previous, dim=-1)
                offset = d.body_link_pos_w[:, self._foot_body_id, :2] - self._two_ground_pt_xy_w
                actual_cross = axis[:, 0] * offset[:, 1] - axis[:, 1] * offset[:, 0]
                missed = valid & (actual_cross.abs() > self.landing_radius_m)
                unlanded_terminal = physical | (self.reset_time_outs.bool() & ~valid)
                pending = self.first_cross_pending_cost
                same_side = torch.where(actual_cross > 0., pending[:, 0], pending[:, 1])
                guidance['first_landing_cross'] = torch.where(
                    unlanded_terminal, pending.sum(-1), torch.where(missed, same_side, 0.))
                first_cross_audit.update(measured_cross_miss=missed, unlanded_terminal=unlanded_terminal)
                self.first_cross_release_step.masked_fill_(valid | unlanded_terminal, -1)
                pending.masked_fill_((valid | unlanded_terminal)[:, None], 0.)
            if self.compression_config is not None:
                tracked = self.retry_compression.tracking.clone()
                speed_error = (d.root_link_lin_vel_w[:, :2]-p.velocity).norm(dim=-1)
                hit = (d.body_link_pos_w[:,self._foot_body_id,:2]-self._two_ground_pt_xy_w).norm(dim=-1) <= self.landing_radius_m
                expired = tracked & ((self.common_step_counter-p.contact_step)*self.step_dt
                                     > self.omni_handoff_config['contact_window_s'] + 1.e-6)
                guidance['retry_compression'] = self.retry_compression.step(
                    speed_error, p.flight, launch, valid, hit, physical, expired | self.reset_time_outs.bool())
                rebound_audit.update(compression_tracked=tracked, compression_error=speed_error)
            if self.support_config is not None:
                foot_error = (d.body_link_pos_w[:, self._foot_body_id, :2] - self._two_ground_pt_xy_w).norm(dim=-1)
                guidance['support_preview_release'] = self.support_preview.settle(valid, physical, foot_error, self.landing_radius_m)
                rebound_audit['support_failed'] = physical | (valid & (foot_error > self.landing_radius_m))
            if self.landing_guidance_config:
                projection_xy=None
                if self.landing_guidance_config.get('projection_origin'):
                    projection_xy=touchdown_projection_origin(d.root_link_pos_w,
                        d.body_link_pos_w[:,self._foot_body_id],d.joint_pos[:,self._spring_joint_id],
                        rebound_audit['normal'])
                    projection_xy=torch.where(moving[:,None],projection_xy,
                        d.body_link_pos_w[:,self._foot_body_id,:2])
                blend_mask=None
                if self.landing_guidance_config.get('long_turn_blend_only'):
                    blend_mask=route_turn & (current_edge.norm(dim=-1)>=.30) & (self.last_attempted_target!=old_index)
                    if guard is not None:
                        blend_mask=blend_mask & ~guard.mask
                precision_mask=None
                narrow=self.landing_guidance_config.get('straight_descent_precision')
                if self.directional_landing or narrow:
                    edge=current-previous
                    prior=route.points[torch.arange(self.num_envs,device=self.device),(i-2).clamp_min(0)]
                    following=self._two_ground_pt1_xy_w-self._two_ground_pt_xy_w
                    straight=(torch.nn.functional.cosine_similarity(previous-prior,edge,dim=-1)>.966) | (
                        torch.nn.functional.cosine_similarity(following,edge,dim=-1)>.966)
                    if narrow:
                        precision_mask=straight & (edge.norm(dim=-1)>=.40) & (self.last_attempted_target!=old_index)
                        if guard is not None:
                            precision_mask &= ~guard.mask
                if self.waypoint_events is not None:
                    from .waypoint_event_costs import retry_rebound_growth
                    first_long = (moving & straight & (self.last_attempted_target != old_index)
                                  & (edge.norm(dim=-1) >= self.waypoint_events['minimum_straight_step_m']))
                    retry_release = launch & self.retry_latched
                    if guard is not None:
                        first_long &= ~guard.mask
                        retry_release &= ~guard.mask
                    opening = torch.zeros_like(first_long)
                    ab = getattr(route, 'ab_pairs', None)
                    if ab is not None:
                        opening = ((old_index == ab.a_index) & ~ab.seen_a & free
                                   & (d.root_link_lin_vel_w[:, 2] > 0))
                    cross, reached = self.straight_first_apex.step(
                        launch, opening, first_long, free, d.body_link_pos_w[:, self._foot_body_id, :2],
                        d.root_link_pos_w[:, :2], d.root_link_lin_vel_w, self._two_ground_pt_xy_w,
                        edge, p.flight, self.common_step_counter, self.step_dt, self.landing_radius_m)
                    growth = retry_rebound_growth(
                        error.sqrt(), self.retry_entry_error, p.flight, retry_release,
                        self.safe_assist['max_release_error_m_s'], self.landing_radius_m, r['launch_weight'])
                    foot_error = d.body_link_pos_w[:, self._foot_body_id, :2] - self._two_ground_pt_xy_w
                    cross_cost, growth_cost = self.waypoint_outcome.step(
                        reached, cross, retry_release, growth, d.root_link_lin_vel_w[:, :2] - p.velocity,
                        valid, physical, foot_error, edge, self.landing_radius_m,
                        self.landing_guidance_config['landing_weight'])
                    guidance['straight_first_apex_cross'] = cross_cost
                    guidance['retry_rebound_growth'] = growth_cost
                    rebound_audit.update(first_straight_apex=reached, retained_release=retry_release,
                                         event_settlement=valid | physical,
                                         event_miss=(valid & (foot_error.norm(dim=-1) > self.landing_radius_m)) | physical)
                powered_clock = None
                first_clock_cap = None
                if self.imu_link_flight_guidance is not None:
                    from .imu_link_flight import collective_cost, sample_ascent
                    from .yaw_thrust_feedback import _thrust_from_duty
                    config = self.imu_link_flight_guidance
                    plant = self._imu_link_plant
                    height, _, _, apex, _, landing_time, gate = sample_ascent(self, config, clearance)
                    first_travel = moving & nonzero & (self.last_attempted_target != old_index)
                    gate *= first_travel
                    powered_clock = (landing_time, gate, config['maximum_clock_change_s'])
                    if self.first_ascent_teaching is not None and not route.evaluation:
                        long_step = (current - previous).norm(dim=-1) >= .35
                        first_clock_cap = (self.first_ascent_teaching['reference_flight_s'] - f.air_steps * self.step_dt,
                                           gate * long_step)
                    target_force, assist, spare, deficit = self._imu_link_collective_reference
                    commanded_duty = (.5 * (1. + self._actions) * plant.voltage_gain[:, None]).clamp(0., 1.)
                    commanded = plant.collective_scale * _thrust_from_duty(commanded_duty).sum(-1)
                    guidance['imu_link_collective_guidance'] = collective_cost(
                        commanded, target_force, assist, first_travel, self.step_dt, config)
                    self.extras.setdefault('log', {}).update({
                        'ImuLink/clock_active_fraction': (gate > 0.).float().mean(),
                        'ImuLink/collective_guidance_fraction': (first_travel & (assist > 0.)).float().mean(),
                        'ImuLink/collective_reference_peak_n': torch.where(first_travel, assist, 0.).max(),
                    })
                landing_terms,guidance_audit=landing_guidance(
                    d.body_link_pos_w[:,self._foot_body_id,:2],clearance,d.root_link_lin_vel_w,
                    d.root_link_pos_w[:,2]-self._terrain.env_origins[:,2],
                    self._two_ground_pt_xy_w,self._desired_pos_w[:,2]-self._terrain.env_origins[:,2],
                    f.phase,moving,valid,abs(self.sim.cfg.gravity[2]),self.step_dt,self.landing_guidance_config,
                    time_floor,projection_xy,blend_mask,precision_mask,powered_clock,first_clock_cap)
                if powered_clock is not None:
                    edge = torch.nn.functional.normalize(current - previous, dim=-1)
                    cross = torch.stack((-edge[:, 1], edge[:, 0]), dim=-1)
                    miss = guidance_audit['projected_foot_xy'] - self._two_ground_pt_xy_w
                    guidance_audit.update(
                        powered_clock_weight=gate, powered_estimated_apex_m=apex,
                        powered_estimated_time_s=landing_time, estimated_rise_m=apex - height,
                        projected_along_error_m=(miss * edge).sum(-1),
                        projected_cross_error_m=(miss * cross).sum(-1),
                        collective_assist_n=torch.where(first_travel, assist, 0.),
                        collective_spare_n=spare, height_deficit_m=deficit)
                guidance.update(landing_terms)
                if self.directional_landing:
                    axis=torch.nn.functional.normalize(edge,dim=-1)
                    residual=d.body_link_pos_w[:,self._foot_body_id,:2]-self._two_ground_pt_xy_w
                    along=(residual*axis).sum(-1)
                    cross=(residual[:,0]*axis[:,1]-residual[:,1]*axis[:,0]).abs()
                    def cost(terms):
                        return (terms['short']*(-along).clamp_min(0)+terms['long']*along.clamp_min(0)
                                +terms['cross']*cross)/self.landing_radius_m
                    loss=torch.where(straight,cost(self.directional_landing['straight']),cost(self.directional_landing['omni']))
                    active=moving & valid & (edge.norm(dim=-1)>=.30)
                    if guard is not None:
                        active=active & ~guard.mask
                    guidance['directional_landing']=torch.where(active,-loss.clamp_max(8.),0.)
                if self.touchdown_cone_weight:
                    # Charge once at measured touchdown, with zero cost throughout
                    # the useful Raibert approach cone and a bounded tail on a miss.
                    cone=self.rebound_config['touchdown_tilt_limit_deg']
                    excess=(torch.rad2deg(tilt)-cone).clamp_min(0.)/(15.-cone)
                    cost=self.touchdown_cone_weight*torch.expm1(-.5*excess.square())
                    guidance['moving_touchdown_cone']=torch.where(moving & valid,cost,0.)
            if self.touchdown_stability_config:
                guidance['moving_touchdown_rate']=touchdown_rate_cost(
                    d.root_ang_vel_b,moving,valid,self.touchdown_stability_config)
            if self.yaw_contact_jump_config is not None:
                from .ring_geometry import rotate
                spin = rotate(d.root_quat_w, d.root_ang_vel_b)[:, 2]
                loaded = d.joint_pos[:, self._spring_joint_id] > .002
                guidance['yaw_contact_jump'] = yaw_contact_jump_cost(
                    self.yaw_precontact_spin, spin, self.yaw_previously_loaded, loaded, self.yaw_contact_jump_config)
            before={name:self._episode_sums[name].clone() for name in REWEIGHTED}
            yaw_audit=None
            if self.yaw_prediction_config:
                yaw_before={k:self._episode_sums[k].clone() for k in ('vertical_yaw_heading','vertical_yaw_rate')}
                heading,heading_rate=heading_motion(d.root_quat_w,d.root_ang_vel_b)
                y=self.vertical_config['yaw_refinement'] if hasattr(self,'vertical_config') else self.course['vertical_pretraining']['yaw_refinement']
                difference=heading-y['target_heading_rad'];heading_error=torch.atan2(difference.sin(),difference.cos())
                future=heading_error+self.yaw_prediction_config['response_horizon_s']*heading_rate
                future=torch.atan2(future.sin(),future.cos())
                yaw_audit=dict(error=heading_error,rate=heading_rate,predicted_error=future)
                if self.all_phase_yaw is not None:
                    # Sample before the parent settles contact or resets a lane.
                    # No phase, goal, near-retry, or survival mask.
                    guidance['all_phase_yaw'] = yaw_envelope_cost(
                        heading_error, heading_rate, d.root_ang_vel_b[:, 2], self.step_dt,
                        self.yaw_prediction_config, self.all_phase_yaw)
                    yaw_audit.update(body_wz=d.root_ang_vel_b[:, 2].clone(),
                                     phase=f.phase.clone(), terminal_physical=physical)
                if self.score_yaw_peak:
                    # Exactly one reset at physical free liftoff, not at goal
                    # switches,repeated contact samples or failed touchdowns.
                    self.rebound_plan.yaw_peak_squared[launch]=0.
                    previous_peak=self.rebound_plan.yaw_peak_squared.clone()
                    peak_mask=rebound_audit['air'] if self.air_yaw_peak_weight else moving
                    peak_weight=self.air_yaw_peak_weight or y['heading_weight']
                    yaw_cost,peak=yaw_peak_increment(heading_error,heading_rate,
                        self.rebound_plan.yaw_peak_squared,peak_mask,
                        self.yaw_prediction_config['response_horizon_s'],
                        self.yaw_prediction_config['peak_heading_goal_deg'],peak_weight)
                    self.rebound_plan.yaw_peak_squared.copy_(peak)
                    guidance['moving_yaw_peak']=yaw_cost
                    yaw_audit.update(peak_previous_squared=previous_peak,peak_excess_squared=peak.clone(),peak_cost=yaw_cost)
            # Parent settles the CURRENT touchdown and only then advances the
            # task queue. All precontact reference values above remain frozen.
            contact_goals=(self._two_ground_pt_xy_w.clone(),self._two_ground_pt1_xy_w.clone()) if self.rebound_plan is not None else None
            air_steps=f.air_steps.clone() if self.rebound_plan is not None else None
            if self.landing_guidance_config and self.long_pair_weight:
                # Snapshot pre-advance history: guarded misses can replace the
                # next target with the measured foot in the parent callback.
                prior = route.points[torch.arange(self.num_envs,device=self.device),(i-2).clamp_min(0)]
                foot_minus_goal = d.body_link_pos_w[:,self._foot_body_id,:2]-self._two_ground_pt_xy_w
                eligible = moving & nonzero & valid & (i>=2) & (self.last_attempted_target!=i)
                guidance['moving_long_pair_along'] = long_pair_landing_cost(
                    previous-prior,current-previous,foot_minus_goal,eligible,self.long_pair_min,
                    self.landing_radius_m,self.long_pair_weight)
            reward=super()._get_rewards()
            if hasattr(self,'last_attempted_target'):
                # The parent scores the completed hop before changing the
                # command. In guarded lanes 'nonzero' captured STEP at entry.
                first=self.last_attempted_target!=old_index
                if self.ws_continuation_forecast is not None:
                    advanced = route.index == old_index + 1
                    hit = guidance_audit['foot_error_m'] <= self.landing_radius_m
                    continuation = first & moving & nonzero & hit & advanced
                    if guard is not None:
                        continuation &= ~guard.mask
                    self.ws_continuation_forecast.touchdown(valid, continuation, route.index)
                if self.adjacent_landing_weight:
                    from .precision_streak import adjacent_landing
                    counted=valid & ~route.stationary_mask
                    if guard is not None:
                        counted=counted & ~guard.mask
                    quality,pair=adjacent_landing(self.previous_landing_quality,self.last_attempted_target,
                        old_index,counted,self._omni_last_strict,guidance_audit['foot_error_m'],self.landing_radius_m)
                    self.previous_landing_quality.copy_(quality)
                    guidance['adjacent_first_landing']=self.adjacent_landing_weight*pair
                if self.precision_streak_weight:
                    from .precision_streak import streak_step
                    counted=valid & ~route.stationary_mask
                    if guard is not None:
                        counted=counted & ~guard.mask
                    next_streak,guidance['precision_streak']=streak_step(self.precision_streak,
                        self.last_attempted_target,old_index,counted,self._omni_last_strict,self.precision_streak_weight)
                    self.precision_streak.copy_(next_streak)
                earned=moving & nonzero & valid & first & self._omni_last_strict
                if self.first_arrival_bonus:
                    guidance['moving_first_arrival']=self.first_arrival_bonus*earned.float()
                if self.center_arrival_bonus:
                    proximity=(1.-guidance_audit['foot_error_m']/self.landing_radius_m).clamp(0.,1.)
                    guidance['moving_center_arrival']=torch.where(earned,self.center_arrival_bonus*proximity,0.)
                self.last_attempted_target.copy_(torch.where(valid,old_index,self.last_attempted_target))
            inherited={name:self._episode_sums[name]-before[name] for name in REWEIGHTED}
            terminal_hold=None
            if guard is not None and 'terminal_hold_tilt_deg' in self.moving_phase_config:
                terminal_hold=guard.mask & route_hold & guard.terminal_hold
                active_hold=moving & terminal_hold
                above=active_hold & (tilt>math.radians(self.moving_phase_config['terminal_hold_tilt_deg']))
                self.extras.setdefault('log',{}).update({
                    'Moving/terminal_hold_env_fraction':active_hold.float().mean(),
                    'Moving/terminal_hold_above_cone_fraction':above.sum()/active_hold.sum().clamp_min(1)})
            deltas=reward_deltas(inherited,moving,weight,tilt,self.step_dt,self.moving_phase_config,terminal_hold)
            if work_extension_audit is not None:
                extra_cost=work_extension_audit['extra_weight']*inherited['vertical_rotation_work']
                deltas['vertical_rotation_work']+=extra_cost
                work_extension_audit.update(base_cost=inherited['vertical_rotation_work'].clone(),additional_cost=extra_cost)
            if rebound_audit is not None:
                # Replace the upright reference only in free descending
                # preparation. Stance and pure-stationary rewards stay intact.
                for name in ('vertical_tilt','attitude_penalty'):
                    deltas[name]+=torch.where(rebound_audit['preparation_active'],-weight*inherited[name],0.)
                p=self.rebound_plan;r=self.rebound_config
                contact=valid&~route.stationary_mask&~physical
                if self.stop_config:
                    self.stop_after_move |= contact & nonzero
                advanced=route.index>old_index
                goal=torch.where(advanced[:,None],contact_goals[1],contact_goals[0])
                ab = getattr(route, 'ab_pairs', None)
                if ab is not None and ab.recovery is not None:
                    # A local hold replaces the old A with the measured foot.
                    # Latch the goal actually published on this contact tick.
                    goal = self._two_ground_pt_xy_w
                if guard is not None:
                    advanced=torch.where(guard.mask,self._omni_last_navigation_advance,advanced)
                    goal=torch.where(guard.mask[:,None],self._two_ground_pt_xy_w,goal)
                ema=r['duration_ema']
                p.flight[contact]=(1-ema)*p.flight[contact]+ema*(air_steps[contact]*self.step_dt).clamp(.6,1.2)
                plan_flight = p.flight
                if self.first_ascent_teaching is not None and not route.evaluation:
                    edge = contact_goals[1] - contact_goals[0]
                    new_long = advanced & (edge.norm(dim=-1) >= .35)
                    plan_flight = torch.where(new_long, p.flight.clamp_max(
                        self.first_ascent_teaching['reference_flight_s']), p.flight)
                p.velocity[contact]=(goal[contact]-d.body_link_pos_w[contact,self._foot_body_id,:2])/plan_flight[contact,None]
                limit=self.landing_guidance_config['speed_limit_m_s']/p.velocity.norm(dim=-1).clamp(min=1e-9)
                p.velocity*=limit.clamp(max=1.)[:,None]
                p.pending[contact]=True;p.contact_step[contact]=self.common_step_counter
                if tracking_audit is not None:
                    config = self.retry_rebound_tracking_config
                    prepared_age = (self.common_step_counter - self.rebound_tracking_prepared_step) * self.step_dt
                    measured_offset = d.body_link_pos_w[:, self._foot_body_id, :2] - contact_goals[0]
                    overshot = (measured_offset * self.rebound_tracking_axis).sum(-1) > self.landing_radius_m
                    same_goal = (goal == contact_goals[0]).all(-1)
                    confirmed = (contact & nonzero & ~advanced & same_goal & overshot
                                 & (self.rebound_tracking_target == old_index)
                                 & (prepared_age <= config['confirm_window_s'] + 1.e-6))
                    self.rebound_tracking_in_support[contact] = confirmed[contact]
                    self.rebound_tracking_target[contact] = -1
                    elapsed = (self.common_step_counter - p.contact_step) * self.step_dt
                    self.rebound_tracking_in_support &= ~launch & ~physical & (elapsed <= config['support_window_s'])
                    support = (self.rebound_tracking_in_support & moving
                               & (d.joint_pos[:, self._spring_joint_id] > .001))
                    measured_axis = (actual[:, :2] * self.rebound_tracking_axis).sum(-1)
                    error_out = (measured_axis - self.rebound_tracking_along).clamp_min(0.)
                    guidance['moving_rebound_approach'] -= torch.where(
                        support, r['approach_weight'] * error_out.square() * self.step_dt, 0.)
                    tracking_audit.update(confirmed_support=confirmed, support_active=support)
                if self.compression_config is not None:
                    retained = moving & nonzero & ~advanced
                    if ab is not None:
                        retained &= ab.recovery.active == ab.recovery.RETRY
                        rebound_audit['compression_promoted'] = self._compression_promoted
                    if guard is not None:
                        retained &= ~guard.mask
                    self.retry_compression.contact(contact, retained, (d.root_link_lin_vel_w[:,:2]-p.velocity).norm(dim=-1))
                if self.retry_capture or self.retry_midflight:
                    retained = moving & ~advanced
                    if ab is not None:
                        retained &= ab.recovery.active == ab.recovery.RETRY
                    if guard is not None:
                        retained &= ~guard.mask
                    self.retry_latched[contact] = retained[contact]
                    if self.waypoint_events is not None:
                        contact_error = (d.root_link_lin_vel_w[:, :2] - p.velocity).norm(dim=-1)
                        self.retry_entry_error[contact] = contact_error[contact]
                if self.retry_capture:
                    self.retry_contact_velocity[contact] = d.root_link_lin_vel_w[contact, :2]
                if self.omni_handoff_config:
                    # Use the confirmed queue outcome and measured foot position,
                    # never the forecast, for the unloading velocity objective.
                    p.handoff_turn[contact] = (moving & advanced & route_turn)[contact]
                    if hasattr(self,'continuation_cosine'):
                        # Old edge and preview were captured before navigation updated the queue.
                        same=(moving & nonzero & advanced & (edge_norm>.1)
                              & ((current_edge*next_edge).sum(-1)>=self.continuation_cosine*current_edge.norm(dim=-1)*edge_norm))
                        axis=next_edge/edge_norm.clamp_min(1e-6)[:,None]
                        p.continuation_axis[contact]=torch.where(same[:,None],axis,0.)[contact]
                        if self.support_config is not None:
                            retained = moving & ~advanced
                            if ab is not None:
                                retained &= ab.recovery.active == ab.recovery.RETRY
                            straight_long = same & (edge_norm >= .4) & (current_edge.norm(dim=-1) >= .4)
                            if guard is not None:
                                retained &= ~guard.mask
                                straight_long &= ~guard.mask
                            self.support_preview.contact(contact, straight_long, retained, d.root_link_lin_vel_w[:, :2])
                        if self.south_release:
                            both_long = (edge_norm>=self.south_min_edge-1e-6) & (current_edge.norm(dim=-1)>=self.south_min_edge-1e-6)
                            self.south_pair[contact] = (same & both_long & (-axis[:,1]>=self.south_cosine))[contact]
                    near = p.velocity.norm(dim=-1) <= self.omni_handoff_config['near_retry_speed_max_m_s']
                    p.handoff_near_retry[contact] = (moving & ~advanced & near)[contact]
                if self.launch_velocity_window_config:
                    p.launch_step[contact]=-1
                p.sample_horizon(contact.nonzero().flatten(),route.evaluation)
                if self.contact_continuity_config:
                    eligible=contact&moving&p.incoming_valid
                    # A successful landing carries exactly its last approach
                    # normal through compression/rebound. On a true miss,
                    # actual navigation retains the old target: prepare its
                    # recovery instead of pushing toward the next waypoint.
                    retry_normal,_=rebound_approach(d.root_link_pos_w,
                        d.body_link_pos_w[:,self._foot_body_id],p.incoming_velocity,p.velocity,p.stance,r)
                    selected=torch.where(advanced[:,None],p.prepared_normal,retry_normal)
                    p.contact_normal[eligible]=selected[eligible]
                    p.contact_reference_valid[contact]=eligible[contact]
                    p.incoming_valid[contact]=False
                    reference_active=moving&p.pending&p.contact_reference_valid&~free
                    for name in ('vertical_tilt','attitude_penalty'):
                        deltas[name]+=torch.where(reference_active,-weight*inherited[name],0.)
                    reference_error=(actual-p.contact_normal).square().sum(-1)
                    guidance['moving_rebound_approach']+=torch.where(reference_active,
                        -r['approach_weight']*reference_error*self.step_dt,0.)
                    rebound_audit.update(contact_reference_weight=reference_active.float(),
                        contact_reference_normal=p.contact_normal.clone(),contact_reference_error=reference_error)
                    # No post-liftoff normal objective; v29 launch-rate,
                    # launch-velocity and all free-flight terms remain intact.
                    p.contact_reference_valid[launch]=False
                if self.transition_repair and self.transition_repair['rebound_consistency']:
                    # Actual navigation, not its precontact confidence, owns
                    # the latched destination. Preserve the last free incoming
                    # velocity: the first contact sample may already contain
                    # part of the impact impulse.
                    incoming=torch.where(p.incoming_valid[:,None],p.incoming_velocity,d.root_link_lin_vel_w)
                    contact_normal,_=rebound_approach(d.root_link_pos_w,
                        d.body_link_pos_w[:,self._foot_body_id],incoming,p.velocity,p.stance,r)
                    p.contact_normal[contact]=contact_normal[contact]
                    p.contact_reference_valid[contact]=moving[contact]
                    p.incoming_valid[contact]=False
                    elapsed=self.common_step_counter-p.launch_step
                    ticks=math.ceil(self.transition_repair['contact_reference_blend_s']/self.step_dt-1e-9)
                    release=(1-elapsed.float()/ticks).clamp(0.,1.)
                    release=torch.where((p.launch_step>=0)&(elapsed>=0),release,0.)
                    reference_weight=torch.where(p.pending,1.,release)
                    reference_weight=torch.where(moving&p.contact_reference_valid&
                        ~rebound_audit['preparation_active'],reference_weight,0.)
                    reference_active=reference_weight>0
                    # Replace contact's upright ANGLE target, retaining all
                    # rate,leg/work and physical termination constraints.
                    for name in ('vertical_tilt','attitude_penalty'):
                        deltas[name]+=torch.where(reference_active,-weight*inherited[name],0.)
                    reference_error=(actual-p.contact_normal).square().sum(-1)
                    guidance['moving_rebound_approach']+=-r['approach_weight']*reference_error*reference_weight*self.step_dt
                    rebound_audit.update(contact_reference_weight=reference_weight,
                        contact_reference_normal=p.contact_normal.clone(),contact_reference_error=reference_error)
            if yaw_audit is not None:
                replacements=dict(vertical_yaw_heading=-y['heading_weight']*future.square()*self.step_dt,
                    vertical_yaw_rate=-y['rate_weight']*heading_rate.square()*self.step_dt)
                for key,value in replacements.items():
                    deltas[key]=torch.where(moving,value-(self._episode_sums[key]-yaw_before[key]),0.)
                if self.rebound_plan is not None:
                    t=self.rebound_plan.totals
                    t[:,4]+=torch.where(moving,heading_error.square()*self.step_dt,0.)
                    t[:,5]+=torch.where(moving,heading_rate.square()*self.step_dt,0.)
                    t[:,6]+=moving.float()*self.step_dt
                    t[:,7]=torch.maximum(t[:,7],torch.where(moving,heading_error.abs(),0.))
            deltas.update(guidance)
            adjustment=torch.zeros_like(reward)
            for name,value in deltas.items():
                self._episode_sums[name]+=value;adjustment+=value
            self._moving_phase_last_audit=dict(moving=moving,weight=weight,
                outward_weight=outward_weight(weight,tilt,self.moving_phase_config),
                route_hold=route_hold,
                legacy_weight=legacy_weight,clearance=clearance,
                inherited=inherited,deltas=deltas,adjustment=adjustment,guidance=guidance_audit,
                impact_alignment=impact_audit,rebound=rebound_audit,yaw_prediction=yaw_audit,
                rotation_prediction=rotation_audit,body_rate=body_rate_audit,
                repeatability_samples=repeatability_samples,
                rotation_work_extension=work_extension_audit,
                outward_tilt_work_active=outward_tilt_work_active)
            self.extras.setdefault('log',{}).update({
                'Moving/phase_reward_delta':adjustment.mean(),
                'Moving/transport_samples':(moving&(weight==0)).float().mean(),
                'Moving/landing_samples':(moving&(weight>0)).float().mean()})
            if self.compression_config is not None and 'compression_promoted' in rebound_audit:
                compression_delta = guidance['retry_compression']
                self.extras['log'].update({
                    'AB/compression_promotions_per_step': rebound_audit['compression_promoted'].sum(),
                    'AB/compression_debits_per_step': (compression_delta < 0).sum(),
                    'AB/compression_returns_per_step': (compression_delta > 0).sum()})
            if guidance_audit is not None:
                self.extras['log'].update({
                    'Moving/velocity_guidance_mean':guidance['moving_velocity_guidance'].mean(),
                    'Moving/landing_error_cost_mean':guidance['moving_landing_error'].mean(),
                    'Moving/requested_velocity_max_m_s':guidance_audit['desired_velocity_xy'].norm(dim=-1).max()})
            if self.touchdown_stability_config:
                self.extras['log']['Moving/touchdown_rate_cost_mean']=guidance['moving_touchdown_rate'].mean()
            if self.overheight_band is not None:
                self.extras['log']['Height/overheight_cost_mean'] = guidance['moving_overheight'].mean()
            if repeatability_active:
                self.extras['log']['AB/repeatability_samples'] = repeatability_samples.sum()
            if rebound_audit is not None:
                self.extras['log'].update({
                    'Moving/rebound_preparation_fraction':(rebound_audit['preparation_weight']>0).float().mean(),
                    'Moving/launch_velocity_cost':guidance['moving_launch_velocity'].mean(),
                    'Moving/cruise_acceleration_cost':guidance['moving_cruise_acceleration'].mean()})
            if first_cross_audit is not None:
                self.extras['log'].update({
                    'Precision/first_cross_active_fraction': first_cross_audit['active'].float().mean(),
                    'Precision/first_cross_settled_cost': guidance['first_landing_cross'].mean(),
                })
            if retry_rebound_audit is not None:
                self.extras['log']['Precision/signed_retry_rebound_active_fraction'] = (
                    retry_rebound_audit['active'].float().mean())
            if tracking_audit is not None:
                self.extras['log'].update({
                    'Precision/rebound_tracking_preparation_fraction': tracking_audit['active'].float().mean(),
                    'Precision/rebound_tracking_support_fraction': tracking_audit['support_active'].float().mean(),
                })
            if yaw_audit is not None:
                count=moving.sum().clamp(min=1)
                self.extras['log']['Moving/world_heading_error_rms_deg']=torch.rad2deg(
                    (torch.where(moving,heading_error.square(),0.).sum()/count).sqrt())
            if rotation_audit is not None:
                self.extras['log']['Moving/predicted_rotation_cost']=guidance['moving_predicted_rotation'].mean()
            if body_rate_audit is not None:
                self.extras['log']['Moving/transport_body_rate_cost']=guidance['moving_transport_body_rate'].mean()
                self.extras['log']['Moving/transport_body_rate_exceedance_fraction']=(
                    body_rate_audit['active']&(body_rate_audit['excess_rate']>0)).float().mean()
            if self.outward_tilt_work_config:
                self.extras['log']['Moving/outward_tilt_work_cost'] = guidance['moving_outward_tilt_work'].mean()
            if self.omni_handoff_config:
                self.extras['log']['Moving/contact_handoff_velocity_cost'] = guidance['moving_contact_handoff_velocity'].mean()
            if self.launch_rotation_config:
                self.extras['log']['Moving/launch_rotation_cost']=guidance['moving_launch_rotation'].mean()
            if work_extension_audit is not None:
                self.extras['log']['Moving/positive_work_extra_cost']=work_extension_audit['additional_cost'].mean()
            if self.transition_repair:
                if self.transition_repair['rebound_consistency']:
                    self.extras['log']['Moving/contact_reference_fraction']=(
                        rebound_audit['contact_reference_weight']>0).float().mean()
            if self.score_yaw_peak:
                self.extras['log']['Moving/yaw_peak_cost']=guidance['moving_yaw_peak'].mean()
            if self.all_phase_yaw is not None:
                self.extras['log']['Yaw/all_phase_cost'] = guidance['all_phase_yaw'].mean()
            return reward+adjustment

    return MovingPhaseEnv
