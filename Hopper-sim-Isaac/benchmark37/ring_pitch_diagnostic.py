"""Optional env0 playback trace and per-hop pitch inspection; no policy changes."""
import argparse
import csv
import json
import math
from pathlib import Path

STATE_COLUMNS = (
    'root_x root_y root_z vx vy vz qw qx qy qz wx wy wz spring_q spring_qdot '
    'action0 action1 action2 action3 motor0 motor1 motor2 motor3 '
    'moment_x moment_y moment_z foot_x foot_y foot_z '
    'target_x target_y next_x next_y phase attempt stage ring_center_z goal_z '
    'route_index thrust_world_x thrust_world_y thrust_world_z'
).split()
TRACE_COLUMNS = ['step','episode','seconds'] + STATE_COLUMNS + ['stage_after','strict','done','passed']


def snapshot(env):
    import torch
    from .ring_geometry import rotate
    d=env._robot.data;f=env._foot_event;j=env._spring_joint_id
    state=torch.cat((
        d.root_link_pos_w[0]-env._terrain.env_origins[0],d.root_link_lin_vel_w[0],
        d.root_quat_w[0],d.root_ang_vel_b[0],d.joint_pos[0,j].reshape(1),d.joint_vel[0,j].reshape(1),
        env._actions[0],env._motor_u[0],env._moment[0,0],
        d.body_link_pos_w[0,env._foot_body_id]-env._terrain.env_origins[0],
        env._two_ground_pt_xy_w[0]-env._terrain.env_origins[0,:2],
        env._two_ground_pt1_xy_w[0]-env._terrain.env_origins[0,:2],
        f.phase[0].reshape(1),
        f.attempt_index[0].reshape(1) if hasattr(f,'attempt_index') else env.stationary_episode_pitch[0,:1],
        env.loop.stage[0].reshape(1) if hasattr(env,'loop') else f.phase.new_tensor([-1]),
        (env.ring.center[0,2]-env._terrain.env_origins[0,2]).reshape(1),env._desired_pos_w[0,2].reshape(1),
        env.omni_route.index[0].reshape(1) if hasattr(env,'omni_route') else f.phase.new_tensor([-1]),
        rotate(d.root_quat_w[0],env._thrust[0,0])))
    values=state.detach().cpu().tolist()
    assert len(values)==len(STATE_COLUMNS)
    return [int(env.common_step_counter),env.preview_episode,float(env.episode_length_buf[0])*env.step_dt]+values


def analyze(path):
    with path.open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    groups={}
    for r in rows:
        r['pitch_deg']=math.degrees(math.asin(max(-1.,min(1.,2*(r['qw']*r['qy']-r['qz']*r['qx'])))))
        r['motor_pitch_difference']=r['motor0']+r['motor1']-r['motor2']-r['motor3']
        if (r['stage']>=2 or r['stage']==-1) and r['phase'] in (2,3):
            groups.setdefault((int(r['episode']),int(r['attempt'])+1),[]).append(r)
    hops=[]
    for (episode,index),samples in groups.items():
        pitch=[r['pitch_deg'] for r in samples];rate=[r['wy'] for r in samples]
        touch=[r for r in samples if r['spring_q']>.002 and r['phase']==3]
        hops.append(dict(episode=episode,jump_index=index,samples=len(samples),
            completed=bool(touch),strict=bool(touch and touch[-1]['strict']),
            pitch_min_deg=min(pitch),pitch_max_deg=max(pitch),pitch_span_deg=max(pitch)-min(pitch),
            pitch_rate_rms_rad_s=math.sqrt(sum(v*v for v in rate)/len(rate)),
            pitch_rate_peak_rad_s=max(map(abs,rate)),
            motor_pitch_difference_peak=max(abs(r['motor_pitch_difference']) for r in samples),
            target_xy_start=[samples[0]['target_x'],samples[0]['target_y']],
            target_xy_end=[samples[-1]['target_x'],samples[-1]['target_y']]))
    complete=[h for h in hops if h['completed']]
    out=dict(scope='one case playback continuation; pitch span10deg is a diagnostic flag, not a task success criterion',
        source=str(path),completed_post_ring_hops=len(complete),
        hops_with_pitch_span_gt10deg=sum(h['pitch_span_deg']>10 for h in complete),
        first_two_post_ring=[h for h in complete if h['jump_index'] in (3,4)],
        later_post_ring=[h for h in complete if h['jump_index']>=5],hops=hops)
    if rows and rows[0]['stage']==-1:
        out['completed_vertical_hops']=out.pop('completed_post_ring_hops')
        out.pop('first_two_post_ring');out.pop('later_post_ring')
    return out,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--check',action='store_true',help='Fail if a completed stationary hop has pitch span>10deg')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();out,rows=analyze(args.trace)
    args.trace.with_suffix('.summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    if args.plot:
        import os
        os.environ.setdefault('MPLCONFIGDIR','/tmp/benchmark37_matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
        for key,ax,label in zip(('pitch_deg','wy','motor_pitch_difference'),axes,
                               ('Pitch (deg)','Pitch body rate (rad/s)','Motor0+1 - motor2-3')):
            ax.plot([r['step']*.01 for r in rows],[r[key] for r in rows],lw=1)
            ax.axhline(0,color='grey',lw=.6);ax.set_ylabel(label);ax.grid(alpha=.2)
            for r in rows:
                if r['strict'] and (r['stage']>=2 or r['stage']==-1):ax.axvline(r['step']*.01,color='#30965e',alpha=.4,lw=.8)
        axes[-1].set_xlabel('Simulation time (s); green lines = strict stationary landings')
        fig.savefig(args.trace.with_suffix('.png'),dpi=150);plt.close(fig)
    if args.check and out['hops_with_pitch_span_gt10deg']:
        raise SystemExit('POST_RING_PITCH_CHECK=FAIL: completed stationary hop pitch span exceeds10deg')


if __name__=='__main__':main()
