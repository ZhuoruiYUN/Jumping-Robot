"""Frozen controlled keyboard diagnostic; separate from all promotion gates."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    config=Path('benchmark37/configs/obs73_quaternion_omni_guarded_keys_course_v38.json')
    proposal=json.loads(config.read_text())
    parser.add_argument('--checkpoint',type=Path,default=Path(proposal['omni']['initial_checkpoint']))
    parser.add_argument('--finish-last-burst',action='store_true',
                        help='Supplemental playback through 65s to score the final three commands issued at 56.6s')
    parser.add_argument('--terminal-hold',choices=('+Y','minusY','four-axis-rebound'))
    parser.add_argument('--hold-report',type=Path)
    parser.add_argument('--hold-trace',type=Path,help='Optional first-episode control-rate NPZ for sim/real phase alignment')
    parser.add_argument('--cases-file',type=Path,default=Path('outputs/benchmark73_quaternion_omni_reversal_v37/eval_seed7001_20261118_094400_585026Z/cases.json'))
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if bool(args.terminal_hold)!=bool(args.hold_report) or (args.terminal_hold and args.finish_last_burst):
        parser.error('Use --terminal-hold with a fresh --hold-report and no --finish-last-burst')
    if args.hold_report and args.hold_report.exists():
        parser.error('Refusing to overwrite hold report')
    if args.hold_trace and (not args.terminal_hold or args.hold_trace.exists() or args.hold_trace==args.hold_report):
        parser.error('A hold trace requires a terminal-hold evaluation and a distinct fresh output file')
    args.kit_args+=' --/exts/omni.kit.telemetry/skipDeferredStartup=true'
    app=AppLauncher(args).app; raw=None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnvCfg
        from .calibration_20261028_v2 import configure
        from .stationary_retention import env_class
        from .moving_phase_reward import wrap_env as motion
        from .crossing_obs73_quat import wrap_env
        from .obs73_omni_evaluation import OmniEvaluation
        from .cardinal_queue import evaluation_spec
        from .guarded_keys import FIELDS
        course=json.loads((args.checkpoint.parent/'ring_config.json').read_text());course['guarded_keys']=proposal['guarded_keys']
        torch.manual_seed(7001); cfg=QuadhopperEnvCfg();configure(cfg);cfg.seed=7001;cfg.scene.num_envs=128;cfg.sim.device=args.device
        horizon=65. if args.finish_last_burst else 60.
        cfg.observation_space=73;cfg.episode_length_s=horizon;cfg.power_model_path=str(Path('Quadhopper_PPO_Reference_SysID/model/quadhopper_memory_power.pt').resolve())
        import tempfile
        with tempfile.TemporaryDirectory(prefix='hopper_guard_') as tmp:
            cfg.csv_log_path=tmp+'/single_env.csv'
            class Probe(wrap_env(motion(env_class(evaluation=True)))):
                guarded_diagnostic=True

                def _pre_physics_step(self,actions):
                    probe=getattr(self,'hold_probe',None)
                    if args.terminal_hold=='four-axis-rebound' and probe is not None:
                        probe.apply_rebound(self)
                    super()._pre_physics_step(actions)

                def _get_rewards(self):
                    probe=getattr(self,'hold_probe',None)
                    if probe is not None:
                        snapshot=probe.before(self)
                        reward=super()._get_rewards()
                        if args.hold_trace:
                            guard=self.key_guard
                            data=self._robot.data
                            hit=guard.count[:,FIELDS.index('landings')]>snapshot[2]
                            values=torch.column_stack((probe.live,hit,guard.terminal_hold,
                                guard.count[:,FIELDS.index('arrived')],
                                data.root_link_pos_w[:,2]-self._terrain.env_origins[:,2],
                                data.root_link_lin_vel_w,data.root_quat_w,data.root_ang_vel_b,self._actions))
                            trace.append(values.detach().cpu().numpy())
                        probe.after(self,snapshot)
                        return reward
                    reward=super()._get_rewards();g=self.key_guard
                    if hasattr(self,'live'):
                        vals=torch.cat((g.count,g.peak_error[:,None],g.peak_rate[:,None],g.peak_yaw[:,None],self.reset_terminated[:,None]),dim=1)
                        self.key_result=torch.where(self.live[:,None,None],g.by_key,self.key_result)
                        self.result=torch.where(self.live[:,None],vals,self.result)
                        self.live &= ~(self.reset_terminated|self.reset_time_outs)
                    return reward
            raw=Probe(cfg,course);env=RslRlVecEnvWrapper(raw)
            raw.cardinal_queue_evaluation_spec=evaluation_spec(course)
            OmniEvaluation(raw).restore_cases(args.cases_file)
            ids=raw._robot._ALL_INDICES;raw.key_guard.reset(ids);raw._bind_targets(ids)
            runner=OnPolicyRunner(env,json.loads((args.checkpoint.parent/'runner_config.json').read_text()),log_dir=None,device=args.device)
            runner.alg.policy.load_state_dict(torch.load(args.checkpoint,map_location=args.device,weights_only=False)['model_state_dict'],strict=True)
            policy=runner.get_inference_policy(device=raw.device);obs=env.get_observations()
            if args.terminal_hold:
                from .terminal_hold_probe import HoldProbe
                trace=[]
                raw.hold_probe=HoldProbe(raw,args.terminal_hold)
                with torch.inference_mode():
                    for tick in range(round(36./raw.step_dt)):
                        obs,_,done,_=env.step(policy(obs));runner.alg.policy.reset(done)
                        if not raw.hold_probe.live.any():break
                assert not runner.alg.optimizer.state
                report=raw.hold_probe.report()
                report.update(checkpoint=str(args.checkpoint.resolve()),cases_file=str(args.cases_file.resolve()),
                              direction='-Y' if args.terminal_hold=='minusY' else args.terminal_hold,optimizer_updates=0,
                              press_times_s=[14.,14.25],horizon_s=36.,seed=7001)
                if args.terminal_hold=='four-axis-rebound':
                    report.update(rebound_against_travel_m_s=.10,disturbed_endpoint_hops=6)
                if args.hold_trace:
                    import numpy as np
                    with args.hold_trace.open('xb') as output:
                        np.savez_compressed(output,samples=np.stack(trace),dt=raw.step_dt,
                            columns='live hit hold arrivals z vx vy vz qw qx qy qz wx wy wz a1 a2 a3 a4'.split(),
                            checkpoint=str(args.checkpoint.resolve()))
                with args.hold_report.open('x') as output:json.dump(report,output,indent=2,allow_nan=False)
                from .terminal_hold_report import format_report
                print(format_report(report,args.hold_report),flush=True)
                return
            raw.live=torch.ones(128,device=raw.device,dtype=torch.bool);raw.result=torch.zeros(128,len(FIELDS)+4,device=raw.device)
            raw.key_result=torch.zeros_like(raw.key_guard.by_key)
            with torch.inference_mode():
                for tick in range(round(horizon/raw.step_dt)+2):
                    obs,_,done,_=env.step(policy(obs));runner.alg.policy.reset(done)
                    if tick%1000==0:print('GUARDED_KEYS_TICK',tick,flush=True)
                    if not raw.live.any():break
            assert not raw.live.any() and not runner.alg.optimizer.state
            result=raw.result.cpu();n=len(FIELDS)
            def group(a):
                out=dict(zip(raw.key_guard.fields,a[:,:n].sum(0).long().tolist()));assert out['requested']==sum(out[k] for k in ('ignored_recover','arrived','cancelled','pending_at_end'))
                out.update(episodes=len(a),deaths=int(a[:,-1].sum()),max_foot_error_m=float(a[:,n].max()),peak_free_rp_rad_s=float(a[:,n+1].max()),peak_free_yaw_deg=float(a[:,n+2].max()),paused_s=round(out.pop('paused_ticks')*raw.step_dt,2))
                return out
            report=dict(checkpoint=str(args.checkpoint.resolve()),case_file=str(args.cases_file),optimizer_updates=0,
                by_command={d:dict(zip(('launched','arrived','strict_arrived','miss'),row)) for d,row in zip(('W_world_pos_x','S_world_neg_x','A_world_pos_y','D_world_neg_y'),raw.key_result.sum(0).tolist())},
                scheduled_presses_per_case=21,not_issued_due_to_early_end=128*21-int(result[:,0].sum()),
                horizon_s=horizon,
                scope=f'fixed 21 scripted presses through 56.6s, scoring to {horizon:g}s; same releases/DR; not promotion',
                total=group(result),by_initial_direction={d:group(result[i::4]) for i,d in enumerate(('+X','-X','+Y','-Y'))})
            print('GUARDED_KEYS_RESULT='+json.dumps(report),flush=True)
    finally:
        if raw is not None:raw.close()
        app.close()


if __name__=='__main__':main()
