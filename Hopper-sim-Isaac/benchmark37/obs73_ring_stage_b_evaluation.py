"""Exactly first128 Stage B episodes; no retries or policy updates."""
import json
import math

import torch

from .ring_evaluation import RingEvaluation
from .metrics import distribution, summarize
from .obs73_ring_stage_b import CASE_VERSION, TASK


class StageBEvaluation(RingEvaluation):
    def restore_cases(self, path):
        super().restore_cases(path)
        e = self.env;ids = e._robot._ALL_INDICES
        e._reset_task_targets(ids)
        from .two_ground_targets import _set_active_goal
        _set_active_goal(e,ids)
        cases = json.loads(path.read_text())['cases']
        e.response_envelope.restore(cases,e.dr_inertia_multi)
        # Saved vertical contexts may have random Z; they do not define this
        # task's ring. Keep the fixed course centre, plus each tile origin.
        e.ring.center[:] = e._terrain.env_origins + e.ring.center.new_tensor(e.course['center_local_m'])

    def freeze_cases(self, folder):
        super().freeze_cases(folder)
        path = folder/'cases.json';data = json.loads(path.read_text())
        data.update(version=CASE_VERSION,task=TASK,post_ring_strict_hops=3,
                    geometry='fixed wide virtual ring; no height randomization')
        self.env.response_envelope.annotate(data['cases'])
        path.write_text(json.dumps(data,indent=2)+'\n')

    def physics(self):
        e = self.env;d = e._robot.data;r = e.ring;f = e._foot_event
        q = d.root_quat_w
        tilt = torch.rad2deg(torch.acos((1-2*q[:,1:3].square().sum(-1)).clamp(-1,1)))
        vertices = r.shape.vertices(d.root_link_pos_w,q,d.joint_pos[:,e._spring_joint_id])
        signed = ((vertices-r.center[:,None,None,:])*r.normal[:,None,None,:]).sum(-1)
        values = torch.cat((d.root_link_pos_w-e._terrain.env_origins,q,d.root_com_lin_vel_w,
            d.root_ang_vel_b,d.joint_pos[:,e._spring_joint_id,None],tilt[:,None],
            signed.amax((1,2))[:,None],e.loop.stage[:,None].float(),f.phase[:,None].float(),
            r.root_crossed[:,None].float(),r.passed[:,None].float(),r.collision[:,None].float()),-1).detach().cpu().tolist()
        for i,a in enumerate(values):
            if self.finished[i]:
                continue
            snap = dict(position_local_m=a[:3],quaternion_wxyz=a[3:7],velocity_world_m_s=a[7:10],
                        angular_velocity_body_rad_s=a[10:13],spring_q_m=a[13],tilt_deg=a[14])
            window = self.window[i]
            if int(a[16]) == e.loop.RING and int(a[17]) in (f.CANDIDATE,f.FLIGHT):
                self.launched[i] = True
                self.confirmed[i] |= int(a[17]) == f.FLIGHT
                if a[15] >= -e.course['axial_thickness_m']/2 and window['entry'] is None:
                    window['before_entry'] = self.previous[i];window['entry'] = snap
                if window['entry'] is not None and window['complete_exit'] is None:
                    window['maximum_tilt_deg'] = max(window['maximum_tilt_deg'],a[14])
                    window['maximum_angular_speed_rad_s'] = max(window['maximum_angular_speed_rad_s'],math.sqrt(sum(x*x for x in a[10:13])))
            if a[18] and window['root_crossing'] is None:
                window['root_crossing'] = snap
            if a[19] and window['complete_exit'] is None:
                window['complete_exit'] = snap
            if a[20] and self.contacts[i] is None:
                self.contacts[i] = dict(before=self.previous[i],current=snap,
                    control_step=int(e.episode_length_buf[i]),task_stage=int(a[16]),flight_phase=int(a[17]))
            self.previous[i] = snap

    def finish(self):
        start = len(self.rows)
        super().finish()
        for row in self.rows[start:]:
            i = row['case_id'];stage = int(self.env.loop.stage[i])
            row.update(first_stationary_strict=stage>=1,ring_landing_strict=stage>=2,
                post1_strict=stage>=3,post2_strict=stage>=4,post3_strict=stage>=5,
                loop_complete=stage==5 and not bool(self.env.loop.failed[i]),
                final_stage=stage,failed_landing=bool(self.env.loop.failed[i]),
                response_group='nominal' if i<64 else 'weaker',
                response_gain=float(self.env.response_envelope.gain[i]),
                stationary_pitch_totals=self.env.stationary_episode_pitch[i].cpu().tolist(),
                vertical_yaw_totals=self.env.vertical_episode_yaw[i].cpu().tolist())
            row['joint_success'] = (row['loop_complete'] and row['full_crossed']
                and not row['ring_collision'] and not row['physical_death']
                and not row['reverse'] and not row['bypass'])
            # This protocol has five true hops and its own explicit post3 key.
            row.pop('joint_and_three_later_strict',None)
            row.pop('joint_and_survive_episode',None)

    def result(self):
        assert len(self.rows)==128 and len({r['case_id'] for r in self.rows})==128
        rows = sorted(self.rows,key=lambda r:r['case_id'])
        keys = ('first_stationary_strict','attempted_takeoff','confirmed_takeoff','root_crossed','full_crossed',
                'ring_landing_strict','post1_strict','post2_strict','post3_strict','loop_complete','joint_success',
                'ring_collision','partial_root_only','reverse','bypass','physical_death','timed_out')
        counts = {key:sum(bool(r[key]) for r in rows) for key in keys}
        events = [event for r in rows for event in r['events']]
        second = [r['events'][1] for r in rows if len(r['events'])>=2]
        pitch = torch.tensor([r['stationary_pitch_totals'] for r in rows]).sum(0).tolist()
        yaw = torch.tensor([r['vertical_yaw_totals'] for r in rows]);duration = yaw[:,2].sum().clamp(min=.01)
        peaks = torch.rad2deg(yaw[:,3])
        result = dict(episodes=128,seed=7001,case_version=CASE_VERSION,task=TASK,
            case_source=self.case_source,benchmark_variant=self.env.course['version'],
            designated_ring_flight=2,post_ring_strict_hops=3,
            denominator='all128 first episodes, maximum15s; initial drop excluded; no retries; joint requires all five strict hops and clean forward full-body hop2 passage',
            counts=counts,rates={key:value/128 for key,value in counts.items()},
            failure_counts_overlapping=dict(first_stationary_missing_or_failed=128-counts['first_stationary_strict'],
                no_full_crossing=128-counts['full_crossed'],missing_ring_touchdown=128-len(second),
                ring_underheight=sum(x['apex_root_link_world_m']<1 for x in second),
                ring_foot_gt10cm=sum(x['foot_error_m']>.1 for x in second),
                ring_tilt_gt15deg=sum(x['touchdown_tilt_deg']>15 for x in second),
                incomplete_post3=128-counts['post3_strict'],timeout=counts['timed_out']),
            strict_landing_rate=sum(x['strict'] for x in events)/max(1,len(events)),
            ring_landing=dict(apex_world_m=distribution([x['apex_root_link_world_m'] for x in second]),
                foot_error_m=distribution([x['foot_error_m'] for x in second]),
                touchdown_tilt_deg=distribution([x['touchdown_tilt_deg'] for x in second])),
            pitch=dict(completed_flights=int(pitch[0]),mean_span_deg=pitch[1]/max(1,pitch[0]),
                span_gt10_rate=pitch[2]/max(1,pitch[0]),mean_rate_rms_rad_s=pitch[3]/max(1,pitch[0])),
            yaw=dict(heading_error_rms_deg=float(torch.rad2deg(torch.sqrt(yaw[:,0].sum()/duration))),
                rate_rms_rad_s=float(torch.sqrt(yaw[:,1].sum()/duration)),
                peak_abs_heading_error_deg=float(peaks.max()),episode_peak_abs_heading_error_p95_deg=float(torch.quantile(peaks,.95))),
            minimum_clearance_m=distribution([r['minimum_signed_clearance_m'] for r in rows if r['minimum_signed_clearance_m'] is not None]),
            full_crossing_clearance_m=distribution([r['minimum_signed_clearance_m'] for r in rows if r['full_crossed'] and r['minimum_signed_clearance_m'] is not None]),
            continuous_jump_metrics=summarize(rows),
            promotion_gate=dict(joint_success_min=116,cases=128,physical_deaths_max=0,
                passed=counts['joint_success']>=116 and counts['physical_death']==0),
            collision_scope='engineering full-body swept geometry, not a physically contacting ring;100mm rotor allowance and2mm numerical margin',
            stage='B; no shrinking or physical-ring promotion implied')
        result['response_strata'] = {}
        for group in ('nominal','weaker'):
            selected = [r for r in rows if r['response_group']==group]
            result['response_strata'][group] = dict(episodes=len(selected),
                counts={key:sum(bool(r[key]) for r in selected) for key in keys})
        return result,rows


def eval_env_class(base):
    class StageBEvalEnv(base):
        def __init__(self,*args,**kwargs):
            self.evaluation = None
            super().__init__(*args,**kwargs)
            self.evaluation = StageBEvaluation(self)

        def _ring_sample(self):
            super()._ring_sample()
            if self.evaluation is not None:
                self.evaluation.physics()

        def _get_rewards(self):
            if self.evaluation is not None:
                self.evaluation.sample()
            reward = super()._get_rewards()
            if self.evaluation is not None:
                self.evaluation.finish()
            return reward

    return StageBEvalEnv
