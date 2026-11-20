"""First128 minimal-loop episodes; distinct from the old first-hop protocol."""
import json
from .ring_evaluation import RingEvaluation


class LoopEvaluation(RingEvaluation):
    def restore_cases(self, path):
        super().restore_cases(path)
        e=self.env
        e._reset_task_targets(e._robot._ALL_INDICES)
        from .two_ground_targets import _set_active_goal
        _set_active_goal(e,e._robot._ALL_INDICES)
        if e.course.get('height_randomization'):
            cases=json.loads(path.read_text())['cases']
            e.restore_height_cases(e._robot._ALL_INDICES,cases)

    def freeze_cases(self, folder):
        super().freeze_cases(folder)
        path=folder/'cases.json';payload=json.loads(path.read_text())
        payload['version']='ring_min_loop_seed7001_initial128_v1'
        if self.env.course.get('height_randomization'):
            from .ring_height_loop import CASE_VERSION,ring_apex_target
            e=self.env;payload['version']=CASE_VERSION
            target=ring_apex_target(e._ring_height,e.height_config).cpu().tolist()
            for i,c in enumerate(payload['cases']):
                c['ring_height_local_m']=float(e._ring_height[i])
                c['ring_hop_apex_target_m']=target[i]
        path.write_text(json.dumps(payload,indent=2)+'\n')

    def finish(self):
        start=len(self.rows)
        super().finish()
        for row in self.rows[start:]:
            i=row['case_id'];stage=int(self.env.loop.stage[i])
            row.update(initial_stationary=stage>=1,ring_landing=stage>=2,
                       post_stationary1=stage>=3,loop_complete=stage>=4,
                       joint_success=bool(self.env.ring_joint_success[i]),final_stage=stage)
            # These original-protocol metrics refer to hop1 and are invalid here.
            row.pop('joint_and_three_later_strict',None)
            row.pop('joint_and_survive_episode',None)
            if self.env.course.get('height_randomization'):
                row['ring_height_local_m']=float(self.env._ring_height[i])

    def result(self):
        assert len(self.rows)==128
        rows=sorted(self.rows,key=lambda r:r['case_id']);n=len(rows)
        keys=('initial_stationary','full_crossed','ring_landing','joint_success',
              'post_stationary1','loop_complete','ring_collision','physical_death','reverse','bypass')
        counts={k:sum(bool(r[k]) for r in rows) for k in keys}
        result=dict(episodes=n,seed=7001,case_version='ring_min_loop_seed7001_initial128_v1',
                    case_source=self.case_source,designated_ring_flight=2,
                    denominator='all128 first episodes; drop excluded; no retries; distinct task protocol',
                    counts=counts,rates={k:v/n for k,v in counts.items()},
                    failure_counts_overlapping={'incomplete_loop':n-counts['loop_complete']},
                    promotion_gate=dict(joint_rate_min=.9,physical_deaths_max=0,
                        passed=counts['loop_complete']/n>=.9 and counts['physical_death']==0))
        if self.env.course.get('height_randomization'):
            from .ring_height_loop import CASE_VERSION
            result['case_version']=CASE_VERSION
            result['height_bins']=[]
            for low,high in ((.8,.9),(.9,1.),(1.,1.1),(1.1,1.2)):
                selected=[r for r in rows if low<=r['ring_height_local_m']<high or
                          (high==1.2 and low<=r['ring_height_local_m']<=high+1e-6)]
                count=sum(r['loop_complete'] for r in selected)
                result['height_bins'].append(dict(min_m=low,max_m=high,cases=len(selected),
                    loop_complete=count,rate=count/len(selected) if selected else None))
        return result,rows


def eval_env_class(base):
    class LoopEvalEnv(base):
        def __init__(self,*a,**kw):
            self.evaluation=None
            super().__init__(*a,**kw)
            self.evaluation=LoopEvaluation(self)

        def _get_rewards(self):
            if self.evaluation is not None:self.evaluation.sample()
            reward=super()._get_rewards()
            if self.evaluation is not None:self.evaluation.finish()
            return reward
    return LoopEvalEnv
