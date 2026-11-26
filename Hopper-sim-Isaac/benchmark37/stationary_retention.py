"""Stationary rehearsal and strict saved-v23 retention checks for new OBS73.

This module never exports or overwrites a deployed model. Passing a finite
simulation gate is a candidate-selection rule, not a hardware guarantee.
"""
import json
import math
from pathlib import Path

import torch

from .obs73_omni import GroundRoute
from .obs73_reference_reward import flight_terms


class RetentionRoute(GroundRoute):
    def __init__(self,n,device,config,seed,evaluation=False,stationary_only=False,training_mixture=None,
                 training_stationary_fraction=.5):
        super().__init__(n,device,config,seed,evaluation,training_mixture=training_mixture)
        ab_only = config.get('ab_pair_only', False)
        if ab_only and (not config.get('ab_pair_training') or training_stationary_fraction != 0.):
            raise ValueError('AB-only courses require AB sampling and zero stationary training lanes')
        if not ab_only and training_stationary_fraction not in (.25,.5):
            raise ValueError('Supported stationary rehearsal fractions are one quarter and one half')
        self.stationary_mask=torch.ones(n,dtype=torch.bool,device=device) if stationary_only else (
            torch.zeros(n,dtype=torch.bool,device=device) if evaluation or ab_only else
            torch.arange(n,device=device)%int(1/training_stationary_fraction)==0)
        self.short_episode_mask=self.stationary_mask
        self.ab_pairs=None
        if config.get('ab_pair_training') and not evaluation and not stationary_only:
            from .ab_pairs import ABPairs
            self.ab_pairs=ABPairs(n,device,seed,focus=config.get('ab_full_pair_focus',False),
                                  all_lanes=ab_only,recovery_config=config.get('ab_miss_recovery'),
                                  paired_axes=config.get('ab_paired_axes', False),
                                  opening_hold_probability=config.get('ab_opening_hold_probability', 0.),
                                  repeated_hold_fraction=config.get('ab_repeated_hold_fraction', 0.))
        if config.get('startup_rehearsal') and not (evaluation or stationary_only or ab_only):
            # Ordinary moving environments; keyboard is slot 1, distance practice is slot 3.
            self.startup_mask=torch.arange(n,device=device)%8==5
            self.short_episode_mask=self.stationary_mask | self.startup_mask
            self.startup_starts=torch.zeros((),dtype=torch.long,device=device)
        self.distance_practice=None
        if config.get('precision_distance_practice') and not (evaluation or stationary_only or ab_only):
            from .precision_distance import DistancePractice
            if config['moving_distance_range_m'][1]!=.5:
                raise ValueError('Distance practice requires the half-metre course')
            self.distance_practice=DistancePractice(n,device)

    def reset(self,ids,common_step_counter=0):
        if self.ab_pairs is not None and self.ab_pairs.all_lanes:
            self.ab_pairs.reset(self, ids)
            self.sampled_max[ids] = self.current_max(common_step_counter)
            if self.adjacent_metrics is not None:
                self.adjacent_metrics.reset(ids)
            return
        super().reset(ids,common_step_counter)
        if self.ab_pairs is not None:
            self.ab_pairs.reset(self,ids)
            self.sequence_kind[ids[self.ab_pairs.mask[ids]]]=0
        if hasattr(self,'startup_starts'):
            self.startup_starts+=self.startup_mask[ids].sum()
        if self.distance_practice is not None:
            self.distance_practice.reset(self,ids)
        self.points[ids[self.stationary_mask[ids]]]=0.
        self.sequence_kind[ids[self.stationary_mask[ids]]]=0


def stationary_reward_delta(z,vz,target,gravity,energy_width,scales,dt,current,baseline,mask):
    actual=flight_terms(z,vz,target,gravity,energy_width,scales,dt,current)
    reference=flight_terms(z,vz,target,gravity,energy_width,scales,dt,baseline)
    return {key:torch.where(mask,reference[key]-actual[key],0.) for key in actual}


def env_class(evaluation=False,stationary_only=False):
    from .obs73_omni import env_class as omni_class

    class RetentionEnv(omni_class(evaluation=evaluation)):
        def _get_dones(self):
            died,timeout=super()._get_dones()
            horizon=self.course.get('long_moving_horizon')
            if horizon and hasattr(self,'omni_route'):
                # Selected release rehearsals use the stationary episode duration
                # with ordinary moving routes, rewards and timeout bootstrapping.
                ticks=math.ceil(horizon['stationary_s']/self.step_dt)
                timeout=timeout|(self.omni_route.short_episode_mask&(self.episode_length_buf>=ticks-1))
            return died,timeout

        def __init__(self,cfg,course,**kwargs):
            c=course['stationary_retention']
            super().__init__(cfg,course,**kwargs)
            self._retention_reference=c['baseline_reference_flight_reward']
            reduction=c.get('rehearsal_margin_reduction_m',0.)
            if not math.isfinite(reduction) or not 0.<=reduction<=min(
                    self._retention_reference.get(k,0.) for k in ('energy_margin_m','height_reward_margin_m')):
                raise ValueError('Stationary rehearsal reward reduction exceeds the original margin')
            if reduction and not evaluation:
                self._retention_reference=dict(self._retention_reference)
                for key in ('energy_margin_m','height_reward_margin_m'):
                    self._retention_reference[key]-=reduction
            self._episode_sums['stationary_retention_margin_delta']=torch.zeros(self.num_envs,device=self.device)

        def _reset_task_targets(self,ids):
            if not hasattr(self,'omni_route'):
                self.omni_route=RetentionRoute(self.num_envs,self.device,self.course['omni'],
                    self.cfg.seed,evaluation=evaluation,stationary_only=stationary_only,
                    training_mixture=self.course.get('moving_training_sequences'),
                    training_stationary_fraction=self.course['stationary_retention']['training_stationary_fraction'])
            super()._reset_task_targets(ids)

        def _get_rewards(self):
            reward=super()._get_rewards()
            d=self._robot.data
            deltas=stationary_reward_delta(d.root_pos_w[:,2],d.root_lin_vel_w[:,2],self._desired_pos_w[:,2],
                abs(self.sim.cfg.gravity[2]),self.cfg.vertical_energy_error_scale,
                {k:getattr(self.cfg,k+'_reward_scale') for k in ('energy_tracking','distance_to_z','apex')},
                self.step_dt,self.reference_reward,self._retention_reference,self.omni_route.stationary_mask)
            total=torch.zeros_like(reward)
            for name,delta in deltas.items():
                self._episode_sums[name]+=delta;total+=delta
            self._episode_sums['stationary_retention_margin_delta']+=total
            self.extras.setdefault('log',{}).update({
                'Retention/stationary_env_fraction':self.omni_route.stationary_mask.float().mean(),
                'Retention/stationary_reward_delta':total.mean()})
            return reward+total

    return RetentionEnv


def compare_stationary(candidate,baseline):
    """No tolerance relaxation: each response stratum must match saved v23.

    Metrics below cover jump5+; fixed128 first episodes include deaths and
    missing coverage. First-four strict count is checked separately.
    """
    groups={}
    for name in ('nominal','weaker'):
        ref=baseline['response_strata'][name]
        got=candidate.get('response_strata',{}).get(name)
        checks={}
        if got:
            checks=dict(episodes=got['episodes']==ref['episodes'],
                enough_later_landings=got['episodes_with_enough_later_landings']==got['episodes'],
                coverage=bool(got['metric_coverage_complete']),
                landings_not_fewer=got['completed_landings']>=ref['completed_landings'],
                strict_rate=got['strict_landing_rate']>=ref['strict_landing_rate'],
                zero_deaths=got['physical_deaths_entire_episode']==0)
            for label,key in (('foot_p95','p95'),('foot_mean','mean'),('foot_p99','p99'),('foot_max','maximum')):
                value=got['foot_error_m'][key]
                checks[label]=value is not None and value<=ref['foot_error_m'][key]
            for label in ('mean_span_deg','span_gt10_rate','mean_rate_rms_rad_s'):
                value=got['pitch'][label]
                checks['pitch_'+label]=value is not None and value<=ref['pitch'][label]
            value=got['apex_world_m']['p05']
            checks['apex_p05']=value is not None and value>=ref['apex_world_m']['p05']
        groups[name]=dict(checks=checks,passed=bool(checks) and all(checks.values()),
            reference=dict(strict_rate_min=ref['strict_landing_rate'],foot_p95_m_max=ref['foot_error_m']['p95'],
                foot_mean_m_max=ref['foot_error_m']['mean'],pitch=ref['pitch']))
    startup={key:candidate.get('counts',{}).get(key,0)>=baseline['counts'][key]
             for key in ('first_strict','first_pair_strict','first_four_strict')}
    first=all(startup.values())
    alive=candidate.get('counts',{}).get('physical_death',128)==0
    cases=candidate.get('episodes')==128 and candidate.get('seed')==7001
    return dict(passed=cases and first and alive and all(g['passed'] for g in groups.values()),
        cases128_seed7001=cases,startup_counts_not_lower=startup,
        first_four_strict_min=baseline['counts']['first_four_strict'],zero_deaths=alive,
        response_groups=groups,
        scope='No observed regression versus saved deployed-v23 simulation result, same cases; not a proof of hardware equivalence')


def eval_env_class(base,root,course):
    from .obs73_stationary import eval_env_class as stationary_evaluator
    reference_path=Path(root)/course['stationary_retention']['baseline_summary']
    reference=json.loads(reference_path.read_text())
    adapted=stationary_evaluator(base)

    class RetentionEvalEnv(adapted):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            original_result=self.evaluation.result
            def result():
                summary,rows=original_result()
                summary['stationary_absolute_convergence_gate']=summary['promotion_gate']
                summary['stationary_retention_gate']=compare_stationary(summary,reference)
                summary['promotion_gate']=summary['stationary_retention_gate']
                if self.course.get('flight_plane_limits'):
                    from .flight_plane import flight_plane_report
                    limits = self.course['flight_plane_limits']
                    report = flight_plane_report(rows,limits,('hold',))
                    summary['flight_plane'] = dict(limits=limits, **report)
                    gate = summary['promotion_gate']
                    gate['flight_plane_passed'] = report['hold']['passed']
                    errors = [ev['foot_error_m'] for row in rows for ev in row['events']]
                    gate['all_touchdowns_within_radius'] = bool(errors) and max(errors)<=self.landing_radius_m
                    gate['passed'] = gate['passed'] and gate['flight_plane_passed'] and gate['all_touchdowns_within_radius']
                summary['stationary_retention_ready']=summary['promotion_gate']['passed']
                summary['retention_reference']=str(reference_path.resolve())
                summary['case_version']='obs73_quaternion_stationary_retention_deployed_v23_seed7001_v1'
                return summary,rows
            self.evaluation.result=result

    return RetentionEvalEnv
