"""Episode-fixed ring Z curriculum; only the designated ring hop changes height."""
import torch

CASE_VERSION='ring_min_loop_height_seed7001_initial128_v1'


def upper_bound(config,control_step,evaluation=False):
    if evaluation:return config['max_z_m']
    update=control_step/config['steps_per_update']
    progress=max(0.,min(1.,(update-config['warmup_updates'])/
                           (config['full_range_update']-config['warmup_updates'])))
    return config['initial_max_z_m']+progress*(config['max_z_m']-config['initial_max_z_m'])


def ring_apex_target(height,config):
    return config['stationary_apex_target_m']+(height-config['reference_ring_z_m']).clamp(min=0.)


def phase_apex_target(stage,height,config):
    return torch.where(stage==1,ring_apex_target(height,config),config['stationary_apex_target_m'])


def env_class(evaluation=False):
    from .ring_min_loop import env_class as base_class

    class RingHeightEnv(base_class()):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.height_config=self.course['height_randomization']
            c=self.height_config
            if not (0<c['min_z_m']<=c['initial_max_z_m']<=c['max_z_m']
                    and c['full_range_update']>c['warmup_updates']>=0 and c['steps_per_update']==256):
                raise ValueError('Invalid ring-height curriculum')
            self._height_rng=torch.Generator(device=self.device).manual_seed(int(self.cfg.seed)+290511)
            self._ring_height=torch.full((self.num_envs,),self.course['center_local_m'][2],device=self.device)

        def _reset_task_targets(self,ids):
            super()._reset_task_targets(ids)
            c=self.height_config
            high=upper_bound(c,self.common_step_counter,evaluation)
            self._ring_height[ids]=c['min_z_m']+(high-c['min_z_m'])*torch.rand(
                len(ids),device=self.device,generator=self._height_rng)
            self._desired_pos_w[ids,2]=c['stationary_apex_target_m']

        def refresh_ring_height(self,ids):
            self.ring.center[ids,2]=self._terrain.env_origins[ids,2]+self._ring_height[ids]

        def restore_height_cases(self,ids,cases):
            # Old cases restore release/DR only; new cases also freeze geometry.
            if any('ring_height_local_m' in c for c in cases):
                if not all('ring_height_local_m' in c for c in cases):raise ValueError('Mixed height case formats')
                h=torch.tensor([c['ring_height_local_m'] for c in cases],device=self.device)
                cfg=self.height_config
                if not torch.isfinite(h).all() or ((h<cfg['min_z_m']-1e-6)|(h>cfg['max_z_m']+1e-6)).any():
                    raise ValueError('Saved ring height outside this task range')
                self._ring_height[ids]=h
            self.refresh_ring_height(ids)

        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if hasattr(self,'_ring_height'):
                self.refresh_ring_height(self._robot._ALL_INDICES if ids is None else ids)

        def _landing_event(self,touch,designated,died):
            super()._landing_event(touch,designated,died)
            self._desired_pos_w[:,2]=phase_apex_target(self.loop.stage,self._ring_height,self.height_config)

        def _get_rewards(self):
            reward=super()._get_rewards()
            self.extras.setdefault('log',{}).update({
                'RingHeight/sampling_upper_m':upper_bound(self.height_config,self.common_step_counter,evaluation),
                'RingHeight/mean_m':self._ring_height.mean()})
            return reward

    return RingHeightEnv
