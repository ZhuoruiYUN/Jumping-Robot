"""v19 task-only observation experiment over the existing spatial task."""
import json
import torch
from pxr import Usd
from isaaclab.utils import configclass
from .spatial_tracking_env import SpatialTrackingEnv, SpatialTrackingPhysical1mEnvCfg
from .tracking_metrics import TrackingMetrics
from .history_foot import OBS_DIM, ObservationHistory, foot_offset_world, observation_frame


def read_foot_geometry(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    data = json.loads(stage.GetDefaultPrim().GetCustomDataByKey('hopper_measurements_json'))
    joint = stage.GetPrimAtPath('/Drone/center_spring_joint')
    if joint.GetAttribute('physics:axis').Get() != 'Z':
        raise ValueError('Foot estimator expects the calibrated Z-axis prismatic joint')
    for name in ('physics:localRot0', 'physics:localRot1'):
        rotation = joint.GetAttribute(name).Get()
        if abs(float(rotation.GetReal()) - 1) > 1e-6 or any(abs(v) > 1e-6 for v in rotation.GetImaginary()):
            raise ValueError('Foot estimator requires aligned joint frames')
    p0 = joint.GetAttribute('physics:localPos0').Get()
    p1 = joint.GetAttribute('physics:localPos1').Get()
    rest = [float(a-b) for a, b in zip(p0, p1)]
    if not data['foot']['spring_leg_origin_at_foot_center']:
        raise ValueError('Expected SpringLeg origin at the foot sphere center')
    if max(abs(a-b) for a, b in zip(rest, data['foot']['center_m'])) > 1e-6:
        raise ValueError('USD joint geometry disagrees with foot metadata')
    return dict(rest_offset_body_m=rest, sphere_radius_m=data['foot']['radius_m'],
                spring_travel_m=data['joint']['travel_m'], reference='Body link origin')


@configclass
class HistoryFootEnvCfg(SpatialTrackingPhysical1mEnvCfg):
    observation_space = OBS_DIM


class HistoryFootEnv(SpatialTrackingEnv):
    def __init__(self, cfg, **kwargs):
        self.foot_geometry = read_foot_geometry(cfg.robot.spawn.usd_path)
        super().__init__(cfg, **kwargs)
        self.observation_history = ObservationHistory(self.num_envs, self.device)
        self.foot_tracking_metrics = TrackingMetrics(self.num_envs, self.device)
        self._foot_body_id = self._robot.find_bodies('SpringLeg')[0][0]
        self.foot_kinematic_error_max = torch.zeros((), device=self.device)

    def foot_position_w(self):
        return self._robot.data.root_pos_w + foot_offset_world(
            self._robot.data.root_quat_w,
            self._robot.data.joint_pos[:, self._spring_joint_id],
            self.foot_geometry['rest_offset_body_m'])

    def _get_observations(self):
        history = getattr(self, 'observation_history', None)
        if history is not None and bool((history.ticks == self.common_step_counter).all()):
            return {'policy': history.frames.flatten(1).clone()}
        obs = super()._get_observations()
        target, _ = self.commands.lookahead()
        target = torch.where(self._initial_drop_pending[:, None], self.commands.start_points(), target)
        frame = observation_frame(obs['policy'], target - self._robot.data.root_pos_w[:, :2],
                                  self.foot_geometry['rest_offset_body_m'],
                                  self.foot_geometry['spring_travel_m'])
        if history is None:
            self.observation_history = ObservationHistory(self.num_envs, self.device)
        obs['policy'] = self.observation_history.update(frame, self.common_step_counter)
        return obs

    def _reset_idx(self, env_ids):
        if hasattr(self, 'observation_history'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self.observation_history.reset(ids)
        if hasattr(self, 'foot_tracking_metrics'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self.foot_tracking_metrics.reset(ids)
        super()._reset_idx(env_ids)

    def _update_cycle_events(self):
        # Latch before the parent advances the waypoint queue.
        target = self.commands.lookahead()[0].clone()
        direction = target - self.commands.anchor_w
        super()._update_cycle_events()
        if not self.cfg.collect_tracking_metrics or not hasattr(self, 'foot_tracking_metrics'):
            return
        foot = self.foot_position_w()
        actual = self._robot.data.body_link_pos_w[:, self._foot_body_id]
        self.foot_kinematic_error_max.copy_(torch.maximum(
            self.foot_kinematic_error_max, (foot-actual).norm(dim=-1).max()))
        ids = self._touchdown_event.nonzero(as_tuple=False).flatten()
        delta = foot[ids, :2] - target[ids]
        d = direction[ids] / direction[ids].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.foot_tracking_metrics.observe(
            ids, self._setup_touchdown_event[ids], delta.norm(dim=-1),
            (delta*d).sum(-1), (delta[:, 0]*d[:, 1]-delta[:, 1]*d[:, 0]).abs(),
            self._settled_apex_height[ids], self._settled_apex_target[ids])

    def foot_summary(self):
        return {**{f'foot_{k}': v for k, v in self.foot_tracking_metrics.summary().items()},
                'foot_kinematic_error_max_m': float(self.foot_kinematic_error_max.item())}
