"""Opt-in post-ring pitch costs; no action filtering or physics changes."""
import math
import torch


def validate(config):
    for key in ('pitch_weight','pitch_rate_weight'):
        value=config[key]
        if not math.isfinite(value) or value<0:
            raise ValueError(f'Invalid post-ring stability coefficient: {key}')


def active_mask(stage,ring_joint_success,reset_buf):
    # Ground queue has already become P_t=P_next=L after hop1. Only start
    # after the designated ring flight has also landed strictly at L.
    return (stage>=2)&ring_joint_success&~reset_buf


def terms(quaternion,pitch_rate,active,step_dt,config):
    w,x,y,z=quaternion.unbind(-1)
    pitch=torch.asin((2*(w*y-z*x)).clamp(-1.,1.))
    return {
        'post_ring_pitch':torch.where(active,-config['pitch_weight']*pitch.square()*step_dt,0.),
        'post_ring_pitch_rate':torch.where(active,-config['pitch_rate_weight']*pitch_rate.square()*step_dt,0.)
    },pitch
