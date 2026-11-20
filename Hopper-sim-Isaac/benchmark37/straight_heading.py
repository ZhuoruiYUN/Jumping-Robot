"""Redistribute only the existing ten-step training heading draw across eight sectors."""
import math

import torch


def weighted_heading(angle, weights):
    if len(weights)!=8 or not all(math.isfinite(value) and 1.<=value<=3. for value in weights):
        raise ValueError('Expected eight bounded positive heading weights')
    width=math.pi/4
    mass=angle.new_tensor(weights)
    cumulative=mass.cumsum(dim=0)
    quantile=(angle+width/2).remainder(2*math.pi)*(cumulative[-1]/(2*math.pi))
    sector=torch.bucketize(quantile,cumulative,right=True).clamp_max(7)
    fraction=(quantile-(cumulative-mass)[sector])/mass[sector]
    return (sector+fraction-.5)*width
