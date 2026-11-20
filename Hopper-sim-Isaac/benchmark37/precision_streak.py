"""Bounded per-touch shaping for uninterrupted sequences of distinct waypoints."""
import torch


def adjacent_landing(previous_quality, previous_index, index, counted, strict, foot_error, radius):
    """Quality of two adjacent first attempts; never depends on accumulated streak length."""
    eligible=counted & strict & (index!=previous_index) & (foot_error<=radius)
    quality=torch.where(eligible,torch.exp(-.5*(foot_error/radius).square()),0.)
    pair=torch.where(counted & (index==previous_index+1),torch.minimum(previous_quality,quality),0.)
    return torch.where(counted,quality,previous_quality),pair


def streak_step(streak, previous, index, touch, strict, weight):
    first=previous!=index
    adjacent=(index==previous+1) | (previous<0)
    earned=touch & strict & first & adjacent
    updated=torch.where(touch,torch.where(earned,(streak+1).clamp_max(10),0),streak)
    bonus=torch.where(earned,weight*(.25+.75*updated.float()/10.),0.)
    return updated,bonus
