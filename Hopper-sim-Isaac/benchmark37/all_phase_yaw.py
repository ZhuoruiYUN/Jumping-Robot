"""Heading objectives and first-episode diagnostics sampled at the control rate."""
import math

import torch


def yaw_envelope_cost(error, heading_rate, body_wz, dt, prediction, config):
    horizon = prediction['response_horizon_s']
    predicted = (error + horizon * heading_rate).abs()
    # A projected forward axis can turn through pi near a side fall. Limit
    # the forecast cost at a half turn; report the original forecast in eval.
    excursion = torch.maximum(error.abs(), predicted).clamp(max=math.pi)
    excess = (excursion - math.radians(config['warning_deg'])).clamp_min(0)
    cost = (config['heading_weight'] * error.square()
            + config['forecast_excess_weight'] * excess.square()
            + config['body_z_rate_weight'] * body_wz.square())
    band = config.get('control_band')
    if band is not None:
        warning = math.radians(band['warning_deg'])
        excess = (excursion - warning).clamp_min(0.)
        # Preserve the corrective slope beyond the band without squaring
        # the already large extrapolations near a physical fall again.
        quadratic_width = math.radians(prediction['peak_heading_goal_deg']) - warning
        quadratic = excess.clamp_max(quadratic_width)
        cost = cost + band['weight'] * (quadratic.square() + 2. * quadratic_width * (excess - quadratic))
    return -dt * cost


class YawEvaluation:
    """Keep the terminal sample and discard later resets on faster lanes."""
    def __init__(self, count, device, prediction):
        self.prediction = prediction
        self.limit = math.radians(prediction['peak_heading_goal_deg'])
        self.live = torch.ones(count, dtype=torch.bool, device=device)
        self.samples = torch.zeros_like(self.live, dtype=torch.long)
        self.squared = torch.zeros_like(self.live, dtype=torch.float64)
        self.peak = torch.zeros_like(self.squared)
        self.predicted_peak = torch.zeros_like(self.squared)
        self.exceeded = torch.zeros_like(self.samples)

    def sample(self, audit, done):
        error = audit['error'].abs()
        forecast = (audit['error'] + self.prediction['response_horizon_s'] * audit['rate']).abs()
        self.samples += self.live
        self.squared += torch.where(self.live, error.square(), 0.)
        self.peak = torch.maximum(self.peak, torch.where(self.live, error, 0.))
        self.predicted_peak = torch.maximum(self.predicted_peak, torch.where(self.live, forecast, 0.))
        self.exceeded += self.live & ((error > self.limit) | (forecast > self.limit))
        self.live &= ~done.bool().reshape_as(self.live)

    def report(self):
        samples = int(self.samples.sum())
        if self.live.any() or not self.samples.all():
            raise ValueError('Expected complete first episodes for every yaw evaluation lane')
        rms = math.degrees(math.sqrt(float(self.squared.sum()) / samples))
        actual = math.degrees(float(self.peak.max()))
        predicted = math.degrees(float(self.predicted_peak.max()))
        if not all(math.isfinite(x) for x in (rms, actual, predicted)):
            raise ValueError('Nonfinite yaw measurement in evaluation')
        return dict(samples=samples, rms_deg=rms, peak_deg=actual, predicted_peak_deg=predicted,
                    outside_current_or_predicted_ticks=int(self.exceeded.sum()),
                    episodes_outside_current_or_predicted=int((self.exceeded > 0).sum()),
                    passed=rms <= self.prediction['rms_heading_goal_deg'] and not bool(self.exceeded.any()))
