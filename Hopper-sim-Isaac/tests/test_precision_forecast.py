import unittest

import torch

from benchmark37.moving_phase_reward import landing_guidance


class ForecastTest(unittest.TestCase):
    def test_blend_only_on_masked_descent_not_on_touch_cost(self):
        foot=torch.zeros(4,2)
        clearance=torch.full((4,),.1)
        velocity=torch.tensor([[.3,0.,-1.],[.3,0.,-1.],[.3,0.,1.],[.3,0.,-1.]])
        args=(foot,clearance,velocity,clearance,foot,clearance,torch.full((4,),3),
              torch.tensor([True,True,True,False]),torch.ones(4,dtype=torch.bool),9.81,.01)
        config=dict(apex_margin_m=.06,minimum_time_s=.15,maximum_time_s=1.2,speed_limit_m_s=1.2,
                    landing_clearance_m=.3,velocity_weight=4.,landing_weight=2.,landing_width_m=.05,
                    objective='projected_touchdown_error_v1',projected_error_width_m=.1,
                    projection_clock_version='zero_at_contact_separate_divisor_v2')
        origin=foot+.04
        old,baseline=landing_guidance(*args,config,projection_xy=origin)
        new,changed=landing_guidance(*args,dict(config,measured_foot_blend_s=.4),projection_xy=origin,
                                     blend_mask=torch.tensor([True,False,True,True]))
        self.assertTrue(torch.equal(changed['projected_foot_xy'][1:],baseline['projected_foot_xy'][1:]))
        self.assertLess(changed['projection_origin_xy'][0].norm(),baseline['projection_origin_xy'][0].norm())
        self.assertTrue(torch.equal(new['moving_landing_error'],old['moving_landing_error']))
