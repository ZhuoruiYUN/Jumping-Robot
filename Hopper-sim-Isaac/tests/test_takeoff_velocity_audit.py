"""CPU-only checks for pre-reset per-hop kinematics and target timing."""

import csv
from pathlib import Path
import runpy
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
Audit = runpy.run_path(str(ROOT / 'Quadhopper_Planner_Random/takeoff_velocity_audit.py'))[
    'TakeoffVelocityAudit']
summarize = runpy.run_path(str(ROOT / 'tools/summarize_takeoff_velocity_audit.py'))['summarize']
summarize_phase = runpy.run_path(str(ROOT / 'tools/summarize_contact_phase_audit.py'))['summarize']

FIELDS = Audit.REQUIRED + ('target_after_events_w_x', 'target_after_events_w_y')
PHASE_FIELDS = FIELDS + Audit.PHASE_REQUIRED


def sample(step, episode=0, reset=False, **overrides):
    values = dict(initial_drop=0, liftoff=0, apex=0, touchdown=0,
                  foot_pos_w_x=0, foot_pos_w_y=0,
                  target_at_action_w_x=.3, target_at_action_w_y=0,
                  next_target_at_action_w_x=.3, next_target_at_action_w_y=.2,
                  root_vel_w_x=.3, root_vel_w_y=0,
                  settled_apex_height_m=1.02, settled_apex_target_m=1,
                  hard_environment=1, global_thrust_scale=.93,
                  target_after_events_w_x=.3, target_after_events_w_y=0)
    values.update(overrides)
    matrix = torch.tensor([[values[name] for name in FIELDS]], dtype=torch.float32)
    return step, torch.tensor([episode]), matrix, torch.tensor([reset])


def phase_sample(step, episode=0, reset=False, **overrides):
    values = dict(initial_drop=0, liftoff=0, apex=0, touchdown=0,
                  foot_pos_w_x=0, foot_pos_w_y=0,
                  target_at_action_w_x=.3, target_at_action_w_y=0,
                  next_target_at_action_w_x=.3, next_target_at_action_w_y=.2,
                  root_vel_w_x=.3, root_vel_w_y=0, root_vel_w_z=0,
                  tilt_deg=0, spring_q_m=0, spring_qdot_m_s=0,
                  spring_contact_proxy=0,
                  settled_apex_height_m=1.02, settled_apex_target_m=1,
                  hard_environment=1, global_thrust_scale=.93,
                  target_after_events_w_x=.3, target_after_events_w_y=0,
                  **{f'motor_pwm_{i}': 500 for i in range(1, 5)})
    values.update(overrides)
    matrix = torch.tensor([[values[name] for name in PHASE_FIELDS]], dtype=torch.float32)
    return step, torch.tensor([episode]), matrix, torch.tensor([reset])


class TakeoffVelocityAuditTests(unittest.TestCase):
    def test_contact_phase_tracks_prior_landing_and_current_landing(self):
        audit = Audit(1, .01, contact_phase=True)
        rng = torch.get_rng_state().clone()
        audit.observe(phase_sample(1, liftoff=1), PHASE_FIELDS)
        audit.observe(phase_sample(2, foot_pos_w_x=.20, root_vel_w_z=-.5), PHASE_FIELDS)
        audit.observe(phase_sample(3, touchdown=1, foot_pos_w_x=.29,
                                   root_vel_w_z=-.1, spring_q_m=.003,
                                   spring_contact_proxy=1), PHASE_FIELDS)
        audit.observe(phase_sample(4, foot_pos_w_x=.29, spring_q_m=.006,
                                   spring_qdot_m_s=.2, spring_contact_proxy=1), PHASE_FIELDS)
        audit.observe(phase_sample(5, liftoff=1, foot_pos_w_x=.30,
                                   target_at_action_w_x=.5, root_vel_w_z=2,
                                   spring_q_m=.001), PHASE_FIELDS)
        audit.observe(phase_sample(6, foot_pos_w_x=.35, target_at_action_w_x=.5,
                                   root_vel_w_z=-.3), PHASE_FIELDS)
        audit.observe(phase_sample(7, touchdown=1, foot_pos_w_x=.49,
                                   target_at_action_w_x=.5, root_vel_w_z=-.2,
                                   spring_q_m=.004, spring_contact_proxy=1), PHASE_FIELDS)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(audit.rows), 2)
        row = audit.rows[1]
        self.assertEqual(row['prior_touchdown_step'], 3)
        self.assertAlmostEqual(row['prior_touchdown_pre_body_vz_m_s'], -.5)
        self.assertAlmostEqual(row['contact_dwell_s'], .02)
        self.assertEqual(row['contact_proxy_samples'], 2)
        self.assertAlmostEqual(row['contact_peak_spring_q_m'], .006)
        self.assertAlmostEqual(row['liftoff_body_vz_m_s'], 2)
        self.assertAlmostEqual(row['touchdown_pre_body_vz_m_s'], -.3)
        self.assertAlmostEqual(row['touchdown_pre_foot_speed_xy_m_s'], 14, places=4)
        with tempfile.TemporaryDirectory() as tmp:
            report = audit.export(tmp)
            self.assertTrue(report['contact_phase_audit'])
            with (Path(tmp) / 'hops.csv').open(newline='') as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(len(exported), 2)
            self.assertEqual(exported[1]['contact_proxy_samples'], '2')
            groups = summarize_phase(exported)
            self.assertEqual(groups['all']['n'], 2)
            self.assertEqual(groups['prior_hit_10cm']['n'], 1)
            self.assertEqual(groups['recovery_prior_miss_under_50cm']['n'], 0)

    def test_contact_phase_does_not_pair_across_reset(self):
        audit = Audit(1, .01, contact_phase=True)
        audit.observe(phase_sample(1, touchdown=1, reset=True,
                                   spring_q_m=.004, spring_contact_proxy=1), PHASE_FIELDS)
        audit.observe(phase_sample(2, episode=1, liftoff=1), PHASE_FIELDS)
        audit.observe(phase_sample(3, episode=1, touchdown=1), PHASE_FIELDS)
        self.assertIsNone(audit.rows[0]['prior_touchdown_step'])
        self.assertIsNone(audit.rows[0]['contact_dwell_s'])

    def test_prestep_target_kept_after_touchdown_advances_queue(self):
        audit = Audit(1, .01)
        rng = torch.get_rng_state().clone()
        audit.observe(sample(1, liftoff=1), FIELDS)
        audit.observe(sample(2, apex=1, foot_pos_w_x=.14), FIELDS)
        audit.observe(sample(3, touchdown=1, foot_pos_w_x=.29,
                             target_after_events_w_x=.8), FIELDS)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(audit.rows), 1)
        row = audit.rows[0]
        self.assertAlmostEqual(row['target_x_m'], .3, places=6)
        self.assertAlmostEqual(row['touchdown_foot_error_m'], .01, places=6)
        self.assertAlmostEqual(row['apex_foot_error_m'], .16, places=6)
        self.assertTrue(row['strict_pair'])
        self.assertAlmostEqual(row['airtime_s'], .02)
        self.assertAlmostEqual(row['reference_vx_m_s'], 15, places=5)
        with tempfile.TemporaryDirectory() as tmp:
            report = audit.export(tmp)
            self.assertEqual(report['rows'], 1)
            with (Path(tmp) / 'hops.csv').open(newline='') as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]['strict_pair'], 'True')
            result = summarize(exported)
            self.assertEqual(result['all']['all']['n'], 1)
            self.assertEqual(result['turn']['left_45_135']['n'], 1)
            self.assertEqual(result['landing_outcome']['under_10cm']['n'], 1)
            self.assertEqual(result['cross_speed']['0.00-0.05']['n'], 1)
            self.assertEqual(result['all']['all']['foot_miss_20cm_rate'], 0)

    def test_previous_miss_and_reset_do_not_cross_episodes(self):
        audit = Audit(1, .01)
        audit.observe(sample(1, liftoff=1), FIELDS)
        audit.observe(sample(2, touchdown=1, foot_pos_w_x=0), FIELDS)
        audit.observe(sample(3, liftoff=1, foot_pos_w_x=.3,
                             target_at_action_w_x=.5), FIELDS)
        self.assertTrue(audit.pending[0]['prior_foot_miss_10cm'])
        audit.observe(sample(4, reset=True), FIELDS)
        self.assertEqual(audit.skipped_reset_before_touchdown, 1)
        audit.observe(sample(5, episode=1, liftoff=1), FIELDS)
        self.assertIsNone(audit.pending[0]['prior_touchdown_foot_error_m'])

    def test_changed_target_excluded(self):
        audit = Audit(1, .01)
        audit.observe(sample(1, liftoff=1), FIELDS)
        audit.observe(sample(2, touchdown=1, target_at_action_w_x=.5), FIELDS)
        self.assertEqual(audit.skipped_target_changed, 1)
        self.assertEqual(audit.rows, [])


if __name__ == '__main__':
    unittest.main()
