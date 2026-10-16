import unittest

from tools.analyze_ab_return_diagnostics import segment


class AbReturnDiagnosticsTest(unittest.TestCase):
    def test_late_motor_pull_is_detected_from_apex_split(self):
        rows = []
        xs = [0.0, .1, .2, .4, .7, .95]
        zs = [.3, .7, 1.0, .9, .6, .3]
        for index, (x, z) in enumerate(zip(xs, zs)):
            spread = .1 if index <= 2 else .3
            rows.append({
                'Post_Time_s': str(index * .01),
                'Post_X': str(x), 'Post_Y': '0', 'Post_Z': str(z),
                'Post_Vx': '.1', 'Post_Vy': '0',
                'Post_Roll_rad': '0', 'Post_Pitch_rad': '0',
                'Target_X': '1', 'Target_Y': '0',
                'LiftoffEvent': str(int(index == 0)),
                'TouchdownEvent': str(int(index == 5)),
                'PolicyActionSpread': str(spread),
                'PolicyActionRate_per_sAbsMean': '2',
                'PolicyActionRate_per_sAbsMax': '4',
                'Motor_PWM_Rate_per_sAbsMean': '200',
            })
        hops = segment(rows)
        self.assertEqual(len(hops), 1)
        self.assertAlmostEqual(hops[0]['apex_progress_fraction'], .2)
        self.assertAlmostEqual(hops[0]['post_apex_progress_fraction'], .75)
        self.assertTrue(hops[0]['late_motor_pull_flag'])


if __name__ == '__main__':
    unittest.main()
