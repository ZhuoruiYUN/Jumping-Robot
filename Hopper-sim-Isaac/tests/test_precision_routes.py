import math
import unittest

from benchmark37.precision_routes import routes, score


class PrecisionGate(unittest.TestCase):
    def setUp(self):
        self.bank = routes(7001, 32)
        self.rows = [dict(case_id=case_id,physical_death=False,timed_out=True,events=[dict(
            target_index=index,target_local_xy_m=point,first_attempt_for_target=True,strict=True,
            foot_error_m=.05,direction=None,commanded_edge_m=0.) for index,point in enumerate(points[:11])])
            for case_id,points in enumerate(self.bank)]
        self.summary = dict(flight_plane={key:{'passed':True} for key in ('hold','travel')},
                            rebound_diagnostics={'yaw_gate':{'passed':True}})

    def test_protocol(self):
        self.assertNotEqual(self.bank, routes(9109,32))
        for case_id,points in enumerate(self.bank):
            if case_id%2:
                self.assertAlmostEqual(math.dist(points[0],points[10]),5.)
            else:
                self.assertEqual(points[3],points[4])
                self.assertLessEqual(math.dist(points[1],points[2]),.100001)
                self.assertLess(math.dist(points[5],points[7]),1e-6)

    def test_gate(self):
        self.assertTrue(score(self.rows,self.summary,7001)['passed'])
        self.rows[0]['events'][5]['foot_error_m']=.050001
        self.rows[1]['events'].pop()
        self.rows[2]['events'][5]['first_attempt_for_target']=False
        self.rows[3]['physical_death']=True
        result=score(self.rows,self.summary,7001)
        self.assertFalse(result['passed'])
        self.assertEqual([group['clean'] for group in result['groups'].values()],[62,62])
        self.assertEqual(result['first_failures'],dict(unprojected_xy=1,missing=1,retry_or_skip=1))
