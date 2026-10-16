"""Pure-Python checks for the two-circle Stage-B qualification decision."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    'score_two_circle_v18_stage_b', ROOT / 'tools' / 'score_two_circle_v18_stage_b.py'
)
assert SPEC is not None and SPEC.loader is not None
SCORE = module_from_spec(SPEC)
SPEC.loader.exec_module(SCORE)


class TwoCircleStageBQualificationTests(unittest.TestCase):
    def _nominal(self):
        return {
            'touchdowns': 150,
            'death_rate': 0.0,
            'evaluation_done_count': 0,
            'first_valid_hit_10cm': 0.8,
            'second_valid_hit_10cm': 0.8,
            'pair_valid_10cm': 0.65,
            'max_consecutive_valid_10cm_min': 5,
            'apex_ge_target_minus_5cm_rate': 0.9,
            'touchdown_tilt_p90_deg': 10.0,
        }

    def _robust(self):
        return {
            'min_touchdowns_per_env': 40,
            'death_rate': 0.03,
            'first_valid_hit_10cm': 0.6,
            'second_valid_hit_10cm': 0.6,
            'pair_valid_10cm': 0.35,
            'max_consecutive_valid_10cm_min': 2,
            'apex_ge_target_minus_5cm_rate': 0.8,
            'touchdown_tilt_p90_deg': 18.0,
        }

    def test_complete_results_can_pass_stage_b_only(self):
        result = SCORE.evaluate(self._nominal(), self._robust())
        self.assertTrue(result['passed'])
        self.assertFalse(result['deployment_authorized'])

    def test_missing_or_failed_metric_cannot_pass(self):
        nominal = self._nominal()
        nominal.pop('second_valid_hit_10cm')
        result = SCORE.evaluate(nominal, self._robust())
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
