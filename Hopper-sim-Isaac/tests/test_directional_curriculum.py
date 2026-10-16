"""CPU-only checks for the V20 signed-turn/zero/reversal curriculum."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    'direction_curriculum', ROOT / 'Quadhopper_Planner_Random' / 'direction_curriculum.py'
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
directional_command_values = MODULE.directional_command_values


class DirectionalCurriculumTests(unittest.TestCase):
    def _at_iteration(self, iteration: float):
        return directional_command_values(
            iteration=iteration,
            duration_iterations=360.0,
            start_turn_deg=25.0,
            final_turn_deg=180.0,
            start_zero_probability=0.0,
            final_zero_probability=0.12,
            final_reverse_probability=0.20,
            reverse_halfwidth_deg=30.0,
        )

    def test_starts_with_existing_narrow_moving_route(self):
        turn, zero, reverse = self._at_iteration(0)
        self.assertAlmostEqual(turn, 25.0)
        self.assertEqual(zero, 0.0)
        self.assertEqual(reverse, 0.0)

    def test_reverse_is_not_sampled_before_its_band_is_valid(self):
        turn, zero, reverse = self._at_iteration(280)
        self.assertLess(turn, 150.0)
        self.assertEqual(reverse, 0.0)
        self.assertGreater(zero, 0.0)

    def test_ends_with_full_directional_task(self):
        turn, zero, reverse = self._at_iteration(360)
        self.assertAlmostEqual(turn, 180.0)
        self.assertAlmostEqual(zero, 0.12)
        self.assertAlmostEqual(reverse, 0.20)


if __name__ == '__main__':
    unittest.main()
