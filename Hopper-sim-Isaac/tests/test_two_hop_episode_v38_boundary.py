"""Keep the terminal safety sample inside the observed contact interval."""

import unittest

from Quadhopper_Planner_Random.two_hop_episode_v38_config import SETTLE_TIME_S


# v31 full contact audit: seed483/control median touchdown-to-liftoff dwell.
# See PATH_FREE_STAGE2_V31_CONTACT_PHASE_AUDIT.md; output files are not tracked.
OBSERVED_CONTACT_DWELL_MEDIAN_S = 0.05


class TwoHopBoundaryTest(unittest.TestCase):
    def test_terminal_sample_precedes_normal_liftoff(self):
        self.assertLessEqual(SETTLE_TIME_S, OBSERVED_CONTACT_DWELL_MEDIAN_S / 2)


if __name__ == '__main__':
    unittest.main()
