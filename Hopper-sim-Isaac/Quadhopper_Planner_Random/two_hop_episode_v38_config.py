"""Simulator-independent thresholds for the two-hop episode boundary."""

# One 100 Hz control sample after touchdown. Historical contact dwell is
# about 0.05 s; a longer hold would count the next normal liftoff as failure.
SETTLE_TIME_S = 0.01
SETTLE_TILT_DEG = 20.0
SETTLE_XY_RATE_RAD_S = 2.0
