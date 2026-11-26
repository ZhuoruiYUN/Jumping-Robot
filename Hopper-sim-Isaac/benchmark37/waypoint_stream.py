"""Online waypoint sources use absolute release-frame XY in metres."""
from importlib import import_module
import math


CARDINAL_STEPS = ((.5, 0.), (-.5, 0.), (0., .5), (0., -.5))


def checked_preview(current, point):
    if point is None:
        return tuple(current)
    x, y = map(float, point)
    if not math.isfinite(x) or not math.isfinite(y) or math.dist(current, (x, y)) > .500001:
        raise ValueError('Next waypoint must be finite XY within 0.5 m of the current waypoint')
    return x, y


def load_source(spec):
    module, separator, name = spec.partition(':')
    if not separator or not module or not name:
        raise ValueError('Expected --waypoint-source module:function')
    source = getattr(import_module(module), name)
    if not callable(source):
        raise ValueError('Waypoint source must be callable')
    return source


def straight_half_metre(case_id, next_index, current):
    """Example source for a continuous +X route."""
    return current[0] + .5, current[1]


def cardinal_half_metre(case_id, next_index, current):
    """Keep each case on one XY axis, interleaving +X, -X, +Y and -Y."""
    dx, dy = CARDINAL_STEPS[case_id % len(CARDINAL_STEPS)]
    return current[0] + dx, current[1] + dy


def opposite_cardinal_half_metre(case_id, next_index, current):
    """Reverse each lane in the cardinal sweep, keeping its physical start."""
    dx, dy = CARDINAL_STEPS[case_id % len(CARDINAL_STEPS)]
    return current[0] - dx, current[1] - dy


def next_preview(source, case_id, next_index, current):
    point = source(case_id, next_index, tuple(current))
    return checked_preview(current, point)
