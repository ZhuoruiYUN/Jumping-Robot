"""Strict-success refinement; retains the v2 zero-curve foot-progress task."""
from .path_free_stage2_v2_env import PathFreeStage2V2Env


class PathFreeStage2V3Env(PathFreeStage2V2Env):
    """Behavior is configured by SETTINGS_V3; a class gives the protocol an explicit task ID."""
