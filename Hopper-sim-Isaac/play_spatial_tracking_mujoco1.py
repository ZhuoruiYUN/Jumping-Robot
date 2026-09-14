"""Run the workspace-level MuJoCo spatial-tracking player from this directory."""
from pathlib import Path
import runpy


PLAYER = Path(__file__).resolve().parents[1] / 'play_spatial_tracking_mujoco1.py'
if not PLAYER.is_file():
    raise FileNotFoundError(f'MuJoCo player is missing: {PLAYER}')

runpy.run_path(str(PLAYER), run_name='__main__')
