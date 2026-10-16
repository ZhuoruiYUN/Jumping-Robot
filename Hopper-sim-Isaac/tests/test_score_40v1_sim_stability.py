import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from score_40v1_sim_stability import score_hops


def test_reset_does_not_create_spurious_touchdown(tmp_path):
    path = tmp_path / 'sim.csv'
    fields = ('Time_s', 'X', 'Y', 'Z', 'Target_X', 'Target_Y',
              'Is_Contact', 'Qw', 'Qx', 'Qy', 'Qz', 'M1', 'M2', 'M3', 'M4')
    samples = [
        (.01, .38, 1), (.02, .50, 0), (.03, 1.00, 0), (.04, .38, 1),
        (.05, .50, 0),
        # Episode reset in mid-flight: its contact is not the prior landing.
        (.01, .38, 1), (.02, .50, 0), (.03, 1.00, 0), (.04, .38, 1),
    ]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for time_s, z, contact in samples:
            writer.writerow(dict(Time_s=time_s, X=0, Y=0, Z=z, Target_X=0,
                                 Target_Y=0, Is_Contact=contact,
                                 Qw=1, Qx=0, Qy=0, Qz=0,
                                 M1=100, M2=100, M3=100, M4=100))
    score = score_hops(path)
    assert score['complete_hops'] == 2
    assert score['episode_resets'] == 1
    assert score['longest_continuous_hops'] == 1
    assert score['touchdown_tilt_ge_15deg_count'] == 0
