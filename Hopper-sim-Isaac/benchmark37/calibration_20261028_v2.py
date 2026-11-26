"""Same calibration values; local link wrenches bypass cached global conversion."""

import torch

from . import calibration_20261028 as v1
from .artifacts import ROOT, sha

PARAMS = v1.PARAMS
ASSET_PATH = v1.ASSET_PATH
SOURCE = v1.SOURCE
SOURCE_SHA256 = v1.SOURCE_SHA256
REVISION = "sysid_20261028_spring_air_v2_local_wrench"
SERIES = "benchmark37_stage4a_calibrated_20261028_v2"
configure = v1.configure
air_force_body = v1.air_force_body
axial_force = v1.axial_force
rotate = v1.rotate
validate_source = v1.validate_source
initialize_runner = v1.initialize_runner
validate_asset = v1.validate_asset
runtime_air_audit = v1.runtime_air_audit


def physics_contract():
    contract = v1.physics_contract()
    contract.update(revision=REVISION,
                    wrench_frame="current root Body -> current link-local; composer is_global=False, no cached world conversion")
    path = ROOT / "benchmark37/calibration_20261028_v2.py"
    contract["files"][str(path.relative_to(ROOT))] = sha(path)
    return contract


class CalibratedAirMixin:
    def _apply_action(self):
        if self._robot.body_names != ["Body", "SpringLeg"]:
            raise RuntimeError("Calibrated air assumes Body/SpringLeg order")
        data = self._robot.data
        root_q = data.root_quat_w[:, None, :].expand(-1, 2, -1)
        velocity = rotate(root_q, data.body_com_lin_vel_w, inverse=True)
        air_b = air_force_body(velocity)
        plant = getattr(self, '_imu_link_plant', None)
        if plant is not None:
            plant.alter_air_force(air_b, velocity)
        force_b = air_b.clone()
        force_b[:, 0:1] += self._thrust
        torque_b = torch.zeros_like(force_b)
        torque_b[:, 0:1] = self._moment
        # The permanent composer caches link poses until reset. Providing world
        # wrenches would therefore reuse an old world->link rotation. Convert
        # explicitly using current link quaternions and write only local data.
        force_link = rotate(data.body_link_quat_w, rotate(root_q, force_b), inverse=True)
        torque_link = rotate(data.body_link_quat_w, rotate(root_q, torque_b), inverse=True)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=[0, 1], forces=force_link, torques=torque_link, is_global=False)
        if not hasattr(self, "_calibration_air_calls"):
            self._calibration_air_calls = 0
            self._calibration_air_max_n = torch.zeros((), device=self.device)
            self._calibration_air_max_power_w = torch.zeros((), device=self.device)
        self._calibration_air_calls += 1
        self._calibration_air_max_n = torch.maximum(
            self._calibration_air_max_n, torch.linalg.vector_norm(air_b.sum(1), dim=-1).max())
        self._calibration_air_max_power_w = torch.maximum(
            self._calibration_air_max_power_w, (air_b * velocity).sum((-1, -2)).max())
