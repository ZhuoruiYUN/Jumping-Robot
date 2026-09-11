import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 使用覆盖层：修正实机辨识的 yaw 惯量（I_ZZ 7.932e-4，旧值 2.306e-3 为错误值）
USD_PATH = os.path.join(CURRENT_DIR, "model", "OriginJumpHopperAsset.usda")

MY_HOPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={"center_spring_joint": 0.0},
        joint_vel={"center_spring_joint": 0.0},
    ),
    actuators={},
)
