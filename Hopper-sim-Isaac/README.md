# Hopper-sim-Isaac

当前任务是 OBS73 全向连续跳。训练状态、冻结课程与 PPO 配置、启动和停止、TensorBoard、AB 回放及验收口径见 [V51 训练交接](TRAINING_V51.md)。

- `benchmark37/` 与 `run_benchmark73_phase_motion.sh` 提供当前训练、评估和回放入口。
- `Quadhopper_PPO_Reference_SysID/` 提供当前仿真的机器人和功耗模型。
- `training_config_v51/` 保存本次交接的主线和 AB 配置快照。
- 本机的 `outputs/` 保存续训和验收所需产物，Git 交接包含源码和冻结配置。在新检出中复用训练状态时，从训练主机复制对应的输出文件。
