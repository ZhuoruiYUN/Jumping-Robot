# Hopper-sim-Isaac

当前 OBS73 任务用全量 AB 两跳回合训练，评估与部署按触地事件连续更新下一落点。启动命令、在线航点接口和验证结果见 [训练交接](TRAINING_V51.md)。

- `benchmark37/` 与 `run_benchmark73_phase_motion.sh` 提供当前训练、评估和回放入口。
- `Quadhopper_PPO_Reference_SysID/` 提供当前仿真的机器人和功耗模型。
- `training_config_v51/` 保存本次交接的主线和 AB 配置快照。
- 本机的 `outputs/` 保存续训和验收所需产物，Git 交接包含源码和冻结配置。在新检出中复用训练状态时，从训练主机复制对应的输出文件。

## V88 基线与穿环预研

直线跳阶段保留 V88 为当前实机基线。课程为 `training_config_v51/v88_continuation_forecast/course.json`，正式权重为 `outputs/benchmark73_precision10_v50/v88_continuation_forecast/train_seed42_batch_20261126_014822_649684173Z/model_49.pt`。`deployment/obs73_v88_imu_link/` 的启动脚本需要同一批次导出的 `quadhopper_step_x05_obs73.onnx`，从训练主机成对复制。完整实机交接和训练产物随本地归档保存。

`benchmark37/TGate_20mm.usda` 和 `benchmark37/TGate_V88_preview.usda` 用于穿环几何预研及 V88 轨迹回放。OBS73 的 59–72 索引已定义穿环指令。V88 地面课程的这组输入为零，独立穿环课程仍在接线阶段。
