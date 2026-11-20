# Hopper OBS73 全向跳训练交接

记录于 2026-11-20，新加坡时间。当前工作目录为 `/home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac`。训练服务已手动停止，已完成到第 45 轮。第 46 轮启动后中断，留下 `model_0.pt`。状态文件在中断时写的是 `phase=training`，实际进程以 `systemctl` 为准。

下面指向 `outputs/` 的链接需要训练主机保留的本机产物。新检出要续训时，将训练主机的活跃 `outputs/` 按原相对路径复制回来。当前状态、配置和服务含训练主机绝对路径，迁移到其他位置时需要更新这些路径。

## 从哪里继续

| 用途 | 文件 | 已知结果 |
|---|---|---|
| 当前自动续训的父权重 | [v50 round_03 的 model_49.pt](outputs/benchmark73_precision10_v50/round_03/train/model_49.pt) | 固定开发集全向 9/64、0.5 m 直线 3/64 完成十跳，128 例失稳 0，飞行姿态门槛未通过 |
| 最近完整主线 | [第 45 轮决策](outputs/benchmark73_precision10_v50/continuous_v51/round_00045/decision.json) | 全向 4/64、直线 3/64、失稳 1，父权重仍为 round_03 |
| 最新 AB 对照和回放 | [model_49.pt](outputs/benchmark73_precision10_v50/continuous_v51/ab_pairs/20261120_070527_471469Z/train/model_49.pt) · [评估](outputs/benchmark73_precision10_v50/continuous_v51/ab_pairs/20261120_070527_471469Z/result.json) | 全向 6/64、直线 2/64。60 秒评估全向失稳 1、直线失稳 5 |

自动服务从 [运行状态](outputs/benchmark73_precision10_v50/continuous_v51/status.json) 读取父权重、反馈和轮次。下次启动时按第 46 轮的配置，从 round_03 的权重和归一化器重新开始这轮。中断产物会保留在同轮的 `train.interrupted-*`，PPO 创建新的优化器。

## 冻结的训练配置

| 配置 | 课程与奖励 | PPO | 环境 | 观测 |
|---|---|---|---|---|
| 主线第 46 轮 | [course.json](training_config_v51/continuous/course.json) | [runner_config.json](training_config_v51/continuous/runner_config.json) | [env_config.json](training_config_v51/continuous/env_config.json) | [observation_contract.json](training_config_v51/continuous/observation_contract.json) |
| 最新 AB，含反跳起飞预测修正 | [course.json](training_config_v51/ab/course.json) | [runner_config.json](training_config_v51/ab/runner_config.json) | [env_config.json](training_config_v51/ab/env_config.json) | [observation_contract.json](training_config_v51/ab/observation_contract.json) |

- 两套均使用 1,024 并行环境，每次 PPO 更新采 256 步，50 次更新共 13,107,200 步。主线的一些诊断轮使用 100 次更新做预算对照。当前 PPO 为单层 256 隐状态 LSTM，学习率 0.0003、自适应调度，熵系数 0.002，训练种子 42。
- 输入为四元数版 OBS73，包括当前和下一目标。输出为四路直接电机动作，100 Hz 控制、200 Hz 物理。每次重置独立采样质量和惯量倍率，均匀分布在 0.9–1.1。
- 自由路线从当前目标出发，步长覆盖 0–0.5 m。自由采样中 10% 为零步长、30% 为 0.5 m，其余在范围内连续采样。连续有位移的相邻步包含同向、直角、反向和全圆随机方向。下一点提前可见，当前目标在触地事件推进。落点验收以足端 XY 为准，圆的直径是 **10 cm**，半径 **5 cm**。
- 第 46 轮含 512 个原地环境、128 个按键指令环境、384 个普通移动环境。原地回合上限 15 秒，移动回合上限 60 秒。普通移动路线中十步往返采样配置为 40.48%，路线类型由重置顺序和覆盖规则决定。
- 最新 AB 含 512 个原地环境、128 个按键指令环境、128 个 AB 环境及 256 个其他移动环境。AB 环境在 B 点发生有效触地并计入该跳奖励后做 timeout 重置。AB 样本一半直接进入 A，一半先做一个移动热身。AB 内 A 为原地、B 为原地、同向 0.5 m、反向 0.5 m、直角、随机方向的采样权重分别为 10%、10%、25%、25%、5%、25%。
- 奖励按每跳的落点、侧向倾角和角速度、原地腿部摆动与电机平衡、yaw 预测误差等项计算。全向与直线开发集各 64 例，初始原地跳和随后十个目标均须首次触地达标、顶点至少 1 m、触地倾角不超过 15°。整段须零倒地，并通过原地和移动飞行姿态及 yaw 门槛，其中原地倾角和移动侧向倾角峰值均限 10°。通过开发集后还要跑新种子留出集、原地保持和四轴按键验收。当前没有获准实机部署的新权重。

## 启动、停止和查看

在训练主机执行。Isaac Sim 为 5.1.0，安装在 `~/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64`，Isaac Lab 位于 `~/Desktop/workspace/IsaacLab`，本次记录的提交为 `3e73d6dd7`。

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

# 恢复自动训练，每轮后评估并根据失败情况生成下一轮
systemctl --user start hopper-precision-continuous.service
journalctl --user -u hopper-precision-continuous.service -f

# 停止队列与当前 Isaac 子进程
systemctl --user stop hopper-precision-continuous.service
```

用户服务定义位于 [benchmark37/hopper-precision-continuous.service](benchmark37/hopper-precision-continuous.service)。它仍配置在用户会话启动目标下。如需以后只手动运行，可执行 `systemctl --user disable --now hopper-precision-continuous.service`。已有 TensorBoard 服务可用 `systemctl --user start hopper-v50-tensorboard.service` 启动，在训练主机访问 `http://127.0.0.1:6011`。[TensorBoard 服务副本](training_config_v51/hopper-v50-tensorboard.service)

复跑一轮完全相同的主线课程，输出到独立目录。若只需要恢复原自动队列，用上面的 service 命令。

```bash
bash run_benchmark73_phase_motion.sh train --headless --profile vram_1024 \
  --ring-config training_config_v51/continuous/course.json \
  --output "$PWD/outputs/benchmark73_precision10_v50/manual_round46_$(date +%Y%m%d_%H%M%S)"
```

再次提交同类型的 AB 对照。自动服务运行时在轮次边界处理。服务停止时由当前终端处理。

```bash
/usr/bin/python3 -u -m benchmark37.train_ab_pairs --ascent-shortfall
```

复看最新 AB 的 0 号反向案例。将 `suite` 改成 `ab_continue` 可看同向两跳。

```bash
trial="$PWD/outputs/benchmark73_precision10_v50/continuous_v51/ab_pairs/20261120_070527_471469Z"
suite=ab_reverse
bash run_benchmark37_ring.sh play \
  --checkpoint "$trial/train/model_49.pt" \
  --ring-config "$trial/$suite/ring_config.json" \
  --cases-file "$trial/$suite/cases.json" \
  --case-id 0 --steps 600 --slow-motion 1.5
```

可视化和训练共用 GPU 锁，若有训练或评估正在运行，回放等当前阶段收尾。TensorBoard 中可看 `Qualification/omni_clean_of_64`、`Qualification/straight_clean_of_64`、`Qualification/travel_cross_peak_deg` 与 `Qualification/physical_deaths_of_128`。历史训练曲线及轮次摘要仍可读。

## 保留依据与归档

工作目录保留当前启动链、任务模块、物理资产、检查脚本和冻结配置。输出保留当前父权重与完整固定开发集、最近八个完整主线轮次和第 46 轮、AB 回放和对照依赖、固定原地及四轴案例、TensorBoard 曲线。旧轮次的轻量决策与摘要仍可查阅。

本机归档目录为 `.archive/20261120_obs73_v51/`，其中 `README.md` 记录源码卷恢复方法。源码卷保留旧代码、旧交接、部署与诊断资料。历史输出压缩卷已永久删除，索引记录此前归档的路径。旧版 Git 跟踪的源码可从提交 `54867f4` 查看或恢复。
