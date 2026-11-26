# Hopper OBS73 全向跳训练交接

更新于 2026-11-22，新加坡时间。训练由用户手动启动。V52 后台训练已停止，用户会话没有该服务的自启动链接。工作目录为 `/home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac`。

下面指向 `outputs/` 的链接需要训练主机保留的本机产物。新检出要续训时，将训练主机的活跃 `outputs/` 按原相对路径复制回来。当前状态、配置和服务含训练主机绝对路径，迁移到其他位置时需要更新这些路径。

## V73 独立 yaw 分配和触地角速度约束

V73 从 V71 短训 `model_3.pt` 初始化，延续全 AB、OBS73 和 LSTM。正常姿态下，独立航向回路替换策略的对角差动，在四电机力限内保持标称滚转和俯仰分量，合推力只在分配所需时、在给定范围内调整。电机延迟及物理力矩模型沿用现有仿真。经测量首次压簧时，新增对世界竖轴角速度突变的事件扣分，留出 0.45 rad/s 容差。它在训练中帮助策略减小触地冲击，航向验收仍使用全阶段实测与 0.14 秒预测峰值均不高于 10°、RMS 不高于 3°，另记录翻倒和足端 5 cm 落点。[V73 课程](training_config_v51/ab_yaw_servo/course.json) · [分配代码](benchmark37/yaw_heading_servo.py)

冻结权重的两组 60 秒四轴回放使用同初态，yaw RMS 从 2.64° 降至 0.54°，超线从 214/256 降到 3/256。255 条存活轨迹的实测峰值最高 7.65°，其中两条预测超线，最高 16.38°。+Y 第 86 例翻倒时实测峰值达到 76.51°，所以全样本硬验收仍未通过。+X 第 52 例实测峰值 1.88°，目标 3 首落 3.48 cm。四轴已触地首落 5 cm 命中率为 +X 90.06%、−X 88.87%、+Y 80.73%、−Y 81.22%。固定开发集的 128 例均未翻倒，实测峰值 7.65°，预测峰值 16.38°。[冻结权重完整对照](outputs/benchmark73_precision10_v50/ab_yaw_servo_v73/static/comparison.json)

V73 零更新全 AB 检查完成 1,227 个控制周期和 57 次有效 B 点结算，捕获 6 次压簧角速度突变扣分。正式训练由用户手动启动。[检查清单](outputs/benchmark73_precision10_v50/ab_yaw_servo_v73/check/manifest.json)

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
b="$PWD/outputs/benchmark73_precision10_v50"
bash run_benchmark73_phase_motion.sh train --headless \
  --ring-config training_config_v51/ab_yaw_servo/course.json \
  --output "$b/ab_yaw_servo_v73/formal_$(date +%Y%m%d_%H%M%S)"
```

## V72 航向纠偏优先

V72 从 V71 短训 `model_3.pt` 初始化，使用 V71 的全 AB 课程、LSTM 和每电机 0.02 N 推力余量。机体继续向外偏转时，若 0.14 秒外推误差超过 6°，逐渐削减与 yaw 反馈相反的策略差动，到 10° 达到完整削减。总推力、滚转、俯仰的标称分配保持原指令，电机的既有延迟和动态力矩继续在物理层生效。训练和评估均在动作进入电机队列前使用此分配。[V72 课程](training_config_v51/ab_yaw_priority/course.json) · [分配计算](benchmark37/yaw_thrust_feedback.py)

逐帧对照使用同一份冻结 V71 权重和相同的 128 路初态、噪声及延迟协议。+X 第 52 例原轨迹在第 581→582 帧压簧时，航向角速度从 −0.11 跳到 −2.44 rad/s，目标 3 的首落偏 13.63 cm。较早介入的 4°～8° 设置把该点首落压到 2.58 cm，并救回 −Y 第 102 例。同一设置在 −Y 第 94 例引入一次早倒。另一种滚转和俯仰保护使第 102 例停在目标 3，连续八次重试仍超 5 cm，留作诊断对照。[V71 总览](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/v71_visual_evaluation.png) · [4°～8° 逐帧对照](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/v72_priority_case_traces.png)

最终选择 6°～10° 设置。两组 60 秒反号回放零翻倒（V71 为 1/256），第 94、102 例均跑满，第 102 例目标 5 首落 3.41 cm。四轴首落 +X、−X、+Y、−Y 分别为 91.16%、90.42%、83.11%、84.06%。四轴 yaw 实测峰值仍为 19.12°，预测峰值由 42.27° 降至 30.69°，RMS 由 2.69° 降至 2.64°。+X 第 52 例的起跳尖峰仍在。航向两种峰值均高于 10° 目标。[最终设置的冻结回放](outputs/benchmark73_precision10_v50/ab_yaw_priority_v72/conservative/comparison.json)

固定开发集零翻倒，长步首落由 82.25% 变为 81.94%，重试由 76.60% 升至 79.26%，实测 yaw 峰值 10.86°、预测峰值 14.58°。最终课程全 AB 零更新检查跑了 16 个 Isaac 环境、1,165 个控制周期和 52 次 B 点有效终止。正式训练由用户手动启动，使用独立的输出目录保存新权重。[开发集](outputs/benchmark73_precision10_v50/ab_yaw_priority_v72/conservative/development/summary.json) · [零更新检查](outputs/benchmark73_precision10_v50/ab_yaw_priority_v72/conservative/check/manifest.json) · [三档分配对照图](outputs/benchmark73_precision10_v50/ab_yaw_priority_v72/v72_allocation_tradeoffs.png)

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
b="$PWD/outputs/benchmark73_precision10_v50"
bash run_benchmark73_phase_motion.sh train --headless \
  --ring-config training_config_v51/ab_yaw_priority/course.json \
  --output "$b/ab_yaw_priority_v72/formal_$(date +%Y%m%d_%H%M%S)"
```

GUI 回放使用原 128 路评估批次和所选案例编号。橙圈为当前目标的 5 cm 容差，紫圈为下一目标，蓝线和绿线分别显示参考与实测 yaw。默认把视角定位到所选环境，随后由 Isaac 视口控制缩放、平移和旋转。需要跟随 Hopper 时加 `--preview-follow-camera`，或在 Viewer Settings 的 Follow Mode 中切换。

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
b="$PWD/outputs/benchmark73_precision10_v50"
ckpt="$b/ab_yaw_reserve_v71/ppo4/model_3.pt"
# 将 ckpt 改为 "$b/ab_yaw_priority_v72/conservative/frozen_v71_weights/model_3.pt" 可看同权重 V72 对照。
bash run_benchmark37_ring.sh eval --checkpoint "$ckpt" --canonical-axis \
  --waypoint-source benchmark37.waypoint_stream:cardinal_half_metre \
  --preview-case 52 --preview-steps 850
bash run_benchmark37_ring.sh eval --checkpoint "$ckpt" --canonical-axis \
  --waypoint-source benchmark37.waypoint_stream:opposite_cardinal_half_metre \
  --preview-case 102 --preview-steps 1380
```

## V71 为独立 yaw 回路预留推力

V71 直接继承 V57 的 LSTM、全 AB 课程、奖励和随机出生分布，从 V57 最终权重和归一化器初始化，使用新 PPO 优化器。环境在每路标称推力两端各留 0.02 N，再用与策略相同的一帧四元数和机体角速度作航向反馈。对角差动保持预留后的标称总推力、滚转和俯仰力矩。推力预留使原始策略的滚转和俯仰控制幅度约缩小 6.4%，并可能改变最多 0.08 N 的合推力。仿真仍使用既有电机延迟、响应和动态 yaw 力矩。OBS73 动作历史记录实际送进电机队列的指令，PPO 仍以原始采样动作计算概率。冻结评估使用负轴方向映射，然后施加同一推力分配。[V71 课程](training_config_v51/ab_yaw_reserve/course.json) · [执行入口](benchmark37/crossing_obs73_quat.py) · [推力分配](benchmark37/yaw_thrust_feedback.py)

16 个 Isaac 环境的零更新检查完成 1,254 个控制周期、57 次 B 点触地结算。对归档 V57 轨迹重算，在预测航向误差绝对值超过 10° 的 7,311 帧里，可施加至少 0.01 N 差动的帧从 399 增至 7,296。闭环验收继续统计 5 cm 足端落点、翻倒，以及全阶段的 yaw RMS、实测和 0.14 秒预测峰值。门槛为 RMS 不高于 3°、两种峰值均不高于 10°。[仿真检查](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/check/manifest.json) · [归档轨迹回算](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/archived_command_authority.json)

四次 PPO 更新共 1,048,576 步，冻结评估 `model_3.pt`。以 V57 加方向映射和无预留 yaw 反馈为对照，两组各 128 例的 60 秒反号回放中，5 cm 首落 +X 从 88.37% 到 91.35%，−X 从 88.30% 到 90.81%，+Y 从 77.55% 到 82.16%，−Y 从 78.87% 到 83.33%。四轴翻倒由 3/256 降到 1/256。剩余翻倒是反号组 −Y 第 102 例，目标 5 落偏后重试，起跳约 40 ms 后倾角达到 91.10°。四轴 yaw RMS 为 2.69°，实测峰值 19.12°、预测峰值 42.27°。实测最大值在原符号组 +X 第 52 例目标 3 首次跳的上升段。该跳离地时已偏 12.78°，首落误差 13.63 cm，重试命中 3.66 cm。[四轴及开发集汇总](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/comparison.json)

固定开发集长步首落由 75.53% 升至 82.25%，同目标重试由 67.26% 升至 76.60%，均为 0/128 翻倒。实测 yaw 峰值 11.00°、预测 17.74°、RMS 2.00°。训练四次的平均回报为 31.28、−19.47、−1.43、−166.25，critic loss 为 8839.96、456.98、1376.28、2475.16。闭环成绩有改善，训练曲线仍在波动，固定验收尚未通过。[V71 开发集](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/development/summary.json) · [短程训练记录](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/ppo4.log)

冻结 V57 只加同样推力预留，在反号组把四轴 yaw 实测峰值由 74.42° 降至 14.40°，翻倒从 1/128 降至 0/128。V71 在同组的实测峰值为 16.67°，翻倒 1/128。冻结带预留的 V57 在开发集长步首落为 79.76%、重试为 83.33%、翻倒 1/128。短训的开发集长步精度和存活有改善，反号组仍保留冻结权重对照。正式训练由用户手动启动。正式权重验收用 `evaluate_reserve.sh` 入口并通过 `V71_CHECKPOINT` 和 `EVAL_OUTPUT` 指定权重和新的输出目录，方向映射以 `--canonical-axis` 启用，预留反馈从课程加载。[冻结反号对照](outputs/benchmark73_precision10_v50/ab_yaw_reserve_v71/frozen_opposite_comparison.json)

## V57 固定航向的方向映射和 yaw 反馈

V70 配对采样四次更新未缩小负向差距。冻结 V57 的映射实验将负轴航段的原始 OBS73 正规化前数据在水平坐标中转半周，保留航向和垂直分量。三组已接受电机历史和输出均交换 F1/F3、F2/F4。原始和变换分支各维护一份同步更新的 LSTM 历史。以世界航点边选择分支，同一目标重试保持选择。两条仿真选项 `--canonical-axis`、`--yaw-feedback` 接入原冻结评估入口。Yaw 反馈将观测航向误差和航向角速度用于受限对角推力差动，并保持标称合推力、滚转和俯仰力矩。[方向映射](benchmark37/axis_canonical_actor.py) · [yaw 反馈](benchmark37/yaw_thrust_feedback.py)

原 V57 的归档轨迹在预测 yaw 超过 10° 的 7,311 帧中有 6,873 帧缺少所需的对角推力余量。该离线证据衡量可用纠偏力矩，闭环成绩另由两轮四轴各 128 例和同开发集给出。[推力余量分析](outputs/benchmark73_precision10_v50/v57_axis_canonical/yaw_headroom.json)

两组同初态反号的 60 秒结果合并后，仅坐标映射使 −X 5 cm 首落由 53.32% 变为 88.67%，−Y 由 42.38% 变为 76.63%。加独立反馈后，+X、−X、+Y、−Y 分别为 88.37%、88.30%、77.55%、78.87%，四轴翻倒由 6 例降至 3 例。开发集长步首落 75.53%，V57 为 64.57%，零翻倒。四轴 yaw 实测峰值仍为 74.42°，存活案例最大实测峰值 19.61°，门槛为 10°。两种控制方式在冻结评估中使用同一 V57 权重。[完整同初态结果](outputs/benchmark73_precision10_v50/v57_axis_canonical/validation.md)

## V70 先原地跳再正反配对

V57 的同初态反号回放确认，连续跳的同序号首次命中率为 +X 88.74%、−X 53.32%、+Y 76.92%、−Y 42.38%。V70 从 V57 的 OBS73 LSTM 初始化，用 V69 的全阶段航向评分（实际航向系数 80、3° 开始额外扣分）和 V66 的早倒结算。[反号配对证据](outputs/benchmark73_precision10_v50/ab_yaw_axis_v69/sign_probe/report.md) · [V70 课程](training_config_v51/ab_paired_axis/course.json)

1,024 个训练环境都用 AB。512 个环境保留原全向 AB 采样。其余 512 个环境把 A 锚定在释放位置，严格命中 A 后才公布 0.50 m 的轴向 B。若 A 落偏，仍按 AB 规则用第二跳重试 A。同一个配对环境在下一回合复用释放位姿、初速度扰动、电机时常、质量惯量和响应倍率，把 B 反号。新一对重新抽取出生参数，正负号先后交替。每控制步的观测噪声、动作采样与延迟独立。运行记录逐方向统计出生、B 已发出、B 已落地和 5 cm 命中次数。[配对采样与目标交接](benchmark37/ab_paired_axis.py)

验收沿用 15 秒固定开发集和两组 60 秒四轴回放。第二组把第一组每个案例的命令反号，同轴正负各有同一批 64 个物理初态。四轴评估的原接口从释放时展示下一目标预瞄。先原地跳后公布 B 的训练行为另由零更新运行检查。落点门槛为 5 cm，yaw RMS ≤3°、实测和 0.14 秒预测峰值均 ≤10°。

V70 四次 PPO 更新已完成。两轮四轴合并首落分别为 +X 88.43%、−X 51.67%、+Y 78.14%、−Y 41.17%，四轴翻倒由 6 例变成 8 例。按实机启动时序冻结回放，−X 和 −Y 的 B 点 5 cm 命中分别由 39/128、33/128 变成 38/128、32/128。开发集长步首落 63.02%、重试 67.62%，零翻倒。原 AB 直接追 A 的出生压力测试早倒 183/1,024。yaw 与短程综合表现未达到验收，保留 V57。[训练和冻结评估结果](outputs/benchmark73_precision10_v50/ab_paired_axis_v70/validation.md)

## V69 固定航向的全向 AB

实机按释放时航向为参考，目标可沿 ±X、±Y 或斜向移动，足端到目标的水平距离须 ≤5 cm。新增 60 秒四方向在线评估，每个方向 32 例、步长 0.50 m，以统一世界参考测 yaw。开发集继续测全向和方向变化。yaw 门槛沿用 RMS ≤3°、实测及 0.14 秒预测峰值均 ≤10°。四轴分别看首落、重试、飞行内 yaw 与落点联合达标和物理翻倒。[轴向路线](benchmark37/waypoint_stream.py) · [V57 四轴计数](outputs/benchmark73_precision10_v50/ab_yaw_axis_v69/v57_cardinal_axis.json)

V69 从 V57 的 OBS73 LSTM 初始化，在全 AB 上做 PPO，使用 V66 的提前翻倒结算。全阶段实际航向平方系数由 40 提至 80。额外对实际误差和外推误差绝对值的较大者，在 3°～10° 对超出 3° 的部分按弧度平方扣分，系数 240，超过 10° 接线性尾部。V57 的航向角速度项、6° 超限项和落点计分仍参与奖励。V68 的 actor 回退机制未接入 V69。[生产航向评分](benchmark37/all_phase_yaw.py) · [V69 课程](training_config_v51/ab_yaw_axis/course.json) · [冻结轨迹回算](outputs/benchmark73_precision10_v50/ab_yaw_axis_v69/reward_review.json)

同一 V57 权重和 1,024 个物理初态，将 A 改为释放位置原地跳、B 预瞄原 A 相对步长，3 秒前早倒从 177 降为 1，978 例观测到 B 落地，其中 789 例落在目标 5 cm 内。原地 A 只用于该诊断，V69 训练仍使用原 AB 采样，便于检验加强航向评分本身。[同初态回放](outputs/benchmark73_precision10_v50/ab_yaw_axis_v69/startup_review.json)

V69 四次 PPO 更新已完成。±X、±Y 首落依次为 89.72%、52.43%、75.77%、39.82%，翻倒总数由 V57 的 5 例降至 2 例。开发集长步首落 / 同目标重试为 64.28% / 70.22%，零翻倒，实测 yaw 峰值 28.53°。四轴全组实测峰值 124.96°、0.14 秒预测峰值 384.84°，均出现在 +Y 第 98 例翻倒过程中。去掉两条失稳轨迹，四轴全组实测峰值仍为 29.68°。V57 保留为基线，V69 课程作为航向惩罚验证。 [四轴与开发集完整结果](outputs/benchmark73_precision10_v50/ab_yaw_axis_v69/validation.md)

## V68 从 V57 约束连续跳时分方向的策略漂移

保留 V57。对齐开发集 43 次退步重试后发现，上一跳触地附近已经有电机和速度差异，顶点才切回参考策略时，34 个普通案例一个也没救回。第 93 例沿正南直跳，在第 254 帧持续越过 10°，到目标 4 首落前第 561 帧达 28.28°。单独减半动作变化惩罚改善短程重试，同时抬高 yaw 峰值；单独减半功率惩罚出现翻倒。[轨迹、分方向配对与消融](outputs/benchmark73_precision10_v50/ab_direction_recovery_v68/diagnosis/diagnosis.md)

V68 从 V57 的 OBS73 LSTM 初始化，全部 PPO 训练环境仍为 AB，保留 V66 的早倒结算和 V67 的分阶段参考。新增的训练检查使用种子 68193 的独立连续轨迹；这批数据已用于训练约束，今后不能作为保留验收集。每次更新按八方向、首跳或重试检查相对 V57 的四电机命令及 yaw 差动，超出短程参考的 1.25 倍时，回退部分 actor 更新。首版在连续 +X 第 65 例翻倒。逐层对照后，最终课程固定 V57 的策略 LSTM、归一化和隐藏层，四电机末层只沿较少用于常规连续跳的特征方向更新；40 个主要方向保留，89 个方向可学习，critic 正常更新。出生阶段在这 89 个方向内额外用当前 rollout 的 V66 参考动作拟合末层；拟合步的参数变化量以上一项 PPO 更新的实际改动量为限。部署仍用单个标准 LSTM。命中沿用足端 ≤5 cm，yaw RMS ≤3°、实测峰值与 0.14 秒预测峰值均 ≤10°。[训练实现](benchmark37/directional_actor_guard.py) · [候选课程](training_config_v51/ab_direction_recovery/course.json)

最终课程已做 4 次 PPO 更新和冻结验证。早倒为 168/1,024（V57 为 177，V67 正式为 137）。开发集长步首落/同目标重试 64.85%/68.60%，连续 +X 为 88.93%/78.81%，两套各 128 例均零物理翻倒。第 55 例目标 4 首落偏 3.62 cm，第 93 例第三次落地才接住，该例全回合 yaw 峰值为 11.16°。开发集及 +X 全组实测峰值分别为 24.11°、22.31°，均未达到 10°。第四次策略更新实收候选步长的 1/32。V57 继续作为保留权重。[完整短程报告及权重核对](outputs/benchmark73_precision10_v50/ab_direction_recovery_v68/validation_startup/report.md)

V68 正式 50 次更新已完成。后 25 次中 24 次完全回退确定性 actor，第 34 次仅保留 1/32。3 秒前早倒 172/1,024，开发集长步首落/重试 64.75%/65.99%，连续 +X 为 89.00%/76.11%。开发集第 13 例在 13.69 秒翻倒，实测和预测 yaw 峰值为 34.38°/122.49°；连续 +X 零翻倒，实测和预测峰值为 18.94°/30.88°。保留 V57。[四组冻结评估与更新步长](outputs/benchmark73_precision10_v50/ab_direction_recovery_v68/train_seed42_20261122_025807_049354Z/evaluation_v68.md)

如要手动启动 V68 的 50 次更新实验，在仓库的 Hopper-sim-Isaac 目录执行下列命令。

```bash
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_direction_recovery/course.json
```

## V67 从 V57 分别学习出生恢复和连续落点控制

V57 仍是保留权重。V66 的 70 个早倒救回初态中有 66 个出生即需跳长步。676 次 +X 首落退步和 18 次重试退步按案例、目标坐标及尝试次数对齐。对于 128 个不同初态上各选的一次失败，在 V66 原轨迹的前次接地切回 V57 可救回 64/111 个首落与 9/17 个重试，到顶点才切则是 10/111 和 0/17。[逐帧轨迹与对照干预](outputs/benchmark73_precision10_v50/ab_spawn_guard_v67/diagnosis/alignment.md)

V67 的单个 OBS73 LSTM 从 V57 初始化，训练仍为全 AB。保留 V66 对物理提前翻倒的终止结算。出生第一条边 ≥0.40 m 且 A 尚未有效触地的前 300 帧，由冻结 V66 提供四电机动作参考；其他帧使用冻结 V57，两者持续读取学生的实际历史并仅在对应回合结束时清空记忆。学生用自己的高斯样本与参考的电机方向修正 PPO 策略优势，参考信号的条件期望为零；环境回报与 critic 目标仍采用原任务结算。评估和部署均只需要学生的标准 checkpoint。yaw 门槛沿用 RMS ≤3°、实测及 0.14 秒预测峰值 ≤10°。正式 50 次更新由用户手动启动。[课程](training_config_v51/ab_spawn_guard/course.json) · [训练实现](benchmark37/ab_phase_reference.py)

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_spawn_guard/course.json
```

两次独立 PPO 更新验证（524,288 步）后，冻结学生的 3 秒前翻倒为 149/1,024，V57 为 177。开发集长步首落/重试为 63.18%/66.91%，连续 +X 为 88.85%/72.95%，两个评估各 128 例均零失稳。连续 +X 首落和重试仍分别低于 V57 0.06、0.95 个百分点，实测 yaw 峰值由 20.42° 升至 22.57°。正式 50 次更新后的表现需要独立验收。[短程验证及逐项计数](outputs/benchmark73_precision10_v50/ab_spawn_guard_v67/validation/report.md)

V67 正式 50 次更新已评估。1,024 个同初态中，3 秒前翻倒为 137 次（V57 为 177、V66 为 135、两次更新为 149）。开发集长步首落/同目标重试为 66.35%/60.62%，连续 +X 为 88.83%/75.27%。两组各 128 例均零翻倒。连续 +X 实测 yaw 峰值从短程 22.57° 降到 19.56°，开发集则升到 28.28°，均超 10° 门槛。V57 继续作为基线，V67 留作少早倒与连续 +X 重试对照。[四组冻结评估与训练曲线](outputs/benchmark73_precision10_v50/ab_spawn_guard_v67/train_seed42_20261121_171625_446941Z/evaluation_v67.md)

## V66 从 V57 修正全 AB 提前翻倒的回报

V66 从已有 V57 的最终 LSTM 及观测归一化器初始化，使用新的 PPO 优化器。1,024 个全 AB 环境，50 次更新。训练负责人手动启动。

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_failure_settlement/course.json
```

从出生起计 300 个 10 ms 控制步，在第 t 步发生物理翻倒时，一次扣除余下时窗的分值。每个未经历时隙 j 的分值为 40 × 0.99^j，j 从 t 到 299。PPO 按原有 γ=0.99 折扣真实终止帧。第 300 步后追加扣分为零。任务仍按原有规则在 B 触地或物理终止时重置，普通时间上限与这个计分时窗独立。正常 AB、同目标恢复及纯超时的新增项均为零。发生在时间上限的物理翻倒按真实终止处理。[课程](training_config_v51/ab_failure_settlement/course.json) · [结算实现](benchmark37/ab_failure_settlement.py)

在 V57 同一批 1,024 个训练初态中，严格完成 AB 的 342 例与 A 实际落偏、重试同目标后严格接住的 67 例用于检验误扣。冷启动 LSTM、MLP 各 1,024 对早倒与晚倒同初态轨迹用于重算。验证数据与扣分量见[回放报告](outputs/benchmark73_precision10_v50/ab_failure_settlement_v66/validation/report.md)。

训练后取 model_49，用冻结开发集 15 秒及连续 +X 60 秒各 128 例复评。在步长 ≥0.40 m 的首次真实落地上统计足端误差 ≤5 cm 的比例；另列严格命中（顶点 ≥1 m、倾角 ≤15°）。V57 参照为开发集 64.57%、连续 +X 88.91%，两组均 0/128 失稳。yaw 验收仍要求全程 RMS ≤3°、实测及 0.14 s 预测峰值 ≤10°。同时读平均回合时长、AB 有效触地数、重试成功数、回报及价值损失；累计分值的尺度已变。

V66 最终权重复评完成，建议继续保留 V57。1,024 个同初态回合中，3 秒前翻倒从 177 减到 135，全部物理翻倒从 177 减到 136。普通 AB 双严格命中从 342 增至 361，同目标落偏后严格接住分别为 67/106、73/115。开发集和连续 +X 的长步 5 cm 首落率分别从 64.57%、88.91% 降至 60.96%、86.06%，连续 +X 重试从 73.90% 降至 68.47%。两组各 128 例均零翻倒，yaw 峰值降低，仍超出验收门槛。V66 critic loss 前后十轮均值下降 60.6%，训练 reward 中段触底后部分回升。[完整冻结评估与训练曲线](outputs/benchmark73_precision10_v50/ab_failure_settlement_v66/train_seed42_20261121_145915_904187Z/evaluation_v66.md)

## V57 全 AB 课程的 LSTM、MLP 对照

本次按用户授权完成两组正式训练与冻结评估。保留已有 V57 权重。两架构均从新权重开始，各 1,024 环境、50 次 PPO 更新、13,107,200 步。训练中 LSTM 与 MLP 分别只有 39 次和 25 次有效落地，严格命中均为零。开发集和连续 +X 各 128 例，两个最终模型均在第一次接地前翻倒。已有 V57 两组均零失稳，连续 +X 长步首落 5 cm 命中率为 88.91%。

用训练分布中 1,024 个相同 AB 初态，冻结第一次与最后一次 PPO 更新后的权重。最终 LSTM 在 1,016 例中更早翻倒且折扣回报更高，MLP 为 1,024 例。另将已有 V57 在每个控制帧的 LSTM 隐状态清零，保留 OBS73 短历史，开发集失稳由 0 变为 119。下一轮优先修正失败轨迹的回报排序，再验证电机历史和预随机状态的覆盖。[评估与训练曲线](outputs/benchmark73_precision10_v50/ab_architecture_comparison/20261121T085523Z/evaluation.md) · [观测与奖励审计](outputs/benchmark73_precision10_v50/ab_architecture_comparison/20261121T085523Z/architecture_audit.md) · [复跑入口](run_ab_architecture_comparison.sh)

## V65 从 V57 加大 yaw 超限惩罚

V65 从 V57 的最终 `model_49.pt` 和归一化器开始，使用新的优化器，训练预算为 50 次 PPO 更新。相对 V57，唯一变化的训练参数是全阶段航向超限权重，从 120 提到 360。每帧取当前绝对航向误差与 0.14 秒预测绝对误差的较大值，超过 6° 的部分按弧度平方扣分。已有的小角度航向和角速度惩罚继续生效。验收仍要求 yaw RMS ≤3°，实测及预测峰值 ≤10°；同时对照 V57 的长步首次命中、重试命中及物理失稳。[课程](training_config_v51/ab_yaw_excursion/course.json)

用训练代码重算 V57 连续 +X 回放，每次确认落地后的前 100 ms 共 68,857 个环境帧，其中 14,924 帧触发更强扣分。V64 相同路线触发 15,120/68,490 帧；同一 V64 原地对照为 3,864/81,327 帧。两项航向误差都在 6° 内时与旧奖励逐帧一致。33 项单元测试通过，并已用冻结的 V57 权重完成 16 环境全 AB 的 Isaac 零更新检查。[回放重算](outputs/benchmark73_precision10_v50/ab_yaw_excursion_v65/yaw_reward_replay.json) · [仿真检查](outputs/benchmark73_precision10_v50/ab_yaw_excursion_v65/ab-check_seed42_20261121_072307_566352Z/manifest.json)

手动启动训练：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_yaw_excursion/course.json
```

**V65 训练后复评。** 两组各 128 例均无翻倒。连续 +X 和开发集的实测 yaw 峰值分别为 18.54° 和 18.95°。连续 +X 触地后前 100 ms 的实测航向越限帧率由 V57 的 1.22% 变为 1.31%；长步首次落入 5 cm 的比例由 88.91% 降为 85.95%。第 31 例目标 19 和第 55 例目标 4 均首次命中。保留 V57。[V65 最终权重评估](outputs/benchmark73_precision10_v50/ab_yaw_excursion_v65/train_seed42_20261121_072826_322689Z/evaluation_v65.md)

## V64 从 V57 训练同目标重试时的早期制动

V57 的 `model_49.pt` 及归一化器作为手动续训父权重。V64 使用新的优化器、原有 1,024 个全 AB 环境和 50 次 PPO 更新，输出到独立目录。OBS73、四电机、动力学、yaw 目标及奖励、长步首跳的落地验收均沿用 V57。[课程](training_config_v51/ab_retry_compression/course.json) · [支撑期反馈](benchmark37/retry_compression.py)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_retry_compression/course.json
```

**41 次新越限重试。** 连续 +X，种子 7001，128 例，每步 0.5 m。误差为实测根连杆水平速度与本次重试锁存的目标离地速度之差的模。确认上次落偏且保留目标的触地帧为 0，每帧 10 ms。纳入触地误差 ≤0.47 m/s、离地误差 >0.47 m/s 且完成下一次落地的尝试。V63 有 41 次，其中 34 次是该目标的第二次尝试，另 7 次是第三至第八次。与 V57 按场景、目标序号、绝对目标坐标和尝试序号对齐，7 次有直接配对，34 次 V57 已提前推进。配对组 V63 命中 2/7，V57 命中 6/7。每次尝试的绝对帧号、确认离地帧、速度、腿轴投影、目标速度和实际落偏保存在[全部 41 次对齐数据](outputs/benchmark73_precision10_v50/ab_retry_compression_v64/diagnosis/aligned_41.json)。

| 与触地相隔的时间 | V63 全部 41 次平均速度误差 | V63 配对 7 次 | V57 配对 7 次 | V63 在该帧首次越限的次数 |
|---|---:|---:|---:|---:|
| 0 ms | 0.360 | 0.397 | 0.484 | 0 |
| +10 ms | 0.344 | 0.384 | 0.393 | 0 |
| +20 ms | 0.398 | 0.427 | 0.297 | 10 |
| +30 ms | 0.487 | 0.502 | 0.237 | 11 |
| +40 ms | 0.557 | 0.566 | 0.230 | 19 |
| +50 ms | 0.552 | 0.560 | 0.224 | 1 |

误差单位为 m/s。按每次误差首次高于上一帧判定开始增长，第 +1/+2/+3 帧分别有 11/29/1 次。第 107 例目标 31 的第二次尝试中，两者入场误差接近。V63 第 5243 帧为 0.378，第 5246 帧升至 0.500，离地为 0.598，落偏 8.23 cm。V57 对应第 4402 帧为 0.367，第 4405 帧为 0.334，离地为 0.335，落偏 4.67 cm。[支撑期对齐图](outputs/benchmark73_precision10_v50/ab_retry_compression_v64/diagnosis/aligned_retry.png)

**为什么制动失败。** 在误差开始增长前，V63 触地前一帧全部 41 次的足端到机身轴沿入场误差方向的投影均为负。触地后 20 ms，34 次的腿轴已扫到放大入场误差的一侧。此时平均弹簧压缩量为 5.46 cm，接近这 41 段的平均最大压缩量。34 次的平均 X 速度从 +0.461 变为 +0.604 m/s，期望仅 +0.051。其余 7 次沿 X 从 −0.127 越过期望 −0.039，到达 +0.451 m/s，属于纠正过头。两类都需要按二维速度矢量计分。两次电机复放的逐跳记录与对应原始评估逐字节相同。在 V63 的 41 次里，触地至离地的平均旋翼水平冲量模约 0.00245 N·s，根连杆水平速度变化模平均 0.241 m/s，腿轴在 +20 ms 与最终速度变化方向的点积 41 次均为正。这些数据支持弹簧支撑方向为主要机制；没有单独测得各接触力分量。V63 原先在离地才检查下降期预测与实测是否同向，41 次中仅 17 次被标记；27 次最终落偏中仅 10 次得到新增扣分。

7 对同目标重试也显示了支撑方向为何相反。触地前 10 ms，V63 足端在机身前方的 X 距离平均为 1.15 cm，V57 为 2.08 cm。到触地后 20 ms，V63 足端已在机身后方 0.88 cm，而 V57 足端仍在前方 0.98 cm。另在 V63 内，用触地水平速度与目标速度的欧氏距离之和贪心选取 41 个互不重复、离地未越限的对照。其触地误差平均 0.362 m/s，与越限组的 0.360 接近；20 ms 时对照已降至 0.285，越限组则升至 0.398 m/s。该对照匹配入场速度，未控制其他状态。

**两项训练改动。**

1. 只在真实移动 A 落偏、确认触地且入场误差在门槛内时，用当前腿轴和世界角速度外推 20 ms 的负载方向，估计假如保留 A，回弹是否会超出 0.47 m/s。沿用 V57 回放标定的 1.81 倍竖直速度变化近似和原有速度上限。满足风险条件时，将原抽中“继续 B”的样本用于同目标重试；原抽中的局部保持及正常命中后的 A→B 照常执行。风险估计只控制训练采样，不决定实际扣分。对 V63 回放的 991 次重试，此筛选命中 41 次新越限中的 23 次，共筛出 36 次。所有分支至多两次有效落地。
2. 对实际保留移动目标且入场误差 ≤0.47 m/s 的重试，在每个支撑帧和确认离地帧用实测误差计算超限量。超限增大时立即扣其增量，回落时返还相应差额，离地后冻结余额。如果该跳在 5 cm 圆内落地，返还余额；落偏或物理终止保留已扣的净额。超时或支撑超过原 0.10 s 窗口且没有确认离地时取消未结算余额。采用 V57 原离地项的权重 4 和 0.10 m 距离尺度。成功跳的未折现新增回报总和为零，PPO 的折扣和信用分配仍会受反馈时刻影响。新增记录为 `Episode_Reward/retry_compression` 及 `AB/compression_promotions_per_step`、`AB/compression_debits_per_step`、`AB/compression_returns_per_step`。

**验证与验收。** 用生产函数重放 V63 的 41 次，全部从实际首次越限帧开始扣分，27 次最终落偏有净扣分，14 次最终命中净扣分为零。V57 全部 862 次 +X 已完成重试里有 36 次新越限，其中 24 次最终落偏有净扣分。首跳不进入该项。固定开发集第 55 例中，V57 目标 4 首次落偏 3.59 cm，目标 5 和 7 的重试均接住；V63 目标 4 从 11.07、5.49 cm 恢复到 3.50 cm，成功重试无净扣分。[重算汇总](outputs/benchmark73_precision10_v50/ab_retry_compression_v64/diagnosis/summary.json)

Isaac 16 环境，V57 父策略零更新运行 1,120 步后再运行 2,263 步。第 3379 帧第 7 环境将预测高风险的 A 落偏改成重试，第 3383 帧实测误差为 0.539 m/s 并触发扣分。扩展段记录到 1 次提升重试采样、75 个跟踪帧、1 次扣分和 1 次正向返还。33 项单元测试通过，包含支撑时减速返还及离地后物理翻倒的结算。[短程仿真](outputs/benchmark73_precision10_v50/ab_retry_compression_v64/verified_check/manifest.json)

按正式四元数 OBS73 和冻结 V57 权重，在 V64 课程跑完整开发集及连续 +X，各 128 例。逐例核对场景、逐跳记录和全程 yaw，均与归档 V57 完全相同。第 31 例目标 19 向 +X 离地，首次落偏 1.29 cm；第 55 例目标 4 首次落偏 3.59 cm，目标 5 和 7 的重试误差分别为 4.13、2.79 cm。早期辅助回放使用了另一种 73 维观测映射，已弃用；下列记录均使用生产四元数映射。两组父策略都超过了 yaw 的 3° RMS 或 10° 峰值验收线。[开发集冻结回归](outputs/benchmark73_precision10_v50/ab_retry_compression_v64/diagnosis/parent_development.json) · [连续 +X 冻结回归](outputs/benchmark73_precision10_v50/ab_retry_compression_v64/diagnosis/parent_stream_straight60.json)

训练后以相同开发集与 +X 路线检验长步首次命中、所有重试命中、从门槛内跨到门槛外的次数及其占比、实际及预测 yaw、全部失稳，单独复核第 31 例目标 19 和第 55 例目标 4。计数随各模型产生的重试总数变化，应同时报告分母。V57 连续 +X 长步首次命中率为 88.91%，yaw RMS / 实际峰值为 3.08° / 20.42°；开发集分别为 64.57%、3.73° / 22.64°。两组均 0/128 失稳。

**V64 训练后复评。** 最终权重连续 +X 新越限 45/1,026（4.39%），实际重试命中 65.89%，第 31 例目标 19 首次命中，第 24 例在目标 7 重试中翻倒。开发集长步首次命中 59.74%，第 55 例目标 4 五次均未接住。保留 V57。[V64、V57、V63 完整对比](outputs/benchmark73_precision10_v50/ab_retry_compression_v64/train_seed42_20261121_065433_907494Z/evaluation_v64.md)

## V63 从 V57 预判弹簧支撑期间的水平速度

V57 的最终权重仍为主线。V63 手动微调从 V57 的 `model_49.pt` 及其观测归一化器开始，使用新的优化器。训练为全量 AB，1,024 个环境、50 次 PPO 更新。配置相对 V57 仅新增支撑期速度预估及其结算，另指定父权重、版本和独立输出目录。控制接口仍为 OBS73 和四路直接电机，沿用相同的物理模型、AB 采样、航向目标和奖励。[课程](training_config_v51/ab_support_preview/course.json) · [预估与结算](benchmark37/support_preview.py)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_support_preview/course.json
```

**连续 +X 第 31 例。** 固定种子 7001，两次 128 环境、60 秒的完整电机回放分别与原版逐跳记录逐字节一致。以目标 18 确认触地为 0 ms；两策略进入该目标的历史状态不同。V62 的推进预瞄权重在 −300 ms 为 0，在 −140 ms 已达 1。到 −10 ms，参考腿轴水平分量为 (+0.054, −0.031)，实测从足端指向根连杆的单位轴仍为 (−0.156, +0.088)。V57 同帧实测为 (−0.066, −0.015)，倾角 3.88°；V62 为 10.33°。0 ms 时两者均正确切换到 (9.5, 0) m。[按触地对齐的轨迹图](outputs/benchmark73_precision10_v50/ab_support_preview_v63/diagnosis/support_alignment.png) · [逐帧动作、推力、弹簧和姿态](outputs/benchmark73_precision10_v50/ab_support_preview_v63/diagnosis/case31_frames.json)

| 第 31 例 | 18 的实测落偏 | 触地 XY 速度 | +30 ms | +70 ms 确认离地 | 19 的结果 |
|---|---:|---|---|---|---|
| V57 | 1.78 cm | (+0.471, +0.038) | (+0.239, −0.036) | (+0.208, −0.068) | 首次落偏 1.29 cm，命中 |
| V62 | 2.99 cm | (+0.428, +0.063) | (−0.219, +0.504) | (−0.466, +0.712) | 离地 0.66 s 后倾角 90.24° |

速度均为世界水平速度，单位 m/s。V62 触地时四电机合推力 0.594 N，V57 为 0.092 N。V62 电机响应时间常数约 68–89 ms，弹簧在确认触地后的 70 ms 区间内压缩至 4.47 cm 并释放。用每帧推力和姿态积分，这 70 ms 的 V62 旋翼水平冲量约为 (−0.0085, +0.0079) N·s，根连杆水平速度的变化已达 (−0.894, +0.648) m/s。腿轴摆向、弹簧压缩和变速帧序支持地面支撑是主要机制。根连杆速度含刚体旋转影响，这组记录不能精确拆分各接触力或识别某个 PPO 更新的因果贡献。

**同目标高误差重试。** 按每次重试的所需离地速度，把触地时已有的水平速度误差作为正方向。V62 的 49 次高误差重试中，−10、+10、+20 ms 的实测腿轴在该方向投影为正的分别有 3、38、45 次，V57 则为 1/42、29/42、35/42。V62 的目标 X 速度平均 +0.033 m/s，触地时实测 +0.372 m/s、+30 ms 为 +0.503 m/s、离地为 +0.547 m/s。仅检查触地前一帧会漏掉压缩时腿轴扫过竖直、把支撑冲量打到错误一侧的情形。[重试对齐数据](outputs/benchmark73_precision10_v50/ab_support_preview_v63/diagnosis/contact_diagnosis.json)

**新训练信号。** 在自由下降、倾角不超过 15°、预计距触地 30–140 ms 的帧，使用角速度外推腿轴至触地，再以足端短时固定近似，估计根连杆向前移动 30 ms 对支撑轴的影响。V57 有 6,141 段满足前一帧自由下降且确认离地的支撑，触地至离地时长中位数 60 ms，竖直速度变化相对上一帧下降速度的中位比为 1.81。用该比值和支撑轴预测水平离地速度，多帧平均。实际触地后以环境真正发布的目标为准。已推进的连续长步仅检查向后或横向失控的离地；保留原目标的重试仅检查实际误差超过 0.47 m/s 且比触地更大的离地，同时要求下降期预估支持相同的偏差方向。下一次实测足端落在 5 cm 圆内就清除记录，落偏或物理终止才扣分。沿用原离地项的权重 4、尺度 10 cm。局部保持、换向、正常幅度的离地和超时清除均无新增扣分。此预估器只服务训练评分。

**验证。** 用同一生产函数重算轨迹，第 31 例预估的离地速度 V62 为 (−0.455, +0.428)，V57 为 (+0.189, −0.094) m/s。V62 在翻倒时计 −4，V57 在成功触地时计 0。V62 的 49 次高误差重试中，24 次在离地时标记，其中 15 次最终落偏并扣分；全部 799 次成功重试的新增扣分均为零。V61 开发集第 55 例目标 4 的成功重试（落偏 2.97 cm）、V62 同例目标 4 的首次命中（4.72 cm）及后续成功恢复均无新增扣分。V57 的 +X 同向长步中，5,096 次有完整前序移动触地的首跳命中全部计零。[冻结轨迹重算](outputs/benchmark73_precision10_v50/ab_support_preview_v63/diagnosis/shadow.json)

Isaac 使用 V57 父策略验证 16 个 AB 环境，共 1,633 步、0 次优化。前 1,120 步记录 1,376 个下降预估样本、58 次离地，全部在速度门槛内；继续 513 步覆盖 2 次离地风险，实测触发 1 次扣分。29 项单元测试通过。训练后的行为须用相同种子复评。V57 参照值为连续 +X 长步首次 88.91%、实际 yaw RMS / 峰值 3.08° / 20.42°、固定开发集 3.73° / 22.64°、两组均 0/128 物理失稳。检查第 31 例目标 19、第 55 例的实际重试、全部已落地重试以及高误差样本构成。[零更新仿真记录](outputs/benchmark73_precision10_v50/ab_support_preview_v63/verified_check/manifest.json)

## V62 从 V57 修正直线首跳侧偏和重试回弹速度增长

保留 V57 为主线。V62 的初始权重及观测归一化器取自 V57 的 `model_49.pt`。在 V57 课程上复用 V61 的临近触地预瞄和受姿态、yaw 约束的重试推力减免。新增评分在当前跳的实际触地或物理失稳时结算。用户手动启动 50 次 PPO 更新。[V62 课程](training_config_v51/ab_waypoint_events/course.json)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_waypoint_events/course.json
```

**对齐依据。** 用种子 7001 重放 V61 两个检查点的连续 +X 和最终权重的开发集，每组 128 例，三份逐跳结果与原始评估逐字节相同。按场景编号、目标序号、实际坐标和尝试序号配对，逐帧例子以各自物理起跳为零。顶点预测使用当时的根位置、平移速度、起跳足端偏移和该跳已锁存的飞行时长，触地误差使用实际足端。+X 同目标的 5,359 个首次长步中，`model_20` 命中 4,665 个，`model_49` 命中 4,474 个。沿 X 单独超出 5 cm 的次数为 428 和 404，沿 Y 超出的次数为 125 和 283。两轴超限允许重叠。两策略均在回合前 30 秒触地的 2,510 对首跳从 2,176 次降为 2,098 次，均在后 30 秒触地的 2,465 对从 2,148 次降为 2,055 次。[成对事件](outputs/benchmark73_precision10_v50/ab_waypoint_events_v62/aligned_v61/paired_events.json)

限定两者上一目标均首次推进，且用离地状态外推的预测点相距不超过 5 cm，仍有 414 对。两模型的平均有符号侧向偏差在起跳预测时为 +0.44 / +0.48 cm，顶点时为 +1.11 / +1.84 cm，实测落地为 +0.63 / +1.44 cm。第 108 例目标 3，顶点预测 Y 偏差从 +1.92 增到 +6.97 cm，实际足端总误差从 1.16 增到 6.52 cm。该对照定位了上升段残留侧偏，不能凭两份冻结策略识别一次特定的 PPO 更新或奖励项造成偏移。

两模型各自起跳水平速度误差大于 0.47 m/s 的已完成重试分别为 33 和 65 次，命中均为 12 次。其平均 X 速度（确认触地 → 重试离地，所需离地速度）分别为 0.366 → 0.527（0.015）和 0.278 → 0.531（0.006）m/s，回弹后仍带有过多向前速度。按 `model_49` 高误差筛选、有同次重试配对的仅 9 个，命中从 7 次降为 3 次；两者都高误差的只有 1 对。第 28 例目标 4，两者进入重试时的速度误差分别为 0.400 / 0.403 m/s，离地时分别为 0.229 / 0.658 m/s，落偏 3.82 / 15.69 cm。[逐帧轨迹图](outputs/benchmark73_precision10_v50/ab_waypoint_events_v62/aligned_v61/aligned_jumps.png) · [轨迹数据](outputs/benchmark73_precision10_v50/ab_waypoint_events_v62/aligned_v61/example_frames.json)

**训练评分。** 首次尝试至少 0.40 m 且相邻边同向的移动目标，在首次物理顶点记录预测横向偏差超出 5 cm 圆剩余宽度的部分，带左右符号。触地落偏时，以顶点和落点同侧超出的较小值计分，尺度为 5 cm，代价上限取原落点项的权重 2。同目标重试在确认保留目标的触地帧记录相对所需离地速度的误差，离地时记录超出触地误差和 0.47 m/s 中较大者的新增部分。乘预计飞行时间换算为距离，仅在最终落偏仍有同向分量时结算，代价上限取原起跳项的权重 4。物理失稳结算本跳已记录的风险，超时重置清除待结算状态。两个新增项在落入 5 cm 圆时均为零。新逻辑不向电机发指令。TensorBoard 项为 `Episode_Reward/straight_first_apex_cross`、`Episode_Reward/retry_rebound_growth`。

**验证。** 用生产评分函数套回 `model_49` 原轨迹，5,478 个首次长步中 756 个实际落偏触发横向项；65 次高误差重试中 35 次实际落偏触发回弹项，命中样本的两项新增代价均为零。开发集第 55 例目标 4 的成功重试从触地速度误差 0.479 m/s 降到离地 0.255 m/s，落偏 2.97 cm，重试实际 yaw 峰值 5.49°，新增代价为零。V57 父权重的 Isaac AB 短程检查为 16 环境、1,120 步、0 更新，覆盖 A/B 顶点 20/16 次、14 次横向扣分、7 次重试离地。延长零更新检查到 2,615 步，15 次重试离地中出现 1 次超阈值，触地进入时误差 2.053 m/s，离地 2.336 m/s，本跳结束时结算了回弹项；另有 11 次重试命中。[延长检查](outputs/benchmark73_precision10_v50/ab_waypoint_events_v62/extended_check/manifest.json) 26 项单元测试通过。正式训练后仍须用相同开发集和连续 +X 检验长步首次命中、高误差重试、实际及预测 yaw、失稳以及第 55 例。[状态重算](outputs/benchmark73_precision10_v50/ab_waypoint_events_v62/aligned_v61/summary.json) · [Isaac 验证](outputs/benchmark73_precision10_v50/ab_waypoint_events_v62/ab-check_seed42_20261121_035446_466984Z/manifest.json)

## V61 从 V57 修正落偏前的交接和重试中的姿态余量

手动候选从 V57 的 `model_49.pt` 和观测归一化器初始化。训练仍为 1,024 个全 AB 环境、50 次更新。V57 是保留基线。[V61 课程](training_config_v51/ab_retry_safe_assist/course.json)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_retry_safe_assist/course.json
```

同场景、同目标坐标和尝试序号的已完成重试中，固定开发集 V57 和 V60 分别命中 95/159 和 86/159，连续 +X 分别为 66/87 和 70/87。按 V60 实测起跳速度误差大于 0.47 m/s 分层，开发集这 10 对从 8 次命中降为 0 次，其余 149 对从 87 次降为 86 次。连续 +X 这 5 对从 3 次降为 2 次，其余 82 对从 63 次升到 68 次。该配对只覆盖双方都出现并完成的重试，全部 +X 重试命中率从 637/862 降为 717/1041。[配对及逐例摘要](outputs/benchmark73_precision10_v50/ab_retry_safe_assist_v61/retry_alignment/case_summary.json)

开发集第 55 例目标 4，V57 第 577 帧首次落点偏差 3.59 cm 并推进。V60 第 570 帧预测将偏 5.38 cm，触地准备仍赋予推进 85% 的权重。第 580 帧实际偏 5.68 cm 并重试，第 586 帧实测水平速度为 (−0.530, +0.510) m/s，目标为 (−0.055, +0.0004) m/s。以第一个纠偏参考方向为固定轴，实测推力投影在第 612 帧转正。第 622 帧训练参考首次改变巡航项，该帧倾角 10.0°，按当下角速度外推 0.14 s 的最大倾角为 26.5°。至第 663 帧预测落偏从窗口开始的 75.5 cm 收到 36.3 cm，倾角增至 29.4°。第 690 帧倾角 92.7°，实际 yaw −5.4°，回合物理终止。参考值是训练信号，冻结回放不能确定哪一帧电机输出由该项学得。

| 已完成的第二次尝试 | V57 顶点预测 / 实际落偏 | V60 顶点预测 / 实际落偏 | V60 起跳 / 首次有效纠偏参考帧 |
|---|---:|---:|---:|
| 开发集 13，目标 1 | 3.42 / 2.20 cm | 14.68 / 8.53 cm | 271 / 307 |
| 连续 +X 80，目标 4 | 11.87 / 9.78 cm | 5.78 / 2.83 cm | 597 / 634 |

V61 有两处训练参考改动，权重沿用 V57。

1. **临近触地时准备保留当前点。** 在预测距触地 140–100 ms 内逐渐切换准备速度的混合规则。距触地 100 ms 以内，预测落点越过 5 cm 验收圆后的过渡带从 2.5 cm 收窄到 0.8 cm。数据依据是 V57 开发集触地前 100 ms 预测误差 5.5–6 cm 的 57 跳，平均推进权重 70%，实际推进 11%。新规则只改变边缘落偏时的反弹准备，圆内位置项仍全额计入，实际触地继续决定换点。对已记录的 V60 状态逐帧重算，第 55 例触地前 100 ms 的推进权重从 85.0% 降到 53.1%。
2. **重试空中只减免已有的同向推力。** 沿预测误差方向，将实际推力换算为水平加速度；仅减免已经产生的同向分量，最多减免参考大小，垂直该方向和多余的分量继续计入巡航代价。零推力代价为零。起跳速度误差超过 0.47 m/s 的整跳按 V57 巡航项计分。该阈值取 V57 开发集和连续 +X 重试起跳速度误差 P95（0.444、0.467 m/s）的较大值向上取整。按 0.14 s 恒角速度预测的最大倾角在 8–15° 内逐步减少减免，实际及外推 yaw 幅值在 6–10° 内也逐步减少减免，取两者的较小余量。首次长步没有该项减免，AB 采样、yaw 权重和实测落点验收沿用 V57。

用同一批 V60 实测帧重算 V61 巡航项，第 55 例 42 帧和第 13 例 40 帧全部恢复基线代价；+X 第 80 例 40 帧全部保有同向减免。该重算使用原轨迹，没有预测训练后命中率。[逐帧数据](outputs/benchmark73_precision10_v50/ab_retry_safe_assist_v61/retry_alignment/aligned_frames.json) · [按起跳对齐的轨迹图](outputs/benchmark73_precision10_v50/ab_retry_safe_assist_v61/retry_alignment/retry_alignment.png)

训练后分别验收固定开发集和连续 +X。以 V57 为基线，比较全部及可配对重试、未完成重试、长步首次 5 cm 命中、全阶段 yaw RMS/峰值/越界率以及物理失稳。单独复看第 55 例，并对照中途和最终权重的 reward、value loss、surrogate loss。V57 父策略在新课程下的 Isaac 零更新检查运行了 16 个环境、1,120 步。274 个下降准备样本降低了推进权重，930 个预测在圆内的普通长步准备样本与 V57 一致，6,505 个普通长步样本的空中重试减免均为零。重试空中有 139 个样本获得减免；此短程检查未触发整跳或姿态拦截，拦截分支另用上述失败轨迹和单元测试覆盖。23 项单元测试通过。[零更新验证](outputs/benchmark73_precision10_v50/ab_retry_safe_assist_v61/ab-check_seed42_20261121_030851_620821Z/manifest.json)

## V60 同目标重试的飞行中段落点反馈

按同一冻结场景、case ID、目标序号、目标 XY 和尝试序号配对，仅分析双方都完成的重试。固定开发集配出 112 对（双方都落偏 19 对），连续 +X 配出 84 对（双方都落偏 9 对）。同一目标的前序落点、姿态和速度可因策略不同而不同。按起跳和顶点事件对齐，沿目标方向及其左侧正交方向分解实测落点。顶点外推使用起跳时足端相对根连杆的水平偏移、顶点根位置/速度和历史飞行时长的剩余值。离线从已记录的位移分解重建该值；训练时只读取当下及过去的状态。[逐对数据](outputs/benchmark73_precision10_v50/ab_retry_midflight_v60/matched_v57_v59/paired_retries.json)

V60 从 V57 最终策略和归一化器做 50 次手动微调。确认同目标重试后，在实际起跳锁存足端相对根连杆的水平偏移，在预计飞行时长的 35%–75% 且处于自由飞行时，逐控制步外推足端落点。若超出 5 cm，按剩余时间计算把预测落点推回容差圆边缘的水平加速度参考，模长最多 0.8 m/s²。原巡航项以零水平加速度为参考；在上述条件内，使用跟踪纠偏加速度的代价并沿用 V57 的权重和接触准备淡出。策略仍以四路电机自主产生动作。首次长步、就地保持和全阶段 yaw 沿用 V57 配置。[课程](training_config_v51/ab_retry_midflight/course.json)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_retry_midflight/course.json
```

训练后以 V57 最终权重作基准，分别比较重试命中率、重试顶点预测误差、长步首次 5 cm 命中率、yaw RMS/峰值/越界率和物理失稳。开发集基线为重试 252/382、长步 483/748、yaw 越界 3.85%、失稳 0/128；连续 +X 基线为重试 637/862、长步 5219/5870、yaw 越界 1.41%、失稳 0/128。

## V59 落偏后的同目标重试

从 V57 最终权重开始，以新的优化器微调 50 次更新。正常长步、全阶段 yaw、随机出生和 AB 采样沿用 V57。确认落偏且保留当前目标后，记录本次触地时根连杆的实测水平速度。接触阶段的速度参考在估计支撑时长内线性过渡到返回原目标所需的速度；支撑项最多计 0.10 秒，确认起跳后继续跟踪目标速度 0.14 秒。同时根据当前四元数和机体系角速度，按恒角速度外推 0.10 秒，惩罚预测最大倾角超出 12° 的部分。速度与倾角分别以 0.20 m/s、10° 为尺度，按控制步长计算，较大误差采用线性尾部。新增两项记录在 TensorBoard 的 `Episode_Reward/retry_capture_velocity` 和 `Episode_Reward/retry_capture_tilt`。[V59 课程](training_config_v51/ab_retry_capture/course.json)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_retry_capture/course.json
```

零更新检查共 16 个环境、1,120 个控制步。新增项覆盖 39 个支撑和 98 个早期飞行样本，6,505 个普通长步控制样本的新项均为零。正式训练后按同一固定开发集、连续 +X 和相同种子评估最终及中途权重。开发集 V57 同目标重试落地在 5 cm 内为 252/382，重试起跳水平速度误差 P95 为 0.444 m/s，长步首次落地 483/748，yaw 越界帧 3.85%，物理失稳 0/128。连续 +X 对应为 637/862、0.467 m/s、5219/5870、1.41%、0/128。每组同时报告实际和预测 yaw 峰值、重试未完成数及全部回合失稳。[仿真验证](outputs/benchmark73_precision10_v50/ab_retry_capture_v59/ab-check_seed42_20261120_195831_401633Z/manifest.json)

## V58 足端落点精度微调

V57 固定开发集每个移动目标的首次有效触地共 911 次，627 次落在 5 cm 内，284 次落偏。284 次落偏的触地倾角均未超过 15°。触地前 100 ms，两组的足端预测距目标平均为 2.9 cm 和 7.9 cm。对全部 911 次落地，预测点到最终实测落点的距离中位数在触地前 250 ms 为原密集引导 1.28 cm、实测足端预测 1.31 cm，在 100 ms 为 1.44 cm、0.50 cm。这支持在下降末段逐步切换到实测足端。[逐跳记录](outputs/benchmark73_precision10_v50/ab_yaw_heading_v57/train_seed42_20261120_181912_898373Z/development/episodes.json)

V58 加载 V57 最终策略与归一化器，以新的优化器做 50 次更新。保留原 AB 课程和全阶段 yaw 项。下降段按估计剩余飞行时间，在最后 0.25 秒逐步从参考腿位置过渡到实测足端，同时将预测误差尺度从 10 cm 收紧至 5 cm，逐步恢复近地引导权重。有效触地的实测足端误差项权重从 2 提到 6。非零移动目标第一次尝试达到原有严格条件（5 cm、1 m 顶点、15° 倾角）另奖 8。落偏重试仍按实测误差计分。[课程配置](training_config_v51/ab_foot_precision/course.json)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_foot_precision/course.json
```

评估按原有固定开发集与连续 +X 60 秒分开报告。优先核对长步首次 5 cm 命中率（V57 开发集 483/748）、全程 yaw 越界帧（3.85%）、物理失稳（0/128）以及训练曲线。正式评估继续使用固定阈值。

## V57 实际航向误差惩罚对照

以 V56 为对照，将每个控制步的实际航向误差平方惩罚权重从 20 调到 40，沿用相同的父权重、出生分布和 50 次 PPO 更新。实际航向与足端落点须分别验收。手动启动命令如下。

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_yaw_heading/course.json
```

往后评估时查看 `summary.json` 的 `training_trend`。它按每十次更新汇总该权重之前的 TensorBoard 训练回报、回合长度、价值损失、PPO 代理损失、策略噪声和全阶段 yaw 奖励。用同一开发集的中途权重与末次权重判断落点、yaw 和失稳是否随更新改善。位置命中率分列开场原点第一跳、每例第一个移动目标和各移动目标首次尝试。

V56 开发集逐帧回放复现了原有的 128 个回合。第 31 例在第 489 帧落偏 48.8 cm、倾斜 31.0°，重试同一目标时第 514 帧触地，倾斜 71.64°，实际 yaw −0.63°，世界航向角速度 −77.37 rad/s，下一帧物理终止。按瞬时航向角速度外推 0.14 秒得到预测幅值 621.28°。[异常帧与训练曲线](outputs/benchmark73_precision10_v50/ab_yaw_spawn_control_v56/train_seed42_20261120_151333_876593Z/v56_training_diagnosis.json)

## V56 出生偏航扰动对照

V56 从与 V54、V55 相同的 `round_03/train/model_49.pt` 开始。沿用 V55 全阶段 yaw 奖励，将随机出生组的初始 yaw 从 ±8° 收回到 ±5°，世界 z 轴角速度从 ±0.85 收回到 ±0.25 rad/s。随机出生占比仍为 50%。V56 与 V55 的对照检验这两项出生扰动的合并影响。V56 与 V54 的对照检验在相同出生范围下添加全阶段 yaw 奖励的影响。[V56 课程](training_config_v51/ab_yaw_spawn_control/course.json)

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_yaw_spawn_control/course.json
```

用相同冻结开发集和连续 +X 60 秒场景比较 `model_49.pt`。按 100 Hz 统计当前 yaw 或外推 0.14 秒后 yaw 超过 10° 的帧占比。两场景的 V54 基线分别为 3.99%、1.36%，V55 分别为 4.63%、1.53%。同时比较实际 RMS、实际与预测峰值、首次落地 5 cm 命中率及物理失稳。全阶段 yaw 验收要求实际 RMS ≤3° 且每一帧的实际与预测误差都 ≤10°。

11 月 21 日评估 V56 的最终权重。两组各 128 例，三版逐例初态一致。开发集越界帧 6,424/190,490（3.37%），实际 RMS 3.63°，实际峰值 38.15°，0.14 秒外推峰值 621.28°，117 例发生越界，2 例失稳。连续 +X 越界帧 10,025/767,872（1.31%），实际 RMS 3.0048°，实际峰值 21.14°，外推峰值 34.67°，128 例均发生越界，失稳 0 例。两组越界帧占比均低于 V54、V55，两组全程 yaw 验收均未通过。[开发集报告](outputs/benchmark73_precision10_v50/ab_yaw_spawn_control_v56/train_seed42_20261120_151333_876593Z/development/summary.json) · [连续 +X 报告](outputs/benchmark73_precision10_v50/ab_yaw_spawn_control_v56/train_seed42_20261120_151333_876593Z/stream_straight60/summary.json)

## V55 全阶段 yaw 与两跳恢复

V54 的固定开发集逐帧回放复现了原始轨迹。第 109 例最后一次重试从触地到释放用了 60 ms，yaw 由 −4.86° 变为 −5.79°，释放水平速度矢量误差为 1.17 m/s。第 115 例在前一次重试上升时已越过 10°，随后一次触地 yaw 为 −31.23°，60 ms 后再次释放时为 −23.00°，水平速度矢量误差为 1.95 m/s。[两例 100 Hz 记录](outputs/benchmark73_precision10_v50/ab_recovery_v54/train_seed42_20261120_125428_001450Z/case109_115_review/control_100hz.csv)

V55 继承 V54 的两跳、同目标重试和实测足端就地保持，从同一 round_03 权重训练 50 次更新。原有 yaw 奖励继续工作，新增项在每个控制步按世界航向误差与机体 z 轴角速度的平方扣分，权重分别为 20 和 2。当前航向和按世界航向角速度外推 0.14 秒的绝对误差取较大值，超过 6° 的部分以权重 120 按平方扣分。三项都使用弧度并乘控制步长。预测幅值在额外奖励中截到 π，验收保留未截断的预测。实际航向取机身前向轴在水平面的投影方向，接近竖直时该投影很小，解读极值时同时看倾角和机体角速度。

50% 的回合采用随机出生，扰动组的初始航向范围扩大到 ±8°，世界 z 轴角速度范围扩大到 ±0.85 rad/s。正式开发集使用冻结初态。从释放到物理终止或超时，全阶段验收要求全样本实际 yaw RMS ≤3°，每一帧实际和预测偏差都 ≤10°。固定开发集 V54 基线为 191,264 个样本，实际 RMS 3.73°，实际峰值 47.92°，预测峰值 108.76°，115/128 例出现实际或预测越界。[课程](training_config_v51/ab_all_phase_yaw/course.json) · [V54 同口径基线](outputs/benchmark73_precision10_v50/ab_recovery_v54/train_seed42_20261120_125428_001450Z/development_all_phase_yaw/summary.json)

手动启动 V55：

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_all_phase_yaw/course.json
```

训练后同时核对评估报告的 `all_phase_yaw`、`same_target_recovery`、首次落地精度和物理失稳。全阶段约束按 100 Hz 采样，真机行为还需单独验证。

## V54 两跳落偏恢复对照

每回合仍从 A 开始。A 首次有效触地在目标半径 5 cm 内且倾角不超过 15° 时，继续去 B。单纯高度不足计为严格失败，但照常推进。对于位置或倾角未达标的 A，50% 继续原 B，25% 重试 A 并保留原 B 预瞄，25% 把实测触地足端 XY 同时设为当前和预瞄点。第二次有效触地结算后结束；物理失稳可以提前结束。恢复沿用真实触地状态和跨跳策略记忆。每回合独立预抽分支，分支仅在 A 落偏时生效。[课程](training_config_v51/ab_recovery/course.json)

V54 和 V53 都从 round_03 的 model_49 开始，均使用权重 40 的自由飞行 yaw 峰值项和 50 次 PPO 更新。手动启动 V54：

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_recovery/course.json
```

TensorBoard 的 `AB/retry_*` 和 `AB/local_hold_*` 分别给出进入恢复、起跳、落地、5 cm 内命中、物理失稳及平均起跳水平速度矢量误差。固定开发集和在线 +X 评估的 `summary.json` 另列 `same_target_recovery`，统计保留原目标后的下一跳。其中速度误差包括有已确认起跳记录的终止航段。按同一口径回算 V53，固定开发集为 266/380 次落在 5 cm 内，平均起跳速度误差 0.279 m/s，恢复中失稳 1 次；连续 +X 为 693/923 次、0.270 m/s、失稳 0 次。评估时同时比较正常目标首次命中、yaw 峰值和全部失稳。

## 11 月 20 日实飞后的 yaw 对照实验

18:39 实飞记录共有 63 次有效落地。19 次落地当下的航向误差超过 10°，41 次按触地前航向角速度外推 0.14 秒后超过 10°。这 41 次中有 22 次的当下航向仍在门槛内。20:05 的 shadow 记录共有 631 个有效策略更新和 0 次有效落地。

V53 与已部署的随机出生 AB 模型从同一份 round_03 父权重开始，均训练 50 次更新。唯一的奖励差异是自由飞行中的 yaw 峰值项。当前和预测航向绝对误差取较大者，超过 10° 的部分按弧度平方计算。每跳仅在这个超限量创下新高时按增量扣分，权重取 40。评估对照固定开发集及 60 秒连续 +X 的 yaw RMS、峰值、严格落点命中和物理失稳次数。[V53 课程](training_config_v51/ab_yaw_peak/course.json)

手动启动 V53 对照训练：

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark73_phase_motion.sh train --headless --ring-config training_config_v51/ab_yaw_peak/course.json
```

## 全量 AB 训练与连续评估

1,024 个训练环境全部直接从目标 A 开始。释放落地后，第一跳看到 A 和 B，首次有效触地结算 A 并切换到 B。第二跳看到 B 和预瞄点 C，首次有效触地结算 B 后按 timeout 重置。A 或 B 落偏仍计为失败尝试。C 只提供下一跳方向，物理失稳可提前终止回合。[当前课程](training_config_v51/ab_only/course.json)

每次训练重置有 50% 概率沿用原释放状态，另 50% 使用随机出生状态。两组释放高度均为 0.8–1.2 m。扰动组在原路线起点周围半径 4 cm 的圆盘内释放，横滚、俯仰各在 ±7° 内，偏航在 ±5° 内。初始水平速率上限为 0.20 m/s，竖直速度在 ±0.10 m/s 内；世界坐标系水平角速度模长上限为 0.60 rad/s，绕竖轴角速度在 ±0.25 rad/s 内。A、B 的位置锚定在原路线坐标系。正式评估继续使用冻结的释放状态。TensorBoard 的 `AB/original_spawn_A_strict_rate`、`AB/perturbed_spawn_A_strict_rate` 和相应的 B 指标分别记录两组已触地尝试的严格命中率。

AB 步长为 0–0.5 m。按采样器的 20 个等概率区间，A 原地、B 原地、满步同向、满步反向、直角和独立随机方向分别占 10%、10%、25%、25%、10%、20%。每次更新采 256 步，手动一轮为 50 次 PPO 更新。输入使用原有四元数 OBS73，输出为四路电机动作。单层 LSTM、归一化器、物理模型与控制频率沿用 V51。[采样器](benchmark37/ab_pairs.py) · [PPO 配置](training_config_v51/ab/runner_config.json)

固定开发集仍考核原地首跳和随后十个目标。在线评估与部署从释放原点开始，首跳须严格达标。之后在有效触地时，足端距目标不超过 5 cm 且倾角不超过 15° 即推进，低于 1 m 的跳仍记为严格失败。落偏或倾角超限会重试当前目标。每次推进把旧预瞄变为当前目标并请求下一点，LSTM 与观测历史跨跳保留。

```bash
cd ~/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

# 零更新检查。加 --offline 只检查课程与固定路线
bash run_benchmark73_phase_motion.sh check --headless

# 手动训练一轮，1,024 环境
bash run_benchmark73_phase_motion.sh train --headless

# 下面用父权重复现验证；训练完成后可将 ckpt 指向新权重
ckpt="$PWD/outputs/benchmark73_precision10_v50/round_03/train/model_49.pt"
bash run_benchmark73_phase_motion.sh eval --headless --checkpoint "$ckpt" \
  --waypoint-source benchmark37.waypoint_stream:straight_half_metre
bash run_benchmark73_phase_motion.sh eval --headless --checkpoint "$ckpt" \
  --precision-suite continuous
```

`--waypoint-source` 接受可导入的 Python 函数 `source(case_id, next_index, current_xy)`。在重置和每次推进时调用，返回释放坐标系下的绝对 XY（米），单步上限 0.5 m，返回 `None` 表示原地跳。需要在两次触地之间替换预瞄时，仿真调用 `env.set_next_waypoints(ids, points_xy)`，部署调用 `stream.set_next(xy)`。更新在控制线程内、取下一帧观测前执行。[在线目标接口](benchmark37/waypoint_stream.py)

部署侧可直接导入 `deployment.obs73_stream.StreamingObservation73`。同一回合复用一个适配器和一个有状态的策略实例。每个 100 Hz 控制周期传入根连杆在释放坐标系中的位置、wxyz 姿态、线速度和机体系角速度，`observe` 会先按旧目标判定触地，再构造新目标窗口的 OBS73。真实回合重置时同时重置两者。[部署适配器](deployment/obs73_stream.py)

```python
from deployment.obs73_stream import StreamingObservation73
from benchmark37.waypoint_stream import straight_half_metre as source

stream = StreamingObservation73(course, source=source)
# 每个控制周期
obs = stream.observe(position, q_wxyz, velocity, omega_body, spring_q=spring_q)
action = policy.step(obs)
stream.accept(action)
```

2026-11-20 零更新验证中，16 个 Isaac 环境运行 384 个控制步，观测到 15 次 B 触地 timeout。用 round_03 父权重做每步 +X 0.5 m 的在线评估，128 例各运行 60 秒，全部推进到第十个移动点之后，共推进 6,069 次，物理失稳 0 次。该记录验证的是目标交接，新的 AB 权重尚待训练和验收。[短程检查](outputs/benchmark73_precision10_v50/ab_only_v52/ab-check_seed42_20261120_084937_089267Z/manifest.json) · [连续评估](outputs/benchmark73_precision10_v50/ab_only_v52/parent_stream_check/summary.json)

## 父权重与 V51 历史记录

| 用途 | 文件 | 已知结果 |
|---|---|---|
| 当前自动续训的父权重 | [v50 round_03 的 model_49.pt](outputs/benchmark73_precision10_v50/round_03/train/model_49.pt) | 固定开发集全向 9/64、0.5 m 直线 3/64 完成十跳，128 例失稳 0，飞行姿态门槛未通过 |
| 最近完整主线 | [第 45 轮决策](outputs/benchmark73_precision10_v50/continuous_v51/round_00045/decision.json) | 全向 4/64、直线 3/64、失稳 1，父权重仍为 round_03 |
| 最新 AB 对照和回放 | [model_49.pt](outputs/benchmark73_precision10_v50/continuous_v51/ab_pairs/20261120_070527_471469Z/train/model_49.pt) · [评估](outputs/benchmark73_precision10_v50/continuous_v51/ab_pairs/20261120_070527_471469Z/result.json) | 全向 6/64、直线 2/64。60 秒评估全向失稳 1、直线失稳 5 |

全量 AB 自动服务首次启动时从 round_03 父权重及归一化器开始，在 `outputs/benchmark73_precision10_v50/ab_only_v52/` 内从第 1 轮独立记录训练、评估和续训状态。旧 V51 的第 45 轮已完成，第 46 轮手动中断时留下 `model_0.pt`。[旧队列状态](outputs/benchmark73_precision10_v50/continuous_v51/status.json)

## V51 历史冻结配置

| 配置 | 课程与奖励 | PPO | 环境 | 观测 |
|---|---|---|---|---|
| 主线第 46 轮 | [course.json](training_config_v51/continuous/course.json) | [runner_config.json](training_config_v51/continuous/runner_config.json) | [env_config.json](training_config_v51/continuous/env_config.json) | [observation_contract.json](training_config_v51/continuous/observation_contract.json) |
| 最新 AB，含反跳起飞预测修正 | [course.json](training_config_v51/ab/course.json) | [runner_config.json](training_config_v51/ab/runner_config.json) | [env_config.json](training_config_v51/ab/env_config.json) | [observation_contract.json](training_config_v51/ab/observation_contract.json) |

- 两套均使用 1,024 并行环境，每次 PPO 更新采 256 步，50 次更新共 13,107,200 步。主线的一些诊断轮使用 100 次更新做预算对照。当前 PPO 为单层 256 隐状态 LSTM，学习率 0.0003、自适应调度，熵系数 0.002，训练种子 42。
- 输入为四元数版 OBS73，包括当前和下一目标。输出为四路直接电机动作，100 Hz 控制、200 Hz 物理。每次重置独立采样质量和惯量倍率，均匀分布在 0.9–1.1。
- 自由路线从当前目标出发，步长覆盖 0–0.5 m。自由采样中 10% 为零步长、30% 为 0.5 m，其余在范围内连续采样。连续有位移的相邻步包含同向、直角、反向和全圆随机方向。下一点提前可见，当前目标在触地事件推进。落点验收以足端 XY 为准，圆的直径是 **10 cm**，半径 **5 cm**。
- 第 46 轮含 512 个原地环境、128 个按键指令环境、384 个普通移动环境。原地回合上限 15 秒，移动回合上限 60 秒。普通移动路线中十步往返采样配置为 40.48%，路线类型由重置顺序和覆盖规则决定。
- V51 混合 AB 对照含 512 个原地环境、128 个按键指令环境、128 个 AB 环境及 256 个其他移动环境。AB 环境在 B 点发生有效触地并计入该跳奖励后做 timeout 重置。AB 样本一半直接进入 A，一半先做一个移动热身。
- 奖励按每跳的落点、侧向倾角和角速度、原地腿部摆动与电机平衡、yaw 预测误差等项计算。全向与直线开发集各 64 例，初始原地跳和随后十个目标均须首次触地达标、顶点至少 1 m、触地倾角不超过 15°。整段须零倒地，并通过原地和移动飞行姿态及 yaw 门槛，其中原地倾角和移动侧向倾角峰值均限 10°。通过开发集后还要跑新种子留出集、原地保持和四轴按键验收。当前没有获准实机部署的新权重。

## 启动、停止和查看

在训练主机执行。Isaac Sim 为 5.1.0，安装在 `~/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64`，Isaac Lab 位于 `~/Desktop/workspace/IsaacLab`，本次记录的提交为 `3e73d6dd7`。

手动训练使用对应版本的课程命令。自动训练服务的[仓库定义](benchmark37/hopper-precision-continuous.service)可供查阅；截至 11 月 21 日，训练主机的用户 systemd 报告该单元为 `not-found`、`inactive`。已有 TensorBoard 服务可用 `systemctl --user start hopper-v50-tensorboard.service` 启动，在训练主机访问 `http://127.0.0.1:6011`。[TensorBoard 服务副本](training_config_v51/hopper-v50-tensorboard.service)

提交一轮全量 AB 训练，并在同一权重上运行固定开发集及 60 秒评估。自动服务运行时在轮次边界处理，服务停止时由当前终端处理。

```bash
/usr/bin/python3 -u -m benchmark37.train_ab_pairs
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
