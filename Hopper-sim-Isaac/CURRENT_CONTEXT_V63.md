# CURRENT_CONTEXT_V63 — short→long transition curriculum（2026-09-07）

## 为什么不继续扩距离

V62 long curriculum 的 100 迭代结果表明孤立首跳已学会：long hit EMA 约 0.787，
long touchdown error EMA 约 0.076m，课程已到 long max 0.435m、turn 67°。

但最后一个 batch 中 `(short touchdowns + long touchdowns) / resets = 1.002`，
且 `live_max_consecutive_hits=1`。随机首相位让约一半 episode 的第一跳就是 long，
所以高 long hit 主要证明 reset 后的孤立 long；它没有证明 short 落地后的第二跳 long。
每个 episode 基本只完成一次 touchdown，随后翻倒。因此暂停几何扩宽，先训练状态衔接。

注意：旧诊断名 `reset_on_long_phase_count` 实际在命令 reset 之前采集，含义是
“终止时处于 long phase”，不是“新 episode 从 long 开始”。保留旧 tag 兼容历史，
新增语义准确的 `reset_while_long_phase_count`。

## V63 课程定义

入口：`run_train_two_hop_transition.sh`，默认从 V62 `model_99.pt` 转移。

- 所有 episode 从 short phase 开始，符合部署 short→long 顺序；
- short 固定 0.00–0.30m，long 固定 0.30–0.435m，turn 固定 67°；
- 不在本阶段增加距离或转向难度；
- 保留 long-only 飞行姿态/角速度 shaping；
- short 下降和 stance 对下一跳方向提供 anticipatory attitude 信号；
- short touchdown 奖励约束下一跳方向速度投影、横向速度和触地姿态；
- prepared landing 奖励提升，完成 short+long 后给 pair hit 奖励；
- termination penalty 从 -12 提至 -40；
- PPO lr 5e-5，action std 0.06，默认 150 迭代；新 experiment token `_transition`。

新增 reset-batch 诊断：pair attempts/hits、conditional second attempts/hits。
所有物理参数、43-D observation 顺序、4-D action 和 recurrent 网络不变。

## 先跑 75 迭代验收

目标不是单跳 hit，而是：

1. `(short_touchdowns + long_touchdowns) / reset_batch_count` 从约 1.00 上升；
2. conditional second attempts 和 hits 持续非零；
3. pair hits/reset 上升；
4. max consecutive hits 达到并稳定超过 1；
5. short/long hit EMA 不发生持续性崩落。

若 75 迭代仍只有约一次 touchdown/reset，暂停检查 touchdown 后的电机历史、RNN hidden
state 与 spring contact/liftoff 状态，不再通过增加奖励权重硬推。
