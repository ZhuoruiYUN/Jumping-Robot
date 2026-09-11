# CURRENT_CONTEXT_V64 — transition V1 failed, V2 bridge（2026-09-07）

## V1 结论

V1 从 V62 model_99 开始，所有 episode 由 short 起步，第二跳直接使用
0.30–0.435m / 67°。75 迭代中：

- touchdown/reset 分段维持 1.006–1.017；
- reset while long phase 为 97.7–99.1%；
- pair hit 和 conditional second hit 全程为 0；
- short hit EMA 从约 0.886 上升到约 0.932；
- long hit EMA 全程为 0；
- prepared landing EMA 从约 0.403 上升到约 0.461，但未转化成第二跳落地。

所以 setup shaping 改善了被测准备量，却无法跨越从“reset 后孤立 long”到
“真实 short touchdown 后 long”的初态差异。V1 checkpoint 不再续训。

## V2

`run_train_two_hop_transition_v2.sh` 仍从 V62 成功的 model_99 分支。short 保持
0–0.30m；第二阶段命令在 400 迭代内从 0.05–0.15m / 15° 缓慢扩到
0.30–0.435m / 67°。学习率由 5e-5 降为 3e-5，其余 transition shaping 保留。

首个 75 迭代结束时预计课程约为 0.097–0.203m / 24.75°。验收首先要求
touchdown/reset > 1.10 且 pair/conditional second hit 持续非零。若近原地第二跳仍
不能落地，停止 reward/distance 调参，转查 touchdown 后 RNN hidden state、action/motor
history、spring contact 与 cycle/liftoff 状态转换。

机器人 Izz、Ktau、电机动力学、观测和动作接口均未改变。
