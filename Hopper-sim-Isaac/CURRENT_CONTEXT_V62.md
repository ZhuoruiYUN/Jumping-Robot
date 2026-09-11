# CURRENT_CONTEXT_V62 — long-hop recovery curriculum（2026-09-07）

## 诊断依据

修复可配置倾斜终止后，从 `_tds_095_av_040` model_199 续跑 30 迭代。
排除启动时的首个异常 reset batch 后，同一窗口聚合结果为：

- reset 126.84，short touchdown 126.40，long touchdown 3.03；
- long touchdown/reset = 2.39%，long hit/reset = 0.10%；
- reset 时 tilt 超过配置阈值（0.95，约 154°）占 100%；
- reset 时平均 XY 角速度范数 9.24 rad/s，绝对 yaw rate 2.30 rad/s；
- low-height reset 和超过 40 rad/s reset 均为 0。

因此主要失败链是 short 落地后发起 long、roll/pitch 翻转、倾斜终止。裸训练路径中
long 飞行稳定性附加项及 pair reward 均为零，short 成功奖励足以支撑高 episode reward，
形成“short 成功、long 翻倒”的局部最优。

## 新课程

入口：`run_train_long_hop_curriculum.sh`。

- 兼容原 43-D observation 和四电机 action，默认从最新诊断 run 的 model_29 转移；
- `randomize_route_phase=True`，多环境 reset 时约一半直接从 long phase 开始；
- short 距离固定 0.00–0.30m；
- long 距离在 400 迭代内从 0.30–0.38m 线性扩到 0.30–0.60m；
- 转向范围同期从 30° 扩到 180°；
- 新增 long-only 的 airborne attitude、XY angular velocity、yaw、yaw rate、motor
  spread、tilt barrier 和 touchdown attitude shaping；
- 倾斜终止恢复到 0.75（约 120°），角速度终止 30 rad/s；
- 学习率 1e-4，默认 500 迭代，新实验 token 为 `_longcur_038_400`，optimizer 重置。

机器人物理、Izz=7.931903e-4、Ktau=1.72e-2、电机响应、动作和观测接口未改。

## 验收

先观察前 50–100 迭代，不以单个 batch 的 long hit 判断：

1. `Diagnostics/reset_on_long_phase_count / reset_batch_count` 应接近 0.5；
2. `long_touchdowns_batch_count / reset_on_long_phase_count` 应明显超过旧值 2.39%；
3. `reset_tilt_over_limit_count / reset_batch_count` 应持续下降；
4. long touchdown error EMA 应从约 0.28m 下降；
5. short hit 不应持续跌破 0.80。

若 100 迭代后 long touchdown 仍低于 10%，暂停，不进入 0.60m 后半段训练。
若课程训练成功，仍需恢复正常 0.5 倾斜阈值做串联 short/long 验证。
