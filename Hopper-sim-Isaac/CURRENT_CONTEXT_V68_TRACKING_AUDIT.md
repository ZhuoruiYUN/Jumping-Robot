# 2026-09-07 追点能力审计（只诊断，未更改训练/物理配置）

## 已复现的事实

1. residual 训练配置为 `force_full_planner=False, planner_reference_blend=0`。这不是禁用 planner。
   `DirectCollocationHopPlanner.replan()` 将第一跳 apex XY 约束为 `(start_xy + Pt)/2`。
   `PlannerCircularEnv._update_reference()` 在支撑期使用 Pt，在整个飞行期使用 apex，并将期望速度置零。
   用当前 planner 和通过 AST 提取的当前 `_update_reference` 做 CPU 调用：
   起点 `(0,0,0.38)`，Pt=`(0.20,0)`，Pt1=`(0.40,0)`，H=`1.0`。
   支撑期参考 `(0.20,0,1.0)`；飞行相位 0、0.5、1 均为 `(0.10000021,0,0.99999851)`；速度均为零。
   stable 前 37 维中的目标、stable XY 奖励、planner XY 奖励追此参考，而落地事件奖励追 Pt。
   在非零位移时存在明确的参考语义冲突；零位移测试无法暴露此问题。

2. 当前 stable 动作历史长度为 5，每步先 append 当前动作，随后执行 `history[-delay_idx]`。
   从实际源码提取此下标表达式，以帧编号执行：配置 0/1/2/3/4 对应动作年龄 4/0/1/2/3 步。
   因此所谓 nominal `play_action_delay=0` 实际为 40 ms 的纯延迟，另加 Tm 一阶滞后。
   旧/新 baseline 以及 planner 共用该实现。不能由此单独断言它解释全部训练失败。
   修正应明确区分 legacy 与正确 delay 定义，并重新验证 checkpoint；本次未静默修改动力学。

3. residual 使用 `teacher_raw + motor_residual`，之后 stable 环境裁剪到 [-1,1]。
   CPU 示例 teacher=1.20、residual=-0.04，裁剪前后实际 PWM 均为 1，修正完全无效。
   最新训练约 46.03% 动作分量越界；residual 均值 0.0023，最大 0.0629。
   这支持部分修正被吞掉的机制，但日志未按相位统计有效修正，不能断言所有修正无效。
   `residual_induced_clip_fraction` 仅测额外引起的越界，并不测修正是否实际生效。

4. 精修 baseline 的 0.08 m XY 奖励宽度/goal 范围是 train_stable 中的配置覆盖，不是权重自带。
   residual 新建 PlannerRandomTwoHopEnvCfg 后仍继承 0.5 m / 0.35 m；另外叠加 planner 和落地奖励。
   不能将其称为完全保留了精修任务目标。

## 不应继续当作定论的历史说法

- 新通道权重更新并不能证明冻结 LSTM 无法表达方向控制；之前提到 GRU 也不准确，迁移实现为 LSTM。
- 平均落点误差 7.8/9.7 cm 不等于随机噪声标准差或无法突破的噪声底。
- 不同 Tm、距离范围、容差、reset、retry/continuous 语义下的 hit rate 不可直接归因比较。
- 训练 reset batch 的 mean_cycle_apex 可能来自未完成周期，不等同于已完成跳跃 apex。

## 对 planner 的判断及下一步

当前 planner 是运动学参考生成器，不含电机、姿态和弹簧接触可行性优化。
两落点任务不必须使用它；现有 blend=0 模式保留了中点参考，却丢掉了向终点运动的时间参考。
建议先做同 teacher / 同 seed / 同 Tm / 同 action-delay / 同距离 / 同容差 / 同采样语义的
零 residual A/B：原中点参考 vs 整跳固定 `(Pt.x, Pt.y, H_apex)` 参考。
这里 H 是绝对 apex 指令，不是要求 root 在地面 Z=0；保留真实 touchdown 事件、下一点 Pt1 及 pair-restart。
此测试仅比较参考，不同时修 delay 或调整奖励，否则不能定位贡献。

若切换为直接落点 PPO：策略输入当前/下一落点、apex 和本体状态；以实际 touchdown 误差为目标，
采用适度的稠密位置引导，保留高度/稳定性/动作代价；去掉中点追踪和全程零速度追踪奖励。
以精修 baseline 初始化端到端四电机策略是可选路线，不应再仅根据猜测交替冻结/解冻或提高距离。
直接落点预期更适合当前任务，但尚未运行匹配的 GPU 对照，不能保证训练成功率。
