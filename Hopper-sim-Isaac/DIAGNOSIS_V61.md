# V61：修正物理后长跳失败复核（2026-09-07）

## 确定的代码错误与本次修复

`PlannerCircularEnv._get_dones()` 调用 stable 父类取得 `died`，父类仍写死
`qx²+qy² > 0.5` 判死。子类虽然计算可配置 `tilt_death`，却只用于计数，实际
`task_died = died`。因此 V60 所称“已修”的放宽倾斜死亡并未生效。

本次将子类物理死亡掩码改为 `low_height_death | tilt_death`，保留父类 timeout、
距离/角速度终止、初始下落豁免、任务完成及 miss 终止。默认 0.5 行为不变。
没有更改机器人模型、惯量、反扭矩系数、动作、奖励或 PPO 参数。

CPU 上提取实际 `_get_dones` 方法执行，检查默认/放宽阈值、低高度、初始下落豁免、
角速度、距离、完成、miss 和 timeout，全部通过。未启动 Isaac/GPU 训练，尚未验证学习效果。

## 最新 TensorBoard 证据

同一实验目录：
`logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_lax_120_lenapex_fixed_100/`

以下为最后 50 个迭代标量的算术均值，不能当作按落地样本加权的成功率：

| run | 最后 step | short hit | long hit | long hit EMA | 平均 episode 步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-09-06_23-01-59 | 799 | 0.90691 | 0.00023 | 0.00904 | 240.479 |
| 2026-09-07_09-52-08 | 199 | 0.92398 | 0 | 0 | 238.2934 |

前一轮最后一个 long EMA=0.04991，只能说明有过成功长跳，不能认定持续上升。
后一轮目录没有 `_tds_095`，未发现带该 token 的训练记录；不能把它当作放宽阈值的对照实验。

## V60 需要纠正的统计解释

- 非 EMA 命中率是 `_reset_idx(env_ids)` 对本批重置环境的 episode 计数求比值，
  随后这些计数清零；不是从训练开始累计。
- 无长跳落地时，分母 clamp 到 1，long hit 和 long touchdown error 都输出 0。
  因此接近零的 long error 不能解释为长跳落点已很准。
- 前一轮末尾 long error=0.01262m，但 long error EMA=0.23144m。
- TensorBoard 的批次比率均值不能直接代入 overall=short*(1-f)+long*f 反推真实落地占比。
  V60 的“约 1% / 99% 提前死亡”没有被本次独立确认。
- EMA 只在对应跳型有落地时更新；没有长跳落地时会停留在旧值。
- 90° 是机体竖轴的总倾角阈值；0.95 对应约 154.2°，并非通常意义上的安全着陆姿态。

## 物理解释的边界

当前任务继承 `Quadhopper_Stable`，其 asset cfg 指向 `OriginJumpHopperAsset.usda`，
覆盖层声明 Izz=7.931903e-4；执行器代码 Ktau=1.72e-2。静态路径一致，
本次未在 PhysX 中读取实际惯量做运行时确认。

按标称数字，Ktau/Izz 从约 23.42 变为 21.68，降低约 7.4%，不是指令 yaw
角加速度增加 2.9 倍。相同外部 yaw 力矩的 1/Izz 灵敏度约增至 2.91 倍。
惯量变化也会影响刚体转动耦合；不能仅从这些数字推出实际翻倒一定由 yaw 引发。
保持修正物理，用逐跳状态/受控对照区分 yaw、roll/pitch、触地和动作饱和。

## 下一步（用户自行运行）

先做修复后 200 迭代的探针，沿用已有 model_799 和 Stage1b 配置，不同时扩宽随机化：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_train_stage1b_lowapex.sh 1024 '' --iterations 200
```

脚本 `${2:-...}` 会在第二参数为空时采用默认 checkpoint；末尾 `--iterations 200`
覆盖脚本内的 1000。此次真正生效的 154° 阈值只用来验证训练终止假设，不能把翻倒后
的落地点命中视作部署成功。观察 long EMA 是否反复更新、误差是否下降、episode 是否
延长，最终仍需在正常终止阈值和冻结权重下验证稳定两跳。不要承诺落地占比自动恢复
40%，也不要只凭 overall hit 挑选模型。

## 200 迭代修复后探针结果（10:37:27 run）

实际带 `_tds_095_av_040` 的运行已完成。每 50 迭代的 long hit EMA 均值依次为
0.04539 / 0.04359 / 0.05240 / 0.05360；非 EMA long hit 非零的迭代数依次为
16 / 20 / 22 / 29。最后一个 long hit=0 不代表后期完全没有成功；但没有稳定突破。
最后 50 迭代 long touchdown error EMA 均值 0.23734m，episode 长度均值
245.7496 步，short hit 均值 0.88422。仅放宽阈值未解决问题。

补充 `Diagnostics/*` 重置前快照，记录 reset batch 大小、长短跳落地次数、长跳
命中次数、没有长跳落地的 episode 数、倾角超过 90°/配置阈值、低高度、角速度
超过阈值的数量，以及绝对 yaw rate 和 XY 角速度范数之和。它们是批次计数/总和，
需在同一窗口聚合后相除，不能把 logger 显示的批次平均计数当成整轮总次数。
重置状态条件可重叠，不是互斥的终止原因，也不能证明 yaw 是首先失稳的轴。
代码只在 reset 路径增加诊断，不改奖励/物理/策略接口，CPU 合成状态检查通过；
Isaac 运行时尚待用户采样验证。

下一步从该 run 的 model_199 续跑 30 迭代收集 Diagnostics，不同时改超参。
