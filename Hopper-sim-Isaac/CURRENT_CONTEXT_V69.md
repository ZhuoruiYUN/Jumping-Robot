# CURRENT CONTEXT V69 — 40_v1 连续原地 1 m 评估与下一步交接（2026-09-14）

> **当前唯一交接入口。** V60 仅保留早期背景；不要把其中过时的 `1.095 m` 高度坐标
> 映射、旧 v12 guard 或旧训练路线带回当前 40_v1 部署/评估链路。

## 1. 当前目标与不可变事实

目标是让 52-D LSTM、四电机直接输出的 40_v1 在实机上持续**原地**跳到物理/Vicon
`1.000 m`，同时保持姿态和落点。不要引入 LQR、不要改 USD、质量、惯量、弹簧、推力曲线、
电机顺序或 100 Hz 控制接口。

- 实机弹簧无压缩 root 高度：`MEASURED_GROUND_ROOT_Z = 0.285 m`。
- Vicon 启动高度 `339.4 mm` 已由用户在 Vicon 坐标系外部平移处理；部署程序不得再把
  `0.3394/0.341` 写成腿长或再叠加高度补偿。
- 当前 40_v1 的 policy/Isaac 指令目标与物理目标均为**绝对 root Z = `1.000 m`**。
  不得把物理 1 m 改成 1.095 m。
- 代码中可见的 `landing_root_height=0.38` 是旧 spatial planner 的仿真落地端点/参考，
  不是实机地面高度，也不是部署坐标转换。
- 已被实机证明可连续运行的 checkpoint：

```text
outputs/tracking_recovery/precision_zero40_08_12_turn30_nominal_v1/model_119.pt
```

这是本轮所有 40_v1 评估与默认启动器的唯一模型来源。历史 mixed 模型和各类 height nudge /
collective residual 训练都不是当前部署候选。

## 2. 已完成的实机映射修正

早先实机跳不到 1 m 的主要可修复错误是把 legacy `0.38 m` 误当成 deployment 的
`landing_root_height`。去掉该错误映射后，实机日志
`/home/terry/Desktop/workspace/quadhopper_deployed_data_20260914_143924.csv`
已经能长时间连续跳；人为终止而非坠机。它仍有后段高度、姿态和落点波动，但不再把高度坐标
搞错。

**部署保持单一固定路径**：不要恢复多模型选择器，也不要要求其他电脑传 `policy_metadata.json`
才能加载一个原本可直接传递的导出包。模型 export 的 metadata 仅在 exporter 内用于校验，
不应成为远端部署机不可恢复的额外依赖。

## 3. 标准 Isaac 连续评估（已实现）

运行：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
./run_eval_40v1_isaac.sh
```

`run_eval_40v1_isaac.sh` 固定使用 `model_119.pt`，1 env、30000 step/300 s、
`zero-hop-probability=1.0`、距离仅用 `0.001` 以满足 waypoint API、目标 `1.0`、
`episode-length-s=301`。因此 300 s 内**没有周期性 episode reset**。

它会导出 `single_env.csv`，再调用保留旧文件名但已改为 Isaac CSV 分析器的：

```text
play_spatial_tracking_mujoco1.py <single_env.csv> --target-height 1.0
```

分析器输出 `analysis/isaac_hops.csv`、`isaac_summary.json`、完整 hop 图和五面板
`isaac_detail_window.png`（高度、姿态、四电机、XY error、功率）。它不是 MuJoCo 代码，
不依赖旧 MuJoCo asset/STL 路径。

### 为什么旧图会出现 1.3–1.4 m 的大尖峰

旧评估的 15 s episode reset 在 reset 时把机器人随机放在 aerial drop 高度：
`landing_root_height 0.38 + [0.65, 1.05] = 1.03–1.43 m`。图中的 1.3–1.4 m 是
**reset 初始状态自由下落**，不是模型起跳到 1.4 m，也不是新 agent/新权重。每次 reset 同时
清空该环境的 LSTM 状态。现在 301 s episode 已排除此伪影；允许 t=0 一次训练式初始 drop。

## 4. 当前最可信的连续仿真结果

输出目录：

```text
outputs/tracking_recovery/isaac_40v1_stationary_1m_20260914_173711/
```

它包含 30000 rows、330 个完整真实 hops，300 s 中没有死亡或周期 reset：

| 指标 | 数值 |
|---|---:|
| apex mean / std | `0.96647 / 0.03649 m` |
| apex median / P10 / P90 | `0.96404 / 0.92707 / 1.00599 m` |
| apex min / max | `0.89325 / 1.13614 m` |
| 严格 `apex >= 1.000 m` | **11.82%**（39/330） |
| `apex >= 0.950 m` | 67.27% |
| 落点误差 mean / median / P90 | `3.30 / 2.07 / 5.62 cm` |
| 5 cm 落点成功率 | 87.88% |
| peak tilt mean / P90 / max | `4.18 / 5.90 / 15.27 deg` |
| apex drift | `+3.95e-5 m/hop`，无后段能量衰减 |

`result.json` 里旧的 `height_ok_rate=1.0` 使用了 `±0.15 m` 宽容差，**不可作为 1 m
达标率**；汇报高度只能使用上表的严格指标或明确标记容差。

结论：连续仿真无长期发散，但 40_v1 本身也未达到严格 1 m。它存在约 `-3.35 cm` 的系统性
低跳和 `3.65 cm` 跳间散布；这不是电压问题、不是坐标转换问题，也不是周期 reset 的假象。

## 5. 当前图中 XY 尖峰和超高的确切定位

在用户展示的 `t=270–300 s` 图中，target 固定不动。红色 XY error 的尖峰来自真实 touchdown
偏移，而不是 target 移动。

| hop | 起跳时刻 | apex | 落点误差 | peak tilt | 解释 |
|---:|---:|---:|---:|---:|---|
| 299 | 270.27 s | 0.948 m | 1.6 cm | 3.5° | 正常跳 |
| 300 | 271.17 s | 1.069 m | 15.7 cm | 10.7° | 第一次大偏移，造成 XY error 上升 |
| 302 | 273.02 s | 1.089 m | 13.6 cm | 15.3° | 姿态恢复加剧，四电机 spread `0.562` |
| 318 | 287.44 s | 0.995 m | 21.9 cm | 12.8° | 第二次同类落点异常，随后恢复 |

它们说明 direct-motor policy 在少数接触/姿态扰动后进入强恢复分支：四电机差动既提供姿态和
水平修正，也不可避免改变总推力，故 XY、姿态和 apex 同时异常。数据足以定位异常跳次，
但**尚不能从现有 CSV 唯一证明**第 300 跳的第一触发是接触边界、动作延迟还是 LSTM 历史积累；
不要把推测写成物理定论。

## 6. 已否决的路线

- 在 40_v1 上盲目加入 height nudge/height gate、增大 apex reward 或训练更久：多次候选都
  未改善严格高度，部分还破坏命中与稳定性；例如 60-iteration nudge 的 native mean
  `0.9624 m`、严格 >=1 m `12.66%`，低于 baseline 的整体高度表现，不能部署。
- `collective residual` 迁移：240 iteration 后 hit≈0、tilt reset 很高、落点约 29 cm，失败。
- 全局增大 deployment `action_scale`：会放大差动、饱和和姿态风险，不是保持高度的正确旋钮。
- 将低电压简单等价为 low-scale thrust 或假定 energy manifold 自动补偿：现有 policy 没有看到
  足以实现闭环能量补偿的可靠在线物理状态；用户判断当前主因是稳定性，而非电压。

## 7. 下一步（先诊断，再训练）

下一项应为**只读的逐跳起飞前诊断**，在固定 seed 的连续 40_v1 eval 中把正常 hop 299 与异常
hop 300（以及 301/302）逐 sample 对齐输出：

1. touchdown 和 liftoff 前后的 root XY/Z、线速度、角速度、roll/pitch/yaw；
2. spring position/velocity、contact 标志和接触时长；
3. policy raw action、实际执行 U1..U4、collective、motor spread、饱和比例；
4. LSTM hidden/cell state 的范数和变化；
5. 异常跳与前一正常跳的逐字段差分。

目的不是再跑训练，而是判定异常首先出现在接触状态、执行动作还是 recurrent memory。确证后再
设计保留四电机接口的解决方案：将 collective 高度维持与 differential 姿态恢复的训练目标/动作
约束解耦，并以连续 300 s 的严格高度、P90 落点和峰值 tilt 作为验收，不能只看训练 reward。

## 8. 工作区和 Git 交接

- 保留锚点见 `ANCHORS.md`；历史 logs、失败 outputs、旧 calibration 产物和无用 launcher 已按
  用户要求清理。不要恢复它们，也不要把 CSV、TensorBoard、checkpoint 或临时分析图提交进 Git。
- 这次应提交的源代码/文档包括：Isaac CSV 分析器、连续评估 launcher、训练/eval 参数修复、
  deployment/export 改动、actuator contract tests、诊断工具、`ANCHORS.md` 及本 V69。
- 提交前确认只提交源代码、脚本、文档和小型 export metadata；大模型、运行输出不入库。
