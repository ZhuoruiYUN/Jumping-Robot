# CURRENT_CONTEXT_V60 — 52-D 原地跳/短跳部署交接（历史总览，更新至 2026-09-11）

> 当前交接入口是 `CURRENT_CONTEXT_V69.md`。本文件保留 2026-09-11 前的背景、标定和
> 失败路线；旧 V60 中 2026-09-06～09-07 的 far two-hop / 43-D raw-PWM 排查已是历史，
> 不可据此选择当前部署模型或重新训练。
>
> 当前目标：在**不使用 LQR、四电机推力直接由 recurrent PPO 输出**的前提下，让 hopper
> 在实机上连续原地/短距离跳跃；优先解决高度、姿态稳定、落点漂移和落地后的恢复。
> 用户自己启动正式训练；Codex 只允许修改、导出、评估或跑短 smoke，不能擅自启动长训练。

## 0. 2026-09-11 物理 1 m / actuator-parity v12 状态

- 用户已授权修复仿真/部署动作语义并把实机物理目标改为 1.000 m；正式训练仍由用户启动。
- 新训练配置：`SpatialTrackingPhysical1mEnvCfg`，保持 52-D/4 电机/LSTM/100 Hz，不改
  USD、质量、惯量、弹簧、推力公式或电机顺序。
- 高度映射：仿真/策略目标 `1.095 m`，减去仿真地面 `0.380` 与实机地面 `0.285` 的
  `0.095 m` 坐标差，部署物理目标恰为 `1.000 m`。
- v12 修正 `delay=0` 为当前最新命令，动作历史改为限幅和整数 PWM 量化后的实际执行动作。
- 仿真和部署共同执行 `height -> spread600 -> yaw diagonal -> spread600 -> yaw diagonal -> floor`
  动作包络。GPU/NumPy 随机 4096 组输入逐电机整数 PWM 完全一致。
- 部署入口只接受带匹配 metadata/SHA 的 v12 export；旧 v11 ONNX 会在电机启动前被拒绝。
- 训练入口：`./run_train_stationary_physical_1m_parity_v1.sh`。
- 已完成 16 env × 1 iteration 训练 smoke；随后 16 env × 1200 step 原地评估：
  `apex=0.9703 m`（策略坐标，尚未训练到 1.095）、XY `3.56 cm`、death `0`。
  该结果只证明新链路可运行，不能作为最终模型部署。

## 1. 不可改变的接口与标定中心

- Policy：52-D observation，ActorCriticRecurrent/LSTM，4 个直接电机动作，100 Hz。
  动作后处理固定为 `u = clip(0.5 * action + 0.5, 0, 1)`。
- 不引入 LQR、Semi-MDP planner 或额外控制层；不要改 USD、推力公式、动作电机映射或
  已标定的物理基线。
- 飞行数据标定：
  - `F = -0.2371 u² + 0.8130 u + 0.0113`
  - 名义电机时间常数 `Tm = 0.0674 s`
  - `Ixx = 1.231252e-3`, `Iyy = 1.286169e-3`,
    `Izz = 7.931903e-4 kg·m²`
  - `Ktau = 1.720762152765e-02`
- 这套 `Izz/Ktau` 修正必须保留。不要退回旧 yaw 物理参数。
- 实机链路日志未显示持续多步命令延迟或 RF 丢包造成的控制中断。训练中的动作延迟随机化
  已被判定为不利于本任务，当前硬件课程关闭它。

## 2. v12 初始化来源与已否决的历史部署模型

### 2.1 v12 正式训练初始化模型（40_v1）

```
outputs/tracking_recovery/precision_zero40_08_12_turn30_nominal_v1/model_119.pt
SHA-256: 2fd72dd57a43ef1e9f4d294c062900e7d87e9ae71d6fb5ad6edd388b2401c024
```

这是 `203431.csv` 实际部署并由 recurrent replay 零误差确认的 40_v1 模型，也是新 launcher
写死的初始化 checkpoint。它的旧 v11 精确原地跳标称评估（256 env × 3000 step）：

| 指标 | 数值 |
|---|---:|
| mean apex | 0.95486 m |
| height_ok_rate | 100%（旧目标 1.000±0.05 m） |
| XY error | 2.1749 cm |
| valid 5 cm hit | 97.3942% |
| pair valid 5 cm | 94.3617% |
| death rate | 0% |

注意这些是旧目标/旧动作历史语义下的参考值，不是 v12 验收结果。

### 2.2 历史 mixed 模型（已实机否决，不再部署）

它曾按用户要求导出并做过实机对照：

```
outputs/tracking_recovery/stationary_hardware_mixed_v1/model_79.pt
SHA-256: 133da4497be540fd8d38f941004b9a65547db20ce2816785914b076302deb48d
deployment/spatial_tracking_export_hardware_mixed_v1/
```

导出已完成：

- ONNX：`deployment/spatial_tracking_export_hardware_mixed_v1/quadhopper_spatial_tracking_policy.onnx`
- PyTorch–ONNX max action error：`1.192e-07`
- ONNX 本地 stateful 推理已成功。
- 部署脚本不再指向该模型；v12 guard 也会拒绝它。

mixed 的离线结果：

| 条件 | apex | height ok | XY error | valid 5 cm | pair valid 5 cm | death |
|---|---:|---:|---:|---:|---:|---:|
| 标称 | 0.92522 m | 99.9512% | 1.8283 cm | 99.0662% | 98.1323% | 0% |
| 无延迟全失配 | 0.88616 m | 81.2401% | 3.6224 cm | 63.6569% | 47.5502% | 0% |

结论：mixed 仅把全失配横向误差改善约 0.8 mm、5 cm 命中提升约 0.6 个百分点，
却比基础模型低约 2.1 mm apex、低约 0.54 个百分点高度合格率。它**不是**
解决“实机跳不高”的离线胜者，实机也未达到高度/稳定性要求。以下 export 只保留为历史：

```
deployment/spatial_tracking_export_stationary_apex_progress_v2/quadhopper_spatial_tracking_policy.onnx
```

## 3. 已证伪/不要重复的训练方向

1. **100% 环境宽失配训练被否决。**
   曾用质量 0.95–1.10、惯量 0.80–1.25、Tm 54–110 ms、推力 0.78–1.08，
   最终命中 EMA 36.4%、落点 EMA 7.28 cm、tilt reset 13.3%。过宽且隐藏的
   系统参数使 policy 只能学习折中动作。
2. **中等但所有环境都失配也无实质高度收益。**
   Tm 54–90 ms、推力 0.87–1.06、质量 0.97–1.07、惯量 0.88–1.15 的
   `stationary_hardware_robust_v2` 相对 source 在相同失配下顶点仅提高约 0.09 mm。
3. **75% 标称 + 25% 失配 mixed 课程**保住了标称表现，但没有改善高度，见上表。
   不要在没有新的实机物理证据时继续扩大/叠加这类随机化。
4. 不要靠提高 reward、增加姿态惩罚或临时高度门来猜测解决实机高度问题。
   当前瓶颈是低推力/接触能量损失与其对 policy 不可见，而不是 endpoint reward 缺失。

## 4. 当前任务代码与已验证行为

### 4.1 空间任务奖励（`Quadhopper_Planner_Random/spatial_tracking_env.py`）

- 对 exact zero-hop 在接近固定落点且下降时，施加轻量姿态、yaw rate、四电机 spread
  软惩罚；不限制真正横向跳或失误后的恢复。
- 静态 apex 用**物理测得的 cycle maximum height**势函数，而非时间轨迹：
  `stationary_apex_progress_scale=100`，`width=0.25 m`。
- 不修改四电机接口、标定力模型、原始任务的落地逻辑。
- 旧 0.05 m potential width 会在低于 0.95 m 时饱和，已修为 0.25 m。

### 4.2 硬件随机化（`train_tracking_recovery.py` / `run_tracking_recovery.py`）

`--hardware-robust` 的中等范围：

| 参数 | 失配范围 |
|---|---:|
| mass multiplier | 0.97–1.07 |
| inertia multiplier | 0.88–1.15 |
| motor Tm | 54–90 ms |
| per-motor thrust scale | 0.87–1.06 |
| thrust curve shape | 0.94–1.06 |
| observation noise | std 0.010 |

- `--hardware-randomization-probability` 控制每次 reset 落入上述失配的环境比例。
- mixed launcher 使用 `0.25`；全失配评估显式用 `1.0`。
- `--hardware-robust` 时 `randomize_action_delay=False`，因为真实日志没有持久多步延迟。
- 随机化对 policy 是隐藏变量；因此评估必须同时看标称和全失配，不能只看训练 reward。

### 4.3 正确的原地跳评估命令

`zero-hop-probability=1.0` 才产生精确 0 m 指令。因为 waypoint generator 要求正的
连续半径/转角接口参数，命令必须保留 `distance=0.001`、`turn=1.0`；它们不会成为
实际指令。不要再次使用 `distance=0` 或 `max-turn=0`。

```bash
python run_tracking_recovery.py eval \
  --checkpoint <MODEL.pt> \
  --root <NEW_EMPTY_OUTPUT_DIR> \
  --num-envs 256 --steps 6000 \
  --distance-min 0.001 --distance-max 0.001 \
  --zero-hop-probability 1.0 --max-turn-angle-deg 1.0 \
  --target-tolerance 0.05
```

全失配附加：

```bash
--hardware-robust \
--hardware-randomization-probability 1.0 \
--observation-noise-std 0.010
```

评估脚本现在每 10% 步数输出 `[SPATIAL-EVAL] progress=.../`，避免 6000-step 无输出
被误判为卡住。若 output root 已存在，即便为空也会拒绝；确认没有 `result.json` 后才删除
空目录，绝不删除训练目录。

## 5. 部署状态、已查实机现象

部署入口：

```
deployment/jump_stationary_spatial52.py
```

高度坐标必须保持：

```
POLICY_GROUND_ROOT_Z = 0.380
MEASURED_GROUND_ROOT_Z = 0.285
REST_LEG_LENGTH = MEASURED_GROUND_ROOT_Z
POLICY_Z_OFFSET = 0.095
POLICY_TARGET_Z = 1.095
PHYSICAL_TARGET_Z = 1.000
```

`MEASURED_GROUND_ROOT_Z=0.285` 是用户实测的弹簧无压缩根节点高度。启动时的
`339.4 mm` 已由用户在 Vicon 外部坐标系中平移处理，部署程序不得再次把 0.3394/0.341
写成腿长，否则会虚构约 5.6 cm 的弹簧压缩。

已分析实机 CSV：

- `quadhopper_deployed_data_20260911_170615.csv`：约 14 个 rebound/rise cycle；
  后段 apex 约 0.639 → 0.509 → 0.426 m，XY drift 约 0.287 → 0.395 → 0.446 m。
  没有 safety transition；是低能量/偏移后的弱反弹，不是 policy 自行切入 hover。
- `quadhopper_deployed_data_20260911_170711.csv`：约 17 cycle，最后
  `Deploy_Safety_Code=3`，即 touchdown tilt abort；末态 tilt 约 31.3°、
  XY error 约 0.140 m、Z 约 0.339 m。
- `quadhopper_deployed_data_20260911_203431.csv`：recurrent replay 已零误差确认实飞模型为
  `precision_zero40_v1`；平均 apex 约 `0.835 m`、平均 XY 约 `0.197 m`，最终同样以
  `Deploy_Safety_Code=3` 触地倾斜保护结束。基站电压在负载下约 `3.60–3.73 V` 波动，
  不能用静置/起始 `3.85 V` 代表跳跃时可用推力。
- 链路没有 packet/command timeout 或 IMU sequence gap 的证据。冲击时 FC flags 32/64
  出现，表示 IMU clipping/vibration；PWM differential 多次达到 600，说明在大 drift
  后 policy 输出已接近差动饱和。
- 记录过电流约 13.7–15.6 A，低电压/接触效率会恶化后段高度，但不能把失败直接归因于
  2.4 GHz 干扰。
- 用户明确：不要加入 position-error abort 或“overrun 保护”；释放初始位置可能不准。
  现有 touchdown tilt safety 保留。

部署检查顺序：

1. 确认 `TEACHER_POLICY_PATH` 指向期望的导出目录。
2. 检查 export 中 checkpoint SHA-256 与训练 checkpoint 一致。
3. 保持上面的高度坐标；0.285 是实机地面根高度，0.380 是 policy/Isaac 地面根高度。
4. 实机测试后分析 CSV 的每一跳 apex、XY drift、tilt、PWM spread、safety code，
   再决定物理标定，而不是先改 reward。

## 6. 下一步建议

1. `stationary_physical_1m_curriculum_v2` 已完成，但不能部署：64-env、1200-step、
   exact zero-hop 筛选的最佳 checkpoint 为 `model_40.pt`，平均 policy apex `0.9445 m`
   （约等于实机 `0.8495 m`），末轮 `model_239.pt` 仅 `0.9405 m`。高度课程没有解决
   低跳局部最优。
2. 下一轮使用 `./run_train_stationary_physical_1m_height_gate_v3.sh`：它从 v2 的
   `model_40.pt` 启动，固定 exact zero-hop，并将有效 apex 门槛随 `1.000 -> 1.095 m`
   课程提升至最终 `1.045 m`。低于门槛的 touchdown 不再获得命中/精度奖励，同时有低高度
   惩罚；这避免了低约 15 cm 但落点稳定的策略拿到主要回报。
3. v3 已完成且同样不能部署：64-env、1200-step、exact zero-hop、最终 1.095 m target
   筛选中最佳高度为 `model_140.pt` 的 `0.9329 m`（约实机 `0.8379 m`），低于 v2
   `model_40.pt` 的 `0.9445 m`。不要将 v3 checkpoint 导出或上实机。
4. 下一次训练前，必须先在 source checkpoint 上逐项记录每跳的实际 apex、接地/离地状态、
   collective PWM 和 gate 事件，确认高度门控的 credit assignment；不得继续只增大 reward
   系数后再启动长训练。
3. 只有 exporter parity、metadata/SHA 和部署端 v12 guard 全部通过后，才允许短实机测试。
4. 实机仍跳不高时，优先从 CSV/台架数据辨识有效集体推力和弹簧/接触能量损失；将辨识出的
   单一物理中心用于训练。不要继续盲目拓宽隐藏随机化。
5. 若确有可在线获取且稳定的物理量（例如已校准的电池电压），才考虑把它作为**明确的**
   observation 扩展；这会改变 52-D policy/ONNX 契约，必须先获得用户确认并从兼容性方案开始。
6. 正式训练仍由用户启动。Codex 只可跑短 smoke、导出、语法/ONNX parity 和评估。

## 7. 工作区状态与注意事项

- 当前未提交的主要改动：
  - `Quadhopper_Planner_Random/spatial_tracking_env.py`
  - `Quadhopper_Stable/actuator_contract.py`
  - `Quadhopper_Stable/quadhopper_env.py`
  - `train_tracking_recovery.py`
  - `run_tracking_recovery.py`
  - `run_train_stationary_physical_1m_parity_v1.sh`
  - `tests/test_actuator_contract.py`
  - `deployment/export_spatial_tracking_policy.py`
  - `deployment/README_SPATIAL_TRACKING_EXPORT.md`
  - `deployment/jump_stationary_spatial52.py`
  - `run_train_stationary_hardware_robust.sh` 及历史 launcher
- `deployment/` 下有多个 export/package/archive 未跟踪文件。不要用通配符提交日志、
  checkpoints、TensorBoard、CSV、包或 archive。
- 所有模型 export 都必须运行 exporter 自带的 ONNX parity 验证；不允许只复制 ONNX。
- 任务若涉及物理、asset、motor mapping 或 force equation，先停下并要求用户明确授权；
  task-level reward、训练随机化、评估和 deployment ONNX 路径可在现有授权内修改。
