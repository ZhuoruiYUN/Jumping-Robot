# CURRENT_CONTEXT_V59

Date: 2026-09-03 update

## 0. 2026-09-03 当前进度快照

### 当前推荐模型：far 0.0-0.3 / 0.3-0.6 m 候选

- 源 checkpoint:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100__ptakeoff_030_080_setup_012_000lcg_000_attw_080_angw_080_yaww_120_heightw_120_blend_100_air_050_fixed_100/2026-09-03_11-19-47/model_79.pt`
- 已导出 ONNX:
  `deployment/two_hop_release_export_far_030_060_m79/quadhopper_v22_policy.onnx`
- 导出 metadata:
  `deployment/two_hop_release_export_far_030_060_m79/policy_metadata.json`
- 导出时同时复制的 checkpoint:
  `deployment/two_hop_release_export_far_030_060_m79/quadhopper_v22_model_100.pt`
  （文件名是旧导出脚本固定命名；metadata 中 `checkpoint_iteration=79` 和
  `source_checkpoint` 是准确的）
- ONNX parity max abs error: `1.192e-07`

### 推荐模型 direct eval 记录

距离分布固定：

- first hop: `0.000-0.300 m`
- second hop: `0.300-0.600 m`
- fixed high apex: `1.0 m`
- `--relative_next_hop`
- `--start_from_random_drop`
- `--planner_reference_blend 1.0`

nominal direct eval:

- `target_hit_rate = 0.7272`
- `short_target_hit_rate = 0.7210`
- `long_target_hit_rate = 0.7344`
- `episode_touchdown_error_m = 0.0779`
- `touchdown_attitude_error_rad = 0.0683`
- `last_apex_height_m = 0.9795`
- `death_rate = 0`

small thrust randomization eval
(`thrust 0.97-1.03`, `shape 0.98-1.02`):

- `target_hit_rate = 0.7442`
- `short_target_hit_rate = 0.7307`
- `long_target_hit_rate = 0.7595`
- `episode_touchdown_error_m = 0.0780`
- `touchdown_attitude_error_rad = 0.0680`
- `last_apex_height_m = 0.9796`
- `death_rate = 0`

### 09-03 试验结论：不要继续 RL polish 覆盖 11:19 模型

已经试过并放弃的分支：

1. touchdown setup velocity:
   第一跳 touchdown 奖励朝下一跳方向的 residual velocity。
   结果 `touchdown_next_velocity_projection_mps` 仍接近 0 或变负，且 hit/姿态退化。
   结论：落地瞬间要求“带势能/动量准备下一跳”不合理，接触时间太短且会干扰落准。
2. long-hop takeoff velocity shaping:
   修复后 reward 生效，但容易追起跳速度，`long_hit` 掉。
   结论：速度奖励过粗，会牺牲落点。
3. airborne setup attitude:
   第一跳空中后半段朝 `P_(t+1)` 轻微预倾，确实生效，但 direct eval
   `target_hit_rate` 和 death 比 11:19 差。
   结论：空中执行方向物理上合理，但当前 reward 写法仍会干扰第一跳落点和姿态安全。
4. damping polish:
   限制 roll/pitch rate、action spread、action rate、轻 yaw rate。
   40 iter 能保持 death 0，但 hit/apex 下降；300 iter 明显训过头。
   结论：继续 RL 阻尼化会把策略推离原有稳定解。
5. free airborne roll/pitch:
   去掉空中 roll/pitch 角度惩罚，保留 touchdown attitude 和 rate damping。
   结果 short 保住但 long 下降。
   结论：不能全拿掉 roll/pitch angle 约束，只能放宽，且当前最好不要再训。

目前判断：`11:19/model_79` 是最均衡远距离模型。可视化中偶尔“发散一下再拉回”的问题，
更适合在部署侧做 action slew-rate limit / 短时异常动作 clip，而不是继续改 policy reward。

### 09-03 代码新增项

- `Quadhopper_Planner_Circular/planner_circular_env.py`
  - thrust randomization support 已存在；
  - setup touchdown reward 项；
  - airborne setup reward 门控；
  - `takeoff_long_hop_only`；
  - `free_airborne_roll_pitch` 相关支持。
- `train_planner_circular.py`
  - 新增 CLI:
    `--setup_hop_finetune`,
    `--airborne_setup_finetune`,
    `--takeoff_long_hop_only`,
    `--attitude_damping_polish`,
    `--free_airborne_roll_pitch`,
    `--thrust_rand_min/max`,
    `--thrust_shape_rand_min/max`,
    `--fixed_action_std`。
- 新增训练脚本：
  - `run_train_setup_hop_far_nominal.sh`
  - `run_train_far_damping_polish.sh`
  - `run_train_far_rate_only_polish.sh`

注意：这些试验脚本用于研究，不代表当前推荐部署模型。当前推荐部署仍是
`2026-09-03_11-19-47/model_79.pt` 导出的 ONNX。

### 部署注意

- 本次只导出，不打包。
- 部署至少需要拷贝：
  - `deployment/two_hop_release_export_far_030_060_m79/quadhopper_v22_policy.onnx`
  - 对应要运行的部署脚本，例如 `deployment/jump_stationary_semimdp.py`
    或 `deployment/jump_two_hop_release.py`
  - `deployment/requirements-runtime.txt`（环境已装好可不拷贝）
- `policy_metadata.json` 建议一起拷贝用于核对，但运行时不需要。
- `quadhopper_v22_model_100.pt` 运行时不需要，只用于追溯。
- 当前部署脚本仍需确认 target/reference 契约：
  `jump_two_hop_release.py` 目前不是训练时的随机 two-hop full-planner 轨迹生成器；
  若要正式飞 `0.0-0.3 / 0.3-0.6` 两跳，需先确认/补齐实机脚本里的 target queue 与
  policy 43-D 观测契约。

---

Date: 2026-09-02

实机部署调试交接文档。本文记录 09-01 至 09-02 的 7 次实飞、
已确认的根因链、当前代码状态，以及给后续 AI/开发者的执行计划。

## 1. 一句话结论

实机上空中姿态/yaw 回路**边际失稳**：策略空中反馈增益按 sim 动力学标定，
实机等效执行增益 ≈1.5–2×，姿态振荡逐跳放大直至翻车。
**部署侧标量旋钮已全部试尽并证伪，下一步必须回到模型层：sim 加推力随机化 +
从 air050 血统做 sim2real 微调。**

## 2. 实飞时间线（7 次，全部日志在用户 Downloads）

| # | 时间 | 脚本/模型 | 旋钮 | 结果 |
|---|---|---|---|---|
| 1 | 09-01 20:13 | stationary / v36 teacher + 69-D | 无保护 | 3 跳漂移发散，60° 硬断油 |
| 2 | 09-01 20:28 | 同上 | 保护已加 | 4 跳，误差 0.3→0.5→1.2 m，60° 断油 |
| 3 | 09-02 10:41 | 同上 + Vicon 速度(A) | A 生效 | 第 1 跳落点 0.06 m ✓；69-D 修正器 7.9° 前倾指令实机放大成 28° 离地侧飞，翻车 |
| 4 | 09-02 11:19 | v36 teacher-only | 摘 69-D | 1 跳：apex 1.18、空中 29°、水平 1.2 m/s，落地 18° 触发保护后侧翻滑行 |
| 5 | 09-02 11:39 | air050(v60 m79) teacher-only | 换模型 | 4 跳：落点 0.04–0.18 m、落地姿态 3–9° ✓；空中倾斜 20→15→24→50°，第 5 跳落地 41° 断油，原地停稳 |
| 6 | 09-02 11:51 | 同上 | action_scale 0.75 | 6 跳：倾斜 6→19→19→27→27→61°，翻车硬断油 |
| 7 | 09-02 12:23 | 同上 | +差动缩放 0.6 | 更差：2 跳 + yaw 甩尾 ±35°/s 加剧，落地 35° 断油，地面翻滚 10 s |

## 3. 已确认的根因链

1. **IMU 削波污染 EKF 速度（已修）**：每次落地冲击 IMU 加速度计削波
   （FC_Status_Flags=32/64），EKF 错过弹跳冲量，整个上升段 Vz 反号
   （Z 在升、Vz −4~−5 m/s）；`PPO_USE_EKF_VELOCITY=True` 时策略吃到坏速度。
   修复：`PPO_USE_EKF_VELOCITY = False`（Vicon 40 Hz 低通速度，方向正确）。
   验证：F3 第 1 跳落点 0.06 m。此修复保留，不再回退。
2. **69-D 修正器实机标定失配（已摘除）**：sim 标定的 7.9° 前倾+2 cm/s 指令
   在实机放大成 28° 离地姿态。`USE_SEMIMDP_PLANNER = False`（teacher-only）。
3. **主问题：策略空中姿态/yaw 回路在实机边际失稳**。位置/落地是低频环
   （落点一直 0.04–0.18 m、落地姿态 3–9°，air050 做得很好）；空中姿态是
   高频环，逐跳泵振荡（倾斜峰值单调递增）。反复验证：等比缩放只延缓不收敛
   （0.85→4 跳、0.75→6 跳），差动单独缩放反而破坏网络内部协调（甩尾加剧）。
   **recurrent 网络内部增益没有部署旋钮可调。**
4. **来源：推力曲线失配**。sim 推力拟合在
   `Quadhopper_Stable/quadhopper_env.py:366`：
   `F = -0.2371*u^2 + 0.8130*u + 0.0113`（每电机，N）。
   实机大概率显著偏离（用户做过推力台拟合，但长期实验后可能漂移，需复测）。
   当前 sim **没有推力随机化**，只有质量随机化（`--safety_mass_min/max`）
   和扰动外力（`planner_circular_env.py:397`）。

## 4. 代码现状（今天改过的文件）

### deployment/jump_stationary_semimdp.py（当前飞行脚本）
- `TEACHER_POLICY_PATH = 'deployment/two_hop_release_export_air050/quadhopper_v22_policy.onnx'`
  （air050 = v60 random_drop_safety `model_79.pt`，43-D recurrent；
  源 checkpoint: logs/rsl_rl/quadhopper_planner_random_two_hop_v60_custom_relative_next_random_drop_safety_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_125_xyw_110_yaww_120_heightw_125_air_050_fixed_100/2026-08-30_14-16-23/model_79.pt）
- `USE_SEMIMDP_PLANNER = False`（teacher-only）
- `PPO_USE_EKF_VELOCITY = False`（Vicon 速度，勿回退）
- `DEPLOY_ACTION_SCALE = 0.75`；`DEPLOY_DIFFERENTIAL_SCALE = 1.0`（已证伪关闭）
- 松手自由落体触发（Vz<-0.5 持续 50 ms）+ 零油门武装（PREHOVER/JUMP_STARTUP）
- 防炸机：60° 硬断油 / 35° 软警告 / 落地误差>0.30 m 或落地姿态>15° 断油回 IDLE /
  高度刹车 / PWM spread/step/yaw-diag 限幅
- 日志含 Flight_Mode / Deploy_Tilt_Deg / Deploy_XY_Error_M / Deploy_PWM_Spread / Deploy_Safety_Code 列

### deployment/jump_two_hop_release.py（两点跳脚本，本轮未实飞）
- H→ARMED 零油门 + 自由落体触发；倾斜保护仅 JUMP 生效
- `TEACHER_POLICY_PATH = 'deployment/two_hop_release_export_v29_model10/quadhopper_v22_policy.onnx'`
  （v29 model_10，sim 已验证 0-0.3/0.3-0.6 两点跳 hit 0.94）
- ⚠️ 已知契约问题：该脚本喂静态目标参考，v29 训练契约是 full-planner 时变轨迹，
  实机部署会退化；正式上两点跳前需把 flight reference 移植进脚本

### 部署包
- `deployment/deployment_update_20260902.zip`（可整包拷贝到飞行电脑）
- 用户飞行电脑与开发机分离，改完文件需手动传输；每次告知具体文件路径

## 5. 执行计划（模型层修复）

### Phase 1：sim 推力随机化（现在就能做，不等硬件）
1. 在 `Quadhopper_Stable/quadhopper_env.py` 推力应用处（约 line 366-378）
   加逐电机推力曲线随机化：乘性幅值噪声 + 曲线形状噪声，作为 cfg 开关；
2. 在 `Quadhopper_Planner_Circular/planner_circular_env.py` 与
   `train_planner_circular.py` 暴露 `--thrust_rand_min/max` 等参数
   （参考现有 `--safety_mass_min/max` 的接法）；
3. 从 air050 `model_79.pt` 起建 sim2real 微调路线：
   - 以现有 sim 曲线为中心、±30% 随机化先跑通；
   - 保持 drop-start + 原地落点分布（stationary）+ 1.0 m 高度命令；
   - 加强 yaw/空中姿态稳定奖励（air050 已有 yawair/air 权重，可加码）；
   - lr 1e-6~2e-6、std ≤0.003（参考 V58 v67 的防塌配置）。

### Phase 2：硬件复测（用户侧）
推力台逐电机复测 F(u) 曲线、电机时间常数（sim 假设 0.125 s）、指令延迟
（sim 假设 3 步/100 Hz）。旧拟合可能已漂移。实测曲线写回 sim 作为标称中心。

### Phase 3：用实测中心微调 + 验收
- 验收协议一律用 `play_planner_circular.py` 的 `[EVAL-DIRECT]` 块
  （256 env、`--max_steps 1700`；注意 V58 里的 `--eval_steps` 参数名已过时）。
- sim 验收：apex 1.0±0.05、touchdown attitude <0.08 rad、death 0，
  且在 ±20–30% 推力随机化下不退化；
- 实机验收：连续 ≥10 跳、空中倾斜 <15°、yaw 角速度 <±20°/s、落地姿态 <10°。

## 6. 给接力 AI 的注意事项

- **不要再改部署侧标量旋钮**（action scale / 差动缩放 / 限幅）——已全部证伪，
  只会延缓发散或破坏网络内部协调。
- 部署侧已有的正确修复（Vicon 速度、零油门武装、松手触发、断油保护）
  保持不动。
- 空中姿态振荡是高频环失稳，落点/落地是低频环正常——分析日志时区分两者；
  每跳空中倾斜峰值的单调递增是判断失稳的关键指标。
- 日志列：Vel_X/Y/Z 是 EKF 速度（会坏），Obs_Teacher_LinVel_* 才是策略实际
  吃到的速度（A 修复后是 Vicon 低通）。
- 训练窗口原始指标（mean_cycle_apex、target_hit_rate 瞬时值）不可信，
  选型只看 direct eval；recurrent 网络从 checkpoint 迁移后微调易塌
  （详见 V58 与项目记忆 project_two_hop_lineage）。
- 另有一条平行线索（v29 两点跳血统）已冻结：sim 逐项复现 hit 0.94、
  误差 0.048 m，等原地跳实机稳定后再启用。
