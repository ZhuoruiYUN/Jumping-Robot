# CURRENT_CONTEXT_V60 — raw-PWM 从零训练三锁排查全记录（2026-09-06 ~ 09-07）

> 交接给 Codex 的工作文档。上接 V59（部署调试、物理修正）。本文件记录修正物理后
> raw-PWM 血统（43-D recurrent，不用 LQR）从零训练的完整诊断链与当前状态。

## 1. 大目标与路线

在修正物理（I_ZZ 7.932e-4、K_tau 1.72e-2、动作延迟 0 步为主——与 LQR 工程实测一致，
已验证逐跳复现实机）下，训练一个能完成 far 部署契约（0-0.3m short / 0.3-0.6m long
两跳追点）的模型，最终覆盖真实工作点带（电机 tau≈0.03-0.125 + 推力 1.15-1.45x）。
路线 = 两阶段课程：Stage 1 温和随机化学会跳+命中 → Stage 2 随机化扩宽到真实工作点带。

## 2. 关键结论：long 跳 hit=0 的三层锁（本阶段核心成果）

从零训练 1500+ 迭代 short hit 0.45-0.90 而 long hit **精确 0.0000**，逐层拆出三层锁：

1. **apex 命中门（已修）**：`train_planner_circular.py` 通用 env 设置里
   `require_apex_tolerance_for_hit = True`（~line 982），从零路径没有 finetune 分支去关它。
   hit 要求 apex ∈ [1.0±0.12] m；long 跳 apex 天然偏出窗口 → 落点再准也永远不 hit。
   修复：新 CLI `--lenient_apex_hit`（token `_lenapex`），route 分支里关掉该门。
   验证：long_target_hit_rate_ema 0 → 0.05（命中开始出现）。
2. **空中翻车判死（已修）**：long 跳起飞后 ~0.3s 翻到 90°+（yaw 失稳级联，修正物理
   下 yaw 扰动灵敏度 2.9 倍）。`planner_circular_env.py::_get_dones` 里
   `tilt_death = roll_pitch_sq > 0.5`（≈90° 判死）→ 99% 的 long 跳在落地前死掉。
   修复：新增 cfg 字段 `tilt_death_sq_threshold`（默认 0.5）+ CLI `--tilt_death_sq_threshold`
   （token `_tds_XXX`），训练用 0.95（≈154°）；`--terminate_angvel_norm` 18→40。
   部署/实机保护阈值不受影响（部署保护在 deployment 脚本里，不在 env cfg）。
3. **死锁后果（修复 2 后自动解开）**：long 落地只占全部落地 ~1%（正常应 ~40-50%）→
   策略从未经历一次完整 long 跳 → long 空中姿态控制永远学不会。
   预期修 2 后 long 落地占比先跳回 ~40%，然后 long hit 开始爬。

### 解读指标的关键技巧
- `long_target_hit_rate` 是**从训练开始累计**的（老数据稀释新进展），
  `long_target_hit_rate_ema` 才是最近窗口趋势——判断进展看 EMA。
- long 落地占比 f 可由 overall≈short×(1−f)+long×f 反推（曾算出 f≈1% 而发现第二层锁）。

## 3. 当前状态（截至 2026-09-07 凌晨）

- **最新 checkpoint**：`model_799`（stage1b v2 = lax_120 + lenapex）——
  overall hit 0.897 / short 0.904 / mean td err 0.059 / reward +323；long EMA 0.05 刚开始爬。
  路径：`logs/rsl_rl/quadhopper_planner_random_two_hop_v59_custom_relative_next_random_drop_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_blend_100_lax_120_lenapex_fixed_100/2026-09-06_23-01-59/model_799.pt`
- **正在/待跑**：stage1b v3 = 上述三修复全开，从 model_799 续训 1000 迭代。
  脚本：`run_train_stage1b_lowapex.sh`（flags：`--low_apex_event_penalty 120
  --lenient_apex_hit --tilt_death_sq_threshold 0.95 --terminate_angvel_norm 40
  --fine_tune_lr 3.0e-4 --iterations 1000`，num_envs 1024，其余与 v2 相同）。
  启动：`./run_train_stage1b_lowapex.sh`（第一个参数 NUM_ENVS，第二个 CKPT）。
- **验证判据**：① long 落地占比回升到 ~40%（最先动的指标）② long_target_hit_rate_ema
  从 0.05 持续爬升 ③ short hit 保持 0.85+。
- **之后**：Stage 2 = 从 stage1b 最优 checkpoint 续训 + 随机化扩宽到
  tau U[0.03,0.125] + 推力 U[1.15,1.45]（参考 `run_train_from_zero_realop.sh` 的随机化
  参数，但**必须从会跳的策略起步**，不能从零）。完成后 3 点 direct eval
  （tm0.03+thr1.3 主点、tm0.06+thr1.3、tm0.125+thr1.0）与实机日志逐跳对比。

## 4. 本轮代码改动（未提交）

- `train_planner_circular.py`：新 CLI `--low_apex_event_penalty`（token `_lax_XXX`）、
  `--lenient_apex_hit`（token `_lenapex`）、`--tilt_death_sq_threshold`（token `_tds_XXX`）；
  route 分支里三处 env_cfg 覆盖（require_apex_tolerance_for_hit=False、
  low_apex_touchdown_event_penalty_scale、tilt_death_sq_threshold）。
- `Quadhopper_Planner_Circular/planner_circular_env.py`：新增 cfg 字段
  `tilt_death_sq_threshold = 0.5`，`_get_dones` 使用之。（插桩代码已全部清除。）
- `run_train_stage1b_lowapex.sh`（重写）、`run_train_stage1_resume.sh`（历史）。
- `experiments/random_two_hop/analyze_1env_hops.py`：1-env CSV 逐跳分析（注意：该文件
  被另一会话改过，带 phase parity 输出，读取前先看内容）。

## 5. 训练运行记录（时间线）

| 时间 | 运行 | 配置 | 结果 |
|---|---|---|---|
| 09-06 10:42 | corrected（256 env） | 默认温和随机化从零 | ✅ it300 apex 1.04 学会跳；it359 被提前终止（后来证明是错的——它其实在正常学习） |
| 09-06 11:39 | realop（256 env） | tau[0.03,0.125]+推力[1.15,1.45] 从零 | ❌ 2000 iters hit 全 0、apex≤0.38（宽随机化从零不可行） |
| 09-06 14:52 | stage1（标称） | --safety_finetune | ❌ 脚本 bug：safety 分支覆盖 lr=2e-6/std=0.01，冻死 |
| 09-06 15:40 | stage1 resume（1024 env） | model_350 精确恢复 | it1011: hit 0.432/short 0.44/long 0（当时误诊为平跳） |
| 09-06 17:37 | stage1b（lax_120） | model_290 起点 | it213: hit 0.459，apex 0.54→0.61（低apex惩罚部分有效，long 仍 0） |
| 09-06 23:01 | stage1b v2（lax_120+lenapex） | model_290→799 | ✅ it799: hit 0.897/short 0.904/long EMA 0→0.05（apex 门修复生效） |
| 09-07 | stage1b v3（+tds_095+angvel40） | model_799 起点，1000 iters | 待跑/待验证 |

## 6. 诊断方法论（给接手的 Codex）

1. **不要在 env step 内部插桩**：在 `_get_rewards`→touchdown 块里做任何额外 GPU 操作
   或 `.cpu()` 同步都会触发 PhysX fabric `DirectGpuHelper` device-side assert
   （it 0 即崩、进程挂死只能 SIGKILL）。要逐跳真相用 1-env CSV 路径：
   **`--num_envs 1` 时 env 才写 `outputs/planner_random_two_hop/on_quadhopper_sim.csv`**
   （100Hz 行、X/Y/Z/Target/quat/contact），用 analyze_1env_hops.py 分段分析。
   注意：1-env trainer 的 PPO 更新会退化策略（看前 20 个 episode）；1-env play
   （冻结权重）本策略会呆坐+翻车死亡（原因未完全查明，别用它当前策略做评估）。
2. **查 env cfg 要追 trainer 的全部赋值点**，不能只看 env 默认值——
   require_apex_tolerance_for_hit 的教训（通用设置 True + 各 finetune 分支 False，
   唯独裸从零路径带着 True）。
3. `--safety_finetune` 会强制 lr=2e-6/entropy=2e-5/std=0.01/num_steps=128，
   在 `--fine_tune_lr` 之后覆盖——**永远不要拿它当"关随机化"开关**。
4. random cfg（`PlannerRandomTwoHopEnvCfg`）里 `advance_route_on_miss=True`、
   `randomize_route_phase=False`、`restart_two_hop_pair=False`；env 默认恰相反。
5. 训练吞吐：1024 envs × 24 steps ≈ 7.7-12.8s/迭代，RTX 5060 Laptop 8GB 显存只剩
   ~80MB，**别在训练时开别的 GPU 程序**；1024 是显存上限（2048 必 OOM）。
6. pkill 自匹配陷阱：后台命令里 pkill 模式会杀掉自己的 shell wrapper，用 `[t]` 前缀。

## 7. 不变的背景（来自 V59 / memory）

- 部署组合：v2_240 ONNX + jump_two_hop_release.py（scale 0.80、落地断油保护、
  VBat 门控 3.90/3.65V）。m79 是当前部署基准。
- 真实工作点：tm≈0.03 + 推力≈1.3x 可复现实机飞行（单独 tm0.125+thr1.3 不能）。
- 血统教训：该 recurrent 血统的 fine-tune 易劣化（v1/v3/v5 被否决，仅 v2_240 成功）；
  部署侧标量旋钮已证伪，勿再试。
- 用户明确：不用 LQR 架构；给训练/评估指令让用户自己跑，不要自动开 GUI。
- 物理修正必须保留：I_ZZ 2.306e-3→7.932e-4（2.9x）、K_tau 5.4e-2→1.72e-2（3.1x），
  旧值指令增益恰好抵消、扰动灵敏度差 2.9 倍（实机 yaw 摆动真因）。
