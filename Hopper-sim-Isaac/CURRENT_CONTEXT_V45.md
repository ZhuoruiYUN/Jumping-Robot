# Quadhopper 随机两跳 / 部署当前上下文

更新时间：2026-08-31（Asia/Shanghai）

## 当前目标

目标已经从“单纯提高仿真随机两跳命中率”转为：

- 两点跳模型在 `P_t == P_t+1` 或接近原地跳时，必须像 baseline 一样连续稳定原地跳。
- 部署时不能靠降低目标高度投机，高度命令仍应按 1.0 m 训练和部署。
- 真实机重点问题是空中姿态不稳、yaw/尾部甩动、roll/pitch 激进纠偏、下方弹簧腿视觉上摆动，以及高度过冲。
- 不能只靠部署代码限幅，最终必须从模型训练上限制激进动作。

## 稳定 Baseline

稳定原地跳 baseline checkpoint：

```text
logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt
```

baseline 评估结果：

```text
full_length_rate=1.0
death_rate=0.0
deploy_success_rate=1.0
good_jump_rate=0.995
mean_apex_error=0.043 m
mean_landing_xy_error=0.061 m
max_xy_error_mean=0.113 m
max_tilt_proxy_mean=0.090
```

baseline 可视化指令：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh play_stable.py \
  --num_envs 9 \
  --checkpoint logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt
```

baseline 的重要特征：

- 多 env play 时目标高度随机为 `1.0-1.5 m`。
- 绿色目标点是固定 `_desired_pos_w`。
- episode 是 15 秒，所以可视化时 agents 跳一会儿会自动刷新。
- 观测里 policy 追的是稳定固定目标，不是随时间变化的轨迹点。

## 当前最好两点跳模型

当前仿真指标最好的两点跳 checkpoint 是：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v29_custom_relative_next_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_10-41-08/model_10.pt
```

不要用同一 run 的 `model_79.pt`。

`model_10.pt` 评估结果：

```text
direct touchdown error = 0.0481 m
target hit rate = 0.9404
short hit rate = 0.9239
long hit rate = 0.9593
touchdown attitude = 0.0628 rad
next velocity error = 0.179 m/s
last apex height = 0.987 m
deploy success = 1.0
death rate = 0.0
```

可视化 full-planner 指令：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh play_planner_circular.py \
  --num_envs 9 \
  --checkpoint logs/rsl_rl/quadhopper_planner_random_two_hop_v29_custom_relative_next_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_10-41-08/model_10.pt \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.0 --short_radius_max 0.3 \
  --long_radius_min 0.3 --long_radius_max 0.6 \
  --height_stage high --height_high 1.0 \
  --relative_next_hop \
  --action_scale 0.90
```

## Planner 与 Baseline 的关键差异

已确认代码差异：

- baseline 的 `_desired_pos_w` 是固定绿色目标点。
- planner 模型继承 baseline 37 维观测，但其中 `_desired_pos_w` 在空中会被更新成 `flight_pos`。
- `flight_pos = apex_pos + planner_weight * (reference_pos - apex_pos)`。
- play 默认 `force_full_planner=True`，所以 planner policy 实际上在空中追随时间变化的 full trajectory reference。
- 这会让出力时机更激进，尤其是短距离/原地跳时，会为了贴轨迹和落点进行 roll/pitch/yaw 修正。

当前判断：

```text
planner 的落点和高度目标是有用的；
full trajectory reference 对真实机部署太硬，可能是甩尾和激进姿态纠偏的重要来源。
```

## 部署日志结论

多次真实部署 CSV 已分析，最后人为停止的倾斜段通常忽略。

共同现象：

- 空中 raw action 经常接近饱和，饱和比例约 55%-60%。
- 高度命令 1.0 m 时，真实机经常超过 1.2 m，甚至到 1.35-1.40 m。
- roll/pitch/yaw 有时被模型用很激进的方式拉回，视觉上不稳。
- XY 会逐渐偏，靠极限姿态纠偏，部署上风险很高。
- 降低 `DEPLOY_ACTION_SCALE` 到 0.85 没有根治，说明是模型习惯问题，不只是输出比例问题。

部署代码已经加过临时保护：

- soft tilt 约 35 度只警告，不直接断电。
- hard tilt 约 60 度才 abort。
- 加了高度过冲 collective brake。
- 加了 airborne PWM spread limit。
- 加了 yaw diagonal PWM diff limit。

这些只是保护，不是最终答案。用户明确要求从模型上限制。

## 失败路线：Static Apex 直接微调 Planner

新增过两个开关：

- `play_planner_circular.py --static_apex_reference`
- `train_planner_circular.py --static_apex_finetune`

目的：让两点跳空中参考更像 baseline 的固定 apex，而不是 full planner trajectory。

结果：从 planner `model_10.pt` 直接切到 static apex reference 微调失败。

v62 run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v62_custom_relative_next_static_apex_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_11-23-26
```

`model_0.pt` static-apex 评估已经崩：

```text
direct hit = 0.097
touchdown error = 0.235 m
touchdown attitude = 0.139 rad
last apex height = 1.226 m
deploy success = 0.156
tilt death events = 346
```

结论：

```text
不是训练 40 轮后才坏；
而是 full-planner 模型一切换 static-apex reference，输入分布马上坏。
不能从已有 two-hop planner checkpoint 直接切 reference。
```

## 当前代码状态

重要修改文件：

- `train_planner_circular.py`
- `play_planner_circular.py`
- `Quadhopper_Planner_Circular/planner_circular_env.py`
- `deployment/jump_two_hop_release.py`

注意：workspace 里还有很多历史改动和未追踪部署导出文件，不要随便 revert。

`train_planner_circular.py` 当前支持：

- `--safety_finetune`
- `--robust_finetune`
- `--static_apex_finetune`
- 初始随机释放高度
- 初始姿态/速度扰动
- 空中 force/torque disturbance
- yaw/airborne/spring/action-spread 等 reward 项
- 对 37D baseline checkpoint 迁移到 43D planner policy
- 对 feed-forward 69D checkpoint 明确报错，不能用于当前 recurrent 43D planner 训练

`play_planner_circular.py` 当前支持：

- `--static_apex_reference`
- `--max_steps`
- direct eval 和 deploy eval 输出

## 下一步建议

不要继续 v62，也不要用 v62 的任何 checkpoint 部署。

下一条更合理路线：

```text
从 37D stable baseline 重新迁移训练 static-apex two-hop，
而不是从 full-planner model 切换参考。
```

训练指令：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh train_planner_circular.py \
  --num_envs 128 \
  --iterations 120 \
  --checkpoint logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.0 --short_radius_max 0.3 \
  --long_radius_min 0.3 --long_radius_max 0.6 \
  --height_stage high --height_high 1.0 \
  --relative_next_hop \
  --static_apex_finetune \
  --safety_attitude_scale 1.15 \
  --safety_angular_vel_scale 1.30 \
  --safety_xy_scale 0.80 \
  --safety_xy_dense_scale 0.35 \
  --safety_yaw_scale 2.00 \
  --safety_height_scale 1.05 \
  --airborne_stability_scale 0.55 \
  --safety_mass_min 1.0 \
  --safety_mass_max 1.0
```

训练后找目录：

```bash
ls -td logs/rsl_rl/quadhopper_planner_random_two_hop_v62_custom_relative_next_static_apex*/* | head
```

优先评估早期 checkpoint：`model_5.pt`、`model_10.pt`、`model_15.pt`，不要默认用最后模型。

评估模板：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh play_planner_circular.py \
  --headless \
  --num_envs 256 \
  --checkpoint <RUN_DIR>/model_10.pt \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.0 --short_radius_max 0.3 \
  --long_radius_min 0.3 --long_radius_max 0.6 \
  --height_stage high --height_high 1.0 \
  --relative_next_hop \
  --static_apex_reference \
  --action_scale 0.90 \
  --max_steps 1700 \
  --no_debug_vis
```

可视化模板：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh play_planner_circular.py \
  --num_envs 9 \
  --checkpoint <RUN_DIR>/model_10.pt \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.0 --short_radius_max 0.3 \
  --long_radius_min 0.3 --long_radius_max 0.6 \
  --height_stage high --height_high 1.0 \
  --relative_next_hop \
  --static_apex_reference \
  --action_scale 0.90
```

判断标准不要只看 hit rate：

- 空中 yaw 是否继续发散或甩尾
- roll/pitch 是否还有大幅极限纠偏
- 下方弹簧腿是否明显摆动
- 高度是否仍然超过 1.2 m
- direct touchdown error 是否保持在 5-8 cm 范围
- deploy death rate 是否为 0

如果从 baseline 重新训练 static-apex 仍然不行，则应考虑放弃 full direct-collocation planner 接口，改成：

```text
baseline 原地跳控制
+ 低频落点偏置 / 目标点偏移
+ 只在落地或支撑期更新下一跳目标
+ 空中不追 full trajectory
```

## 最新训练结果：baseline 重新迁移 static-apex 也失败

新 run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v62_custom_relative_next_static_apex_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_11-32-15
```

该 run 从 stable baseline 迁移得到 `initial_policy_43d.pt`，完整训练到 `model_119.pt`。

训练末尾 iteration 119/120：

```text
Mean reward = -1740.31
target_hit_rate = 0.0170
touchdown_error_ema = 0.4159 m
mean_cycle_apex_height = 0.5416 m
episode_touchdown_error = 0.3536 m
short_hit_rate = 0.0257
long_hit_rate = 0.0082
deploy 相关风险信号：landing_error=-15.8804, target_miss=-23.2100
```

直接评估早期 checkpoint，协议为 static-apex reference、`action_scale=0.90`、256 env、1700 steps：

```text
model_5:
  direct target_hit_rate = 0.0116
  direct touchdown_error = 0.3858 m
  short/long error = 0.3316 / 0.4474 m
  last_apex_height = 0.5871 m
  deploy death_rate = 0.7852

model_10:
  direct target_hit_rate = 0.0095
  direct touchdown_error = 0.3689 m
  short/long error = 0.3198 / 0.4242 m
  last_apex_height = 0.5829 m
  deploy death_rate = 0.7383

model_15:
  direct target_hit_rate = 0.0062
  direct touchdown_error = 0.4038 m
  short/long error = 0.3451 / 0.4712 m
  last_apex_height = 0.5725 m
  deploy death_rate = 0.7734
```

结论：

```text
不是后期 checkpoint 退化；
从 baseline 重新迁移 static-apex two-hop 后，前 5-15 轮已经落入低跳/短跳坏模式。
这条 v62 static-apex 路线不要部署，也不建议继续挑 model_119。
```

关键诊断：

- 高度命令仍是 1.0 m，但评估 apex 只有约 0.57-0.59 m，训练末尾平均 apex 也只有 0.54 m。
- touchdown along error 为约 -0.35 到 -0.39 m，说明策略系统性落短。
- 姿态指标本身不算爆炸（touchdown attitude 约 0.052 rad），但 deploy death 仍约 74%-79%，主要表现为 tilt death。
- 这说明当前 reward/任务耦合允许模型通过低跳、短跳、少出力来换取局部收益；`height_progress` 给了正收益，但 `apex_error/shortfall` 没有强到能守住 1.0 m apex。

下一步不要继续 static-apex 直接 PPO。更合理的方向：

```text
先保住 baseline 的 1.0 m 原地跳高度/节律，
再以低频目标偏置方式引入 XY 落点，
并把 touchdown/target hit 只在有效高度跳跃后计分。
```

推荐下一轮改动：

- 在 reward 中加入 height-gated touchdown/hit：apex 未达到例如 0.85-0.90 m 时，不给 target_hit / landing_precision 的主要正奖励，并额外罚低跳命中。
- 明确提高 apex_shortfall / apex_error 权重，先让 `last_apex_height_m` 回到 0.95-1.05 m。
- 对 static-apex 训练做 curriculum：先 `short_radius_max=0.0` 或 0.05 复现 baseline 原地跳，再放到 0.15、0.30、0.60，而不是一开始就混合短/长目标。
- 考虑把空中 reference 进一步简化为“固定 1.0 m apex + 支撑期/落地前目标偏置”，不要在空中持续追 XY 轨迹。

## v63/v64 复查：问题不是 reward gate，而是 43D planner 接口已破坏 baseline

v63 改动：

- `static_apex_min_valid_apex=0.88`
- touchdown 正奖励按有效 apex gate
- 低 apex touchdown 额外惩罚
- static-apex smooth distance curriculum

v63 run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v63_custom_rel_static_hgate_smooth_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_11-52-19
```

评估结果仍失败：

```text
model_5:   hit=0.0080, apex=0.5957 m, touchdown_error=0.3776 m, deploy death=0.7422
model_10:  hit=0.0059, apex=0.5771 m, touchdown_error=0.3824 m, deploy death=0.7578
model_30:  hit=0.0104, apex=0.5825 m, touchdown_error=0.3884 m, deploy death=0.7812
model_119: hit=0.0116, apex=0.5982 m, touchdown_error=0.3993 m, deploy death=0.7539
```

v64 改动：

- 新增 `--static_apex_height_probe`
- 极短目标：`short/long = 0.0-0.05 m`
- 关闭 XY、landing、target hit、prepared landing、streak、circle complete 等正向任务奖励
- 关闭 dynamics/action-delay randomization
- 降低 PPO 学习率和探索，目标只验证 static-apex 43D 输入能否保住 baseline 1.0 m 原地跳

v64 run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v64_custom_rel_static_height_probe_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_12-10-45
```

关键评估：

```text
model_0, action_scale=0.90:
  direct hit = 0.1600
  direct touchdown_error = 0.1920 m
  last_apex_height = 0.5721 m
  deploy death = 0.7109

model_0, action_scale=1.00:
  direct hit = 0.1822
  direct touchdown_error = 0.1824 m
  last_apex_height = 0.6265 m
  deploy death = 0.7617

model_79, action_scale=1.00:
  direct hit = 0.2282
  direct touchdown_error = 0.1725 m
  last_apex_height = 0.6864 m
  deploy death = 0.6875
```

v64 训练末尾虽然 `target_hit_rate_ema≈0.72`，但 `mean_cycle_apex_height=0.2687 m`、
`target_hit_rate=0`、`successful_waypoints=0`。这是事件/EMA 口径与 reset
统计混在一起后的假乐观；不能作为可部署信号。

最终结论：

```text
model_0 刚从 37D stable baseline 迁移到 43D planner policy 后就已经低跳。
这说明不是 PPO 后期学坏，也不是 action_scale=0.90 造成；
而是 43D planner/random_two_hop/static-apex 接口本身没有保持 baseline 的输入/状态合同。
```

下一步不要继续 v63/v64 static-apex PPO。优先做接口审计：

- 对比 `play_stable.py` baseline 与 `play_planner_circular.py --static_apex_reference` 的前 1-2 秒观测，特别是 37D stable obs 中的 `_desired_pos_w`、root frame error、高度命令、episode reset 初始状态。
- 检查 37D→43D migration 是否真的保持前 37D recurrent 输入权重不变、新 6D 输入权重置零。
- 检查 planner static-apex 的 stance reference：当前 stance 阶段 `_desired_pos_w` 的 XY 被设为 `p_t`，即使半径 0.05 m，也不同于 baseline 固定目标；这可能改变支撑期姿态/起跳节律。
- 下一版应先做一个“baseline-compatible 43D wrapper”：前 37D 观测必须与 `play_stable.py` bit-level 或近似一致，新 6D 置零/旁路；确认 `model_0` apex 回到 0.95-1.05 m 后，再谈 XY 偏置。

## v65 关键修复：支撑期 stable target z 必须保持 apex 高度

migration 离线检查通过：

```text
memory_a.rnn.weight_ih_l0: first37 max diff=0.0, new6 max abs=0.0
memory_c.rnn.weight_ih_l0: first37 max diff=0.0, new6 max abs=0.0
```

因此低跳不是 37D→43D 权重迁移导致。

真正根因：

```text
stable baseline:
  支撑期 37D obs 的 pos_error 追 _desired_pos_w.z = 1.0-1.5 m apex target

planner static-apex:
  支撑期 _desired_pos_w.z 被设成 landing_root_height = 0.38 m
```

同一个 baseline policy 在 planner 环境里看到“目标在地面”，所以 model_0
低跳是输入合同被破坏，不是 PPO reward 问题。

已加代码开关：

- `PlannerCircularEnvCfg.stance_reference_uses_apex_height`
- `play_planner_circular.py --baseline_compatible_stance_reference`
- `train_planner_circular.py --baseline_compatible_stance_reference`

实现效果：当该开关启用时，stance 阶段 `stance_pos[:, 2]` 使用
`_active_target_height`，使 inherited 37D stable obs 的高度目标与 baseline 兼容。

直接用 v64 `model_0` 复测，除了打开新 play 开关外不改 checkpoint：

```text
原 v64 model_0, action_scale=1.00:
  direct hit = 0.1822
  touchdown_error = 0.1824 m
  last_apex_height = 0.6265 m
  deploy death = 0.7617

v64 model_0 + --baseline_compatible_stance_reference, action_scale=1.00:
  direct hit = 0.3638
  touchdown_error = 0.1274 m
  touchdown_along_error = -0.0229 m
  last_apex_height = 0.9442 m
  max_consecutive_hits = 2.8828
  deploy success = 0.8438
  deploy death = 0.1562
```

这是强阳性结果：支撑期 target z 修复后，迁移后的 `model_0` 立即接近
baseline 高度，落短问题基本消失。

v65 冒烟训练也已通过并创建目录：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_height_probe_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_15-02-01/model_0.pt
```

下一步训练应从 v65 height probe 开始，先保持 `0.0-0.05 m` 极短目标。
选择 checkpoint 时仍优先看 `model_0/5/10/20`，并且评估必须带：

```text
--baseline_compatible_stance_reference
```

## v65 200 轮 height probe 结果

v65 200-iteration run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_height_probe_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_15-06-29
```

训练末尾 `model_199` 日志：

```text
Mean reward = 2052.95
mean_cycle_apex_height = 0.3285 m   # reset/episode 口径偏低，不作选型主依据
target_hit_rate_ema = 0.1640
touchdown_error_ema = 0.1912 m
live_max_consecutive_hits = 6.14
```

direct/deploy 评估协议：`0.0-0.05 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference`。

```text
model_0:
  hit = 0.3565
  touchdown_error = 0.1277 m
  along_error = -0.0229 m
  apex = 0.9491 m
  max_consecutive_hits = 2.6758
  deploy success = 0.8438
  deploy death = 0.1562

model_5:
  hit = 0.3544
  touchdown_error = 0.1250 m
  along_error = -0.0288 m
  apex = 0.9548 m
  max_consecutive_hits = 2.5586
  deploy success = 0.8516
  deploy death = 0.1484

model_20:
  hit = 0.3664
  touchdown_error = 0.1252 m
  along_error = -0.0200 m
  apex = 0.9561 m
  max_consecutive_hits = 2.6914
  deploy success = 0.8438
  deploy death = 0.1562

model_50:
  hit = 0.3517
  touchdown_error = 0.1296 m
  along_error = -0.0248 m
  apex = 0.9393 m
  max_consecutive_hits = 2.5898
  deploy success = 0.8281
  deploy death = 0.1719
```

结论：

```text
v65 baseline-compatible stance reference 修复是有效的；
PPO probe 没有再次打坏 1.0 m 高度，但 0.0-0.05 m 极短阶段也没有显著降低 tilt death。
```

当前最佳选择：

- 若优先 death：`model_5.pt`，deploy death `0.1484`。
- 若优先 hit/apex：`model_20.pt`，hit `0.3664`、apex `0.9561 m`。
- 两者差距很小；建议用 `model_20.pt` 作为下一阶段 `0.0-0.15 m` 的起点。

下一阶段不要从 stable baseline 重新开始，应该从 v65 `model_20.pt` 续训，
保留 `--baseline_compatible_stance_reference`，把半径扩到：

```text
short_radius = 0.0-0.15 m
long_radius = 0.0-0.15 m
```

## v65 0.0-0.15 m base-stance 续训结果

从 v65 height-probe `model_20.pt` 续训，目标半径扩到：

```text
short_radius = 0.0-0.15 m
long_radius = 0.0-0.15 m
```

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_15-26-24
```

训练末尾 `model_119` 日志：

```text
Mean reward = -906.96
target_hit_rate = 0.2273
touchdown_error_ema = 0.1936 m
touchdown_attitude_error = 0.1116 rad
live_max_consecutive_hits = 3.0
```

direct/deploy 评估协议：`0.0-0.15 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference`。

```text
model_0:
  hit = 0.2357
  touchdown_error = 0.1603 m
  along_error = -0.0707 m
  apex = 0.9590 m
  max_consecutive_hits = 1.7852
  deploy success = 0.8398
  deploy death = 0.1602

model_20:
  hit = 0.2392
  touchdown_error = 0.1561 m
  along_error = -0.0625 m
  apex = 0.9338 m
  max_consecutive_hits = 1.7266
  deploy success = 0.8203
  deploy death = 0.1797

model_119:
  hit = 0.2484
  touchdown_error = 0.1545 m
  along_error = -0.0785 m
  apex = 0.9401 m
  max_consecutive_hits = 1.8867
  deploy success = 0.8398
  deploy death = 0.1602
```

结论：

```text
0.0-0.15 m 阶段保持了 v65 的高度修复，apex 仍在 0.93-0.96 m。
PPO 续训只带来很小的 hit/error 改善，没有降低 deploy death。
```

当前 0.15 m 最佳可选：

- 若看整体指标，`model_119.pt` 略优：hit `0.2484`、error `0.1545 m`、apex `0.9401 m`、death `0.1602`。
- 若优先高度裕度，`model_0.pt` 更稳：apex `0.9590 m`、death 同为 `0.1602`，但 hit 较低。

下一步建议先用 `model_119.pt` 作为 `0.0-0.25 m` 或 `0.0-0.30 m` 的起点，
但不要期待 static-apex PPO 自己解决 death；death 约 15%-16% 可能来自
deploy eval 的 tilt gate/episode 判据，需要单独可视化或对比 baseline deploy eval。

## v65 0.0-0.25 m base-stance 续训结果

从 v65 `0.0-0.15 m` run 的 `model_119.pt` 续训，目标半径扩到：

```text
short_radius = 0.0-0.25 m
long_radius = 0.0-0.25 m
```

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_15-38-24
```

训练末尾 `model_119` 日志：

```text
Mean reward = -1075.52
target_hit_rate = 0.0000
target_hit_rate_ema = 0.0988
touchdown_error_ema = 0.2594 m
touchdown_attitude_error = 0.0963 rad
live_max_consecutive_hits = 4.0
```

direct/deploy 评估协议：`0.0-0.25 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference`。

```text
model_0:
  hit = 0.1475
  touchdown_error = 0.2010 m
  along_error = -0.1393 m
  lateral_abs_error = 0.0789 m
  apex = 0.9296 m
  max_consecutive_hits = 1.1641
  deploy success = 0.7695
  deploy death = 0.2305

model_20:
  hit = 0.1409
  touchdown_error = 0.2015 m
  along_error = -0.1341 m
  lateral_abs_error = 0.0828 m
  apex = 0.9500 m
  max_consecutive_hits = 1.1367
  deploy success = 0.7852
  deploy death = 0.2148

model_119:
  hit = 0.1352
  touchdown_error = 0.2055 m
  along_error = -0.1390 m
  lateral_abs_error = 0.0820 m
  apex = 0.9239 m
  max_consecutive_hits = 1.0703
  deploy success = 0.7461
  deploy death = 0.2539
```

结论：

```text
0.25 m 阶段仍保住了大部分高度，apex 约 0.92-0.95 m；
但 XY 落点明显变差，沿目标方向系统性 short landing 约 13-14 cm；
deploy death 升到 21%-25%，超过当前可接受范围。
PPO 从 model_0 训到 model_119 没有改善 0.25 m，反而略微退化。
```

当前 0.25 m 最佳 checkpoint：

```text
model_20.pt
```

原因：death 最低 `0.2148`，apex 最高 `0.9500 m`，虽然 hit 不是最高但差距很小。

下一步不建议直接扩到 `0.30 m`；更合理是先处理 0.25 m 的系统性 short landing：

- 继续从 0.25 m `model_20.pt` 起步；
- 把落点/投影落点权重加一点，或加前向 short-landing 修正；
- 目标是先把 `touchdown_along_error_m` 从约 `-0.13 m` 拉到 `-0.05 m` 以内，再考虑扩半径。

## v65 0.25 m landing compensation 0.08 结果

从 v65 `0.0-0.25 m` run 的 `model_20.pt` 续训，保持半径不变，
加入 `--landing_compensation 0.08`。

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_15-54-01
```

训练末尾 `model_119` 日志：

```text
Mean reward = -646.11
episode length = 799.79
target_hit_rate = 0.0000
target_hit_rate_ema = 0.2495
touchdown_error_ema = 0.1938 m
mean_cycle_apex_height = 0.1185 m
low_apex_touchdown = -4.1327
touchdown_attitude_error = 0.2132 rad
```

direct/deploy 评估协议：`0.0-0.25 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference`。评估时没有额外
`--landing_compensation`，只看训练后的 policy。

```text
model_0:
  hit = 0.1290
  touchdown_error = 0.2043 m
  along_error = -0.1489 m
  lateral_abs_error = 0.0768 m
  apex = 0.9155 m
  max_consecutive_hits = 1.0391
  deploy success = 0.7930
  deploy death = 0.2070

model_20:
  hit = 0.1422
  touchdown_error = 0.2046 m
  along_error = -0.1432 m
  lateral_abs_error = 0.0806 m
  apex = 0.9144 m
  max_consecutive_hits = 1.0508
  deploy success = 0.7734
  deploy death = 0.2227

model_119:
  hit = 0.1305
  touchdown_error = 0.2035 m
  along_error = -0.1325 m
  lateral_abs_error = 0.0859 m
  apex = 0.9392 m
  max_consecutive_hits = 1.1250
  deploy success = 0.8008
  deploy death = 0.1992
```

结论：

```text
训练时 landing_compensation=0.08 没有把系统性 short landing 修回来；
沿向误差仍约 -0.13 到 -0.15 m。
它让 model_119 的 death 略降到 19.9%，但 hit 降到 13.1%，不如预期。
不要把这轮作为 0.30 m 扩展起点。
```

当前可临时保留：

- 若只看 0.25 m death，`15-54-01/model_119.pt` 勉强最好，death `0.1992`。
- 若看整体 hit/apex，上一轮无 compensation 的 `15-38-24/model_20.pt` 仍更健康。

下一步应先做 eval-time compensation sweep，而不是继续 PPO 盲训：

```text
用上一轮 15-38-24/model_20.pt，
分别评估 --landing_compensation 0.04 / 0.08 / 0.12。
如果 eval-time 补偿也不能修 along error，说明问题不在目标前推幅度，而在 static-apex 接口缺少前向速度/冲量 authority。
```

## v65 eval-time compensation sweep 与 0.04 训练结果

对上一轮无 compensation 的 `15-38-24/model_20.pt` 做 eval-time compensation sweep。
关键结果：

```text
eval-time --landing_compensation 0.04:
  hit = 0.2164
  touchdown_error = 0.1673 m
  along_error = -0.1053 m
  lateral_abs_error = 0.0736 m
  apex = 0.9377 m
  max_consecutive_hits = 1.5625
  deploy success = 0.8203
  deploy death = 0.1797

eval-time --landing_compensation 0.08:
  hit = 0.1856
  touchdown_error = 0.1784 m
  along_error = -0.1129 m
  lateral_abs_error = 0.0764 m
  apex = 0.9111 m
  max_consecutive_hits = 1.3359
  deploy success = 0.7422
  deploy death = 0.2578
```

结论：

```text
eval-time compensation 有用，0.04 是目前更合适的临时值；
0.08 开始破坏高度/安全性。
```

随后从无 compensation 的 `15-38-24/model_20.pt` 开始，用
`--landing_compensation 0.04` 训练 80 轮。

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_16-11-51
```

训练末尾 `model_79` 日志：

```text
Mean reward = -732.74
episode length = 1171.58
target_hit_rate = 0.0741
target_hit_rate_ema = 0.2500
touchdown_error_ema = 0.2118 m
mean_cycle_apex_height = 0.2658 m
touchdown_attitude_error = 0.1259 rad
successful_waypoints = 4.0
```

direct/deploy 评估协议：`0.0-0.25 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference`。评估时没有额外
`--landing_compensation`，只看训练后的 policy。

```text
model_0:
  hit = 0.1445
  touchdown_error = 0.1996 m
  along_error = -0.1386 m
  lateral_abs_error = 0.0805 m
  apex = 0.9348 m
  max_consecutive_hits = 1.0781
  deploy success = 0.7500
  deploy death = 0.2500

model_20:
  hit = 0.1235
  touchdown_error = 0.2075 m
  along_error = -0.1466 m
  lateral_abs_error = 0.0785 m
  apex = 0.9122 m
  max_consecutive_hits = 0.9844
  deploy success = 0.7695
  deploy death = 0.2305

model_79:
  hit = 0.1593
  touchdown_error = 0.1938 m
  along_error = -0.1262 m
  lateral_abs_error = 0.0829 m
  apex = 0.9373 m
  max_consecutive_hits = 1.1992
  deploy success = 0.7656
  deploy death = 0.2344
```

结论：

```text
0.04 compensation 作为 eval-time/deploy-time 几何偏置是有帮助的；
但把它放进训练后，PPO 没有把收益内化到 policy，反而整体不如直接 eval-time 0.04。
当前不要再训练 landing_compensation=0.04/0.08。
```

当前 0.25 m 最佳候选：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_15-38-24/model_20.pt
```

使用时临时带：

```text
--landing_compensation 0.04
```

下一步更合理方向：

- 先可视化该组合，判断 tilt death 是否是少数 env 的统计问题还是明显姿态风险。
- 若画面可接受，可以把 `0.25 m + compensation 0.04` 当作保守部署/实机试验候选。
- 训练上不要继续让 PPO 学 compensation 常数；应改成结构性前向 authority，例如支撑/起跳期轻量 motor residual 或显式 landing velocity/impulse bias。

## v65 0.20 m landing velocity 0.35 结果

用户反馈 `0.25 m + static-apex + compensation` 可视化跳得不好、精度低、偶尔不起跳，
因此不再推进该候选。

从 v65 `0.0-0.15 m` run 的 `model_119.pt` 续训，目标半径改为：

```text
short_radius = 0.0-0.20 m
long_radius = 0.0-0.20 m
landing_velocity_scale = 0.35
```

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_120_angw_140_xyw_070_xyd_025_yaww_200_heightw_120_hg088_la180_air_070_yawair_spring_mass_100_100_fixed_100/2026-08-31_16-27-48
```

训练末尾 `model_79` 日志：

```text
Mean reward = -836.03
episode length = 1257.63
target_hit_rate = 0.0423
target_hit_rate_ema = 0.1434
touchdown_error_ema = 0.2152 m
mean_cycle_apex_height = 0.4625 m
touchdown_attitude_error = 0.1729 rad
successful_waypoints = 3.0
```

direct/deploy 评估协议：`0.0-0.20 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference --landing_velocity_scale 0.35`。

```text
model_0:
  hit = 0.1952
  touchdown_error = 0.1784 m
  along_error = -0.1119 m
  lateral_abs_error = 0.0751 m
  apex = 0.9363 m
  next_velocity_error = 0.1085 m/s
  max_consecutive_hits = 1.4414
  deploy success = 0.7891
  deploy death = 0.2109

model_20:
  hit = 0.1799
  touchdown_error = 0.1765 m
  along_error = -0.1058 m
  lateral_abs_error = 0.0759 m
  apex = 0.9453 m
  next_velocity_error = 0.1073 m/s
  max_consecutive_hits = 1.3828
  deploy success = 0.8164
  deploy death = 0.1836

model_79:
  hit = 0.1872
  touchdown_error = 0.1763 m
  along_error = -0.1040 m
  lateral_abs_error = 0.0772 m
  apex = 0.9469 m
  next_velocity_error = 0.1101 m/s
  max_consecutive_hits = 1.4453
  deploy success = 0.8359
  deploy death = 0.1641
```

结论：

```text
landing_velocity_scale=0.35 对速度和起跳可靠性有帮助；
0.20 m model_79 的 death 回到约 16.4%，apex 约 0.947 m，明显比 0.25 m 更适合作为安全候选。
但 hit 只有约 18.7%，沿向仍系统性落短约 10.4 cm，精度仍不足。
```

当前较稳候选：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_120_angw_140_xyw_070_xyd_025_yaww_200_heightw_120_hg088_la180_air_070_yawair_spring_mass_100_100_fixed_100/2026-08-31_16-27-48/model_79.pt
```

部署/可视化必须带：

```text
--landing_velocity_scale 0.35
```

下一步建议：

- 先可视化 `0.20 m model_79`，确认是否还会“不起跳”。
- 若视觉上明显好于 0.25 m，再做一个 `landing_velocity_scale=0.50` 的短训/评估，看能否把沿向误差从 `-0.104 m` 继续拉近。
- 如果 0.50 增加 tilt 或破坏高度，就退回 0.35，不再靠 velocity scale 加码。

## 用户可视化反馈：仍有 thrust 不足/不起跳

用户观察到：

```text
0.20/0.25 m static-apex 候选仍然跳得不好、精度低；
不少时候 thrust 不足，甚至飞不起来。
```

复查 reset 逻辑后发现重要差异：

```text
之前 v65 static-apex 训练命令没有带 --start_from_random_drop。
因此 planner reset 默认直接在 landing_root_height=0.38 m 的地面/支撑高度开始。

stable baseline reset 则是在目标高度附近加随机 drop_height，从空中释放：
default_root_state[:, 2] = desired_pos_w.z + drop_height
```

planner 里已经有相应开关：

```text
--start_from_random_drop
--initial_drop_height_min
--initial_drop_height_max
```

下一步应先恢复 baseline 风格的空中随机释放，而不是继续加大
`landing_velocity_scale` 或 `landing_compensation`。建议从 v65 height-probe
`model_20.pt` 重新做一个短阶段：

```text
0.0-0.05 m + static_apex_height_probe + baseline_compatible_stance_reference
+ start_from_random_drop
```

目标是验证从空中随机释放后，是否能恢复“每次都能落地后继续跳起来”的节律。

## v65 airborne reset height probe 成功

按上述判断，重新加入 baseline 风格空中随机释放：

```text
--start_from_random_drop --initial_drop_height_min 0.8 --initial_drop_height_max 1.5
```

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_height_probe_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_16-44-51
```

训练末尾 `model_119` 日志：

```text
Mean reward = 12256.68
episode length = 14999.00
target_hit_rate = 0.2815
target_hit_rate_ema = 0.7864
touchdown_error_ema = 0.0760 m
mean_cycle_apex_height = 0.8407 m
touchdown_attitude_error = 0.0891 rad
successful_waypoints = 43.6271
termination_p = 0.0000
```

direct/deploy 评估协议：`0.0-0.05 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference --start_from_random_drop`
和 `initial_drop_height=0.8-1.5`。

```text
model_0:
  hit = 0.5897
  touchdown_error = 0.0944 m
  along_error = -0.0110 m
  lateral_abs_error = 0.0496 m
  apex = 1.0012 m
  apex_error = 0.0430 m
  next_velocity_error = 0.0871 m/s
  max_consecutive_hits = 5.0547
  deploy success = 1.0000
  deploy death = 0.0000

model_20:
  hit = 0.5900
  touchdown_error = 0.0969 m
  along_error = -0.0035 m
  lateral_abs_error = 0.0499 m
  apex = 0.9957 m
  apex_error = 0.0415 m
  next_velocity_error = 0.0905 m/s
  max_consecutive_hits = 5.5117
  deploy success = 1.0000
  deploy death = 0.0000

model_119:
  hit = 0.5434
  touchdown_error = 0.1034 m
  along_error = 0.0099 m
  lateral_abs_error = 0.0503 m
  apex = 0.9791 m
  apex_error = 0.0427 m
  next_velocity_error = 0.0895 m/s
  max_consecutive_hits = 5.2461
  deploy success = 1.0000
  deploy death = 0.0000
```

结论：

```text
airborne reset 是关键修复。加入 baseline 风格空中随机释放后，
static-apex height probe 恢复到稳定连续跳：apex 接近 1.0 m，death 0，
沿向误差接近 0，连续命中约 5。
```

当前最佳 checkpoint：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_height_probe_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_16-44-51/model_20.pt
```

下一步扩距离必须保留：

```text
--start_from_random_drop
--initial_drop_height_min 0.8
--initial_drop_height_max 1.5
```

建议从 `model_20.pt` 扩到 `0.0-0.15 m`，不要直接跳到 `0.20/0.25 m`。

## v65 airborne reset 0.15 m 结果

从 airborne reset height probe `model_20.pt` 续训，目标半径扩到：

```text
short_radius = 0.0-0.15 m
long_radius = 0.0-0.15 m
```

继续保留：

```text
--start_from_random_drop --initial_drop_height_min 0.8 --initial_drop_height_max 1.5
--static_apex_reference --baseline_compatible_stance_reference
```

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_16-52-53
```

训练末尾 `model_119` 日志：

```text
Mean reward = -114.03
episode length = 6673.00
target_hit_rate = 0.0619
target_hit_rate_ema = 0.1964
touchdown_error_ema = 0.2116 m
mean_cycle_apex_height = 0.2214 m
touchdown_attitude_error = 0.1616 rad
successful_waypoints = 7.0
```

direct/deploy 评估协议：`0.0-0.15 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference --start_from_random_drop`
和 `initial_drop_height=0.8-1.5`。

```text
model_0:
  hit = 0.3747
  touchdown_error = 0.1338 m
  along_error = -0.0557 m
  lateral_abs_error = 0.0621 m
  apex = 0.9921 m
  apex_error = 0.0374 m
  next_velocity_error = 0.1540 m/s
  max_consecutive_hits = 2.8711
  deploy success = 1.0000
  deploy death = 0.0000

model_20:
  hit = 0.3665
  touchdown_error = 0.1344 m
  along_error = -0.0481 m
  lateral_abs_error = 0.0681 m
  apex = 0.9963 m
  apex_error = 0.0382 m
  next_velocity_error = 0.1515 m/s
  max_consecutive_hits = 3.0156
  deploy success = 1.0000
  deploy death = 0.0000

model_119:
  hit = 0.3671
  touchdown_error = 0.1332 m
  along_error = -0.0615 m
  lateral_abs_error = 0.0642 m
  apex = 0.9945 m
  apex_error = 0.0397 m
  next_velocity_error = 0.1572 m/s
  max_consecutive_hits = 2.8164
  deploy success = 1.0000
  deploy death = 0.0000
```

结论：

```text
airborne reset 路线在 0.15 m 仍稳定：apex 约 0.99-1.00 m，deploy death 0。
hit 从 0.05 m 的约 0.59 降到约 0.37，这是距离扩展后的正常精度损失；
但不再出现之前大规模不起跳/tilt death。
```

当前 0.15 m 最佳选择：

- `model_20.pt`：整体最平衡，along error 最小、apex 最高、max consecutive hits 最高。
- `model_0.pt`：hit 略高，但连续命中和沿向误差略弱。

下一步可以扩到 `0.0-0.20 m`，必须继续保留 airborne reset。

## v65 airborne reset 0.20 m 结果

从 airborne reset `0.15 m` run 的 `model_20.pt` 续训，目标半径扩到：

```text
short_radius = 0.0-0.20 m
long_radius = 0.0-0.20 m
```

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_17-03-06
```

训练末尾 `model_119` 日志：

```text
Mean reward = -488.86
episode length = 6480.59
target_hit_rate = 0.0748
target_hit_rate_ema = 0.0975
touchdown_error_ema = 0.2256 m
mean_cycle_apex_height = 0.2623 m
touchdown_attitude_error = 0.1138 rad
successful_waypoints = 8.0
```

direct/deploy 评估协议：`0.0-0.20 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference --start_from_random_drop`
和 `initial_drop_height=0.8-1.5`。

```text
model_0:
  hit = 0.2848
  touchdown_error = 0.1577 m
  along_error = -0.0857 m
  lateral_abs_error = 0.0734 m
  apex = 0.9973 m
  apex_error = 0.0418 m
  next_velocity_error = 0.1943 m/s
  max_consecutive_hits = 2.2734
  deploy success = 1.0000
  deploy death = 0.0000

model_20:
  hit = 0.2789
  touchdown_error = 0.1583 m
  along_error = -0.0861 m
  lateral_abs_error = 0.0718 m
  apex = 0.9867 m
  apex_error = 0.0445 m
  next_velocity_error = 0.1936 m/s
  max_consecutive_hits = 2.2188
  deploy success = 1.0000
  deploy death = 0.0000

model_119:
  hit = 0.2971
  touchdown_error = 0.1564 m
  along_error = -0.0773 m
  lateral_abs_error = 0.0718 m
  apex = 0.9983 m
  apex_error = 0.0420 m
  next_velocity_error = 0.1872 m/s
  max_consecutive_hits = 2.4531
  deploy success = 1.0000
  deploy death = 0.0000
```

结论：

```text
airborne reset 路线在 0.20 m 仍保住了稳定跳跃：deploy death 0，apex 约 0.99-1.00 m。
这说明之前不起跳问题已解决。
精度继续随半径下降，hit 约 0.28-0.30，沿向仍短约 7.7-8.6 cm。
```

当前 0.20 m 最佳 checkpoint：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_17-03-06/model_119.pt
```

下一步不要急着训练 0.25 m。先对该 `model_119.pt` 做 eval-time
`landing_compensation 0.03-0.05` sweep，看能否把 along error 从约 `-0.077 m`
拉近，同时保持 death 0。

## v65 airborne reset 0.20 m eval-time compensation sweep

对 0.20 m 最佳 checkpoint：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_17-03-06/model_119.pt
```

做 eval-time `landing_compensation` sweep，评估协议继续保留：

```text
0.0-0.20 m
--static_apex_reference
--baseline_compatible_stance_reference
--start_from_random_drop --initial_drop_height_min 0.8 --initial_drop_height_max 1.5
--action_scale 1.00
```

结果：

```text
landing_compensation=0.03:
  hit = 0.3026
  touchdown_error = 0.1531 m
  along_error = -0.0757 m
  lateral_abs_error = 0.0713 m
  apex = 0.9963 m
  max_consecutive_hits = 2.4219
  deploy success = 1.0000
  deploy death = 0.0000

landing_compensation=0.04:
  hit = 0.3355
  touchdown_error = 0.1430 m
  along_error = -0.0765 m
  lateral_abs_error = 0.0682 m
  apex = 1.0023 m
  max_consecutive_hits = 2.6289
  deploy success = 1.0000
  deploy death = 0.0000

landing_compensation=0.05:
  hit = 0.3295
  touchdown_error = 0.1463 m
  along_error = -0.0652 m
  lateral_abs_error = 0.0718 m
  apex = 1.0008 m
  max_consecutive_hits = 2.5508
  deploy success = 1.0000
  deploy death = 0.0000
```

结论：

```text
0.20 m + airborne reset 可以稳定使用 eval-time compensation。
0.04 是当前最佳折中：hit 最高、touchdown error 最低、连续命中最高，death 仍为 0。
0.05 虽然 along error 更小，但 lateral/error/hit 略差。
```

当前 0.20 m 可视化/候选配置：

```text
model = 17-03-06/model_119.pt
landing_compensation = 0.04
start_from_random_drop = true
```

下一步建议先可视化该组合；若视觉上稳定，再从同一个 `model_119.pt`
扩到 `0.0-0.25 m`，训练时仍保留 airborne reset。是否把
`landing_compensation=0.04` 放进训练需要谨慎；更稳的做法是先无 compensation
训练 0.25 m，再做 eval-time compensation sweep。

## v65 0.20 m trajectory-mix 长训结果

用户反馈：0.20 m 能跳稳，但追点效果很差，感觉没有跟随轨迹。

代码复查发现：

```text
static_apex_finetune 默认并不是立即追 full trajectory。
env 中 planner_weight:
  0-100 equivalent iterations: 基本 pure static apex
  100-400 equivalent iterations: 线性混入 full planner trajectory
  400 后: full planner trajectory
```

因此之前 120 轮训练多数时间仍然像“稳定原地跳 + 弱落点偏置”，
追轨迹感弱是预期内的。随后尝试在 0.20 m 上长训 300 轮，并提高 XY/landing dense reward：

```text
safety_xy_scale = 1.20
safety_xy_dense_scale = 0.70
```

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_120_xyd_070_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_17-21-34
```

训练后段日志：

```text
Mean reward = -533.72
episode length = 14061.33
target_hit_rate = 0.0909
target_hit_rate_ema = 0.1459
touchdown_error_ema = 0.2127 m
mean_cycle_apex_height = 0.8091 m
touchdown_attitude_error = 0.1161 rad
successful_waypoints = 14.0
termination_p = 0.0000
```

direct/deploy 评估协议：`0.0-0.20 m` short/long、`action_scale=1.00`、
`--static_apex_reference --baseline_compatible_stance_reference --start_from_random_drop`
和 `initial_drop_height=0.8-1.5`。

```text
model_120:
  hit = 0.2934
  touchdown_error = 0.1546 m
  along_error = -0.0874 m
  lateral_abs_error = 0.0705 m
  apex = 0.9922 m
  max_consecutive_hits = 2.3477
  deploy success = 1.0000
  deploy death = 0.0000

model_180:
  hit = 0.2843
  touchdown_error = 0.1596 m
  along_error = -0.0852 m
  lateral_abs_error = 0.0752 m
  apex = 0.9923 m
  max_consecutive_hits = 2.2812
  deploy success = 1.0000
  deploy death = 0.0000

model_240:
  hit = 0.2827
  touchdown_error = 0.1550 m
  along_error = -0.0950 m
  lateral_abs_error = 0.0664 m
  apex = 0.9919 m
  max_consecutive_hits = 2.2227
  deploy success = 1.0000
  deploy death = 0.0000

model_299:
  hit = 0.2670
  touchdown_error = 0.1619 m
  along_error = -0.0920 m
  lateral_abs_error = 0.0700 m
  apex = 0.9802 m
  max_consecutive_hits = 2.1406
  deploy success = 1.0000
  deploy death = 0.0000
```

对比上一轮 0.20 m 最佳：

```text
17-03-06/model_119 + eval-time landing_compensation=0.04:
  hit = 0.3355
  touchdown_error = 0.1430 m
  apex = 1.0023 m
  deploy death = 0.0000
```

结论：

```text
长训 + 更高 XY reward + 默认 100-400 trajectory blend 没有提升追点。
后段越训越退，model_120 已经不如旧候选，model_299 更差。
所以问题不只是 target reward 太小，也不只是训练轮数不够；
更像是 trajectory reference 混入太晚/太弱，或 static-apex 接口对水平追点的控制 authority 不足。
```

已新增训练参数：

```text
--static_apex_curriculum_iterations
--full_planner_curriculum_iterations
```

用于直接控制从 pure static apex 到 full planner reference 的混入速度。

下一步建议试更早、更明确的 trajectory blend：

```text
static_apex_curriculum_iterations = 20
full_planner_curriculum_iterations = 120
```

仍从稳定的 `17-03-06/model_119.pt` 起步，保持 0.20 m 和 airborne reset。

## 2026-08-31 追点能力下一步

20->120 curriculum 的后续评估没有超过旧最佳：

```text
static-apex eval, 0.20 m:
  model_40  hit ~= 0.282, touchdown_error ~= 0.159 m, apex ~= 0.998 m, death = 0
  model_80  hit ~= 0.272, touchdown_error ~= 0.163 m, apex ~= 0.992 m, death = 0
  model_120 hit ~= 0.275, touchdown_error ~= 0.160 m, apex ~= 0.989 m, death = 0

full-planner eval, model_159:
  hit ~= 0.021
  apex ~= 0.639 m
  death ~= 0.262
```

结论：不是简单“目标点奖励太小”。pure static-apex 很稳但追点 authority 弱；
full planner 参考直接启用会破坏高度和姿态，不能部署。

已加中间档参数：

```text
--planner_reference_blend FLOAT
```

语义：固定混合 `0.0 = stationary apex`，`1.0 = full planner trajectory`。
该参数已接入 `play_planner_circular.py`、`train_planner_circular.py` 和
`PlannerCircularEnvCfg.planner_reference_blend`，并通过 `python -m py_compile`。

当前接受的对照最佳仍是：

```text
checkpoint:
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_17-03-06/model_119.pt
```

上面路径缺少 `xyw_080_xyd_035` 字段，正确 checkpoint 是：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_attw_115_angw_130_xyw_080_xyd_035_yaww_200_heightw_105_hg088_la180_air_055_yawair_spring_mass_100_100_fixed_100/2026-08-31_17-03-06/model_119.pt

eval flags:
--baseline_compatible_stance_reference
--start_from_random_drop --initial_drop_height_min 0.8 --initial_drop_height_max 1.5
--landing_compensation 0.04

0.20 m result:
hit ~= 0.3355, touchdown_error ~= 0.1430 m, apex ~= 1.002 m, death = 0
```

下一步先不要训练，先 eval-time sweep：

```text
planner_reference_blend = 0.05, 0.10, 0.15, 0.20
```

接受条件：

```text
deploy death = 0
last_apex_height_m >= 0.95
target_hit_rate > 0.335
touchdown_error_m <= 0.143 m, 或至少明显降低 along/lateral 某一项且不损害稳定
```

如果 sweep 不提升，说明 trajectory-reference 这条路暂时不行，下一步应转向
stance/impulse-level 的落点偏置，而不是继续加 full trajectory 或单纯加大奖励。

## 2026-08-31 partial planner-reference sweep

使用旧最佳 checkpoint：

```text
17-03-06/model_119.pt
landing_compensation = 0.04
random drop = 0.8-1.5 m
distance = 0.0-0.20 m
```

eval-time `--planner_reference_blend` 结果：

```text
blend 0.05:
  hit = 0.3228
  touchdown_error = 0.1439 m
  short/long hit = 0.3322 / 0.3123
  apex = 0.9888 m
  deploy death = 0

blend 0.10:
  hit = 0.3287
  touchdown_error = 0.1424 m
  short/long hit = 0.3343 / 0.3225
  apex = 0.9727 m
  deploy death = 0

blend 0.15:
  hit = 0.3267
  touchdown_error = 0.1389 m
  short/long hit = 0.3364 / 0.3160
  apex = 0.9650 m
  deploy death = 0

blend 0.20:
  hit = 0.3348
  touchdown_error = 0.1353 m
  short/long hit = 0.3490 / 0.3192
  apex = 0.9510 m
  deploy death = 0
```

结论：

```text
partial trajectory reference 有效降低 touchdown error；
但 hit rate 仍基本没有超过旧最佳 0.3355，且 blend=0.20 已把 apex 拉到 0.951 m。
不能再继续加 blend，否则很可能重演 full-planner 低跳/tilt failure。
```

下一步建议：

```text
只把 blend=0.15 或 0.20 当训练探针。
优先选 0.15 更稳，0.20 追点稍好但高度余量小。
如果训练后 apex 低于 0.95 或 deploy death > 0，立刻停这条线。

## 2026-08-31 blend=0.05 training probe failed

训练命令使用：

```text
--planner_reference_blend 0.05
--safety_height_scale 1.50
--static_apex_min_valid_apex 0.92
--static_apex_low_apex_penalty 320
```

训练末尾 iteration 59/60：

```text
target_hit_rate = 0.1091
target_hit_rate_ema = 0.1348
touchdown_error_ema = 0.2017 m
mean_cycle_apex_height_m = 0.1208 m
episode_touchdown_error = 0.2785 m
touchdown_attitude_error = 0.1640 rad
low_apex_touchdown = -1.3335
apex_shortfall = -4.7332
apex_error = -9.2226
```

结论：

```text
fixed blend training 失败。即使 blend=0.05 且加重高度惩罚，策略仍然退化到低跳/贴地追点。
不要继续 blend 训练，不要部署该 run。
```

下一步方向：

```text
不要再从 full trajectory/blend 方向压 PPO。
应改 reward 结构：低于有效 apex 的 touchdown 不能获得 target_hit、landing_precision、
prepared_landing、streak_progress 等任何正向落点奖励；低 apex touchdown penalty 也应
按 touchdown event 强惩罚，而不是被 episode averaging 稀释。

或者改接口：保留 baseline static apex，仅在 stance / takeoff 阶段加入低频水平落点偏置，
不要让空中 reference 追 time-parameterized trajectory。

## 2026-08-31 reward gate patch

为阻止低跳/贴地追点，已修改 `Quadhopper_Planner_Circular/planner_circular_env.py`：

```text
新增 cfg:
  gate_dense_xy_rewards_by_apex = False
  low_apex_touchdown_event_penalty_scale = 0.0

static-apex finetune 中启用:
  gate_dense_xy_rewards_by_apex = True
  low_apex_touchdown_event_penalty_scale = -80.0 * height_scale
```

具体行为：

```text
planner_xy 正奖励乘 apex_reward_gate
projected_landing 正奖励乘 apex_reward_gate
landing_precision 继续由 valid_apex_touchdown_event gate
target_hit 本来已经要求 valid_apex
prepared_landing / streak_progress 额外乘 touchdown_reward_gate
low_apex_touchdown = shortfall penalty + invalid-apex-touchdown 固定事件罚分
```

保留 `planner_z` 不 gate，因为它本身提供高度追踪信号。

静态检查：

```text
python -m py_compile train_planner_circular.py play_planner_circular.py Quadhopper_Planner_Circular/planner_circular_env.py
passed
```

下一步建议先从旧最佳 checkpoint 训练 pure static-apex gate 版本，不加 blend；
先验证 gate 能否保住高度和原有稳定性，再考虑加 `--planner_reference_blend 0.05`。

## 2026-08-31 first reward-gate train failed

pure static-apex gate 版本训练 80 轮末尾：

```text
target_hit_rate = 0.1333
target_hit_rate_ema = 0.1166
touchdown_error_ema = 0.2112 m
mean_cycle_apex_height_m = 0.4161 m
touchdown_attitude_error = 0.1511 rad
low_apex_touchdown = -7.1349
planner_position = 1.6848
planner_velocity = 0.3253
planner_xy = 0.2124
planner_z = 2.1940
goal_bonus = 3.1614
```

结论：

```text
第一版 gate 有效果但不够。apex 从 blend=0.05 训练的 0.1208 m 回到 0.4161 m，
但仍远低于 1.0 m，且 planner_position/planner_velocity 仍提供正向 tracking 奖励。
```

已追加第二版 gate patch：

```text
planner_position 正奖励也乘 apex_reward_gate
planner_velocity 正奖励也乘 apex_reward_gate
low_apex_touchdown_event_penalty_scale 从 -80.0 * height_scale 加重到 -240.0 * height_scale
```

静态检查通过：

```text
python -m py_compile train_planner_circular.py play_planner_circular.py Quadhopper_Planner_Circular/planner_circular_env.py
```

下一步继续从旧最佳 checkpoint 训练 pure static-apex，不加 blend，先看 `mean_cycle_apex_height_m`
能否回到至少 0.8 m 以上。

## 2026-08-31 parent XY rewards disabled: still failed

关掉 inherited stable reward 中的：

```text
distance_to_xy_reward_scale = 0
xy_progress_reward_scale = 0
goal_bonus_scale = 0
```

训练 80 轮末尾：

```text
distance_to_xy = 0
xy_progress = 0
goal_bonus = 0
target_hit_rate = 0.1538
target_hit_rate_ema = 0.1130
touchdown_error_ema = 0.2217 m
mean_cycle_apex_height_m = 0.2001 m
low_apex_touchdown = -12.6897
planner_position = 0.7738
planner_velocity = 0.1520
planner_xy = 0.1072
planner_z = 1.0820
```

结论：

```text
父类 XY/goal 正奖励不是唯一问题。即使关掉这些奖励，旧 43D policy 在 static-apex
继续 PPO 微调时仍会快速破坏高跳模式。
这条 reward-only fine-tune 方向不要继续长训。
```

下一步先筛该 run 的早期 checkpoint：

```text
model_5, model_10, model_20, model_40
```

如果早期 checkpoint 也没有超过旧最佳，则停止 43D policy 继续 PPO 微调，改成结构性方案：

```text
冻结/保留 baseline 高跳 policy；
不要让 PPO 继续改坏起跳高度；
在 stance/takeoff 层面学习或注入低频 XY 落点偏置。
```

## 2026-08-31 gate run early checkpoint eval

第二版 gate run 的早期 checkpoint 静态评估：

```text
model_5:
  hit = 0.3327
  touchdown_error = 0.1434 m
  short/long hit = 0.3409 / 0.3236
  apex = 1.0016 m
  deploy death = 0

model_10:
  hit = 0.3218
  touchdown_error = 0.1453 m
  apex = 0.9986 m
  deploy death = 0

model_20:
  hit = 0.3187
  touchdown_error = 0.1469 m
  apex = 1.0016 m
  deploy death = 0

model_40:
  hit = 0.3224
  touchdown_error = 0.1459 m
  apex = 0.9986 m
  deploy death = 0

model_79:
  hit = 0.3217
  touchdown_error = 0.1492 m
  apex = 1.0032 m
  deploy death = 0
```

对比旧最佳：

```text
17-03-06/model_119 + landing_compensation=0.04:
  hit ~= 0.3355
  touchdown_error ~= 0.1430 m
  apex ~= 1.002 m
  deploy death = 0
```

结论：

```text
早期 checkpoint 可以保住高度，但没有提升追点；后期训练会坍到低跳。
reward gate 修补不足以突破旧最佳。停止这条 43D PPO 微调路线。
```

## 2026-08-31 takeoff XY bias patch

开始尝试结构性方案 B：不让空中追 full trajectory，只在 stance/takeoff 阶段给水平瞄准偏置。

新增参数：

```text
--takeoff_xy_bias_gain
--takeoff_xy_bias_max
```

接入文件：

```text
Quadhopper_Planner_Circular/planner_circular_env.py
play_planner_circular.py
train_planner_circular.py
```

实现语义：

```text
start_xy = commands.start_points()
p_t = current waypoint
stance desired xy = p_t + clamp((p_t - start_xy) * gain, max=takeoff_xy_bias_max)
```

只影响 stance / contact phase 的 desired XY；真实 waypoint 和命中判定不变。
空中仍可以用 `--static_apex_reference` 追 stationary apex，不启用 full trajectory。

静态检查：

```text
python -m py_compile train_planner_circular.py play_planner_circular.py Quadhopper_Planner_Circular/planner_circular_env.py
passed
```

下一步先 eval-time sweep 旧最佳 checkpoint，不训练。目标是补偿旧最佳约 `-0.075 m`
的 along undershoot，同时保持：

```text
death = 0
apex >= 0.95 m
hit > 0.3355
touchdown_error < 0.143 m
```

## 2026-08-31 takeoff XY bias eval sweep

旧最佳 checkpoint + static-apex + baseline-compatible stance + random drop +
`landing_compensation=0.04`，0.0-0.20 m random two-hop。

```text
gain 0.20, max 0.08:
  hit = 0.3330
  touchdown_error = 0.1466 m
  along_error = -0.0672 m
  lateral_abs_error = 0.0707 m
  apex = 1.0007 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3900 m

gain 0.35, max 0.08:
  hit = 0.3359
  touchdown_error = 0.1441 m
  along_error = -0.0776 m
  lateral_abs_error = 0.0671 m
  apex = 0.9993 m
  deploy death = 0
  deploy max_xy_error_mean = 0.4025 m

gain 0.50, max 0.08:
  hit = 0.3357
  touchdown_error = 0.1444 m
  along_error = -0.0705 m
  lateral_abs_error = 0.0687 m
  apex = 1.0015 m
  deploy death = 0
  deploy max_xy_error_mean = 0.4149 m

gain 0.75, max 0.08:
  hit = 0.3310
  touchdown_error = 0.1447 m
  along_error = -0.0716 m
  lateral_abs_error = 0.0704 m
  apex = 0.9957 m
  deploy death = 0
  deploy max_xy_error_mean = 0.4298 m
```

对比旧最佳：

```text
hit ~= 0.3355
touchdown_error ~= 0.1430 m
apex ~= 1.002 m
deploy death = 0
deploy max_xy_error_mean ~= 0.357 m
```

结论：

```text
takeoff_xy_bias 很安全，保住高度和 death=0；
但对追点提升只有噪声级，且 deploy max_xy_error_mean 变差。
不要把它作为新默认部署参数。可保留代码开关备用。
```

## 2026-08-31 landing compensation sweep

旧最佳 checkpoint + static-apex + baseline-compatible stance + random drop，
0.0-0.20 m random two-hop，不加 takeoff bias。

```text
comp 0.00:
  hit = 0.2971
  touchdown_error = 0.1564 m
  along_error = -0.0773 m
  lateral_abs_error = 0.0718 m
  apex = 0.9983 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3740 m

comp 0.02:
  hit = 0.3030
  touchdown_error = 0.1498 m
  along_error = -0.0808 m
  lateral_abs_error = 0.0715 m
  apex = 1.0023 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3685 m

comp 0.04:
  hit = 0.3355
  touchdown_error = 0.1430 m
  along_error = -0.0765 m
  lateral_abs_error = 0.0682 m
  apex = 1.0023 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3570 m

comp 0.06:
  hit = 0.3386
  touchdown_error = 0.1419 m
  along_error = -0.0651 m
  lateral_abs_error = 0.0698 m
  apex = 0.9979 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3588 m

comp 0.08:
  hit = 0.3647
  touchdown_error = 0.1364 m
  along_error = -0.0619 m
  lateral_abs_error = 0.0672 m
  apex = 1.0014 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3425 m

comp 0.10:
  hit = 0.3672
  touchdown_error = 0.1364 m
  along_error = -0.0532 m
  lateral_abs_error = 0.0698 m
  apex = 1.0034 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3455 m
```

结论：

```text
landing_compensation 是目前真正有效的准度提升方向。
comp=0.08 和 0.10 都显著超过旧最佳 0.04；
0.10 hit 最高，0.08 deploy max_xy_error 更低且略保守。
建议新默认候选用 comp=0.08，激进候选用 comp=0.10。
```

## 2026-08-31 landing compensation fine sweep

继续细扫 `landing_compensation=0.08-0.24`，仍使用旧最佳 checkpoint、
static-apex、baseline-compatible stance、random drop、0.0-0.20 m random two-hop。

```text
comp 0.08:
  hit = 0.3647
  touchdown_error = 0.1364 m
  along_error = -0.0619 m
  lateral_abs_error = 0.0672 m
  next_velocity_error = 0.2578 m/s
  apex = 1.0014 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3425 m

comp 0.09:
  hit = 0.3510
  touchdown_error = 0.1367 m
  along_error = -0.0573 m
  lateral_abs_error = 0.0698 m
  next_velocity_error = 0.2657 m/s
  apex = 1.0020 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3420 m

comp 0.10:
  hit = 0.3672
  touchdown_error = 0.1364 m
  along_error = -0.0532 m
  lateral_abs_error = 0.0698 m
  next_velocity_error = 0.2767 m/s
  apex = 1.0034 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3455 m

comp 0.11:
  hit = 0.3737
  touchdown_error = 0.1326 m
  along_error = -0.0522 m
  lateral_abs_error = 0.0691 m
  next_velocity_error = 0.2848 m/s
  apex = 1.0024 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3407 m

comp 0.12:
  hit = 0.3783
  touchdown_error = 0.1311 m
  along_error = -0.0514 m
  lateral_abs_error = 0.0667 m
  next_velocity_error = 0.2957 m/s
  apex = 1.0068 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3351 m

comp 0.14:
  hit = 0.3951
  touchdown_error = 0.1303 m
  along_error = -0.0452 m
  lateral_abs_error = 0.0700 m
  next_velocity_error = 0.3163 m/s
  apex = 1.0018 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3391 m

comp 0.16:
  hit = 0.3949
  touchdown_error = 0.1273 m
  along_error = -0.0410 m
  lateral_abs_error = 0.0690 m
  next_velocity_error = 0.3375 m/s
  apex = 1.0047 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3336 m

comp 0.18:
  hit = 0.3868
  touchdown_error = 0.1282 m
  along_error = -0.0393 m
  lateral_abs_error = 0.0702 m
  next_velocity_error = 0.3606 m/s
  apex = 1.0031 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3368 m

comp 0.20:
  hit = 0.4084
  touchdown_error = 0.1240 m
  along_error = -0.0338 m
  lateral_abs_error = 0.0692 m
  next_velocity_error = 0.3806 m/s
  apex = 1.0056 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3321 m

comp 0.24:
  hit = 0.3932
  touchdown_error = 0.1256 m
  along_error = -0.0236 m
  lateral_abs_error = 0.0729 m
  next_velocity_error = 0.4236 m/s
  apex = 1.0066 m
  deploy death = 0
  deploy max_xy_error_mean = 0.3309 m
```

结论：

```text
comp=0.20 是当前仿真准度最佳候选：
  hit 从旧最佳 0.3355 提升到 0.4084
  touchdown_error 从 0.1430 m 降到 0.1240 m
  apex 仍约 1.0 m
  deploy death = 0

comp=0.24 已开始过补偿/落地质量变差：
  hit 回落到 0.3932
  prepared_landing_rate 明显下降
  next_velocity_error 升到 0.4236 m/s

部署保守候选可用 comp=0.16；
仿真准度候选/下一轮验证用 comp=0.20。
```

## 2026-08-31 compensation multi-seed validation

多 seed 随机 waypoint 验证：`seed = 7, 42, 101, 202, 303`。

配置：

```text
旧最佳 checkpoint 17-03-06/model_119.pt
static-apex reference
baseline-compatible stance reference
random drop 0.8-1.5 m
distance 0.0-0.20 m
action_scale = 1.0
```

`landing_compensation=0.16`:

```text
seed 7:   hit 0.4065, error 0.1258 m, apex 1.0039, death 0
seed 42:  hit 0.3949, error 0.1273 m, apex 1.0047, death 0
seed 101: hit 0.3862, error 0.1299 m, apex 1.0098, death 0
seed 202: hit 0.3922, error 0.1267 m, apex 1.0001, death 0
seed 303: hit 0.3868, error 0.1278 m, apex 1.0070, death 0

mean hit ~= 0.3933
mean touchdown_error ~= 0.1275 m
mean next_velocity_error ~= 0.3356 m/s
mean deploy max_xy_error ~= 0.3292 m
```

`landing_compensation=0.20`:

```text
seed 7:   hit 0.3998, error 0.1252 m, apex 0.9999, death 0
seed 42:  hit 0.4084, error 0.1240 m, apex 1.0056, death 0
seed 101: hit 0.3994, error 0.1257 m, apex 1.0045, death 0
seed 202: hit 0.4009, error 0.1240 m, apex 1.0032, death 0
seed 303: hit 0.4213, error 0.1225 m, apex 1.0080, death 0

mean hit ~= 0.4059
mean touchdown_error ~= 0.1243 m
mean next_velocity_error ~= 0.3782 m/s
mean deploy max_xy_error ~= 0.3278 m
```

结论：

```text
comp=0.20 是当前最佳多 seed 候选：
  平均 hit 比 comp=0.16 高约 1.3 个百分点
  平均 touchdown error 低约 3.2 mm
  apex 稳定在 1.0 m
  deploy death 全 0
  deploy max_xy_error 平均也略好

代价：
  next_velocity_error 从约 0.336 m/s 增到约 0.378 m/s。

当前推荐：
  仿真/短距离 0.0-0.20 m 默认用 comp=0.20
  若真实机担心落地速度/下一跳准备，保守用 comp=0.16
```

## 2026-08-31 longer-distance diagnosis

验证用户判断：“姿态锁太严重，距离远一点无法在空中倾斜/产生足够水平位移。”

配置：

```text
旧最佳 checkpoint 17-03-06/model_119.pt
static-apex reference
baseline-compatible stance reference
random drop 0.8-1.5 m
landing_compensation = 0.20
action_scale = 1.0
seed = default 42
```

结果：

```text
radius 0.20-0.30:
  hit = 0.0767
  touchdown_error = 0.2598 m
  along_error = -0.1510 m
  lateral_abs_error = 0.1316 m
  touchdown_attitude_error = 0.0838 rad
  next_velocity_error = 0.5764 m/s
  apex = 0.9923 m
  deploy death = 0

radius 0.30-0.40:
  hit = 0.0365
  touchdown_error = 0.3557 m
  along_error = -0.2532 m
  lateral_abs_error = 0.1532 m
  touchdown_attitude_error = 0.0948 rad
  next_velocity_error = 0.7282 m/s
  apex = 0.9904 m
  deploy death = 0

radius 0.40-0.50:
  hit = 0.0219
  touchdown_error = 0.4565 m
  along_error = -0.3646 m
  lateral_abs_error = 0.1717 m
  touchdown_attitude_error = 0.1020 rad
  next_velocity_error = 0.8858 m/s
  apex = 0.9730 m
  deploy death = 0.0156
```

结论：

```text
用户判断成立。远距离失败不是跳不高，apex 仍约 1 m；
主要是沿目标方向系统性欠跳，且 touchdown attitude error 仍很小。
这说明当前 static-apex 稳定策略姿态/水平速度 authority 不足：
它会保持机身很正，但不愿或不会用足够倾斜换水平位移。

下一步若要提升 0.2m 以上距离，不能再靠 landing_compensation；
需要训练/控制一个 phase-dependent tilt/velocity preparation：
  起跳和上升早期允许朝目标方向倾斜/产生水平速度；
  下降和 touchdown 前强制收正。
```

## 2026-08-31 phase-tilt finetune patch

针对“姿态锁太严重，远距离沿目标方向欠跳”的诊断，新增分阶段起跳倾斜训练项。

代码开关：

```text
train_planner_circular.py:
  --phase_tilt_finetune
  --takeoff_tilt_deg
  --takeoff_tilt_phase_end
  --takeoff_tilt_reward_scale

play_planner_circular.py:
  同名参数，用于 eval 时记录 reward/debug 项；旧 checkpoint 行为不会因此改变。
```

环境配置：

```text
PlannerCircularEnvCfg.takeoff_tilt_rad
PlannerCircularEnvCfg.takeoff_tilt_phase_end
PlannerCircularEnvCfg.takeoff_tilt_min_height
PlannerCircularEnvCfg.takeoff_tilt_reward_scale
PlannerCircularEnvCfg.takeoff_tilt_penalty_scale
```

reward 逻辑：

```text
takeoff_tilt_mask =
  cycle active
  root z velocity > 0.05
  flight phase <= takeoff_tilt_phase_end
  root z >= takeoff_tilt_min_height

在该 mask 内：
  奖励 body z-axis 朝目标方向倾斜 takeoff_tilt_deg
  惩罚 takeoff_tilt_error
  暂时关闭 airborne upright attitude penalty

在下降/触地阶段：
  原 touchdown attitude / prepared landing / tilt barrier 仍然生效，
  不允许一路歪着落地。
```

默认建议只短训测试，不要直接长训：

```text
checkpoint = 当前 static-apex 最稳短距模型 17-03-06/model_119.pt
训练距离 = 0.10-0.30 m
landing_compensation = 0.20
takeoff_tilt_deg = 8
takeoff_tilt_phase_end = 0.40
takeoff_tilt_reward_scale = 0.75
iterations = 60
```

筛选标准：

```text
先看 model_5 / model_10 / model_20 / model_40 / model_59。
不要默认相信最后模型。

0.20-0.30 m eval 目标：
  apex 仍 >= 0.90 m，最好 0.95-1.05 m
  death_rate = 0
  touchdown_attitude_error 不明显超过 0.10 rad
  along_error 绝对值显著小于旧值 0.151 m
  direct target_hit_rate 显著超过旧值 0.077

如果 hit 没上升但 attitude/tilt death 变差，说明倾斜奖励太强；
降低 takeoff_tilt_deg 到 5 或 reward_scale 到 0.4。
如果姿态仍几乎不倾、along_error 仍大，提升 takeoff_tilt_deg 到 10 或 reward_scale 到 1.0。
```

## 2026-09-01 conservative phase-tilt eval result

保守 phase-tilt run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100__ptilt_04_025lcg_000_attw_120_angw_140_xyw_080_xyd_035_yaww_200_heightw_200_hg095_la500_air_065_yawair_spring_mass_100_100_fixed_100/2026-09-01_09-54-44
```

训练配置要点：

```text
distance = 0.00-0.20 m
landing_compensation = 0.20
takeoff_tilt_deg = 4
takeoff_tilt_phase_end = 0.25
takeoff_tilt_reward_scale = 0.25
static_apex_min_valid_apex = 0.95
static_apex_low_apex_penalty = 500
safety_height_scale = 2.0
```

挑战区 `0.20-0.30 m` 评估：

```text
model_5:
  hit = 0.0784
  touchdown_error = 0.2595 m
  along_error = -0.1539 m
  lateral_abs_error = 0.1319 m
  apex = 0.9893 m
  death = 0.0039

model_10:
  hit = 0.0723
  touchdown_error = 0.2584 m
  along_error = -0.1462 m
  lateral_abs_error = 0.1335 m
  apex = 0.9986 m
  death = 0

model_20:
  hit = 0.0768
  touchdown_error = 0.2571 m
  along_error = -0.1521 m
  lateral_abs_error = 0.1288 m
  apex = 0.9986 m
  death = 0

model_39:
  hit = 0.0743
  touchdown_error = 0.2564 m
  along_error = -0.1505 m
  lateral_abs_error = 0.1299 m
  apex = 0.9954 m
  death = 0
```

旧模型 `17-03-06/model_119.pt` 在同一区间的基准：

```text
hit = 0.0767
touchdown_error = 0.2598 m
along_error = -0.1510 m
apex = 0.9923 m
death = 0
```

结论：

```text
保守 phase-tilt 守住了高度，但没有改善 0.20-0.30 m 欠跳。
沿目标方向误差仍约 -0.15 m，说明 policy 没获得额外水平速度/位移能力。
单纯加 early takeoff attitude reward 不够。
```

稳区 `0.00-0.20 m` 评估：

```text
model_10:
  hit = 0.3941
  touchdown_error = 0.1260 m
  along_error = -0.0313 m
  lateral_abs_error = 0.0720 m
  apex = 1.0006 m
  death = 0

model_20:
  hit = 0.3808
  touchdown_error = 0.1301 m
  along_error = -0.0313 m
  lateral_abs_error = 0.0742 m
  apex = 1.0019 m
  death = 0

model_39:
  hit = 0.3931
  touchdown_error = 0.1278 m
  along_error = -0.0311 m
  lateral_abs_error = 0.0718 m
  apex = 1.0014 m
  death = 0
```

旧多 seed 短距最佳约：

```text
hit = 0.4059
touchdown_error = 0.1243 m
apex ~= 1.0 m
death = 0
```

结论：

```text
保守 phase-tilt 没有毁掉短距稳定性，但也略低于旧最佳。
当前不要部署/采用 ptilt_04_025 作为新主模型。
继续使用旧 17-03-06/model_119.pt + landing_compensation=0.20 作为短距候选。
```

下一步方向：

```text
问题不是“不会稍微倾斜”，而是没有形成可控的水平速度/支撑期推离能力。
下一步应从动作/观测/奖励上引入 takeoff horizontal velocity 或 foothold-style 支撑期目标，
而不是继续调 takeoff_tilt reward。
```

## 2026-09-01 phase-takeoff finetune patch

针对保守 phase-tilt 无效的问题，新增“分段起跳”训练开关：

```text
--phase_takeoff_finetune
```

核心变化：

```text
起跳/上升早期：
  使用 takeoff_tilt_mask 作为 phase mask
  按比例放松 airborne_attitude penalty
  按比例放松 airborne_angvel_xy penalty
  按比例放松 tilt_barrier penalty
  奖励 root_vel_xy 在目标方向上的投影速度接近 takeoff_velocity_target

下降/触地：
  原 touchdown_attitude_error / prepared_landing / tilt_barrier 仍生效
  不允许歪着落地
```

新增环境配置：

```text
takeoff_phase_attitude_relax
takeoff_phase_angvel_relax
takeoff_phase_tilt_barrier_relax
takeoff_velocity_target_mps
takeoff_velocity_width_mps
takeoff_velocity_reward_scale
takeoff_velocity_penalty_scale
```

新增训练/评估 CLI：

```text
--phase_takeoff_finetune
--takeoff_velocity_target
--takeoff_velocity_reward_scale
--takeoff_phase_attitude_relax
--takeoff_phase_angvel_relax
--takeoff_phase_tilt_barrier_relax
```

新增 reward 日志：

```text
Episode_Reward/takeoff_velocity
Episode_Reward/takeoff_velocity_error
```

建议第一条短训：

```text
distance = 0.00-0.25 m
takeoff_velocity_target = 0.30 m/s
takeoff_velocity_reward_scale = 0.60
takeoff_tilt_deg = 5
takeoff_tilt_reward_scale = 0.15
takeoff_tilt_phase_end = 0.30
relax attitude/angvel/tilt_barrier = 0.45 / 0.30 / 0.50
iterations = 50
```

判据：

```text
如果 0.20-0.30m 的 along_error 从 -0.15m 改到 -0.10m 以内，
且 apex >= 0.95m、death=0、touchdown_attitude_error <= 0.10rad，
说明分段起跳方向有效。

如果 along_error 不变，但 takeoff_velocity reward 有明显值，
说明速度目标可能太低或 phase mask 太短。

如果 apex 塌或 touchdown_attitude_error 明显上升，
说明 relax/速度 reward 太强。
```

## 2026-09-01 phase-takeoff eval result

phase-takeoff run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100__ptakeoff_030_060lcg_000_attw_120_angw_140_xyw_080_xyd_035_yaww_200_heightw_200_hg095_la500_air_065_yawair_spring_mass_100_100_fixed_100/2026-09-01_10-17-59
```

训练末尾 `49/50` 的 episode 窗口为空，不能看全 0 项；EMA 为：

```text
touchdown_error_ema = 0.1843 m
target_hit_rate_ema = 0.1017
prepared_landing_rate_ema = 0.0716
```

`0.20-0.30 m` direct/deploy eval：

```text
model_0:
  hit = 0.0749
  error = 0.2564 m
  along_error = -0.1522 m
  lateral_abs_error = 0.1256 m
  apex = 0.9984 m
  death = 0

model_5:
  hit = 0.0754
  error = 0.2606 m
  along_error = -0.1531 m
  lateral_abs_error = 0.1338 m
  apex = 0.9970 m
  death = 0

model_10:
  hit = 0.0793
  error = 0.2571 m
  along_error = -0.1487 m
  lateral_abs_error = 0.1299 m
  apex = 0.9931 m
  death = 0

model_20:
  hit = 0.0716
  error = 0.2582 m
  along_error = -0.1523 m
  lateral_abs_error = 0.1292 m
  apex = 0.9936 m
  death = 0

model_49:
  hit = 0.0824
  error = 0.2567 m
  along_error = -0.1464 m
  lateral_abs_error = 0.1298 m
  apex = 1.0051 m
  death = 0
```

结论：

```text
phase-takeoff 安全性和高度守住了，但追点只从 hit ~= 0.077 到 0.082、
along_error 从 -0.151 m 到 -0.146 m，属于噪声级小改善，不是实质突破。

单靠 reward 调姿态/速度无法明显改变已有 stable policy 的水平位移能力。
下一步需要改变 action/reference/目标接口，例如支撑期目标偏置、落点补偿课程、
或显式把期望起跳水平速度作为 observation/command，而不是只放在 reward 里。
```

## 2026-09-01 takeoff XY bias curriculum patch

新增支撑期目标偏置课程：

```text
PlannerCircularEnvCfg.takeoff_xy_bias_curriculum_iterations
train_planner_circular.py --takeoff_xy_bias_curriculum_iterations
```

已有参数仍表示最终值：

```text
--takeoff_xy_bias_gain
--takeoff_xy_bias_max
```

实现：

```text
effective_bias_fraction = clamp(equivalent_iteration / takeoff_xy_bias_curriculum_iterations, 0, 1)
effective_bias_gain = takeoff_xy_bias_gain * effective_bias_fraction
effective_bias_max = takeoff_xy_bias_max * effective_bias_fraction
stance p_t = physical p_t + clipped((p_t - start_xy) * effective_bias_gain, effective_bias_max)
```

注意：

```text
这个 bias 只改变支撑期 inherited target/reference；
物理 waypoint / touchdown success target 不变。
所以它是 foothold-style command bias，不是把命中标准前移。
```

建议第一条训练：

```text
从旧最佳 17-03-06/model_119.pt 开始
distance = 0.00-0.25 m
landing_compensation = 0.20
takeoff_xy_bias_gain = 0.35
takeoff_xy_bias_max = 0.04 m
takeoff_xy_bias_curriculum_iterations = 35
iterations = 70
不同时开 phase_tilt/phase_takeoff
```

目标：

```text
先看是否能在 0.20-0.30m 将 along_error 从 -0.151m 推到 -0.10m 附近。
如果短距 0.00-0.20m 明显损坏，说明 bias 过大或 ramp 太快。
如果完全没变化，下一版把 max 提到 0.06m 或 gain 提到 0.50。
```

## 2026-09-01 takeoff XY bias curriculum eval result

run：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v65_custom_rel_static_base_stance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_tb_035_004_tbc_035_attw_120_angw_140_xyw_080_xyd_035_yaww_200_heightw_200_hg095_la500_air_065_yawair_spring_mass_100_100_fixed_100/2026-09-01_10-29-07
```

训练末尾：

```text
mean_cycle_apex_height = 0.3897 m
target_hit_rate = 0.1364
touchdown_error_ema = 0.1697 m
target_hit_rate_ema = 0.1638
low_apex_touchdown = -31.7826
```

末尾训练窗口显示低跳严重，但 eval 中 checkpoint 仍能保住 apex，说明训练日志和 direct eval 口径仍要分开看。

`0.20-0.30 m` eval，带最终 bias `gain=0.35 max=0.04`：

```text
model_0:
  hit = 0.0787
  error = 0.2566 m
  along = -0.1486 m
  apex = 0.9959 m
  death = 0
  deploy max_xy = 0.6646 m

model_10:
  hit = 0.0813
  error = 0.2551 m
  along = -0.1434 m
  apex = 0.9935 m
  death = 0
  deploy max_xy = 0.6658 m

model_20:
  hit = 0.0840
  error = 0.2568 m
  along = -0.1517 m
  apex = 0.9936 m
  death = 0
  deploy max_xy = 0.6771 m

model_35:
  hit = 0.0797
  error = 0.2534 m
  along = -0.1573 m
  apex = 0.9884 m
  death = 0
  deploy max_xy = 0.6672 m

model_50:
  hit = 0.0806
  error = 0.2553 m
  along = -0.1478 m
  apex = 0.9932 m
  death = 0
  deploy max_xy = 0.6764 m

model_69:
  hit = 0.0774
  error = 0.2549 m
  along = -0.1442 m
  apex = 0.9963 m
  death = 0
  deploy max_xy = 0.6700 m
```

旧模型同区间基准：

```text
hit ~= 0.0767
error ~= 0.2598 m
along ~= -0.1510 m
apex ~= 0.992 m
death = 0
deploy max_xy ~= 0.63-0.64 m in comparable biased tests; old short-distance deploy max_xy ~= 0.33 m
```

结论：

```text
bias curriculum 没有实质突破。
best hit 只到 0.084，along error 只改善到约 -0.143m，仍远离目标 -0.10m。
同时 deploy max_xy_error_mean 升到约 0.66-0.68m，说明支撑期目标前移会扩大轨迹偏差风险。

不建议采用该 run 作为新主模型。
```

## 2026-09-01 自适应顶点高度 planner

用户提出固定 `height_high=1.0m` 不适合路径优化：远一点的点应该允许更高/更长滞空轨迹，而原地/近距离跳仍保持 1m 稳定跳。已实现距离自适应 apex height：

```text
--adaptive_apex_height
--adaptive_height_min
--adaptive_height_max
--adaptive_height_distance_start
--adaptive_height_distance_end
--adaptive_height_noise
```

环境中 `_height_commands()` 现在可根据当前 hop 距离和下一跳距离生成 `current_height, next_height`：

```text
H(d) = adaptive_height_min + clamp((d - distance_start)/(distance_end - distance_start), 0, 1)
       * (adaptive_height_max - adaptive_height_min)
```

并且修正了 gated touchdown/dense XY reward 的 valid-apex threshold：`alternate_target_heights` 或 `adaptive_apex_height` 打开时都使用动态门槛：

```text
valid_apex_threshold = max(active_target_height - apex_tolerance, landing_root_height + 0.05)
```

日志新增：

```text
Metrics/command_apex_height_m      # mean over reset envs
Metrics/command_apex_height_min_m
Metrics/command_apex_height_max_m
```

编译检查已通过：

```text
python -m py_compile train_planner_circular.py play_planner_circular.py Quadhopper_Planner_Circular/planner_circular_env.py
```

下一步建议先跑固定 `0.24-0.26m` 的 adaptive-height skill probe，再 eval `0.24-0.26m` 和 `0.00-0.20m`：

```text
核心假设：
固定 1m apex 时，0.25m 目标需要太大的水平速度且策略陷入竖直跳局部最优。
把 0.25m command 映射到约 1.14-1.16m apex，能增加滞空时间，让同样姿态/速度限制下的水平落点更可达。
```

## 2026-09-01 adaptive height + fixed takeoff velocity 结果

固定 `0.24-0.26m`、adaptive height 到约 `1.146m` 的训练/eval 显示：

```text
command_apex_height_m ~= 1.146m
last_apex_height_m ~= 1.13-1.14m in eval
target_hit_rate ~= 0.075-0.079
touchdown_along_error ~= -0.19m
```

比旧固定 1m apex 的 `0.25m` 结果没有改善，甚至 along error 更短。结论：问题不是单纯飞行时间不够，而是策略没有把更高 apex 转换成水平起跳/水平位移。

随后固定 `takeoff_velocity_target=0.34`、持续 early-ascent velocity shaping 的训练结果：

```text
target_hit_rate = 0.0551
episode_touchdown_error = 0.3234m
mean_cycle_apex_height = 0.9795m
command_apex_height = 1.1461m
takeoff_velocity = 14.4872
takeoff_velocity_error = -4.5400
lateral_vel_p = -1.7119
landing_error = -50.0764
projected_landing_error = -0.8136
low_apex_touchdown = -28.5575
```

该 run 明显更差：持续速度塑形会和落点/稳定奖励拉扯，造成 lateral 和 projected landing 惩罚变大，并且不少周期 apex 仍不足。

已新增两个更保守的开关：

```text
--takeoff_velocity_from_planner   # 目标速度来自当前 planner 第一段水平速度
--takeoff_velocity_event_reward   # 只在 liftoff 瞬间给 takeoff velocity 奖励/惩罚
```

目的：把水平速度塑形从“整个起跳阶段持续催速度”改成“离地瞬间速度对齐”，降低过冲/横漂风险。

该版本训练后仍然坏：

```text
target_hit_rate = 0.0521
episode_touchdown_error = 0.3215m
mean_cycle_apex_height = 0.9760m
command_apex_height = 1.1463m
landing_error ~= -49.78
low_apex_touchdown ~= -32.96
```

结论：仅靠 reward shaping 仍没有学出水平冲量。下一步已实现 47-D observation：

```text
--planner_velocity_observation
```

在原 43-D observation 后追加 4 维：

```text
planner takeoff vxy in body frame
planner landing vxy in body frame
```

checkpoint migration 已支持 `target_obs_dim=47`，可从 37-D/42-D/43-D 扩展到 47-D，新增列零初始化。训练/eval 新 47-D 模型时必须都带 `--planner_velocity_observation`。

## 2026-09-01 takeoff impulse 子阶段

用户同意尝试分阶段。已新增：

```text
--takeoff_impulse_finetune
```

该 flag 是 `--static_apex_finetune` 的子模式，不是互斥大模式；必须与 `--static_apex_finetune` 一起使用。核心改动：

```text
takeoff_velocity_from_planner = True
takeoff_velocity_event_reward = True
```

只在 liftoff event 奖励 planner 第一段水平速度对齐，弱化 `target_miss` 和 `landing_error`，降低姿态/横向速度/投影落点惩罚，保留较强 apex/low-apex 约束。目的不是立刻提高 hit rate，而是先确认 `touchdown_along_error` 能从 `-0.22m` 明显向 `-0.10m` 或更小移动。

PPO 对该阶段也更主动：

```text
37-D baseline 起训: lr=3e-5, entropy=8e-5, fixed action std=0.05
非 37-D 起训: lr=2e-6, entropy=2e-5, fixed action std=0.015
```

编译通过：

```text
python -m py_compile train_planner_circular.py play_planner_circular.py Quadhopper_Planner_Circular/planner_circular_env.py Quadhopper_Planner_Circular/checkpoint_migration.py
```

## 2026-09-01 relative-next 47-D resume 修复

从 `custom_rel_static..._pvelobs...` checkpoint 继续训练时报错：

```text
ValueError: memory_a.rnn.weight_ih_l0 has observation width 47, expected 43
```

原因有两个：

```text
1. absolute_next_to_relative_state_dict() 只接受 43-D，但 47-D pvelobs 也应支持同样 37:41 列转换。
2. source_uses_relative_next 只识别路径里的 "relative_next"，没有识别当前实验名里的 "rel_static"，导致已 relative 的 checkpoint 被误判为 absolute，尝试重复转换。
```

已修复：

```text
absolute_next_to_relative_state_dict() accepts 43-D or 47-D.
source_uses_relative_next detects "rel_static".
```

编译通过：

```text
python -m py_compile train_planner_circular.py Quadhopper_Planner_Circular/checkpoint_migration.py
```
