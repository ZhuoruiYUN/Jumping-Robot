# 冻结的 PPO-jump-reference 37 维原版

环境、资产配置、PPO 与模型文件逐字节来自用户提供的 ZIP。不要直接编辑这些文件；改任务或物理请另建配置/分支。

完整逐维合同、奖励/物理/重置差异、晋级门槛与手动命令见 [BENCHMARK37_CONTRACT_AND_STAGES.md](../BENCHMARK37_CONTRACT_AND_STAGES.md)。来源锁为 [source_lock.json](../benchmark37/source_lock.json)。

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
bash run_benchmark37.sh offline
bash run_benchmark37.sh smoke
```

本阶段保留原版 1.8–3m 释放与下落电机响应，作为来源对照；尚不满足实机 0.8–1.2m、首次触地后启动的条件。短测通过后才手动正式训练；所有输出独立保存到 `outputs/benchmark37/`，原有模型不覆盖。

本机7.5GiB显存上，原4096环境训练实际在第一轮前CUDA OOM，失败manifest保留。请先手动运行`bash run_benchmark37.sh fit-smoke --seed 42`检验1024环境容量；完成后再运行`bash run_benchmark37.sh train-fit --seed 42`。该档位维持原任务与总交互数，PPO批量和更新次数不同，详情见合同文档。
