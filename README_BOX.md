# G1 搬箱训练

本说明用于训练 G1 的可选搬箱策略。箱子质量固定为 `0.5 kg`，训练数据使用
完整 LaFAN 与 G1 搬箱数据清单：

```text
configs/data/lafan_g1_largebox.yaml
```

处理后的搬箱训练和推理数据已经随仓库提供：

```text
humanoidverse/data/g1_largebox_g1fit_train_near10s_ufo.pkl
humanoidverse/data/g1_largebox_g1fit_full_ufo.pkl
```

数据内部使用仓库相对路径，不依赖原始的本机 Downloads 目录。
原始约 `47.1 × 45.9 × 40.8 cm` 的网格与 G1 重定向姿态最高穿模约 14 cm；
`g1fit` 数据使用经过 31,422 帧真实 MJLab 动力学认证的
`25.9 × 25.2 × 22.4 cm` 可抱持尺寸，质量仍为 `0.5 kg`。碰撞代理不可见，
可视 OBJ 与代理尺寸严格一致，因此 play 中不会再出现一大一小两层箱子。

策略使用统一的条件式时序流程：

- walk：4 帧箱子窗口严格为零，保留 LaFAN 行走/动作能力。
- carry：actor 只接收可部署的箱子相对位置、6D 旋转、尺寸及其 4 帧历史；
  接触、`ever_lifted`、`dropped` 不进入 actor。
- 目标位置作为 3 维高层 task command 写入 z 的保留尾部，不进入普通观测；因此
  reward mode 可以指定新落点，而当前箱子状态仍不进入 Backward/z 编码器。
- 一个判别器同时处理两类数据。机器人时序分支始终开启，箱子交互分支只在
  carry 门控下开启；训练使用 50/50 walk/carry 采样和机器人/箱子错配负样本。
- task reward 决定接近、拾取、搬运、放置和掉落重捡；判别器只负责动作风格和
  人箱同步。大箱子的接近/抬升奖励按 OBB 表面、尺寸、旋转和底面高度计算。
- 搬箱动作按 TokenHSI 风格从接近、抓取、运输、放置的物理安全参考状态随机开始，
  默认概率为 `10% / 20% / 50% / 20%`。阶段标签和安全掩码只用于 reset，
  不进入 actor 或判别器观测。

## 单卡 RTX 5090

32 GiB 显存下将 512 万容量 replay 保存在 CPU，并使用后台预取和
GPU-native rollout：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh --agent fb --task carry_box --gpu-ids single --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_stage_rsi_v3 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cpu --buffer-prefetch 2 --buffer-pin-memory-threads 2 --gpu-native-rollout --runtime-timing-every 25 --fail-fast-diagnostics
```

该目录第一次启动时迁移旧 carry checkpoint 的兼容权重；旧 19 维箱子输入只映射
语义相同的位姿列，新时序列从零开始，新的条件式判别器重新初始化。优化器、
replay 和训练计数从零开始。之后再次使用同一个 `--work-dir` 会自动续训。

性能计时确认完成后，可以把 `--runtime-timing-every 25` 改为
`--runtime-timing-every 0`，消除计时采样带来的同步开销。

## 八卡 H200

### 1. 分布式 smoke

先验证八个 NCCL worker、CUDA replay 和搬箱环境：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --smoke --work-dir /tmp/ufo_h200x8_carry_smoke_v3 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 1 --fail-fast-diagnostics
```

日志应覆盖 `rank=0/8` 到 `rank=7/8`。smoke 会自动限制为每卡 16 个环境，
因此应看到 `global_parallel_envs=128`、`buffer_storage=cuda` 和
`gpu_native_rollout=True`。

### 2. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_stage_rsi_v3_h200x8 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 100 --fail-fast-diagnostics
```

这里的 `--num-envs` 和 `--buffer-size` 都是每个 rank 的数值：

- 每卡 1024 个环境，全局共 8192 个环境。
- 每卡 512 万 replay，全局共 4096 万。
- replay checkpoint 约每卡 25 GiB，八卡合计约 200 GiB。
- 如果只需要全局合计 512 万 replay，改用 `--buffer-size 640000`。

八卡稳定配置默认关闭分布式 `torch.compile`。smoke 通过后，可以在正式
命令末尾加入 `--compile` 做对照测试；每个 rank 会使用独立的
Inductor/Triton 临时缓存。完成计时分析后，将
`--runtime-timing-every 100` 改为 `0`。

## 实时检查策略

训练产生 checkpoint 后，可直接在 MJLab 界面播放搬箱轨迹，不生成视频：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.tracking_inference --model-folder runs/ufo_fb_g1_carry_box_stage_rsi_v3 --data-manifest configs/data/lafan_g1_largebox.yaml --dataset g1_largebox --motion-list 0 --device cuda:0 --headless false --save-mp4 false --export-onnx false --disable-dr true --disable-obs-noise true
```

实时界面默认同时显示两套状态：原色是策略 rollout，青色是逐帧同步的原始
机器人、箱子和目标点，默认沿世界坐标 Y 轴平移 1.5 m。需要重叠 ghost 对照时
加入 `--live-reference-offset 0 0 0`；只看策略时加入 `--live-reference false`。

将 `--dataset g1_largebox` 改成 `--dataset lafan`，可检查没有箱子观测时的
普通运动路径。

## Reward mode 搬箱

checkpoint replay 中可以直接推断 `carry-pick`、`carry-transport`、
`carry-place`、`carry-recover` 和端到端 `carry-full` 的 z。下面在 MJLab
界面实时执行端到端任务，并把世界坐标 `(1.5, 0.0, 0.21)` 设为新落点：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.reward_inference --model-folder runs/ufo_fb_g1_carry_box_stage_rsi_v3 --data-manifest configs/data/lafan_g1_largebox.yaml --dataset g1_largebox --device cuda:0 --tasks carry-full --carry-target 1.5 0.0 0.21 --headless false --save-mp4 false --export-onnx false --disable-dr true --disable-obs-noise true
```

箱子掉落后门控不会关闭；策略仍看到箱子位姿历史与同一目标 task z，因此恢复行为
由 `carry-recover`/`carry-full` 奖励和人箱联合风格共同训练，而不是依赖部署时不可得
的接触或 `dropped` 标志。
