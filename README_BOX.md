# G1 搬箱训练

本说明用于训练 G1 的可选搬箱策略。箱子质量固定为 `0.5 kg`，训练数据使用
完整 LaFAN 与 G1 搬箱数据清单：

```text
configs/data/lafan_g1_largebox.yaml
```

处理后的搬箱训练和推理数据已经随仓库提供：

```text
humanoidverse/data/g1_largebox_train_near10s_ufo.pkl
humanoidverse/data/g1_largebox_full_ufo.pkl
```

数据内部使用仓库相对路径，不依赖原始的本机 Downloads 目录。

策略在没有箱子观测时执行普通运动；提供箱子观测时执行接近、拿取和搬运。
箱子状态只进入 actor、F、critic 和 auxiliary critic，不进入 Backward/z 编码器
或风格判别器。

## 单卡 RTX 5090

32 GiB 显存下将 512 万容量 replay 保存在 CPU，并使用后台预取和
GPU-native rollout：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh --agent fb --task carry_box --gpu-ids single --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cpu --buffer-prefetch 2 --buffer-pin-memory-threads 2 --gpu-native-rollout --runtime-timing-every 25
```

该目录第一次启动时只加载 `runs/ufo_fb_g1/checkpoint` 的模型权重，优化器、
replay 和训练计数从零开始。之后再次使用同一个 `--work-dir` 会自动续训。

性能计时确认完成后，可以把 `--runtime-timing-every 25` 改为
`--runtime-timing-every 0`，消除计时采样带来的同步开销。

## 八卡 H200

### 1. 分布式 smoke

先验证八个 NCCL worker、CUDA replay 和搬箱环境：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --smoke --work-dir /tmp/ufo_h200x8_carry_smoke --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 1
```

日志应覆盖 `rank=0/8` 到 `rank=7/8`。smoke 会自动限制为每卡 16 个环境，
因此应看到 `global_parallel_envs=128`、`buffer_storage=cuda` 和
`gpu_native_rollout=True`。

### 2. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_h200x8 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 100
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
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.tracking_inference --model-folder runs/ufo_fb_g1_carry_box --data-manifest configs/data/lafan_g1_largebox.yaml --dataset g1_largebox --motion-list 0 --device cuda:0 --headless false --save-mp4 false --export-onnx false --disable-dr true --disable-obs-noise true
```

将 `--dataset g1_largebox` 改成 `--dataset lafan`，可检查没有箱子观测时的
普通运动路径。
