# G1 搬箱训练

本说明用于训练 G1 的可选搬箱策略。箱子质量固定为 `0.5 kg`，训练数据使用
完整 LaFAN 与 G1 搬箱数据清单：

```text
configs/data/lafan_g1_largebox.yaml
```

搬箱训练和推理数据已经随仓库提供：

```text
humanoidverse/data/g1_largebox_train_near10s_ufo.pkl
humanoidverse/data/g1_largebox_full_ufo.pkl
```

数据内部使用仓库相对路径，不依赖原始的本机 Downloads 目录。
参考数据已经和 G1 一起完成重定向，箱体轨迹必须原样使用；训练端不再对它做
第二次尺寸缩放或贴地平移。箱体保持原始约 `47.1 × 45.9 × 40.8 cm`，质量为
`0.5 kg`。碰撞代理与 OBJ 使用同一原始边界，碰撞代理不可见，因此 play 中只
显示一层 OBJ。

正式训练只接受上面列出的原生 PKL；加载时会验证数据没有经过额外的箱体几何
重定向，从入口阻止重复缩放或轨迹平移。

策略使用统一的条件式时序流程：

- walk：箱子窗口严格为零，保留 LaFAN 行走/动作能力。
- carry：actor 接收可部署的箱子相对位置、6D 旋转和尺寸；窗口为当前帧加
  4 个过去帧，与机器人时序窗口对齐。
  接触、`ever_lifted`、`dropped` 不进入 actor。
- 箱子到落点的 3 维 heading-frame 位移使用独立 `goal_obs`。它进入 actor、F、B
  和 critic；FB 的 256 维全部保持同一种技能/奖励语义。
- B 同时编码机器人、箱子和目标状态，因此 tracking、goal 和 reward inference
  能区分“靠近、拿起、运输、放下”，不再只靠机器人姿态猜测箱子阶段。
- 一个判别器同时处理两类数据。机器人时序分支始终开启，箱子交互分支只在
  carry 门控下开启；AMP 判别器不读取目标坐标或 z，避免用命令泄漏代替判断动作
  风格。训练使用 50/50 walk/carry 采样和机器人/箱子错配负样本。
- task reward 决定接近、拾取、搬运、放置和掉落重捡；判别器只负责动作风格和
  人箱同步。大箱子的接近/抬升奖励按 OBB 表面、尺寸、旋转和底面高度计算。
- 默认直接从原始参考序列随机初始化，并用参考前置帧填充机器人和箱子历史，避免
  中段 RSI 被误表示成全零历史；训练和 play 不依赖额外的阶段认证数据。

## 单卡 RTX 5090

32 GiB 显存下将 512 万容量 replay 保存在 CPU，并使用后台预取和
GPU-native rollout：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh --agent fb --task carry_box --gpu-ids single --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_native_goal_v4 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cpu --buffer-prefetch 2 --buffer-pin-memory-threads 2 --gpu-native-rollout --runtime-timing-every 25
```

必须使用新的 work-dir，并只从原始 locomotion checkpoint
`runs/ufo_fb_g1/checkpoint` 初始化。不含 `goal_obs` 的 carry checkpoint 只允许
inference，训练入口会拒绝用它续训或作为 `--init-from`。初始化时保留兼容的机器人
权重，新增的箱子/目标输入从零权重开始；优化器、replay 和训练计数从零开始。之后
再次使用同一个新 work-dir 才会正常续训。

性能计时确认完成后，可以把 `--runtime-timing-every 25` 改为
`--runtime-timing-every 0`，消除计时采样带来的同步开销。

## 八卡 H200

### 1. 分布式 smoke

先验证八个 NCCL worker、CUDA replay 和搬箱环境：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --smoke --work-dir /tmp/ufo_h200x8_carry_native_goal_v4_smoke --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 1 --fail-fast-diagnostics
```

日志应覆盖 `rank=0/8` 到 `rank=7/8`。smoke 会自动限制为每卡 16 个环境，
因此应看到 `global_parallel_envs=128`、`buffer_storage=cuda` 和
`gpu_native_rollout=True`。

### 2. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_native_goal_v4_h200x8 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 100 --fail-fast-diagnostics
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
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.tracking_inference --model-folder runs/ufo_fb_g1_carry_box_native_goal_v4 --data-manifest configs/data/lafan_g1_largebox.yaml --dataset g1_largebox --motion-list 0 --device cuda:0 --headless false --save-mp4 false --export-onnx false --disable-dr true --disable-obs-noise true --live-reference true --live-reference-offset 0 1.5 0
```

实时界面默认同时显示两套状态：原色是策略 rollout，青色是逐帧同步的原始
机器人、箱子和目标点，默认沿世界坐标 Y 轴平移 1.5 m。需要重叠 ghost 对照时
加入 `--live-reference-offset 0 0 0`；只看策略时加入 `--live-reference false`。

将 `--dataset g1_largebox` 改成 `--dataset lafan`，可检查没有箱子观测时的
普通运动路径。

使用已有兼容 checkpoint 直接检查原始参考数据时，下面这条命令是基准。所需的
观测宽度会按 checkpoint 自动恢复；原色是策略 rollout，青色是同步原始数据：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.tracking_inference --model-folder runs/ufo_fb_g1_carry_box_conditional_amp_v2 --data-path humanoidverse/data/g1_largebox_full_ufo.pkl --device cuda:0 --motion-list 60 --headless false --save-mp4 false --disable-dr true --disable-obs-noise true --export-onnx false --live-reference true --live-reference-offset 0 1.5 0
```

## Reward mode 搬箱

checkpoint replay 中可以直接推断 `carry-pick`、`carry-transport`、
`carry-place`、`carry-recover` 和端到端 `carry-full` 的 z。下面在 MJLab
界面实时执行端到端任务，并把世界坐标 `(1.5, 0.0, 0.21)` 设为新落点：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.reward_inference --model-folder runs/ufo_fb_g1_carry_box_native_goal_v4 --data-manifest configs/data/lafan_g1_largebox.yaml --dataset g1_largebox --device cuda:0 --tasks carry-full --carry-target 1.5 0.0 0.21 --headless false --save-mp4 false --export-onnx false --disable-dr true --disable-obs-noise true
```

箱子掉落后门控不会关闭；策略仍看到箱子位姿历史和更新后的 `goal_obs`，因此恢复
行为由 `carry-recover`/`carry-full` 奖励和人箱联合风格共同训练，而不是依赖部署时
不可得的接触或 `dropped` 标志。
