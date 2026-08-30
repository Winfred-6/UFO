<h1 align="center">
  UFO：面向人形机器人控制的无监督强化学习框架
</h1>

<p align="center">
  <a href="https://roboparty.github.io/UFO/"><img alt="Website" src="https://img.shields.io/badge/Website-roboparty.github.io%2FUFO-2563eb?style=for-the-badge" /></a>
  <a href="https://youtu.be/uJPcLdn9sNA"><img alt="Video" src="https://img.shields.io/badge/Video-Demo-7c3aed?style=for-the-badge" /></a>
  <a href="https://roboparty.github.io/UFO/assets/UFO.pdf"><img alt="PDF" src="https://img.shields.io/badge/PDF-Available-0f766e?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="./assets/UFO.png" alt="UFO" height="72" style="vertical-align: middle;" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./assets/rplab_logo.png" alt="ROBO PARTY LAB Logo" height="72" style="vertical-align: middle;" />
</p>

## UFO 是什么？

UFO 是一个面向人形机器人控制、源代码可用的科研框架。`main` 分支主要用于 MJLab 训练、RobotState 数据导入、tracking/goal/reward inference，以及 ONNX 导出。`deploy` 分支用于 Unitree G1 实机部署和遥操作运行时。

当前最完整、测试最充分的路线是 Unitree G1。新机器人适配已经有实验性接口，但需要用户准备目标机器人的 MuJoCo XML、可选 URDF，以及已经 retarget 到该机器人的 RobotState motion data。UFO 不会自动把人类动作或其他机器人的动作 retarget 到新机器人；不同机器人之间也不能直接复用同一个 checkpoint。

## 当前支持范围

| 功能 | 状态 |
| --- | --- |
| G1 训练 | 支持，测试最充分 |
| RobotState CSV / NPZ / `ufo_pkl` | 支持 |
| 多数据源 manifest | 支持 |
| Tracking inference | Robot-config aware |
| Goal inference | 支持 robot config；非 G1 需要机器人专属 goal JSON |
| Reward inference | G1 支持完整默认任务；非 G1 当前主要支持 root/locomotion 任务 |
| 实机部署 / 遥操作 | 使用 [`deploy` 分支](https://github.com/Roboparty/UFO/tree/deploy) |
| 自动 motion retargeting | 不支持 |
| 跨机器人复用同一个 checkpoint | 不支持 |

> [!NOTE]
> `main` 分支：训练、数据导入、推理、ONNX 导出。
> `deploy` 分支：G1 实机部署和遥操作运行时。

## 路线 A：Unitree G1 快速开始

### 1. 安装环境

```bash
git clone https://github.com/Roboparty/UFO.git
cd UFO
```

安装 [`uv`](https://docs.astral.sh/uv/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

或者：

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

安装项目环境：

```bash
uv sync
```

在 Linux 和 Windows 上，lockfile 会安装官方 PyTorch 2.7.1 CUDA 12.8
wheel，其中包含 RTX 5090 等 NVIDIA Blackwell GPU 所需的 `sm_120`
支持。旧的 `cu126` 安装如果只列出到 `sm_90`，会在创建训练环境前报
`no kernel image is available`。

### 可选：W&B logging

W&B logging 是可选功能。如需启用，请先登录，按需设置自己的 entity，然后在训练命令中加入 `--use-wandb --wandb-run-name ...`：

```bash
uv run wandb login
export WANDB_ENTITY=your_entity   # optional
# 然后加入 --use-wandb --wandb-run-name ufo_fb_g1
```

### 2. 下载 G1 LaFAN 数据

大数据不放在 Git 仓库中。使用下面的命令下载默认的 G1 LaFAN 数据：

```bash
bash scripts/download_data.sh g1_lafan
ls -lh humanoidverse/data/lafan_29dof_10s-clipped.pkl
```

### 3. Smoke test

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_g1
```

### 4. FB 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 100 \
  --buffer-size 5120000
```

### 5. TeCH 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tech \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_tech_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 10 \
  --buffer-size 5120000
```

TeCH 在早期 UFO 版本中曾经叫 TLDR。`--agent tldr` 仍然保留为 `--agent tech` 的兼容 alias，但已经不推荐继续使用。

### 可选：G1 搬箱任务

`--task carry_box` 是独立开关，不会改变原来的 `task=motion` 环境。未指定
数据参数时会自动使用 `configs/data/lafan_g1_largebox.yaml`：完整 LaFAN
与仓库内处理好的 174 条 G1/箱子配对轨迹按 0.50/0.50 均衡采样，不再依赖
本机 Downloads 路径。箱子是 MJLab 中的 500 g 自由刚体，并带有可视化
目标框。

搬箱扩展保持原始 UFO 网络语义：actor、F、B、critic、aux critic 和原始
z-conditioned discriminator 仅额外读取 `object_obs`。它包含箱子相对位置、6D
旋转、尺寸的当前帧与四帧历史；无箱子的 LaFAN 数据对应全零窗口。接触状态与目标
位置不进入策略观测，完整 256 维 z 不做覆盖，也不增加 task command、目标分支或
专用判别器。接近、拿起、搬运、放置和恢复使用原有 auxiliary reward 机制。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --task carry_box \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_g1_carry_box_minimal_v1 \
  --data-manifest configs/data/lafan_g1_largebox.yaml \
  --init-from runs/ufo_fb_g1/checkpoint \
  --update-z-every-step 100 \
  --buffer-size 5120000
```

实时查看搬箱 tracking（不保存视频）：

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_g1_carry_box_minimal_v1 \
  --data-path humanoidverse/data/g1_largebox_full_ufo.pkl \
  --motion-list 60 \
  --device cuda:0 \
  --headless false \
  --save-mp4 false \
  --export-onnx false \
  --disable-dr true \
  --disable-obs-noise true \
  --live-reference true \
  --live-reference-offset 0 1.5 0
```

原色是策略，青色是逐帧同步的原始机器人和箱体；需要重叠观察时，把
`--live-reference-offset` 改成 `0 0 0`。

单张 32 GiB RTX 5090 建议保持 1024 个环境，并使用 CPU replay、后台
预取和 GPU-native rollout：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh --agent fb --task carry_box --gpu-ids single --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_minimal_v1 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cpu --buffer-prefetch 2 --buffer-pin-memory-threads 2 --gpu-native-rollout --runtime-timing-every 25
```

八张 H200 可以把每个 rank 的 replay 直接放在对应 GPU，去掉 CPU
拷贝和预取线程：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --smoke --work-dir /tmp/ufo_h200x8_carry_minimal_v1_smoke --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 1
```

smoke 确认日志中出现 `rank=0/8` 到 `rank=7/8` 后，再启动正式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_minimal_v1_h200x8 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 100
```

这里的 `--buffer-size` 是每卡容量。八卡稳定配置默认关闭分布式
`torch.compile`；H200 smoke 通过后可加 `--compile`，每个 rank 会使用独立
的 Inductor/Triton 临时缓存。计时确认完成后，将
`--runtime-timing-every` 改为 `0` 可消除采样同步开销。
每卡 512 万容量的 replay checkpoint 约为 25 GiB，八卡合计约 200 GiB；
如果只需要全局合计 512 万容量，可改成 `--buffer-size 640000`，但每个
rank 可看到的历史窗口也会相应缩短。

### 安全停止与续训

训练默认启用退出前 checkpoint。需要暂停时运行：

```bash
uv run python scripts/request_training_stop.py --work-dir runs/ufo_fb_g1_carry_box_minimal_v1
```

该命令会等待一个完整 rollout/参数更新边界，再保存模型、优化器、replay buffer
和精确计数；看到 `Safe stop complete` 后训练进程才会退出。之后以相同训练命令和
相同 `--work-dir` 启动即可续训。`Ctrl-C` 与 `SIGTERM` 也走相同安全保存路径；
`SIGKILL`、断电或系统崩溃无法触发退出保存。

### Replay buffer 的内存与吞吐

在线 replay 和 expert replay 的大容量存储默认放在 CPU RAM。UFO 使用
[TorchRL ReplayBuffer](https://docs.pytorch.org/rl/0.8/reference/generated/torchrl.data.ReplayBuffer.html)
与 `SliceSampler` 保持轨迹边界，并且不重复保存每一帧的 next observation。
采样、页锁定和 CPU 到 GPU 的传输会在后台提前执行：

```text
CPU LazyTensorStorage -> SliceSampler -> pinned TensorDict batch -> async H2D -> GPU optimizer
```

默认配置等价于：

```bash
./run_train.sh ... \
  --buffer-storage cpu \
  --buffer-prefetch 2 \
  --buffer-pin-memory-threads 2
```

可选存储模式：

- `cpu`（默认）：主机内存足够时推荐使用，显存中只保留正在训练的
  minibatch。
- `memmap`：主机内存紧张时写入
  `<work-dir>/replay_memmap/rank_<rank>`；建议使用高速本地 NVMe。
- `cuda`：用于小 buffer 对照实验。默认 512 万 transition 的 G1 buffer
  不适合放在 32 GiB 显存中。

按当前 G1 FB 字段估算，5,120,000 帧的在线 replay 每个 rank 约占
24.7 GiB 主机内存，此外还需要 expert data 和进程运行空间。多 GPU
训练通常是每张 GPU 一个 rank，因此 RAM 或 memmap 容量也会随 rank
数倍增。`--buffer-size` 必须能被 `--num-envs` 整除。`--smoke` 会自动把
buffer 容量限制到短 rollout 的规模，不再分配正式训练的完整容量。

长时间训练前可以先运行 replay benchmark：

```bash
uv run python scripts/benchmark_replay.py \
  --storage all \
  --batch-size 1024 \
  --prefetch 2
```

### 6. Tracking inference

推理时建议使用 full motion sequences，不要使用裁剪后的 training clips：

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_g1 \
  --data-path /path/to/full_motions.pkl \
  --device cuda:0 \
  --headless \
  --save-mp4 \
  --motion-list 0
```

输出会写到 `<model-folder>/tracking_inference/`。

### 7. ONNX 导出说明

在 tracking inference 命令中加入 `--export-onnx true` 可以导出 robot-config-aware ONNX policy 和 metadata JSON。导出的 ONNX 和当前 checkpoint 的机器人、动作维度、观测维度绑定，不能直接用于其他机器人。

## 路线 B：适配新机器人

这条路线是 experimental。你需要先准备：

1. 目标机器人的 MuJoCo XML；
2. 可选的匹配 URDF；
3. 已经适配或 retarget 到目标机器人的 RobotState 数据。

UFO 不负责自动把人类动作或其他机器人动作 retarget 到新机器人。用户需要先通过 `hhtools`、GMR 或自定义 retargeting pipeline 得到目标机器人的 RobotState 数据，再导入 UFO。

### 1. 生成 robot config 草稿

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml /path/to/robot.xml \
  --urdf /path/to/robot.urdf \
  --name my_robot \
  --out configs/robots/my_robot.yaml \
  --hydra-out humanoidverse/config/robot/my_robot/my_robot_auto.yaml
```

如果没有 URDF，可以省略 `--urdf`。URDF 只是辅助信息；MuJoCo XML 仍然是 qpos/qvel、action layout 和 actuator order 的 source of truth。

### 2. 人工检查 robot config

自动生成的配置只是草稿。大规模训练前必须人工检查 base body、control-joint order、feet、hands、key bodies、initial state、PD gains、actuator limits、contact bodies，以及和 reward/termination 相关的语义。

### 3. 构建 RobotState data manifest

```bash
uv run python -m humanoidverse.tools.data_build \
  --robot configs/robots/my_robot.yaml \
  --source "/path/to/motions/*.csv" \
  --format robot_state_csv \
  --name my_motion \
  --fps 50 \
  --clip-seconds 10 \
  --out configs/data/my_motion_auto_build.yaml \
  --rebuild-cache
```

无表头 CSV 支持两种格式：`root_pos` xyz、`root_quat` xyzw、随后是 XML/control-joint order 的 DOF position；也可以在最前面增加可选的 `time` 列。

如只想检查 CSV schema，可以先运行 `humanoidverse.tools.data_inspect`。

### 4. Smoke training

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/my_robot.yaml \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_my_robot
```

## 常见注意事项

- G1 是当前最完整、测试最充分的路径。
- 新机器人适配是 experimental，通常还需要调 controller、reward、contact 和 termination 语义。
- `main` 分支用于 training、data import、inference、ONNX export。
- `deploy` 分支当前主要面向 G1 实机部署和遥操作。
- 非 G1 的 goal inference 需要机器人专属 goal JSON。
- 非 G1 的 reward inference 当前主要支持 root/locomotion 任务，除非额外补充机器人语义。
- TeCH 曾经叫 TLDR，`--agent tldr` 仍是兼容 alias。
- 不同机器人之间不能直接复用同一个 checkpoint。

## 多数据源技能注入

UFO 支持基于 manifest 的多数据源混合。每个数据源之间的采样比例保持固定，prioritized sampling 在每个数据源内部进行。这适合在保持基础动作分布的同时注入少量稀有高敏捷技能，例如 cartwheel。可以参考 `configs/data/example_mix.yaml`。

## 文档链接

- [Import Wizard](docs/import_wizard.md)：RobotState schema、数据检查和数据构建。
- [Robot-Config Training](docs/robot_config_training.md)：实验性的 robot-aware training 初始化说明。
- [Training and Inference](docs/TRAIN_INFERENCE.md)：更多训练和推理命令。
- [Deploy branch](https://github.com/Roboparty/UFO/tree/deploy)：G1 实机部署和遥操作运行时。

## 引用 / 许可证

如果你在研究中使用了 UFO，请引用：

```bibtex
@misc{ufo2026,
  author       = {{RoboParty Lab Team}},
  title        = {UFO: An Unsupervised Reinforcement Learning Framework for Humanoid Control},
  year         = {2026},
  howpublished = {\url{https://github.com/Roboparty/UFO}},
  note         = {Project page: \url{https://roboparty.github.io/UFO/}}
}
```

在 LaTeX 中，将上述条目加入 `.bib` 文件后，使用 `\cite{ufo2026}` 即可。

UFO 当前基于 CC BY-NC 4.0 发布，主要面向非商业科研使用，具体以 [LICENSE](LICENSE) 文件为准。
