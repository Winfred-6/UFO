# G1 搬箱训练

搬箱功能是原始 UFO 的最小扩展：加入箱体数据、动态箱体、`object_obs` 和箱体
辅助奖励，不改变 FB、AMP、256 维 z、actor/critic 或 expert-z rollout 的语义。

## 数据与箱体

训练清单：

```text
configs/data/lafan_g1_largebox.yaml
```

仓库内数据：

```text
humanoidverse/data/g1_largebox_train_near10s_ufo.pkl
humanoidverse/data/g1_largebox_full_ufo.pkl
```

箱体轨迹按原生坐标、旋转和尺寸直接使用，不再进行二次几何重定向、缩放或贴地
平移。箱体约为 `47.1 × 45.9 × 40.8 cm`，质量 `0.5 kg`；碰撞代理与 OBJ
使用同一边界，play 中只显示一层 OBJ。

## 相对原始 UFO 的改动边界

- actor、F、B、critic、aux critic 和原始 z-conditioned discriminator 额外读取
  `object_obs`。
- `object_obs` 是箱子相对机器人位置、6D 旋转、尺寸的当前帧和四帧历史；无箱子
  的 LaFAN 数据对应全零窗口。
- 接触、`ever_lifted`、`dropped` 和目标点不进入策略观测。
- z 始终是原始完整 256 维；不存在 z 尾部覆盖、task command、`goal_obs`、目标
  编码器或专用 conditional discriminator。
- expert-z 采样、FB 更新、AMP 更新和 tracking z 平滑完全沿用原始 UFO。
- 接近、拿起、搬运、放置、恢复和速度限制通过原有 aux critic 训练。目标位置只在
  环境内部用于辅助奖励和训练后的 reward relabel。

## 单卡 RTX 5090

使用新的 work-dir，从原始 locomotion checkpoint 只初始化兼容模型权重：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh --agent fb --task carry_box --gpu-ids single --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_minimal_v1 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cpu --buffer-prefetch 2 --buffer-pin-memory-threads 2 --gpu-native-rollout --runtime-timing-every 25
```

新增 object-input 列从零初始化；优化器、replay 和训练计数从零开始。不要在这个
work-dir 中续训此前带 `goal_obs`、z 尾部命令或专用判别器的 checkpoint。

性能确认后可将 `--runtime-timing-every 25` 改为 `0`。

### 安全停止与续训

训练默认启用 `--save-on-exit`。需要暂停时不要直接杀进程，运行：

```bash
uv run python scripts/request_training_stop.py --work-dir runs/ufo_fb_g1_carry_box_minimal_v1
```

命令会等待当前完整训练更新结束，保存模型、优化器、replay buffer 和精确训练计数，
看到 `Safe stop complete` 后训练进程会正常退出。以后用相同训练命令和相同
`--work-dir` 启动即可从该 checkpoint 继续；已有 checkpoint 时 `--init-from` 不会
重新覆盖续训状态。`Ctrl-C` 和 `SIGTERM` 也会触发同一流程，但上述命令能明确等待
保存确认。`SIGKILL`、断电和机器崩溃无法触发退出保存，只能恢复最近一次已落盘的
checkpoint。

## 八卡 H200

先运行 smoke：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --smoke --work-dir /tmp/ufo_h200x8_carry_minimal_v1_smoke --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 1 --fail-fast-diagnostics
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_minimal_v1_h200x8 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 100 --fail-fast-diagnostics
```

`--num-envs` 和 `--buffer-size` 都是每个 rank 的值。八卡稳定配置默认关闭分布式
`torch.compile`；smoke 通过后再按需加 `--compile`。

## Tracking 对比

训练完成后同时查看策略 rollout 和原始机器人/箱体数据：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.tracking_inference --model-folder runs/ufo_fb_g1_carry_box_minimal_v1 --data-path humanoidverse/data/g1_largebox_full_ufo.pkl --device cuda:0 --motion-list 60 --headless false --save-mp4 false --disable-dr true --disable-obs-noise true --export-onnx false --live-reference true --live-reference-offset 0 1.5 0
```

原色为策略，青色为逐帧同步参考。需要重叠对比时使用
`--live-reference-offset 0 0 0`。

## Reward inference

replay 中保存的辅助奖励可用于推断阶段技能：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m humanoidverse.reward_inference --model-folder runs/ufo_fb_g1_carry_box_minimal_v1 --data-manifest configs/data/lafan_g1_largebox.yaml --dataset g1_largebox --device cuda:0 --tasks carry-pick carry-transport carry-place --headless false --save-mp4 false --export-onnx false --disable-dr true --disable-obs-noise true
```

`carry-pick`、`carry-transport` 和 `carry-place` 都通过原始 FB reward inference
产生完整 z。任意运行时落点应由 TokenHSI/上层规划器根据目标重复选择或更新 z，
而不是向基础 UFO 网络增加目标观测。
