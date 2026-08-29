# UFO Setup, Training, and Inference

This document mirrors the repository quick start with a little more context for training and inference runs.

## Install

```bash
uv sync
```

For W&B logging, authenticate before launching multi-process training:

```bash
uv run wandb login
# or
export WANDB_API_KEY=your_wandb_api_key
```

## Defaults

The default training configuration is defined in `humanoidverse/train.py`:

- `--num-envs`: `1024` environments per GPU.
- `--num-env-steps`: `192000000` global environment steps.
- `--data-path`: `humanoidverse/data/lafan_29dof_10s-clipped.pkl`.
- `--work-dir`: `runs/ufo`.
- `--checkpoint-every-steps`: `3200000` global environment steps.
- `--buffer-size`: `5120000` transitions per training rank.
- `--buffer-storage`: `cpu`; alternatives are `memmap` and `cuda`.
- `--buffer-prefetch`: `2` batches staged ahead of the optimizer.
- `--buffer-pin-memory-threads`: `2` threads for page-locking sampled batches.
- `--gpu-native-rollout`: enabled; MJLab observations/actions remain on CUDA.
- `--runtime-timing-every`: `0`; set a positive sampling interval to report the runtime breakdown.
- `--update-z-every-step`: defaults to `100` for FB and `10` for TeCH.

All of these can be overridden from the command line.

## Replay storage

The default TorchRL replay keeps the full online and expert datasets in CPU
RAM, reconstructs valid one-step transitions with `SliceSampler`, and
prefetches pinned TensorDict minibatches to the training GPU. This avoids a
full-capacity CUDA allocation while overlapping most sampling and PCIe transfer
work with the preceding optimizer update.

For current G1 FB observations, a 5.12-million-frame online buffer is about
24.7 GiB per rank before expert-data and process overhead. Use
`--buffer-storage memmap` with fast local NVMe if aggregate host RAM is
insufficient, especially for multi-GPU runs. The memmap directory is
`<work-dir>/replay_memmap/rank_<rank>`. `--buffer-storage cuda` remains
available for small-buffer performance comparisons.

Run the storage microbenchmark with:

```bash
uv run python scripts/benchmark_replay.py --storage all --batch-size 1024
```

Compare the legacy NumPy rollout path with the GPU-native path using the real
MJLab carry environment:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_rollout.py --num-envs 1024 --steps 30
```

## FB Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_fb_8gpu
```

### Optional G1 carry-box task

`--task carry_box` enables an isolated MJLab task extension with a 0.5 kg
rigid box and goal marker. Training uses
`configs/data/lafan_g1_largebox.yaml`: LaFAN and all paired G1/large-box
trajectories are balanced at 0.50/0.50. The paired data is already retargeted
with G1 and is used at the native OBJ scale without another geometry resize or
ground-trajectory shift.

The actor receives current plus four past frames of relative box position,
6-D rotation, and size, together with an explicit heading-frame `goal_obs`.
Actor, F, B, and critics consume `object_obs` and `goal_obs`; the complete
256-dimensional z keeps one FB skill/reward meaning. The gated temporal AMP
discriminator handles walk style and carry robot/box synchronization without
reading goal coordinates or z as shortcuts.

Initialize model weights from the existing locomotion run while starting a new
optimizer, replay, and step counter:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh \
  --agent fb \
  --task carry_box \
  --gpu-ids single \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_g1_carry_box_native_goal_v4 \
  --data-manifest configs/data/lafan_g1_largebox.yaml \
  --init-from runs/ufo_fb_g1/checkpoint \
  --update-z-every-step 100 \
  --buffer-size 5120000 \
  --buffer-storage cpu \
  --buffer-prefetch 2 \
  --buffer-pin-memory-threads 2 \
  --gpu-native-rollout \
  --runtime-timing-every 25
```

The repository includes the processed full and near-10-second G1/large-box
PKLs under `humanoidverse/data/`; `configs/data/lafan_g1_largebox.yaml` points
to those portable native paths and does not depend on a machine-local source
folder. Formal training rejects object data marked as having undergone another
geometry retarget. Use a fresh work directory and initialize only from
`runs/ufo_fb_g1/checkpoint`; carry checkpoints without `goal_obs` are
inference-only.

For one 32 GiB RTX 5090, keep the 5.12-million-frame replay in host RAM. This
machine profile keeps the already-tested 1024 environments and samples a timing
breakdown every 25 rollout iterations:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh --agent fb --task carry_box --gpu-ids single --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_native_goal_v4 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cpu --buffer-prefetch 2 --buffer-pin-memory-threads 2 --gpu-native-rollout --runtime-timing-every 25
```

For eight H200 GPUs, each rank can keep its replay directly on its local GPU.
This removes the rollout-to-replay CPU copy and does not need CPU prefetching.
`--buffer-size` is per rank, so the command below uses 5.12 million frames on
each H200. Distributed `torch.compile` remains auto-disabled for the stable
profile; add `--compile` only after the H200 smoke run passes. Inductor and
Triton temporary caches are isolated per rank when compilation is enabled.
At this capacity, replay checkpoints are roughly 25 GiB per rank (about 200
GiB across eight ranks), so provision local checkpoint storage accordingly.
Use `--buffer-size 640000` if a 5.12-million-frame aggregate replay is desired
instead, recognizing that this shortens each rank's local replay horizon.

First verify the eight-worker/NCCL path with the bounded smoke profile:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --smoke --work-dir /tmp/ufo_h200x8_carry_native_goal_v4_smoke --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 1
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_train.sh --agent fb --task carry_box --gpu-ids all --num-envs 1024 --num-env-steps 192000000 --work-dir runs/ufo_fb_g1_carry_box_native_goal_v4_h200x8 --data-manifest configs/data/lafan_g1_largebox.yaml --init-from runs/ufo_fb_g1/checkpoint --update-z-every-step 100 --buffer-size 5120000 --buffer-storage cuda --buffer-prefetch 0 --buffer-pin-memory-threads 0 --gpu-native-rollout --runtime-timing-every 100
```

The timing output reports `env_step_gpu_ms`, `cpu_copy_ms`, `replay_extend_ms`,
`replay_sample_ms`, `fb_ms`, `critic_ms`, `aux_critic_ms`, `actor_ms`, and the
overall agent-update time. Set `--runtime-timing-every 0` after profiling for
the lowest possible synchronization overhead.

## TeCH Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tech \
  --gpu-ids all \
  --use-wandb \
  --wandb-run-name ufo_tech_8gpu
```

TeCH was previously exposed as the TLDR preset in early UFO versions. `--agent tldr` is kept as a deprecated compatibility alias for `--agent tech`.

## Tracking Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --motion-list 20
```

For live MJLab playback of a carry trajectory (no video or ONNX export):

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_g1_carry_box_native_goal_v4 \
  --data-manifest configs/data/lafan_g1_largebox.yaml \
  --dataset g1_largebox \
  --motion-list 0 \
  --device cuda:0 \
  --headless false \
  --save-mp4 false \
  --export-onnx false \
  --disable-dr true \
  --disable-obs-noise true \
  --live-reference true \
  --live-reference-offset 0 1.5 0
```

Live tracking shows the policy in its normal colors and a frame-synchronized
cyan source robot, box, and target 1.5 m to its side. Use
`--live-reference-offset 0 0 0` for a ghost overlay, or
`--live-reference false` to restore policy-only playback.

Use `--dataset lafan` with the same checkpoint to exercise the no-box path;
the five-frame object observation is then exactly zero and the inactive box is
kept at its collision-safe parking position outside the active scene.

When `--export-onnx` is enabled, `tracking_inference` exports a
robot-config-aware policy ONNX next to the checkpoint. The policy input split is
derived from the checkpoint model's `obs_space` and actor `input_filter`, and a
metadata JSON records the robot name, robot config path, XML path, controlled
joints, actor input dimensions, z dimension, actor observation dimension, and
output action dimension.

The exported ONNX is tied to the checkpoint's robot, action, and observation
dimensions. One checkpoint cannot be reused across different robots. The deploy
branch remains G1-only unless a robot-specific deploy configuration is created.

## Goal Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.goal_inference \
  --model-folder runs/ufo \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --save-mp4 \
  --export-onnx
```

`goal_inference` accepts `--robot-config`, `--data-manifest`, `--dataset`, and
`--rebuild-motion-cache` with the same manifest behavior as tracking inference.
If no robot config is provided, it defaults to `configs/robots/g1_29dof.yaml`.
For G1, omitting `--goal-json` keeps the existing
`goal_frames_lafan29dof.json` fallback. For non-G1 robots, pass a goal JSON
generated for the selected robot; the G1 goal JSON is not shared across robot
morphologies.

## Reward Inference

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.reward_inference \
  --model-folder runs/ufo \
  --device cuda:0 \
  --headless \
  --disable-dr \
  --disable-obs-noise \
  --buffer-rank 0 \
  --num-samples 150000 \
  --n-inferences 1 \
  --save-mp4 \
  --export-onnx
```

`reward_inference` also accepts `--robot-config`, `--data-manifest`,
`--dataset`, and `--rebuild-motion-cache`. G1 keeps the full default reward task
set. For non-G1 robots, the first robot-config-aware path is limited to
robot-config-aware rollout/relabel setup and root/locomotion tasks such as
`move-ego-*` and `rotate-z-*`; arm, crouch, sit-on-ground, and other
G1-semantics tasks require robot-specific reward semantics and are rejected
early. The exported ONNX and reward/goal outputs remain tied to the checkpoint's
robot, action, and observation dimensions. The deploy branch remains G1-only
unless a robot-specific deploy config is created.
