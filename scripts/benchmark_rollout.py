"""Compare legacy NumPy and GPU-native MJLab rollout transfer paths."""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils._pytree import tree_map

from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.utils.motion_data import prepare_motion_manifest


def _to_gpu(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if hasattr(value, "dtype"):
        return torch.as_tensor(value, device=device)
    return value


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _measure(env, action: torch.Tensor, *, steps: int, gpu_native: bool) -> float:
    if gpu_native:
        observation, _ = env.reset(to_numpy=False)
    else:
        observation, _ = env.reset(to_numpy=True)
    torch.cuda.synchronize(action.device)
    start = time.perf_counter()
    for _ in range(steps):
        if gpu_native:
            result = env.step(action, to_numpy=False)
            # Simulate the single unavoidable write to CPU replay.
            tree_map(_to_cpu, result)
            observation = result[0]
        else:
            # Reproduce the old CPU observation -> CUDA policy -> CPU action path.
            tree_map(lambda value: _to_gpu(value, str(action.device)), observation)
            result = env.step(action.detach().cpu().numpy(), to_numpy=True)
            observation = result[0]
    torch.cuda.synchronize(action.device)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-manifest", default="configs/data/lafan_g1_largebox.yaml")
    args = parser.parse_args()

    manifest = prepare_motion_manifest(args.data_manifest)
    buffer_size = max(args.num_envs * 2, args.num_envs * ((10240 + args.num_envs - 1) // args.num_envs))
    cfg = build_ufo_mjlab_config(
        device=args.device,
        work_dir="/tmp/ufo_rollout_transfer_bench",
        num_envs=args.num_envs,
        num_env_steps=buffer_size,
        seed=4728,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        agent="fb",
        data_path=manifest.train_data_paths,
        data_mix_weights=manifest.train_data_weights,
        buffer_size=buffer_size,
        task="carry_box",
        disable_dr=True,
        disable_obs_noise=True,
    )
    env, _ = cfg.env.build(num_envs=args.num_envs)
    action = torch.zeros((args.num_envs, env.single_action_space.shape[0]), device=args.device)
    try:
        env.reset(to_numpy=False)
        for _ in range(args.warmup_steps):
            env.step(action, to_numpy=False)
        torch.cuda.synchronize(action.device)

        legacy_seconds = _measure(env, action, steps=args.steps, gpu_native=False)
        native_seconds = _measure(env, action, steps=args.steps, gpu_native=True)
        print(
            {
                "num_envs": args.num_envs,
                "steps": args.steps,
                "legacy_ms_per_step": legacy_seconds * 1000.0 / args.steps,
                "gpu_native_ms_per_step": native_seconds * 1000.0 / args.steps,
                "speedup": legacy_seconds / native_seconds,
                "legacy_env_fps": args.num_envs * args.steps / legacy_seconds,
                "gpu_native_env_fps": args.num_envs * args.steps / native_seconds,
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
