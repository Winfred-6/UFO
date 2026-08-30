"""Stress inactive carry boxes for longer than one training episode.

The carry scene always contains a dynamic box, including object-free LAFAN
environments.  This diagnostic verifies that the masked/inactive box remains
finite and stationary instead of interacting with the infinite ground plane.
It does not construct an agent or perform any learning update.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from humanoidverse.train import build_ufo_mjlab_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "humanoidverse/data/lafan_29dof_10s-clipped.pkl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ufo_inactive_box_stress"))
    args = parser.parse_args()

    cfg = build_ufo_mjlab_config(
        device=args.device,
        work_dir=str(args.work_dir),
        num_envs=args.num_envs,
        num_env_steps=args.num_envs * args.steps,
        seed=args.seed,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        agent="fb",
        data_path=str(args.data_path.expanduser().resolve()),
        data_mix_weights=None,
        buffer_size=args.num_envs * 2,
        compile_agent=False,
        disable_dr=False,
        disable_obs_noise=False,
        task="carry_box",
        fail_fast_diagnostics=True,
    )
    carry_cfg = cfg.env.carry_box
    cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"carry_box": carry_cfg})})
    env, _ = cfg.env.build(num_envs=args.num_envs)
    core = env.base_env
    try:
        env.reset(to_numpy=False)
        if bool(torch.any(core.object_valid > 0.5).item()):
            raise RuntimeError("Inactive-box stress requires object-free motions only")
        generator = torch.Generator(device=core.device)
        generator.manual_seed(args.seed)
        action_shape = (args.num_envs, env.single_action_space.shape[0])
        for step in range(1, args.steps + 1):
            actions = torch.rand(action_shape, generator=generator, device=core.device) * 2.0 - 1.0
            env.step(actions, to_numpy=False)
            if step == 1 or step % 100 == 0 or step == args.steps:
                object_speed = torch.linalg.vector_norm(core.object_lin_vel, dim=-1).amax()
                object_ang_speed = torch.linalg.vector_norm(core.object_ang_vel, dim=-1).amax()
                robot_distance = torch.linalg.vector_norm(
                    core.object_pos - core.robot_root_states[:, :3], dim=-1
                ).amin()
                print(
                    f"[inactive-box-stress] step={step}/{args.steps} "
                    f"max_object_speed={float(object_speed.item()):.6f} "
                    f"max_object_ang_speed={float(object_ang_speed.item()):.6f} "
                    f"min_robot_distance={float(robot_distance.item()):.6f}",
                    flush=True,
                )
        print("[inactive-box-stress] PASS: all inactive boxes remained finite", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
