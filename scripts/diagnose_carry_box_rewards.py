"""Measure carry reward semantics on every expert reference frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from humanoidverse.agents.envs.carry_box import (
    CARRY_STAGE_NAMES,
    adaptive_carry_thresholds,
    box_collision_geometry,
    carry_task_terms,
    hand_box_surface_geometry,
)
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.utils.motion_data.object_physics import classify_carry_stages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "humanoidverse/data/g1_largebox_full_ufo.pkl"


def _summary(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    tensor = torch.cat(values).float()
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "max": float(tensor.max().item()),
        "positive_fraction": float((tensor > 1.0e-6).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ufo_carry_reward_diagnostic"))
    args = parser.parse_args()

    data_path = args.data_path.expanduser().resolve()
    motion_count = len(joblib.load(data_path))
    cfg = build_ufo_mjlab_config(
        device=args.device,
        work_dir=str(args.work_dir),
        num_envs=motion_count,
        num_env_steps=motion_count,
        seed=4728,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        agent="fb",
        data_path=str(data_path),
        data_mix_weights=None,
        buffer_size=motion_count,
        compile_agent=False,
        disable_dr=True,
        disable_obs_noise=True,
        task="carry_box",
    )
    env, _ = cfg.env.build(num_envs=motion_count)
    core = env.base_env
    try:
        core._motion_lib.load_all_motions()
        lib = core._motion_lib
        motion_ids = torch.arange(motion_count, device=core.device, dtype=torch.long)
        frame_counts = lib._motion_num_frames.long()
        max_frames = int(frame_counts.max().item())
        diagnostic_phase = lib.object_phase.clone()
        for motion_id in range(lib._num_motions):
            start = int(lib.length_starts[motion_id].item())
            count = int(frame_counts[motion_id].item())
            valid = lib.object_valid[start : start + count, 0] > 0.5
            if not torch.any(valid) or torch.any(diagnostic_phase[start : start + count, 0][valid] > 0):
                continue
            phases = classify_carry_stages(
                lib.object_pos[start : start + count].detach().cpu().numpy(),
                lib.object_quat[start : start + count].detach().cpu().numpy(),
                lib.object_goal_pos[start : start + count].detach().cpu().numpy(),
                lib.object_valid[start : start + count].detach().cpu().numpy(),
                fps=1.0 / float(lib._motion_dt[motion_id].item()),
                collision_center=np.asarray(core.carry_box_cfg.collision_center, dtype=np.float32),
                half_extents=np.asarray(core.carry_box_cfg.half_extents, dtype=np.float32),
                lift_height_m=float(core.carry_box_cfg.lift_height),
                goal_tolerance_m=float(core.carry_box_cfg.goal_tolerance),
            )
            diagnostic_phase[start : start + count] = torch.as_tensor(
                phases,
                device=core.device,
                dtype=diagnostic_phase.dtype,
            )
        carry_scales = {
            key: value
            for key, value in build_fb_agent(device="cpu", compile=False, carry_box=True).aux_rewards_scaling.items()
            if key.startswith("carry_") or key == "box_overspeed_penalty"
        }
        keys = (*carry_scales, "lift_fraction", "grasp_quality", "transport_gate", "weighted_task_reward")
        samples = {
            stage: {key: [] for key in keys}
            for stage in range(1, len(CARRY_STAGE_NAMES))
        }
        previous_hand_distance = torch.zeros(motion_count, device=core.device)
        previous_goal_distance = torch.zeros(motion_count, device=core.device)
        previous_lift_fraction = torch.zeros(motion_count, device=core.device)
        ever_lifted = torch.zeros(motion_count, device=core.device, dtype=torch.bool)

        for frame in range(max_frames):
            active = frame < frame_counts
            if not torch.any(active):
                continue
            times = torch.minimum(frame * lib._motion_dt, lib._motion_lengths)
            motion = lib.get_motion_state(motion_ids, times, offset=core.env_origins)
            hand_pos = motion["rg_pos"][:, core.hand_body_indices]
            hand_distance = hand_box_surface_geometry(
                hand_pos=hand_pos,
                object_pos=motion["object_pos"],
                object_quat_xyzw=motion["object_quat"],
                cfg=core.carry_box_cfg,
            )[0].mean(dim=-1)
            goal_distance = torch.linalg.vector_norm(motion["object_goal_pos"] - motion["object_pos"], dim=-1)
            _center, bottom_height, _axes = box_collision_geometry(
                object_pos=motion["object_pos"],
                object_quat_xyzw=motion["object_quat"],
                cfg=core.carry_box_cfg,
            )
            lift_height = adaptive_carry_thresholds(core.carry_box_cfg)["lift_height"]
            lift_fraction = torch.clamp(
                (bottom_height - core.env_origins[:, 2]) / lift_height,
                min=0.0,
                max=1.0,
            )
            if frame == 0:
                previous_hand_distance.copy_(hand_distance)
                previous_goal_distance.copy_(goal_distance)
                previous_lift_fraction.copy_(lift_fraction)

            valid = motion["object_valid"][:, 0] * active.float()
            aux, state = carry_task_terms(
                hand_pos=hand_pos,
                bilateral_contact=torch.zeros(motion_count, dtype=torch.bool, device=core.device),
                object_pos=motion["object_pos"],
                object_quat_xyzw=motion["object_quat"],
                object_lin_vel=motion["object_lin_vel"],
                object_ang_vel=motion["object_ang_vel"],
                goal_pos=motion["object_goal_pos"],
                valid=valid,
                ground_height=core.env_origins[:, 2],
                ever_lifted=ever_lifted,
                prev_hand_distance=previous_hand_distance,
                prev_goal_distance=previous_goal_distance,
                prev_lift_fraction=previous_lift_fraction,
                cfg=core.carry_box_cfg,
            )
            weighted = torch.zeros(motion_count, device=core.device)
            for key, scale in carry_scales.items():
                weighted += float(scale) * aux[key]
            frame_ids = torch.minimum(
                torch.full_like(frame_counts, frame),
                frame_counts - 1,
            )
            phase = diagnostic_phase[lib.length_starts.long() + frame_ids, 0].round().long()
            finite_values = [*aux.values(), *state.values(), weighted]
            if any(value.is_floating_point() and not torch.isfinite(value).all() for value in finite_values):
                raise FloatingPointError(f"Non-finite expert carry reward at frame={frame}")
            for stage in range(1, len(CARRY_STAGE_NAMES)):
                mask = active & (phase == stage)
                if not torch.any(mask):
                    continue
                for key in carry_scales:
                    samples[stage][key].append(aux[key][mask].detach().cpu())
                for key in ("lift_fraction", "grasp_quality", "transport_gate"):
                    samples[stage][key].append(state[key][mask].detach().cpu())
                samples[stage]["weighted_task_reward"].append(weighted[mask].detach().cpu())

            previous_hand_distance[active] = state["hand_distance"][active]
            previous_goal_distance[active] = state["goal_distance"][active]
            previous_lift_fraction[active] = state["lift_fraction"][active]
            ever_lifted[active] = state["ever_lifted"][active]

        report = {
            "data_path": str(data_path),
            "motions": motion_count,
            "thresholds": adaptive_carry_thresholds(core.carry_box_cfg),
            "reward_scales": carry_scales,
            "stages": {
                CARRY_STAGE_NAMES[stage]: {
                    key: _summary(values)
                    for key, values in samples[stage].items()
                }
                for stage in range(1, len(CARRY_STAGE_NAMES))
            },
        }
        print(report, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
