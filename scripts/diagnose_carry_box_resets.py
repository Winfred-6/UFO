"""Stress-test carry-box reference resets in the real MJLab environment.

This diagnostic deliberately bypasses the learning algorithm.  It loads every
motion in a carry-box PKL, resets one environment to every reference frame, and
checks the simulator state before and after a small number of zero-action
physics steps.  The report ties geometry penetrations or state explosions back
to an exact motion key and frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np
import torch

from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl
from humanoidverse.utils.torch_utils import my_quat_rotate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "humanoidverse/data/g1_largebox_train_near10s_ufo.pkl"


def _motion_count(path: Path) -> int:
    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty motion dictionary in {path}")
    return len(data)


def _iter_tensors(value: Any, prefix: str):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _iter_tensors(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _iter_tensors(child, f"{prefix}[{index}]")


def _nonfinite(value: Any, prefix: str) -> list[str]:
    failures: list[str] = []
    for name, tensor in _iter_tensors(value, prefix):
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
            failures.append(f"{name}: shape={tuple(tensor.shape)} first_bad={bad[0].tolist()}")
    return failures


def _core_state(core) -> dict[str, torch.Tensor]:
    names = (
        "robot_root_states",
        "dof_pos",
        "dof_vel",
        "body_pos",
        "body_rot",
        "body_vel",
        "body_ang_vel",
        "torques",
        "contact_forces",
        "object_pos",
        "object_quat",
        "object_lin_vel",
        "object_ang_vel",
        "hand_box_force",
    )
    return {name: getattr(core, name) for name in names if hasattr(core, name)}


def _box_min_corner_z(
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    *,
    collision_center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> torch.Tensor:
    device = object_pos.device
    dtype = object_pos.dtype
    center = torch.tensor(collision_center, device=device, dtype=dtype)
    half = torch.tensor(half_extents, device=device, dtype=dtype)
    signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    local_corners = center + signs * half
    count = object_pos.shape[0]
    rotated = my_quat_rotate(
        object_quat_xyzw[:, None, :].expand(-1, 8, -1).reshape(-1, 4),
        local_corners[None, :, :].expand(count, -1, -1).reshape(-1, 3),
    ).reshape(count, 8, 3)
    return (object_pos[:, None, :] + rotated)[:, :, 2].amin(dim=1)


def _point_box_signed_distance(
    point_world: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    *,
    collision_center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> torch.Tensor:
    """Signed point-to-OBB distance: positive outside, negative inside."""

    point_local = _point_box_local(
        point_world,
        object_pos,
        object_quat_xyzw,
        collision_center=collision_center,
    )
    half = torch.tensor(half_extents, device=point_world.device, dtype=point_world.dtype)
    q = torch.abs(point_local) - half
    outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.minimum(torch.amax(q, dim=-1), torch.zeros_like(outside))
    return outside + inside


def _point_box_local(
    point_world: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    *,
    collision_center: tuple[float, float, float],
) -> torch.Tensor:
    """Transform a world point into collision-center-relative box coordinates."""

    object_quat_inv = object_quat_xyzw.clone()
    object_quat_inv[:, :3] *= -1.0
    point_local = my_quat_rotate(object_quat_inv, point_world - object_pos)
    center = torch.tensor(collision_center, device=point_world.device, dtype=point_world.dtype)
    return point_local - center


def _target_states(core, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    motion = core._motion_lib.get_motion_state(motion_ids, motion_times, offset=core.env_origins)
    root_states = torch.cat([motion["root_pos"], motion["root_rot"], motion["root_vel"], motion["root_ang_vel"]], dim=-1)
    dof_states = torch.stack([motion["dof_pos"], motion["dof_vel"]], dim=-1)
    object_states = torch.cat(
        [
            motion["object_pos"],
            motion["object_quat"],
            motion["object_lin_vel"],
            motion["object_ang_vel"],
        ],
        dim=-1,
    )
    return (
        {
            "root_states": root_states,
            "dof_states": dof_states,
            "object_states": object_states,
            "object_valid": motion["object_valid"],
            "object_goal_pos": motion["object_goal_pos"],
        },
        motion,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--box-scale",
        type=float,
        nargs="+",
        default=(1.0,),
        metavar="S",
        help="One uniform scale or three XYZ scales applied around the mesh origin.",
    )
    parser.add_argument("--physics-steps", type=int, default=2)
    parser.add_argument(
        "--min-hand-clearance",
        type=float,
        default=0.02,
        help="Minimum signed wrist-origin clearance from the collision OBB for a reset frame.",
    )
    parser.add_argument("--max-post-step-object-speed", type=float, default=2.0)
    parser.add_argument("--max-post-step-object-angular-speed", type=float, default=10.0)
    parser.add_argument("--max-post-step-body-speed", type=float, default=20.0)
    parser.add_argument("--max-post-step-dof-speed", type=float, default=100.0)
    parser.add_argument("--max-post-step-contact-force", type=float, default=10000.0)
    parser.add_argument("--max-post-step-hand-box-force", type=float, default=5000.0)
    parser.add_argument("--max-post-step-ground-penetration", type=float, default=0.005)
    parser.add_argument(
        "--min-stage-hand-clearance",
        type=float,
        default=-0.10,
        help="Minimum wrist-origin signed OBB clearance for staged reference resets.",
    )
    parser.add_argument(
        "--max-stage-robot-penetration",
        type=float,
        default=0.03,
        help="Maximum shallow box/robot contact depth allowed during a staged reset.",
    )
    parser.add_argument("--max-stage-object-speed", type=float, default=8.0)
    parser.add_argument("--max-stage-object-angular-speed", type=float, default=30.0)
    parser.add_argument("--only-frame", type=int, default=None)
    parser.add_argument(
        "--write-reset-mask",
        type=Path,
        default=None,
        help="Write a copy of the input PKL with the measured object_reset_valid mask.",
    )
    parser.add_argument(
        "--write-stage-reset-mask",
        type=Path,
        default=None,
        help="Write object_stage_reset_valid after phase-aware dynamic certification.",
    )
    parser.add_argument(
        "--project-box-above-ground",
        action="store_true",
        help="Diagnose a hypothetical per-frame vertical projection before changing the dataset.",
    )
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ufo_carry_box_reset_diagnostic"))
    args = parser.parse_args()
    if len(args.box_scale) not in (1, 3):
        parser.error("--box-scale accepts either one uniform value or three XYZ values")
    box_scale = tuple(float(value) for value in args.box_scale)
    if len(box_scale) == 1:
        box_scale = box_scale * 3
    if any(not 0.0 < value <= 1.0 for value in box_scale):
        parser.error("every --box-scale value must be in (0, 1]")
    if (args.write_reset_mask is not None or args.write_stage_reset_mask is not None) and args.only_frame is not None:
        parser.error("writing a reset mask requires a full-frame scan")
    if args.write_reset_mask is not None and args.project_box_above_ground:
        parser.error("sanitize the PKL first; a hypothetical projection cannot be written as a reset mask")
    if args.write_stage_reset_mask is not None and args.project_box_above_ground:
        parser.error("sanitize the PKL first; a hypothetical projection cannot be written as a staged reset mask")

    data_path = args.data_path.expanduser().resolve()
    num_envs = _motion_count(data_path)
    cfg = build_ufo_mjlab_config(
        device=args.device,
        work_dir=str(args.work_dir),
        num_envs=num_envs,
        num_env_steps=num_envs,
        seed=args.seed,
        use_wandb=False,
        wandb_run_name=None,
        disable_eval_prioritization=True,
        agent="fb",
        data_path=str(data_path),
        data_mix_weights=None,
        buffer_size=num_envs,
        compile_agent=False,
        disable_dr=True,
        disable_obs_noise=True,
        task="carry_box",
    )
    base_carry_cfg = cfg.env.carry_box
    diagnostic_carry_cfg = base_carry_cfg.model_copy(
        update={
            "require_safe_reset_mask": False,
            # This diagnostic is also the explicit tool for examining legacy
            # resized datasets; formal training remains fail-closed.
            "require_native_reference_geometry": False,
            "half_extents": tuple(
                float(value) * scale for value, scale in zip(base_carry_cfg.half_extents, box_scale)
            ),
            "collision_center": tuple(
                float(value) * scale for value, scale in zip(base_carry_cfg.collision_center, box_scale)
            ),
        }
    )
    diagnostic_env_cfg = cfg.env.model_copy(update={"carry_box": diagnostic_carry_cfg})
    cfg = cfg.model_copy(update={"env": diagnostic_env_cfg})
    env, _ = cfg.env.build(num_envs=num_envs)
    core = env.base_env
    try:
        core._motion_lib.load_all_motions()
        lib = core._motion_lib
        if lib._num_motions != num_envs:
            raise RuntimeError(f"Expected {num_envs} loaded motions, got {lib._num_motions}")

        motion_ids = torch.arange(num_envs, device=core.device, dtype=torch.long)
        motion_keys = list(lib.curr_motion_keys)
        frame_counts = lib._motion_num_frames.long()
        max_frames = int(frame_counts.max().item())
        cfg_box = core.carry_box_cfg
        mj_model = core.mjlab_env.sim.mj_model
        geom_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"geom_{index}" for index in range(mj_model.ngeom)]
        box_geom_ids = {index for index, name in enumerate(geom_names) if "carry_box_collision" in name}
        if len(box_geom_ids) != 1:
            raise RuntimeError(f"Expected exactly one carry-box collision geom, got {box_geom_ids}")
        box_geom_id = next(iter(box_geom_ids))
        print(
            f"[diagnostic] box_geom={box_geom_id}:{geom_names[box_geom_id]} "
            f"ground_geoms={[name for name in geom_names if 'terrain' in name.lower() or 'ground' in name.lower()]}",
            flush=True,
        )

        penetration_frames = 0
        tested_frames = 0
        hand_origin_inside_frames = 0
        tested_hand_origins = 0
        clearance_samples: list[torch.Tensor] = []
        hand_local_samples: list[torch.Tensor] = []
        hand_midpoint_local_samples: list[torch.Tensor] = []
        torso_local_samples: list[torch.Tensor] = []
        clearance_thresholds = (-0.05, -0.02, 0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
        clearance_counts = {threshold: 0 for threshold in clearance_thresholds}
        preliminary_clearance_counts = {threshold: 0 for threshold in clearance_thresholds}
        safe_reset_counts = torch.zeros(num_envs, dtype=torch.long, device=core.device)
        stage_reset_counts = torch.zeros((num_envs, 5), dtype=torch.long, device=core.device)
        preliminary_safe_count = 0
        dynamic_rejection_counts = {
            "box_ground_penetration": 0,
            "box_robot_penetration": 0,
            "object_speed": 0,
            "object_angular_speed": 0,
            "body_speed": 0,
            "dof_speed": 0,
            "contact_force": 0,
            "hand_box_force": 0,
        }
        deepest = (float("inf"), "", -1)
        deepest_hand_origin = (float("-inf"), "", -1, "", ())
        safe_reset_mask = torch.zeros((num_envs, max_frames), dtype=torch.bool, device=core.device)
        stage_reset_mask = torch.zeros((num_envs, max_frames), dtype=torch.bool, device=core.device)
        largest: dict[str, tuple[float, str, int, int]] = {
            "object_speed": (0.0, "", -1, -1),
            "object_ang_speed": (0.0, "", -1, -1),
            "body_speed": (0.0, "", -1, -1),
            "dof_speed": (0.0, "", -1, -1),
            "contact_force": (0.0, "", -1, -1),
            "hand_box_force": (0.0, "", -1, -1),
        }
        deepest_contacts: dict[str, tuple[float, str, int, str, str]] = {
            "box_ground": (float("inf"), "", -1, "", ""),
            "box_robot": (float("inf"), "", -1, "", ""),
            "any": (float("inf"), "", -1, "", ""),
        }

        def update_peak(name: str, values: torch.Tensor, frame: int, physics_step: int) -> None:
            flat = values.reshape(num_envs, -1).amax(dim=1)
            value, env_id = torch.max(flat, dim=0)
            scalar = float(value.item())
            if scalar > largest[name][0]:
                index = int(env_id.item())
                largest[name] = (scalar, motion_keys[index], frame, physics_step)

        def update_contacts(
            frame: int,
            phase: str,
            active: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            robot_penetration = torch.zeros(num_envs, dtype=torch.bool, device=core.device)
            ground_penetration = torch.zeros(num_envs, dtype=torch.bool, device=core.device)
            robot_min_distance = torch.full((num_envs,), float("inf"), device=core.device)
            ground_min_distance = torch.full((num_envs,), float("inf"), device=core.device)
            sim_data = core.mjlab_env.sim.data
            contact_count = int(sim_data.nacon[0].item())
            if contact_count == 0:
                return robot_penetration, ground_penetration, robot_min_distance, ground_min_distance
            geom = sim_data.contact.geom[:contact_count].long()
            world = sim_data.contact.worldid[:contact_count].long()
            dist = sim_data.contact.dist[:contact_count]
            valid = (world >= 0) & (world < num_envs) & active[world.clamp(0, num_envs - 1)]
            valid &= (geom[:, 0] >= 0) & (geom[:, 1] >= 0) & torch.isfinite(dist)
            if not torch.any(valid):
                return robot_penetration, ground_penetration, robot_min_distance, ground_min_distance

            valid_indices = valid.nonzero(as_tuple=False).flatten()
            any_offset = torch.argmin(dist[valid_indices])
            any_index = int(valid_indices[int(any_offset.item())].item())

            def record(category: str, contact_index: int) -> None:
                value = float(dist[contact_index].item())
                if value >= deepest_contacts[category][0]:
                    return
                env_id = int(world[contact_index].item())
                ids = geom[contact_index].tolist()
                pair = f"{geom_names[ids[0]]} <-> {geom_names[ids[1]]}"
                deepest_contacts[category] = (value, motion_keys[env_id], frame, phase, pair)

            record("any", any_index)
            box_mask = valid & ((geom[:, 0] == box_geom_id) | (geom[:, 1] == box_geom_id))
            if not torch.any(box_mask):
                return robot_penetration, ground_penetration, robot_min_distance, ground_min_distance
            for category, want_ground in (("box_ground", True), ("box_robot", False)):
                candidates = box_mask.nonzero(as_tuple=False).flatten()
                selected: list[int] = []
                for contact_index in candidates.tolist():
                    ids = geom[contact_index].tolist()
                    other_id = ids[1] if ids[0] == box_geom_id else ids[0]
                    other_name = geom_names[other_id].lower()
                    is_ground = "terrain" in other_name or "ground" in other_name
                    env_id = int(world[contact_index].item())
                    contact_distance = float(dist[contact_index].item())
                    if is_ground and contact_distance < -float(args.max_post_step_ground_penetration):
                        ground_penetration[env_id] = True
                    elif not is_ground and contact_distance < -1.0e-4:
                        robot_penetration[env_id] = True
                    if is_ground:
                        ground_min_distance[env_id] = torch.minimum(
                            ground_min_distance[env_id],
                            dist[contact_index],
                        )
                    else:
                        robot_min_distance[env_id] = torch.minimum(
                            robot_min_distance[env_id],
                            dist[contact_index],
                        )
                    if is_ground == want_ground:
                        selected.append(contact_index)
                if selected:
                    selected_tensor = torch.tensor(selected, device=dist.device, dtype=torch.long)
                    offset = torch.argmin(dist[selected_tensor])
                    record(category, selected[int(offset.item())])
            return robot_penetration, ground_penetration, robot_min_distance, ground_min_distance

        frames = range(max_frames) if args.only_frame is None else (args.only_frame,)
        for frame in frames:
            if frame < 0 or frame >= max_frames:
                raise ValueError(f"--only-frame must be in [0, {max_frames - 1}], got {frame}")
            active = frame < frame_counts
            if not torch.any(active):
                continue
            active_ids = active.nonzero(as_tuple=False).flatten()
            times = torch.minimum(frame * lib._motion_dt, lib._motion_lengths)
            targets, motion = _target_states(core, motion_ids, times)
            failures = _nonfinite(targets, "reference")
            if failures:
                raise FloatingPointError(f"Non-finite reference at frame={frame}: {failures}")

            min_corner_z = _box_min_corner_z(
                motion["object_pos"],
                motion["object_quat"],
                collision_center=cfg_box.collision_center,
                half_extents=cfg_box.half_extents,
            )
            if args.project_box_above_ground:
                ground_lift = torch.clamp(-min_corner_z, min=0.0)
                targets["object_states"][:, 2] += ground_lift
                targets["object_goal_pos"][:, 2] += ground_lift
                motion["object_pos"][:, 2] += ground_lift
                motion["object_goal_pos"][:, 2] += ground_lift
                min_corner_z = min_corner_z + ground_lift

            hand_clearances = []
            frame_hand_local = []
            for hand_name in cfg_box.hand_body_names:
                hand_index = core.body_names.index(hand_name)
                local_point = _point_box_local(
                    motion["rg_pos"][:, hand_index],
                    motion["object_pos"],
                    motion["object_quat"],
                    collision_center=cfg_box.collision_center,
                )
                frame_hand_local.append(local_point)
                signed_clearance = _point_box_signed_distance(
                    motion["rg_pos"][:, hand_index],
                    motion["object_pos"],
                    motion["object_quat"],
                    collision_center=cfg_box.collision_center,
                    half_extents=cfg_box.half_extents,
                )
                hand_clearances.append(signed_clearance)
                tested_hand_origins += int(active.sum().item())
                hand_origin_inside_frames += int(((signed_clearance < 0.0) & active).sum().item())
                active_penetration = -signed_clearance[active]
                margin_value, margin_offset = torch.max(active_penetration, dim=0)
                margin_env = int(active_ids[int(margin_offset.item())].item())
                if float(margin_value.item()) > deepest_hand_origin[0]:
                    deepest_hand_origin = (
                        float(margin_value.item()),
                        motion_keys[margin_env],
                        frame,
                        hand_name,
                        (float(signed_clearance[margin_env].item()),),
                    )
            frame_hand_local = torch.stack(frame_hand_local, dim=1)
            hand_local_samples.append(frame_hand_local[active].detach().cpu())
            hand_midpoint_local_samples.append(frame_hand_local[active].mean(dim=1).detach().cpu())
            torso_index = core.body_names.index("torso_link")
            torso_local = _point_box_local(
                motion["rg_pos"][:, torso_index],
                motion["object_pos"],
                motion["object_quat"],
                collision_center=cfg_box.collision_center,
            )
            torso_local_samples.append(torso_local[active].detach().cpu())
            min_hand_clearance = torch.stack(hand_clearances, dim=-1).amin(dim=-1)
            active_clearance = min_hand_clearance[active]
            clearance_samples.append(active_clearance.detach().cpu())
            for threshold in clearance_thresholds:
                clearance_counts[threshold] += int((active_clearance >= threshold).sum().item())
            active_min_z = min_corner_z[active]
            tested_frames += int(active.sum().item())
            penetration_frames += int((active_min_z < 0.0).sum().item())
            min_value, active_offset = torch.min(active_min_z, dim=0)
            min_env = int(active_ids[int(active_offset.item())].item())
            if float(min_value.item()) < deepest[0]:
                deepest = (float(min_value.item()), motion_keys[min_env], frame)

            env.reset(to_numpy=False, target_states=targets)
            failures = _nonfinite(_core_state(core), "reset")
            if failures:
                raise FloatingPointError(f"Non-finite simulator reset motion={motion_keys[min_env]} frame={frame}: {failures}")
            robot_penetration, _, robot_min_distance, _ = update_contacts(frame, "reset", active)
            grounded = min_corner_z <= 2.0e-3
            reference_object_speed = torch.linalg.vector_norm(motion["object_lin_vel"], dim=-1)
            reference_object_angular_speed = torch.linalg.vector_norm(motion["object_ang_vel"], dim=-1)
            stationary = (reference_object_speed <= 0.25) & (reference_object_angular_speed <= 1.0)
            preliminary_safe = active & grounded & stationary & ~robot_penetration
            preliminary_safe_count += int(preliminary_safe.sum().item())
            for threshold in clearance_thresholds:
                preliminary_clearance_counts[threshold] += int(
                    (preliminary_safe & (min_hand_clearance >= threshold)).sum().item()
                )
            safe_reset = preliminary_safe & (min_hand_clearance >= float(args.min_hand_clearance))
            reference_stage = motion["object_phase"][:, 0].round().long()
            stage_safe_reset = (
                active
                & (motion["object_valid"][:, 0] > 0.5)
                & (reference_stage > 0)
                & (min_corner_z >= -float(args.max_post_step_ground_penetration))
                & (min_hand_clearance >= float(args.min_stage_hand_clearance))
                & (robot_min_distance >= -float(args.max_stage_robot_penetration))
                & (reference_object_speed <= float(args.max_stage_object_speed))
                & (reference_object_angular_speed <= float(args.max_stage_object_angular_speed))
            )
            frame_rejections = {
                name: torch.zeros(num_envs, dtype=torch.bool, device=core.device)
                for name in dynamic_rejection_counts
            }

            hold_action = (
                core.dof_pos - (core.default_dof_pos + core.default_dof_pos_offset)
            ) / torch.clamp(core.action_target_scale, min=1.0e-6)
            if bool(core.config.robot.control.normalize_action):
                hold_action *= float(core.config.robot.control.normalize_action_from) / float(
                    core.config.robot.control.normalize_action_to
                )
            hold_action = torch.clamp(
                hold_action,
                -float(core.config.robot.control.action_clip_value),
                float(core.config.robot.control.action_clip_value),
            )

            for physics_step in range(1, args.physics_steps + 1):
                result = env.step(hold_action, to_numpy=False)
                state = _core_state(core)
                failures = _nonfinite((state, result), "step")
                if failures:
                    bad_envs = (~torch.isfinite(core.object_pos)).any(dim=1).nonzero(as_tuple=False).flatten().tolist()
                    bad_keys = [motion_keys[index] for index in bad_envs]
                    raise FloatingPointError(
                        f"Non-finite physics state frame={frame} physics_step={physics_step} bad_motions={bad_keys}: {failures}"
                    )
                step_robot_penetration, step_ground_penetration, step_robot_min_distance, _ = update_contacts(
                    frame, f"step_{physics_step}", active
                )
                object_speed = torch.linalg.vector_norm(core.object_lin_vel, dim=-1)
                object_ang_speed = torch.linalg.vector_norm(core.object_ang_vel, dim=-1)
                body_speed = torch.linalg.vector_norm(core.body_vel, dim=-1).amax(dim=1)
                dof_speed = torch.abs(core.dof_vel).amax(dim=1)
                contact_force = torch.linalg.vector_norm(core.contact_forces, dim=-1).amax(dim=1)
                hand_box_force = core.hand_box_force.amax(dim=1)
                step_criteria = {
                    "box_ground_penetration": ~step_ground_penetration,
                    "box_robot_penetration": ~step_robot_penetration,
                    "object_speed": object_speed <= float(args.max_post_step_object_speed),
                    "object_angular_speed": object_ang_speed <= float(args.max_post_step_object_angular_speed),
                    "body_speed": body_speed <= float(args.max_post_step_body_speed),
                    "dof_speed": dof_speed <= float(args.max_post_step_dof_speed),
                    "contact_force": contact_force <= float(args.max_post_step_contact_force),
                    "hand_box_force": hand_box_force <= float(args.max_post_step_hand_box_force),
                }
                for name, criterion in step_criteria.items():
                    frame_rejections[name] |= ~criterion
                    safe_reset &= criterion
                stage_safe_reset &= (
                    (~step_ground_penetration)
                    & (step_robot_min_distance >= -float(args.max_stage_robot_penetration))
                    & (object_speed <= float(args.max_stage_object_speed))
                    & (object_ang_speed <= float(args.max_stage_object_angular_speed))
                    & (body_speed <= float(args.max_post_step_body_speed))
                    & (dof_speed <= float(args.max_post_step_dof_speed))
                    & (contact_force <= float(args.max_post_step_contact_force))
                    & (hand_box_force <= float(args.max_post_step_hand_box_force))
                )
                update_peak("object_speed", object_speed[:, None], frame, physics_step)
                update_peak("object_ang_speed", object_ang_speed[:, None], frame, physics_step)
                update_peak("body_speed", torch.linalg.vector_norm(core.body_vel, dim=-1), frame, physics_step)
                update_peak("dof_speed", torch.abs(core.dof_vel), frame, physics_step)
                update_peak("contact_force", torch.linalg.vector_norm(core.contact_forces, dim=-1), frame, physics_step)
                update_peak("hand_box_force", core.hand_box_force, frame, physics_step)

            safe_reset_counts += safe_reset.long()
            safe_reset_mask[:, frame] = safe_reset
            stage_reset_mask[:, frame] = stage_safe_reset
            for stage in range(1, 5):
                stage_reset_counts[:, stage] += (stage_safe_reset & (reference_stage == stage)).long()
            for name, rejected in frame_rejections.items():
                dynamic_rejection_counts[name] += int((preliminary_safe & rejected).sum().item())

            if args.only_frame is not None or frame % 25 == 0 or frame + 1 == max_frames:
                print(f"[diagnostic] frame={frame + 1}/{max_frames}", flush=True)

        clearance_values = torch.cat(clearance_samples) if clearance_samples else torch.empty(0)
        clearance_quantiles = (
            {
                str(quantile): float(torch.quantile(clearance_values, quantile).item())
                for quantile in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
            }
            if clearance_values.numel()
            else {}
        )

        def coordinate_quantiles(samples: list[torch.Tensor]) -> dict[str, list[float]]:
            if not samples:
                return {}
            values = torch.cat(samples).reshape(-1, 3)
            return {
                str(quantile): [float(value) for value in torch.quantile(values, quantile, dim=0)]
                for quantile in (0.0, 0.05, 0.5, 0.95, 1.0)
            }

        report = {
            "data_path": str(data_path),
            "box_scale": list(box_scale),
            "collision_center": list(cfg_box.collision_center),
            "half_extents": list(cfg_box.half_extents),
            "motions": num_envs,
            "tested_frames": tested_frames,
            "penetration_frames": penetration_frames,
            "penetration_fraction": penetration_frames / max(tested_frames, 1),
            "deepest_min_corner_z": deepest,
            "hand_origin_inside_frames": hand_origin_inside_frames,
            "hand_origin_inside_fraction": hand_origin_inside_frames / max(tested_hand_origins, 1),
            "deepest_hand_origin": deepest_hand_origin,
            "minimum_hand_clearance_m": float(args.min_hand_clearance),
            "hand_clearance_quantiles_m": clearance_quantiles,
            "hand_local_xyz_quantiles_m": coordinate_quantiles(hand_local_samples),
            "hand_midpoint_local_xyz_quantiles_m": coordinate_quantiles(hand_midpoint_local_samples),
            "torso_local_xyz_quantiles_m": coordinate_quantiles(torso_local_samples),
            "hand_clearance_counts": {str(key): value for key, value in clearance_counts.items()},
            "preliminary_safe_clearance_counts": {
                str(key): value for key, value in preliminary_clearance_counts.items()
            },
            "preliminary_safe_count": preliminary_safe_count,
            "dynamic_safe_count": int(safe_reset_counts.sum().item()),
            "staged_dynamic_safe_count": int(stage_reset_counts.sum().item()),
            "staged_safe_frames": {
                name: int(stage_reset_counts[:, stage].sum().item())
                for stage, name in enumerate(("inactive", "approach", "pickup", "transport", "place"))
            },
            "staged_safe_motions": {
                name: int((stage_reset_counts[:, stage] > 0).sum().item())
                for stage, name in enumerate(("inactive", "approach", "pickup", "transport", "place"))
            },
            "dynamic_rejection_counts": dynamic_rejection_counts,
            "deepest_contacts": deepest_contacts,
            "safe_reset_criteria": {
                "ground_gap_m": 2.0e-3,
                "initial_object_linear_speed_mps": 0.25,
                "initial_object_angular_speed_radps": 1.0,
                "minimum_hand_clearance_m": float(args.min_hand_clearance),
                "max_robot_penetration_m": 1.0e-4,
                "post_step_count": int(args.physics_steps),
                "max_post_step_object_speed_mps": float(args.max_post_step_object_speed),
                "max_post_step_object_angular_speed_radps": float(args.max_post_step_object_angular_speed),
                "max_post_step_body_speed_mps": float(args.max_post_step_body_speed),
                "max_post_step_dof_speed_radps": float(args.max_post_step_dof_speed),
                "max_post_step_contact_force_n": float(args.max_post_step_contact_force),
                "max_post_step_hand_box_force_n": float(args.max_post_step_hand_box_force),
                "max_post_step_ground_penetration_m": float(args.max_post_step_ground_penetration),
            },
            "stage_reset_criteria": {
                "minimum_hand_clearance_m": float(args.min_stage_hand_clearance),
                "max_robot_penetration_m": float(args.max_stage_robot_penetration),
                "max_object_speed_mps": float(args.max_stage_object_speed),
                "max_object_angular_speed_radps": float(args.max_stage_object_angular_speed),
                "post_step_count": int(args.physics_steps),
                "max_body_speed_mps": float(args.max_post_step_body_speed),
                "max_dof_speed_radps": float(args.max_post_step_dof_speed),
                "max_contact_force_n": float(args.max_post_step_contact_force),
                "max_hand_box_force_n": float(args.max_post_step_hand_box_force),
                "max_ground_penetration_m": float(args.max_post_step_ground_penetration),
            },
            "motions_without_safe_reset": [
                motion_keys[index] for index in (safe_reset_counts == 0).nonzero(as_tuple=False).flatten().tolist()
            ],
            "safe_reset_count_min": int(safe_reset_counts.min().item()),
            "safe_reset_count_median": int(torch.median(safe_reset_counts).item()),
            "safe_reset_count_max": int(safe_reset_counts.max().item()),
            "largest_after_physics_step": largest,
            "nonfinite_detected": False,
        }
        if args.write_reset_mask is not None:
            output_path = args.write_reset_mask.expanduser().resolve()
            data = joblib.load(data_path)
            if set(data) != set(motion_keys):
                raise RuntimeError("Loaded MotionLib keys do not match input PKL keys")
            mask_cpu = safe_reset_mask.detach().cpu().numpy()
            for motion_index, motion_key in enumerate(motion_keys):
                frame_count = int(frame_counts[motion_index].item())
                record = data[motion_key]
                record["object_reset_valid"] = mask_cpu[motion_index, :frame_count, None].astype(np.float32)
                metadata = dict(record.get("metadata") or {})
                metadata["object_reset_safety"] = {
                    "method": "mjlab_dynamic_collision_scan_v2",
                    "ground_tolerance_m": 2.0e-3,
                    "max_linear_speed_mps": 0.25,
                    "max_angular_speed_radps": 1.0,
                    "max_robot_penetration_m": 1.0e-4,
                    "minimum_hand_clearance_m": float(args.min_hand_clearance),
                    "post_step_count": int(args.physics_steps),
                    "max_post_step_object_speed_mps": float(args.max_post_step_object_speed),
                    "max_post_step_object_angular_speed_radps": float(args.max_post_step_object_angular_speed),
                    "max_post_step_body_speed_mps": float(args.max_post_step_body_speed),
                    "max_post_step_dof_speed_radps": float(args.max_post_step_dof_speed),
                    "max_post_step_contact_force_n": float(args.max_post_step_contact_force),
                    "max_post_step_hand_box_force_n": float(args.max_post_step_hand_box_force),
                    "max_post_step_ground_penetration_m": float(args.max_post_step_ground_penetration),
                    "valid_frames": int(safe_reset_counts[motion_index].item()),
                }
                record["metadata"] = metadata
            dump_ufo_pkl(data, output_path, "carry_box_reset_mask")
            report["reset_mask_output"] = str(output_path)
        if args.write_stage_reset_mask is not None:
            output_path = args.write_stage_reset_mask.expanduser().resolve()
            data = joblib.load(data_path)
            if set(data) != set(motion_keys):
                raise RuntimeError("Loaded MotionLib keys do not match input PKL keys")
            mask_cpu = stage_reset_mask.detach().cpu().numpy()
            stage_counts_cpu = stage_reset_counts.detach().cpu().numpy()
            for motion_index, motion_key in enumerate(motion_keys):
                frame_count = int(frame_counts[motion_index].item())
                record = data[motion_key]
                if "object_phase" not in record:
                    raise ValueError(
                        f"motion={motion_key!r} has no object_phase; run prepare_carry_box_curriculum.py first"
                    )
                record["object_stage_reset_valid"] = mask_cpu[
                    motion_index, :frame_count, None
                ].astype(np.float32)
                metadata = dict(record.get("metadata") or {})
                metadata["object_stage_reset_safety"] = {
                    "method": "mjlab_phase_dynamic_collision_scan_v1",
                    **report["stage_reset_criteria"],
                    "valid_frames_by_stage": stage_counts_cpu[motion_index].tolist(),
                }
                record["metadata"] = metadata
            dump_ufo_pkl(data, output_path, "carry_box_stage_reset_mask")
            report["stage_reset_mask_output"] = str(output_path)
        print(report, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
