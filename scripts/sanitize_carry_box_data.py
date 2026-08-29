"""Project carry-box trajectories out of the ground before physics use."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.agents.envs.carry_box import DEFAULT_LARGEBOX_MESH, DEFAULT_LARGEBOX_MESH_SCALE
from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl
from humanoidverse.utils.motion_data.object_physics import (
    mesh_axis_aligned_bounds,
    retarget_object_collision_geometry,
    sanitize_object_ground_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_LARGEBOX_MESH)
    parser.add_argument("--mesh-scale", type=float, nargs=3, default=DEFAULT_LARGEBOX_MESH_SCALE)
    parser.add_argument("--transition-clearance", type=float, default=0.12)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    source_center, source_half_extents = mesh_axis_aligned_bounds(args.mesh)
    mesh_scale = np.asarray(args.mesh_scale, dtype=np.float32)
    if mesh_scale.shape != (3,) or np.any(~np.isfinite(mesh_scale)) or np.any(mesh_scale <= 0.0):
        parser.error("--mesh-scale must contain three positive finite values")
    center = source_center * mesh_scale
    half_extents = source_half_extents * mesh_scale
    data = joblib.load(input_path)
    total_frames = 0
    projected_frames = 0
    maximum_lift = 0.0
    for motion_key, record in data.items():
        if "object_valid" not in record:
            continue
        retargeted_pos, geometry_shift = retarget_object_collision_geometry(
            record["object_pos"],
            record["object_quat"],
            source_collision_center=source_center,
            source_half_extents=source_half_extents,
            target_collision_center=center,
            target_half_extents=half_extents,
            transition_clearance_m=float(args.transition_clearance),
        )
        object_pos, object_lin_vel, object_goal_pos, lift = sanitize_object_ground_trajectory(
            retargeted_pos,
            record["object_quat"],
            fps=float(record["fps"]),
            collision_center=center,
            half_extents=half_extents,
        )
        record["object_pos"] = object_pos
        record["object_lin_vel"] = object_lin_vel
        record["object_goal_pos"] = object_goal_pos
        record.pop("object_reset_valid", None)
        record.pop("object_stage_reset_valid", None)
        record.pop("object_phase", None)
        metadata = dict(record.get("metadata") or {})
        metadata["object_ground_projection"] = {
            "applied_frames": int(np.count_nonzero(lift > 0.0)),
            "max_lift_m": float(np.max(lift, initial=0.0)),
        }
        metadata["object_geometry_retarget"] = {
            "method": "ground_anchored_mesh_scale_v1",
            "mesh_scale": mesh_scale.tolist(),
            "transition_clearance_m": float(args.transition_clearance),
            "vertical_shift_min_m": float(np.min(geometry_shift, initial=0.0)),
            "vertical_shift_max_m": float(np.max(geometry_shift, initial=0.0)),
            "source_collision_center": source_center.tolist(),
            "source_half_extents": source_half_extents.tolist(),
            "target_collision_center": center.tolist(),
            "target_half_extents": half_extents.tolist(),
        }
        record["metadata"] = metadata
        total_frames += len(object_pos)
        projected_frames += int(np.count_nonzero(lift > 0.0))
        maximum_lift = max(maximum_lift, float(np.max(lift, initial=0.0)))

    dump_ufo_pkl(data, output_path, "carry_box_ground_sanitizer")
    print(
        {
            "input": str(input_path),
            "output": str(output_path),
            "motions": len(data),
            "frames": total_frames,
            "projected_frames": projected_frames,
            "max_lift_m": maximum_lift,
            "collision_center": center.tolist(),
            "half_extents": half_extents.tolist(),
            "mesh_scale": mesh_scale.tolist(),
        }
    )


if __name__ == "__main__":
    main()
