"""Add deterministic carry-stage labels before physics certification.

The generated labels are training-only reset metadata.  They do not enter the
actor observation or discriminator input.  ``object_stage_reset_valid`` starts
at zero for every object frame and must subsequently be populated by
``diagnose_carry_box_resets.py --write-stage-reset-mask``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from humanoidverse.agents.envs.carry_box import (
    CarryBoxConfig,
    DEFAULT_LARGEBOX_MESH,
    DEFAULT_LARGEBOX_MESH_SCALE,
)
from humanoidverse.utils.motion_data.adapters import dump_ufo_pkl
from humanoidverse.utils.motion_data.object_physics import classify_carry_stages, mesh_axis_aligned_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_LARGEBOX_MESH)
    parser.add_argument("--mesh-scale", type=float, nargs=3, default=DEFAULT_LARGEBOX_MESH_SCALE)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    data = joblib.load(input_path)
    center, half_extents = mesh_axis_aligned_bounds(args.mesh)
    mesh_scale = np.asarray(args.mesh_scale, dtype=np.float32)
    if mesh_scale.shape != (3,) or np.any(~np.isfinite(mesh_scale)) or np.any(mesh_scale <= 0.0):
        parser.error("--mesh-scale must contain three positive finite values")
    center = center * mesh_scale
    half_extents = half_extents * mesh_scale
    cfg = CarryBoxConfig(
        collision_center=tuple(float(value) for value in center),
        half_extents=tuple(float(value) for value in half_extents),
    )
    thresholds = {
        "lift_height_m": max(float(cfg.lift_height), 0.5 * float(half_extents[2])),
        "goal_tolerance_m": max(
            float(cfg.goal_tolerance),
            0.70 * float(np.linalg.norm(half_extents[:2])),
        ),
    }
    stage_counts = np.zeros(5, dtype=np.int64)
    for motion_key, record in data.items():
        if "object_valid" not in record:
            continue
        phases = classify_carry_stages(
            record["object_pos"],
            record["object_quat"],
            record["object_goal_pos"],
            record["object_valid"],
            fps=float(record["fps"]),
            collision_center=center,
            half_extents=half_extents,
            lift_height_m=thresholds["lift_height_m"],
            goal_tolerance_m=thresholds["goal_tolerance_m"],
        )
        record["object_phase"] = phases
        # Certification is deliberately fail-closed.  A geometry label alone
        # never authorizes a dynamic mid-trajectory reset.
        record["object_stage_reset_valid"] = np.zeros_like(phases, dtype=np.float32)
        counts = np.bincount(phases[:, 0].astype(np.int64), minlength=5)
        stage_counts += counts
        metadata = dict(record.get("metadata") or {})
        metadata["carry_stage_labels"] = {
            "method": "obb_clearance_goal_v1",
            "stage_counts": counts.tolist(),
            **thresholds,
        }
        record["metadata"] = metadata

    dump_ufo_pkl(data, output_path, "carry_box_stage_labels")
    print(
        {
            "input": str(input_path),
            "output": str(output_path),
            "motions": len(data),
            "stage_counts": stage_counts.tolist(),
            "collision_center": center.tolist(),
            "half_extents": half_extents.tolist(),
            "mesh_scale": mesh_scale.tolist(),
            **thresholds,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
