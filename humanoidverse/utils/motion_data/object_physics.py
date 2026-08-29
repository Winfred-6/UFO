"""Physics-consistency helpers for rigid-object reference trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_GROUND_CLEARANCE_M = 1.0e-3
CARRY_STAGE_INACTIVE = 0
CARRY_STAGE_APPROACH = 1
CARRY_STAGE_PICKUP = 2
CARRY_STAGE_TRANSPORT = 3
CARRY_STAGE_PLACE = 4


def mesh_axis_aligned_bounds(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the local-space center and half extents of an OBJ vertex cloud."""

    path = Path(mesh_path).expanduser().resolve()
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            vertex = np.fromstring(line[2:], sep=" ", dtype=np.float64)
            if vertex.shape != (3,) or not np.all(np.isfinite(vertex)):
                raise ValueError(f"OBJ={path} contains an invalid vertex line: {line.rstrip()!r}")
            minimum = vertex.copy() if minimum is None else np.minimum(minimum, vertex)
            maximum = vertex.copy() if maximum is None else np.maximum(maximum, vertex)
    if minimum is None or maximum is None:
        raise ValueError(f"OBJ={path} contains no vertices")
    center = ((minimum + maximum) * 0.5).astype(np.float32)
    half_extents = ((maximum - minimum) * 0.5).astype(np.float32)
    if np.any(half_extents <= 0.0):
        raise ValueError(f"OBJ={path} has degenerate bounds: half_extents={half_extents.tolist()}")
    return center, half_extents


def oriented_box_min_corner_z(
    object_pos: np.ndarray,
    object_quat_xyzw: np.ndarray,
    *,
    collision_center: np.ndarray,
    half_extents: np.ndarray,
) -> np.ndarray:
    """Compute the lowest world-space corner for every oriented box pose."""

    pos = np.asarray(object_pos, dtype=np.float64)
    quat = np.asarray(object_quat_xyzw, dtype=np.float64)
    center = np.asarray(collision_center, dtype=np.float64)
    half = np.asarray(half_extents, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"object_pos must have shape [T, 3], got {pos.shape}")
    if quat.shape != (len(pos), 4):
        raise ValueError(f"object_quat_xyzw must have shape [{len(pos)}, 4], got {quat.shape}")
    if center.shape != (3,) or half.shape != (3,) or np.any(half <= 0.0):
        raise ValueError(f"Invalid collision bounds center={center.shape}, half_extents={half.tolist()}")

    signs = np.asarray(
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
        dtype=np.float64,
    )
    corners = center[None, :] + signs * half[None, :]
    rotation_matrices = Rotation.from_quat(quat).as_matrix()
    rotated_corners = np.einsum("tij,cj->tci", rotation_matrices, corners)
    return pos[:, 2] + np.min(rotated_corners[:, :, 2], axis=1)


def project_oriented_box_above_ground(
    object_pos: np.ndarray,
    object_quat_xyzw: np.ndarray,
    *,
    collision_center: np.ndarray,
    half_extents: np.ndarray,
    clearance_m: float = DEFAULT_GROUND_CLEARANCE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift only invalid poses so no collision-box corner starts below z=0."""

    if not np.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError(f"clearance_m must be finite and non-negative, got {clearance_m}")
    projected = np.asarray(object_pos, dtype=np.float32).copy()
    minimum_z = oriented_box_min_corner_z(
        projected,
        object_quat_xyzw,
        collision_center=collision_center,
        half_extents=half_extents,
    )
    lift = np.maximum(float(clearance_m) - minimum_z, 0.0).astype(np.float32)
    projected[:, 2] += lift
    return projected, lift


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """Forward finite difference with a repeated final sample."""

    result = np.zeros_like(values, dtype=np.float32)
    if len(values) <= 1:
        return result
    result[:-1] = np.diff(values, axis=0) * float(fps)
    result[-1] = result[-2]
    return result


def sanitize_object_ground_trajectory(
    object_pos: np.ndarray,
    object_quat_xyzw: np.ndarray,
    *,
    fps: float,
    collision_center: np.ndarray,
    half_extents: np.ndarray,
    goal_window_seconds: float = 0.2,
    clearance_m: float = DEFAULT_GROUND_CLEARANCE_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a trajectory above ground and recompute dependent quantities."""

    projected, lift = project_oriented_box_above_ground(
        object_pos,
        object_quat_xyzw,
        collision_center=collision_center,
        half_extents=half_extents,
        clearance_m=clearance_m,
    )
    linear_velocity = finite_difference(projected, fps)
    goal_window = max(1, int(round(float(goal_window_seconds) * float(fps))))
    goal = np.median(projected[-goal_window:], axis=0).astype(np.float32)
    goal_pos = np.repeat(goal[None, :], len(projected), axis=0)
    return projected, linear_velocity, goal_pos, lift


def retarget_object_collision_geometry(
    object_pos: np.ndarray,
    object_quat_xyzw: np.ndarray,
    *,
    source_collision_center: np.ndarray,
    source_half_extents: np.ndarray,
    target_collision_center: np.ndarray,
    target_half_extents: np.ndarray,
    transition_clearance_m: float = 0.12,
    clearance_m: float = DEFAULT_GROUND_CLEARANCE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-anchor a resized object while preserving its airborne origin.

    Scaling a mesh around its local origin makes a grounded smaller box float.
    A global vertical offset would instead pull the held reference away from
    the hands.  This function exactly preserves the source bottom height on
    grounded frames, smoothly fades that correction through the pickup/place
    transition, and leaves clearly airborne frames unchanged.
    """

    if not np.isfinite(transition_clearance_m) or transition_clearance_m <= 0.0:
        raise ValueError("transition_clearance_m must be positive and finite")
    source_bottom = oriented_box_min_corner_z(
        object_pos,
        object_quat_xyzw,
        collision_center=source_collision_center,
        half_extents=source_half_extents,
    )
    target_bottom = oriented_box_min_corner_z(
        object_pos,
        object_quat_xyzw,
        collision_center=target_collision_center,
        half_extents=target_half_extents,
    )
    source_ground = float(np.quantile(source_bottom, 0.02))
    source_clearance = np.maximum(source_bottom - source_ground, 0.0)
    blend = np.clip(source_clearance / float(transition_clearance_m), 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    correction = ((source_bottom - target_bottom) * (1.0 - blend)).astype(np.float32)
    retargeted = np.asarray(object_pos, dtype=np.float32).copy()
    retargeted[:, 2] += correction
    retargeted, projection = project_oriented_box_above_ground(
        retargeted,
        object_quat_xyzw,
        collision_center=target_collision_center,
        half_extents=target_half_extents,
        clearance_m=clearance_m,
    )
    correction += projection
    return retargeted, correction


def classify_carry_stages(
    object_pos: np.ndarray,
    object_quat_xyzw: np.ndarray,
    object_goal_pos: np.ndarray,
    object_valid: np.ndarray,
    *,
    fps: float,
    collision_center: np.ndarray,
    half_extents: np.ndarray,
    lift_height_m: float,
    goal_tolerance_m: float,
    pickup_lead_seconds: float = 0.5,
) -> np.ndarray:
    """Label approach, pickup, transport, and place phases from object motion.

    Labels are training internals used only for reference-state reset sampling;
    they are never exposed to the policy.  Boundaries are derived from OBB
    bottom clearance and target distance, making the procedure invariant to
    box yaw, trajectory translation, and augmentation naming.
    """

    pos = np.asarray(object_pos, dtype=np.float64)
    quat = np.asarray(object_quat_xyzw, dtype=np.float64)
    goal = np.asarray(object_goal_pos, dtype=np.float64)
    valid = np.asarray(object_valid, dtype=np.float64).reshape(-1) > 0.5
    if len(pos) == 0 or quat.shape != (len(pos), 4) or goal.shape != (len(pos), 3):
        raise ValueError("carry stage inputs must have synchronized non-empty [T, ...] shapes")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")

    stages = np.full((len(pos),), CARRY_STAGE_INACTIVE, dtype=np.int64)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return stages[:, None].astype(np.float32)

    bottom = oriented_box_min_corner_z(
        pos,
        quat,
        collision_center=np.asarray(collision_center),
        half_extents=np.asarray(half_extents),
    )
    ground_level = float(np.quantile(bottom[valid], 0.02))
    clearance = np.maximum(bottom - ground_level, 0.0)
    goal_distance_xy = np.linalg.norm(goal[:, :2] - pos[:, :2], axis=-1)
    active_clearance = clearance[valid]
    robust_carry_height = max(
        float(lift_height_m),
        0.65 * float(np.quantile(active_clearance, 0.90)),
    )
    lift_onset_threshold = max(0.02, 0.15 * float(lift_height_m))
    onset_candidates = np.flatnonzero(valid & (clearance >= lift_onset_threshold))
    if len(onset_candidates) == 0:
        stages[valid] = CARRY_STAGE_APPROACH
        return stages[:, None].astype(np.float32)

    lift_onset = int(onset_candidates[0])
    lead_frames = max(1, int(round(float(pickup_lead_seconds) * float(fps))))
    pickup_start = max(int(valid_indices[0]), lift_onset - lead_frames)

    transport_candidates = np.flatnonzero(valid & (clearance >= robust_carry_height))
    transport_start = int(transport_candidates[0]) if len(transport_candidates) else lift_onset
    transport_start = max(transport_start, pickup_start + 1)

    place_candidates = np.flatnonzero(
        valid
        & (np.arange(len(pos)) > transport_start)
        & (goal_distance_xy <= float(goal_tolerance_m))
    )
    if len(place_candidates):
        place_start = int(place_candidates[0])
    else:
        last_valid = int(valid_indices[-1])
        place_start = max(transport_start + 1, last_valid - lead_frames)
    place_start = min(place_start, int(valid_indices[-1]))

    frame_ids = np.arange(len(pos))
    stages[valid & (frame_ids < pickup_start)] = CARRY_STAGE_APPROACH
    stages[valid & (frame_ids >= pickup_start) & (frame_ids < transport_start)] = CARRY_STAGE_PICKUP
    stages[valid & (frame_ids >= transport_start) & (frame_ids < place_start)] = CARRY_STAGE_TRANSPORT
    stages[valid & (frame_ids >= place_start)] = CARRY_STAGE_PLACE
    return stages[:, None].astype(np.float32)
