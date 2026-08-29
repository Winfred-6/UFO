"""Reader for paired HHTools G1 robot and rigid-object CSV trajectories."""

from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_data.object_physics import (
    finite_difference,
    mesh_axis_aligned_bounds,
    sanitize_object_ground_trajectory,
)
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_to_ufo_motion
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec

_AUGMENTATION_SUFFIX = re.compile(r"_(?P<kind>rot|trans)_(?P<index>\d+)$")


def _resolve_source_roots(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    base_dir: Path | None,
) -> list[Path]:
    specs = [path_spec] if isinstance(path_spec, (str, os.PathLike)) else list(path_spec)
    roots: list[Path] = []
    for raw_spec in specs:
        expanded = Path(str(raw_spec)).expanduser()
        candidates = [expanded] if expanded.is_absolute() else [*([] if base_dir is None else [base_dir / expanded]), Path.cwd() / expanded]
        matches: list[Path] = []
        for candidate in candidates:
            matches.extend(Path(match) for match in glob.glob(str(candidate)))
            if matches:
                break
        if not matches:
            raise FileNotFoundError(f"Paired robot/object CSV source does not exist: {raw_spec}")
        roots.extend(path.resolve() for path in matches)

    unique_roots = sorted(set(roots))
    non_dirs = [path for path in unique_roots if not path.is_dir()]
    if non_dirs:
        raise ValueError(f"Paired robot/object CSV sources must be directories, got: {non_dirs}")
    return unique_roots


def _read_commented_csv(path: Path) -> tuple[dict[str, str], list[str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    table_lines: list[str] = []
    with path.open("r", newline="") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                key_value = stripped[1:].strip().split(":", 1)
                if len(key_value) == 2:
                    metadata[key_value[0].strip()] = key_value[1].strip()
                continue
            table_lines.append(raw_line)
    if not table_lines:
        raise ValueError(f"CSV file contains no table rows: {path}")
    reader = csv.DictReader(table_lines)
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
    if not fieldnames or not rows:
        raise ValueError(f"CSV file contains no data rows: {path}")
    return metadata, fieldnames, rows


def _matrix(rows: list[dict[str, str]], columns: list[str], path: Path) -> np.ndarray:
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"CSV file={path} is missing columns: {missing}")
    try:
        return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV file={path} contains a non-numeric value in columns={columns}") from exc


def _times_and_fps(
    rows: list[dict[str, str]],
    path: Path,
    metadata: dict[str, str],
    configured_fps: float | int | None,
) -> tuple[np.ndarray, float]:
    times = _matrix(rows, ["time"], path).reshape(-1).astype(np.float64)
    if times.size < 2:
        raise ValueError(f"CSV file={path} needs at least two time samples")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError(f"CSV file={path} time values must be strictly increasing")
    inferred_fps = 1.0 / float(np.median(dt))
    metadata_fps = float(metadata["sample_rate"]) if "sample_rate" in metadata else None
    fps = float(configured_fps) if configured_fps is not None else (metadata_fps or inferred_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"CSV file={path} has invalid fps={fps}")
    if abs(inferred_fps - fps) / fps > 0.01:
        raise ValueError(f"CSV file={path} time-derived fps={inferred_fps:.6f} disagrees with fps={fps:.6f}")
    return times, fps


def _normalize_quaternions(quat_xyzw: np.ndarray, path: Path) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).copy()
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-8):
        raise ValueError(f"CSV file={path} contains zero or non-finite quaternions")
    quat /= norms
    for frame in range(1, len(quat)):
        if float(np.dot(quat[frame - 1], quat[frame])) < 0.0:
            quat[frame] *= -1.0
    return quat


def _angular_velocity_world(quat_xyzw: np.ndarray, fps: float) -> np.ndarray:
    result = np.zeros((len(quat_xyzw), 3), dtype=np.float32)
    if len(quat_xyzw) <= 1:
        return result
    rotations = Rotation.from_quat(quat_xyzw.astype(np.float64))
    delta = rotations[1:] * rotations[:-1].inv()
    result[:-1] = (delta.as_rotvec() * float(fps)).astype(np.float32)
    result[-1] = result[-2]
    return result


def _sequence_metadata(sequence_name: str) -> tuple[str, str, int | None]:
    match = _AUGMENTATION_SUFFIX.search(sequence_name)
    if match is None:
        return sequence_name, "original", None
    return sequence_name[: match.start()], match.group("kind"), int(match.group("index"))


def _sequence_directories(root: Path) -> list[Path]:
    candidates = sorted({path.parent for path in root.rglob("object_0_largebox.csv")})
    if not candidates and (root / "object_0_largebox.csv").exists():
        candidates = [root]
    if not candidates:
        raise ValueError(f"No object_0_largebox.csv files were found below {root}")
    return candidates


def load_paired_hhtools_csv(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    """Convert every paired sequence below ``path_spec`` to the UFO schema.

    Each sequence directory must contain ``<directory-name>.csv`` for the G1
    state, ``object_0_largebox.csv`` for the rigid object, and optionally the
    referenced OBJ mesh. Robot and object timestamps are required to align.
    """

    converted: dict[str, Any] = {}
    for root in _resolve_source_roots(path_spec, base_dir=base_dir):
        for sequence_dir in _sequence_directories(root):
            sequence_name = sequence_dir.name
            robot_path = sequence_dir / f"{sequence_name}.csv"
            object_path = sequence_dir / "object_0_largebox.csv"
            mesh_path = sequence_dir / "largebox_cleaned_simplified.obj"
            if not robot_path.exists():
                raise FileNotFoundError(f"Missing paired robot CSV: {robot_path}")

            robot_meta, robot_fields, robot_rows = _read_commented_csv(robot_path)
            object_meta, _object_fields, object_rows = _read_commented_csv(object_path)
            robot_times, motion_fps = _times_and_fps(robot_rows, robot_path, robot_meta, fps)
            object_times, object_fps = _times_and_fps(object_rows, object_path, object_meta, fps)
            if len(robot_times) != len(object_times) or not np.allclose(robot_times, object_times, atol=1.0e-6, rtol=0.0):
                raise ValueError(
                    f"Robot/object timestamps do not align for sequence={sequence_name}: "
                    f"robot_frames={len(robot_times)}, object_frames={len(object_times)}"
                )
            if abs(motion_fps - object_fps) > 1.0e-5:
                raise ValueError(f"Robot/object fps do not match for sequence={sequence_name}: {motion_fps} vs {object_fps}")

            dof_columns = [f"dof_{joint_name}" for joint_name in robot_spec.control_joint_names]
            missing_dofs = [column for column in dof_columns if column not in robot_fields]
            if missing_dofs:
                raise ValueError(f"Robot CSV={robot_path} is missing control-joint columns: {missing_dofs}")
            root_pos = _matrix(robot_rows, ["root_x", "root_y", "root_z"], robot_path)
            root_quat = _normalize_quaternions(
                _matrix(robot_rows, ["root_qx", "root_qy", "root_qz", "root_qw"], robot_path), robot_path
            )
            dof_pos = _matrix(robot_rows, dof_columns, robot_path)

            base_sequence_id, augmentation, augmentation_index = _sequence_metadata(sequence_name)
            robot_motion = RobotStateMotion(
                motion_key=sequence_name,
                root_pos=root_pos,
                root_quat=root_quat,
                dof_pos=dof_pos,
                fps=motion_fps,
                joint_names=list(robot_spec.control_joint_names),
                source=source_name,
                metadata={
                    "path": str(robot_path),
                    "reader": "robot_state_object_csv",
                    "base_sequence_id": base_sequence_id,
                    "augmentation": augmentation,
                    "augmentation_index": augmentation_index,
                },
            )
            record = robot_state_to_ufo_motion(robot_motion, robot_spec, source_name)

            object_pos = _matrix(object_rows, ["pos_x", "pos_y", "pos_z"], object_path)
            object_quat = _normalize_quaternions(
                _matrix(object_rows, ["quat_x", "quat_y", "quat_z", "quat_w"], object_path), object_path
            )
            if mesh_path.exists():
                collision_center, half_extents = mesh_axis_aligned_bounds(mesh_path)
                object_pos, object_lin_vel, object_goal_pos, ground_lift = sanitize_object_ground_trajectory(
                    object_pos,
                    object_quat,
                    fps=motion_fps,
                    collision_center=collision_center,
                    half_extents=half_extents,
                )
            else:
                object_lin_vel = finite_difference(object_pos, motion_fps)
                goal_window = max(1, int(round(0.2 * motion_fps)))
                goal_pos = np.median(object_pos[-goal_window:], axis=0).astype(np.float32)
                object_goal_pos = np.repeat(goal_pos[None, :], len(object_pos), axis=0)
                ground_lift = np.zeros(len(object_pos), dtype=np.float32)
            record.update(
                {
                    "object_pos": object_pos,
                    "object_quat": object_quat,
                    "object_lin_vel": object_lin_vel,
                    "object_ang_vel": _angular_velocity_world(object_quat, motion_fps),
                    "object_valid": np.ones((len(object_pos), 1), dtype=np.float32),
                    "object_goal_pos": object_goal_pos,
                    "object_name": str(object_meta.get("object", "largebox")),
                    "object_mesh_path": str(mesh_path.resolve()) if mesh_path.exists() else None,
                    "base_sequence_id": base_sequence_id,
                    "augmentation": augmentation,
                }
            )
            metadata = dict(record.get("metadata") or {})
            metadata.update(
                {
                    "object_path": str(object_path),
                    "object_mesh_path": str(mesh_path.resolve()) if mesh_path.exists() else None,
                    "object_quat_order": "xyzw",
                    "base_sequence_id": base_sequence_id,
                    "augmentation": augmentation,
                    "augmentation_index": augmentation_index,
                    "object_ground_projection": {
                        "applied_frames": int(np.count_nonzero(ground_lift > 0.0)),
                        "max_lift_m": float(np.max(ground_lift, initial=0.0)),
                    },
                }
            )
            record["metadata"] = metadata
            if sequence_name in converted:
                raise ValueError(f"Duplicate paired sequence key={sequence_name} while loading source={source_name}")
            converted[sequence_name] = record

    return validate_ufo_motion_dict(converted, source_name)
