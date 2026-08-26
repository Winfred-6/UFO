"""Optional rigid-box task components for the MJLab UFO environment."""

from pathlib import Path
from typing import Literal

import torch

from humanoidverse.agents.base import BaseConfig
from humanoidverse.utils.torch_utils import (
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_to_tan_norm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LARGEBOX_MESH = PROJECT_ROOT / "humanoidverse/data/objects/largebox/largebox_cleaned_simplified.obj"
OBJECT_OBS_DIM = 19


class CarryBoxConfig(BaseConfig):
    """Configuration isolated behind ``enabled`` so legacy environments stay unchanged."""

    name: Literal["CarryBoxConfig"] = "CarryBoxConfig"
    enabled: bool = False
    mass_kg: float = 0.5
    half_extents: tuple[float, float, float] = (0.235577105, 0.229365065, 0.20394774)
    collision_center: tuple[float, float, float] = (0.001494335, -0.000715375, 0.00575559)
    visual_mesh_path: str = str(DEFAULT_LARGEBOX_MESH)
    hand_body_names: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    park_depth: float = 5.0
    position_clip: float = 5.0
    linear_velocity_clip: float = 10.0
    angular_velocity_clip: float = 20.0
    grasp_force_threshold: float = 1.0
    approach_sigma: float = 0.35
    lift_height: float = 0.12
    goal_tolerance: float = 0.25
    place_height_tolerance: float = 0.12
    linear_speed_limit: float = 3.0
    angular_speed_limit: float = 6.0


def make_carry_box_spec(cfg: CarryBoxConfig):
    """Build a floating 0.5 kg box with primitive collision and mesh visualization."""
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="carry_box")
    body.add_freejoint(name="carry_box_freejoint")
    body.add_geom(
        name="carry_box_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(cfg.collision_center),
        size=list(cfg.half_extents),
        mass=float(cfg.mass_kg),
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.53, 0.28, 0.10, 1.0],
    )

    mesh_path = Path(cfg.visual_mesh_path).expanduser().resolve()
    if mesh_path.exists():
        mesh = spec.add_mesh(name="largebox_visual_mesh", file=str(mesh_path))
        body.add_geom(
            name="carry_box_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name,
            contype=0,
            conaffinity=0,
            mass=0.0,
            rgba=[0.66, 0.38, 0.15, 1.0],
        )
    return spec


def make_carry_target_spec(cfg: CarryBoxConfig):
    """Build a collision-free translucent goal marker, auto-wrapped as mocap by MJLab."""
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="carry_target")
    body.add_geom(
        name="carry_target_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(cfg.collision_center),
        size=list(cfg.half_extents),
        contype=0,
        conaffinity=0,
        mass=0.0,
        rgba=[0.12, 0.85, 0.30, 0.18],
    )
    return spec


def build_carry_box_scene_parts(cfg: CarryBoxConfig, *, control_decimation: int):
    """Return optional MJLab entity and sensor configs for the carry task."""
    if not cfg.enabled:
        return {}, ()

    from mjlab.entity import EntityCfg
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

    parked_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, -float(cfg.park_depth)),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    )
    entities = {
        "carry_box": EntityCfg(spec_fn=lambda: make_carry_box_spec(cfg), init_state=parked_state),
        "carry_target": EntityCfg(spec_fn=lambda: make_carry_target_spec(cfg), init_state=parked_state),
    }
    sensors = (
        ContactSensorCfg(
            name="hand_box_contact",
            primary=ContactMatch(mode="body", pattern=cfg.hand_body_names, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="carry_box", entity="carry_box"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=int(control_decimation),
        ),
    )
    return entities, sensors


def object_observation(
    *,
    base_pos: torch.Tensor,
    base_quat_xyzw: torch.Tensor,
    base_lin_vel_world: torch.Tensor,
    base_ang_vel_world: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    object_lin_vel_world: torch.Tensor,
    object_ang_vel_world: torch.Tensor,
    goal_pos: torch.Tensor,
    valid: torch.Tensor,
    cfg: CarryBoxConfig,
) -> torch.Tensor:
    """Encode the box and goal in the robot heading frame.

    The 19-D vector is fully masked to zero when no object command is active,
    which makes object presence itself the task switch while keeping it out of z.
    """
    heading_inv = calc_heading_quat_inv(base_quat_xyzw, w_last=True)
    rel_pos = my_quat_rotate(heading_inv, object_pos - base_pos).clamp(-cfg.position_clip, cfg.position_clip)
    rel_quat = quat_mul(heading_inv, object_quat_xyzw, w_last=True)
    rel_rot_6d = quat_to_tan_norm(rel_quat, w_last=True)
    rel_lin_vel = my_quat_rotate(heading_inv, object_lin_vel_world - base_lin_vel_world).clamp(
        -cfg.linear_velocity_clip, cfg.linear_velocity_clip
    )
    rel_ang_vel = my_quat_rotate(heading_inv, object_ang_vel_world - base_ang_vel_world).clamp(
        -cfg.angular_velocity_clip, cfg.angular_velocity_clip
    )
    goal_delta = my_quat_rotate(heading_inv, goal_pos - object_pos).clamp(-cfg.position_clip, cfg.position_clip)
    mask = valid.float().reshape(-1, 1)
    observation = torch.cat([mask, rel_pos, rel_rot_6d, rel_lin_vel, rel_ang_vel, goal_delta], dim=-1)
    if observation.shape[-1] != OBJECT_OBS_DIM:
        raise RuntimeError(f"Expected carry object observation dim={OBJECT_OBS_DIM}, got {observation.shape[-1]}")
    return observation * mask
