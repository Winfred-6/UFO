"""Optional rigid-box task components for the MJLab UFO environment."""

from pathlib import Path
from typing import Literal

import pydantic
import torch

from humanoidverse.agents.base import BaseConfig
from humanoidverse.utils.torch_utils import (
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_rotate_inverse,
    quat_to_tan_norm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LARGEBOX_MESH = PROJECT_ROOT / "humanoidverse/data/objects/largebox/largebox_cleaned_simplified.obj"
# The source box (roughly 47 x 46 x 41 cm) intersects the retargeted G1 torso
# by as much as 14 cm.  A physics scan over all 31,422 reference frames found
# 55% to be the largest conservative uniform proxy that supplies certified
# frames in every carry phase without admitting deep torso penetration.
DEFAULT_LARGEBOX_MESH_SCALE = (0.55, 0.55, 0.55)
_SOURCE_LARGEBOX_HALF_EXTENTS = (0.235577105, 0.229365065, 0.20394774)
_SOURCE_LARGEBOX_COLLISION_CENTER = (0.001494335, -0.000715375, 0.00575559)
DEFAULT_LARGEBOX_HALF_EXTENTS = tuple(
    value * scale for value, scale in zip(_SOURCE_LARGEBOX_HALF_EXTENTS, DEFAULT_LARGEBOX_MESH_SCALE)
)
DEFAULT_LARGEBOX_COLLISION_CENTER = tuple(
    value * scale for value, scale in zip(_SOURCE_LARGEBOX_COLLISION_CENTER, DEFAULT_LARGEBOX_MESH_SCALE)
)
OBJECT_FRAME_DIM = 12
OBJECT_HISTORY_STEPS = 4
OBJECT_OBS_DIM = OBJECT_FRAME_DIM * OBJECT_HISTORY_STEPS
TASK_COMMAND_DIM = 3
CARRY_STAGE_INACTIVE = 0
CARRY_STAGE_APPROACH = 1
CARRY_STAGE_PICKUP = 2
CARRY_STAGE_TRANSPORT = 3
CARRY_STAGE_PLACE = 4
CARRY_STAGE_COUNT = 5
CARRY_STAGE_NAMES = ("inactive", "approach", "pickup", "transport", "place")


class CarryBoxConfig(BaseConfig):
    """Configuration isolated behind ``enabled`` so legacy environments stay unchanged."""

    name: Literal["CarryBoxConfig"] = "CarryBoxConfig"
    enabled: bool = False
    require_safe_reset_mask: bool = True
    fail_fast_diagnostics: bool = False
    diagnostic_max_object_linear_speed: float = 20.0
    diagnostic_max_object_angular_speed: float = 100.0
    diagnostic_max_body_linear_speed: float = 50.0
    diagnostic_max_body_angular_speed: float = 200.0
    diagnostic_max_dof_speed: float = 200.0
    diagnostic_max_torque: float = 1.0e5
    diagnostic_max_contact_force: float = 1.0e5
    diagnostic_max_relative_position: float = 50.0
    mass_kg: float = 0.5
    half_extents: tuple[float, float, float] = DEFAULT_LARGEBOX_HALF_EXTENTS
    collision_center: tuple[float, float, float] = DEFAULT_LARGEBOX_COLLISION_CENTER
    visual_mesh_path: str = str(DEFAULT_LARGEBOX_MESH)
    visual_mesh_scale: tuple[float, float, float] = DEFAULT_LARGEBOX_MESH_SCALE
    # The source OBJ/data convention uses local -Z as the visible top face.
    visual_up_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    hand_body_names: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    # Keep inactive dynamic boxes on the ground, far from the robot.  Parking
    # below an infinite plane creates a deep contact that can explode after
    # hundreds of Warp/MuJoCo steps even though object observations are masked.
    park_distance: float = 100.0
    park_ground_clearance: float = 1.0e-3
    # Retained only so older serialized configs still validate.  New code must
    # not place a dynamic box below the ground plane.
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
    # TokenHSI-style reference-state initialization.  The source-level data
    # mix already supplies locomotion examples, so these probabilities are
    # normalized over object-bearing approach/pickup/transport/place frames.
    stage_reset_curriculum: bool = True
    stage_reset_probabilities: tuple[float, float, float, float] = (0.10, 0.20, 0.50, 0.20)
    upright_success_degrees: float = 30.0

    @pydantic.field_validator(
        "half_extents",
        "collision_center",
        "visual_mesh_scale",
        "visual_up_axis",
        "hand_body_names",
        "stage_reset_probabilities",
        mode="before",
    )
    @classmethod
    def _restore_json_tuple_fields(cls, value):
        """Restore tuples that JSON necessarily serializes as arrays."""

        if isinstance(value, list):
            return tuple(value)
        return value

    @pydantic.model_validator(mode="before")
    @classmethod
    def _migrate_legacy_geometry(cls, value):
        """Migrate the old unscaled default box while preserving custom boxes.

        Carry checkpoints written before ``visual_mesh_scale`` existed stored
        the source OBJ bounds directly.  Replaying one of those checkpoints
        with the G1-fit trajectories would therefore restore the 47 cm source
        box and render it over the smaller physical reference.  Only that
        exact legacy default is migrated; arbitrary user-specified bounds keep
        their original size and merely infer a matching mesh scale.
        """

        if not isinstance(value, dict) or "half_extents" not in value:
            return value
        restored = dict(value)
        half_extents = tuple(float(item) for item in restored["half_extents"])
        collision_center = tuple(
            float(item) for item in restored.get("collision_center", _SOURCE_LARGEBOX_COLLISION_CENTER)
        )
        legacy_default = (
            "visual_mesh_scale" not in restored
            and len(half_extents) == 3
            and len(collision_center) == 3
            and all(
                abs(actual - expected) <= 1.0e-8
                for actual, expected in zip(half_extents, _SOURCE_LARGEBOX_HALF_EXTENTS)
            )
            and all(
                abs(actual - expected) <= 1.0e-8
                for actual, expected in zip(collision_center, _SOURCE_LARGEBOX_COLLISION_CENTER)
            )
        )
        if legacy_default:
            restored["half_extents"] = DEFAULT_LARGEBOX_HALF_EXTENTS
            restored["collision_center"] = DEFAULT_LARGEBOX_COLLISION_CENTER
            restored["visual_mesh_scale"] = DEFAULT_LARGEBOX_MESH_SCALE
        elif "visual_mesh_scale" not in restored and len(half_extents) == 3:
            restored["visual_mesh_scale"] = tuple(
                half / source for half, source in zip(half_extents, _SOURCE_LARGEBOX_HALF_EXTENTS)
            )
        return restored

    @pydantic.model_validator(mode="after")
    def _validate_stage_curriculum(self):
        up_norm = sum(float(value) ** 2 for value in self.visual_up_axis) ** 0.5
        if abs(up_norm - 1.0) > 1.0e-6:
            raise ValueError("visual_up_axis must be a unit vector")
        if len(self.stage_reset_probabilities) != CARRY_STAGE_COUNT - 1:
            raise ValueError(
                "stage_reset_probabilities must contain approach/pickup/transport/place weights"
            )
        if any((not float(value) >= 0.0) for value in self.stage_reset_probabilities):
            raise ValueError("stage_reset_probabilities must be non-negative")
        if sum(float(value) for value in self.stage_reset_probabilities) <= 0.0:
            raise ValueError("stage_reset_probabilities must sum to a positive value")
        return self


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
        # Physics proxy only.  Rendering this OBB as well as the mesh produced
        # the misleading nested/two-layer box in tracking and reward play.
        rgba=[0.0, 0.0, 0.0, 0.0],
    )

    mesh_path = Path(cfg.visual_mesh_path).expanduser().resolve()
    if mesh_path.exists():
        mesh = spec.add_mesh(
            name="largebox_visual_mesh",
            file=str(mesh_path),
            scale=list(cfg.visual_mesh_scale),
        )
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


def parked_box_local_position(cfg: CarryBoxConfig) -> tuple[float, float, float]:
    """Return a stable, collision-safe local pose for an inactive box."""

    body_z = float(cfg.half_extents[2]) - float(cfg.collision_center[2]) + float(cfg.park_ground_clearance)
    return float(cfg.park_distance), 0.0, body_z


def build_carry_box_scene_parts(cfg: CarryBoxConfig, *, control_decimation: int):
    """Return optional MJLab entity and sensor configs for the carry task."""
    if not cfg.enabled:
        return {}, ()

    from mjlab.entity import EntityCfg
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

    parked_state = EntityCfg.InitialStateCfg(
        pos=parked_box_local_position(cfg),
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
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    valid: torch.Tensor,
    cfg: CarryBoxConfig,
) -> torch.Tensor:
    """Encode one deployable box frame in the robot heading frame.

    Only relative position, relative 6-D rotation, and physical size enter the
    policy observation.  Contact, velocity, goal, lift/drop state, and an
    explicit valid flag are deliberately absent.  A walk task is represented
    by an exactly zero frame; a carry task keeps this gate open for the whole
    episode, including after a drop.
    """
    heading_inv = calc_heading_quat_inv(base_quat_xyzw, w_last=True)
    rel_pos = my_quat_rotate(heading_inv, object_pos - base_pos).clamp(-cfg.position_clip, cfg.position_clip)
    rel_quat = quat_mul(heading_inv, object_quat_xyzw, w_last=True)
    rel_rot_6d = quat_to_tan_norm(rel_quat, w_last=True)
    full_extents = torch.as_tensor(cfg.half_extents, device=base_pos.device, dtype=base_pos.dtype).mul(2.0)
    full_extents = full_extents.unsqueeze(0).expand(base_pos.shape[0], -1)
    mask = valid.float().reshape(-1, 1)
    observation = torch.cat([rel_pos, rel_rot_6d, full_extents], dim=-1)
    if observation.shape[-1] != OBJECT_FRAME_DIM:
        raise RuntimeError(f"Expected carry object frame dim={OBJECT_FRAME_DIM}, got {observation.shape[-1]}")
    return observation * mask


def task_latent_command(
    *,
    base_pos: torch.Tensor,
    base_quat_xyzw: torch.Tensor,
    goal_pos: torch.Tensor,
    valid: torch.Tensor,
    cfg: CarryBoxConfig,
) -> torch.Tensor:
    """Return a normalized target command that is injected into reward/task z.

    This is a high-level command, not an actor observation feature.  It carries
    the desired target relative to the robot heading while the current box pose
    remains exclusively in ``object_obs``.
    """

    heading_inv = calc_heading_quat_inv(base_quat_xyzw, w_last=True)
    target_rel = my_quat_rotate(heading_inv, goal_pos - base_pos)
    target_rel = target_rel.clamp(-cfg.position_clip, cfg.position_clip) / max(float(cfg.position_clip), 1.0e-6)
    command = target_rel * valid.float().reshape(-1, 1)
    if command.shape[-1] != TASK_COMMAND_DIM:
        raise RuntimeError(f"Expected carry task command dim={TASK_COMMAND_DIM}, got {command.shape[-1]}")
    return command


def temporal_object_history(frames: torch.Tensor, history_steps: int = OBJECT_HISTORY_STEPS) -> torch.Tensor:
    """Build newest-to-oldest fixed history without crossing an episode start."""

    if frames.ndim != 2 or frames.shape[-1] != OBJECT_FRAME_DIM:
        raise ValueError(f"Expected object frames [T, {OBJECT_FRAME_DIM}], got {tuple(frames.shape)}")
    if history_steps <= 0:
        raise ValueError("history_steps must be positive")
    history = frames.new_zeros((frames.shape[0], history_steps, frames.shape[-1]))
    for lag in range(history_steps):
        if lag == 0:
            history[:, lag] = frames
        elif lag < frames.shape[0]:
            history[lag:, lag] = frames[:-lag]
    return history.reshape(frames.shape[0], history_steps * frames.shape[-1])


def box_collision_geometry(
    *,
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    cfg: CarryBoxConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return world collision center, bottom height, and rotated half axes."""

    batch_size = object_pos.shape[0]
    center_offset = torch.as_tensor(cfg.collision_center, device=object_pos.device, dtype=object_pos.dtype)
    center_offset = center_offset.unsqueeze(0).expand(batch_size, -1)
    center = object_pos + my_quat_rotate(object_quat_xyzw, center_offset)

    half_extents = torch.as_tensor(cfg.half_extents, device=object_pos.device, dtype=object_pos.dtype)
    scaled_axes = torch.eye(3, device=object_pos.device, dtype=object_pos.dtype) * half_extents.unsqueeze(0)
    scaled_axes = scaled_axes.unsqueeze(0).expand(batch_size, -1, -1)
    rotated_axes = my_quat_rotate(
        object_quat_xyzw[:, None, :].expand(-1, 3, -1).reshape(-1, 4),
        scaled_axes.reshape(-1, 3),
    ).reshape(batch_size, 3, 3)
    vertical_radius = rotated_axes[..., 2].abs().sum(dim=-1)
    bottom_height = center[:, 2] - vertical_radius
    return center, bottom_height, rotated_axes


def hand_box_surface_geometry(
    *,
    hand_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    cfg: CarryBoxConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-hand OBB surface distance and opposite-side grasp quality."""

    if hand_pos.ndim != 3 or hand_pos.shape[1:] != (2, 3):
        raise ValueError(f"Expected two hand positions [N, 2, 3], got {tuple(hand_pos.shape)}")
    center, _bottom_height, _axes = box_collision_geometry(
        object_pos=object_pos,
        object_quat_xyzw=object_quat_xyzw,
        cfg=cfg,
    )
    local = quat_rotate_inverse(
        object_quat_xyzw[:, None, :].expand(-1, 2, -1).reshape(-1, 4),
        (hand_pos - center[:, None, :]).reshape(-1, 3),
        w_last=True,
    ).reshape(hand_pos.shape[0], 2, 3)
    half_extents = torch.as_tensor(cfg.half_extents, device=hand_pos.device, dtype=hand_pos.dtype)
    signed_axis_distance = local.abs() - half_extents.view(1, 1, 3)
    outside = torch.relu(signed_axis_distance)
    outside_distance = torch.linalg.vector_norm(outside, dim=-1)
    # A point inside an OBB is not on its surface.  The old zero distance for
    # every interior wrist rewarded deep box/robot interpenetration.  Measure
    # distance to the nearest face on the inside and Euclidean distance on the
    # outside instead.
    inside_distance = torch.relu(-signed_axis_distance.amax(dim=-1))
    surface_distance = outside_distance + inside_distance

    normalized_local = local / half_extents.clamp_min(1.0e-6).view(1, 1, 3)
    directions = torch.nn.functional.normalize(normalized_local, dim=-1, eps=1.0e-6)
    opposite_side_quality = torch.relu(-(directions[:, 0] * directions[:, 1]).sum(dim=-1))
    return surface_distance, opposite_side_quality


def adaptive_carry_thresholds(cfg: CarryBoxConfig) -> dict[str, float]:
    """Return OBB-size-aware distances used by every carry reward phase."""

    full_x, full_y, full_z = (2.0 * float(value) for value in cfg.half_extents)
    horizontal_diagonal = max((full_x**2 + full_y**2) ** 0.5, 1.0e-6)
    full_diagonal = max((full_x**2 + full_y**2 + full_z**2) ** 0.5, 1.0e-6)
    return {
        "approach_sigma": max(0.06, min(float(cfg.approach_sigma), 0.25 * horizontal_diagonal)),
        "lift_height": max(float(cfg.lift_height), 0.25 * full_z),
        "goal_tolerance": max(float(cfg.goal_tolerance), 0.35 * horizontal_diagonal),
        "place_height_tolerance": max(float(cfg.place_height_tolerance), 0.25 * full_z),
        "goal_progress_scale": max(0.04, 0.10 * horizontal_diagonal),
        "goal_shape_scale": max(horizontal_diagonal, 0.25),
        "full_diagonal": full_diagonal,
    }


def carry_task_terms(
    *,
    hand_pos: torch.Tensor,
    bilateral_contact: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat_xyzw: torch.Tensor,
    object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor,
    goal_pos: torch.Tensor,
    valid: torch.Tensor,
    ground_height: torch.Tensor,
    ever_lifted: torch.Tensor,
    prev_hand_distance: torch.Tensor,
    prev_goal_distance: torch.Tensor,
    prev_lift_fraction: torch.Tensor,
    cfg: CarryBoxConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Compute dense, phase-continuous carry terms and internal diagnostics.

    The reward follows the same high-level decomposition as TokenHSI
    (approach, hand-object alignment, carry-to-target, put-down) while using an
    oriented-box surface and size-normalized distances.  Contacts and episode
    memory remain reward internals; they never enter the deployable actor
    observation.
    """

    valid = valid.reshape(-1).float()
    valid_bool = valid > 0.5
    thresholds = adaptive_carry_thresholds(cfg)
    hand_distance_each, opposite_side_quality = hand_box_surface_geometry(
        hand_pos=hand_pos,
        object_pos=object_pos,
        object_quat_xyzw=object_quat_xyzw,
        cfg=cfg,
    )
    hand_distance = hand_distance_each.mean(dim=-1)
    per_hand_proximity = torch.exp(-hand_distance_each / thresholds["approach_sigma"])
    surface_proximity = torch.sqrt(torch.clamp(per_hand_proximity.prod(dim=-1), min=0.0, max=1.0))
    geometric_grasp = surface_proximity * opposite_side_quality
    grasp_quality = torch.maximum(0.5 * geometric_grasp, bilateral_contact.reshape(-1).float())

    _center, box_bottom_height, _axes = box_collision_geometry(
        object_pos=object_pos,
        object_quat_xyzw=object_quat_xyzw,
        cfg=cfg,
    )
    lift_clearance = box_bottom_height - ground_height.reshape(-1)
    lift_fraction = torch.clamp(lift_clearance / thresholds["lift_height"], min=0.0, max=1.0)
    # Smoothstep avoids an abrupt transport gate at the old binary lift
    # threshold, while still assigning almost no carry credit to a kicked box.
    lift_gate = lift_fraction.square() * (3.0 - 2.0 * lift_fraction)
    lift_progress = torch.clamp((lift_fraction - prev_lift_fraction) / 0.10, min=-1.0, max=1.0)
    support_quality = 0.20 + 0.80 * grasp_quality
    lifted_now = lift_fraction >= 1.0
    next_ever_lifted = ever_lifted | (lifted_now & valid_bool)

    goal_distance = torch.linalg.vector_norm(goal_pos - object_pos, dim=-1)
    goal_progress = torch.clamp(
        (prev_goal_distance - goal_distance) / thresholds["goal_progress_scale"],
        min=-1.0,
        max=1.0,
    )
    goal_proximity = torch.exp(-2.0 * torch.square(goal_distance / thresholds["goal_shape_scale"]))
    transport_gate = lift_gate * support_quality

    approach_progress = torch.clamp(
        (prev_hand_distance - hand_distance) / max(thresholds["approach_sigma"] * 0.25, 1.0e-3),
        min=-1.0,
        max=1.0,
    )
    # This single pick term remains checkpoint-compatible, but now contains a
    # dense grasp target and positive lift potential before the binary lifted
    # event.  It therefore supplies a direction from a pre-grasp reset.
    pick_reward = (
        0.35 * grasp_quality
        + 0.45 * lift_fraction * support_quality
        + 0.20 * torch.relu(lift_progress) * support_quality
    ).clamp(0.0, 1.0)
    transport_reward = transport_gate * (0.80 * goal_progress + 0.20 * goal_proximity)

    mesh_up = object_pos.new_tensor(cfg.visual_up_axis).expand(object_pos.shape[0], -1)
    object_up = my_quat_rotate(object_quat_xyzw, mesh_up)
    upright = object_up[:, 2] >= torch.cos(
        object_pos.new_tensor(float(cfg.upright_success_degrees) * torch.pi / 180.0)
    )
    object_speed = torch.linalg.vector_norm(object_lin_vel, dim=-1)
    placed = (
        next_ever_lifted
        & (goal_distance < thresholds["goal_tolerance"])
        & (
            torch.abs(box_bottom_height - ground_height.reshape(-1) - float(cfg.park_ground_clearance))
            < thresholds["place_height_tolerance"]
        )
        & (object_speed < 0.5)
        & upright
        & valid_bool
    )
    dropped = (
        next_ever_lifted
        & (lift_fraction < 0.35)
        & (goal_distance >= thresholds["goal_tolerance"])
        & valid_bool
    )

    linear_excess = torch.relu(object_speed - float(cfg.linear_speed_limit))
    angular_excess = torch.relu(torch.linalg.vector_norm(object_ang_vel, dim=-1) - float(cfg.angular_speed_limit))
    aux = {
        "carry_approach": surface_proximity * valid,
        "carry_approach_progress": approach_progress * valid,
        "carry_pick": pick_reward * valid,
        "carry_transport_progress": transport_reward * valid,
        "carry_success": placed.float() * valid,
        "carry_recovery_progress": torch.relu(approach_progress) * dropped.float(),
        "carry_drop_penalty": dropped.float(),
        "box_overspeed_penalty": (linear_excess.square() + 0.1 * angular_excess.square()) * valid,
    }
    state = {
        "hand_distance": hand_distance,
        "goal_distance": goal_distance,
        "lift_fraction": lift_fraction,
        "grasp_quality": grasp_quality,
        "transport_gate": transport_gate,
        "ever_lifted": next_ever_lifted,
        "dropped": dropped,
        "placed": placed,
    }
    return aux, state
