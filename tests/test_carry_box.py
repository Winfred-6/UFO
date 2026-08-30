from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import mujoco

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
from humanoidverse.agents.envs.carry_box import (
    OBJECT_FRAME_DIM,
    OBJECT_OBS_DIM,
    CarryBoxConfig,
    assert_native_reference_geometry,
    carry_task_terms,
    hand_box_surface_geometry,
    make_carry_box_spec,
    object_observation,
    parked_box_local_position,
    temporal_object_history,
)
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.mjlab_reward_relabel import RewardWrapperHV
from humanoidverse.training.workspace import _copy_with_inserted_features
from humanoidverse.utils.motion_data.object_physics import (
    classify_carry_stages,
    mesh_axis_aligned_bounds,
    oriented_box_min_corner_z,
    sanitize_object_ground_trajectory,
)
from humanoidverse.utils.motion_data.paired_object_csv import load_paired_hhtools_csv
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.motion_lib.motion_lib_base import _optional_object_state_from_motion_file
from humanoidverse.utils.robot_spec import load_robot_spec


def _write_tiny_robot(root: Path) -> Path:
    (root / "tiny.xml").write_text(
        """
<mujoco model="tiny">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1"/>
    <motor name="joint2_motor" joint="joint2"/>
  </actuator>
</mujoco>
""".strip()
    )
    config = root / "tiny.yaml"
    config.write_text(
        "\n".join(
            [
                "name: tiny",
                "xml_path: tiny.xml",
                "base_body: base",
                "root_quat_order: xyzw",
                "coordinate_system: z_up",
                "dof_unit: rad",
                "control_joints:",
                "  mode: all_actuated",
                "feet: [link2]",
                "hands: []",
                "key_bodies: [base, link1, link2]",
                "default_dof_pos: {}",
            ]
        )
    )
    return config


def _write_paired_sequence(root: Path) -> None:
    sequence = root / "seq_original"
    sequence.mkdir()
    robot_path = sequence / "seq_original.csv"
    with robot_path.open("w", newline="") as stream:
        stream.write("# sample_rate: 50\n")
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "time",
                "root_x",
                "root_y",
                "root_z",
                "root_qx",
                "root_qy",
                "root_qz",
                "root_qw",
                "dof_joint1",
                "dof_joint2",
            ],
        )
        writer.writeheader()
        for frame in range(3):
            writer.writerow(
                {
                    "time": frame * 0.02,
                    "root_x": 0.0,
                    "root_y": 0.0,
                    "root_z": 1.0,
                    "root_qx": 0.0,
                    "root_qy": 0.0,
                    "root_qz": 0.0,
                    "root_qw": 1.0,
                    "dof_joint1": frame * 0.1,
                    "dof_joint2": frame * 0.2,
                }
            )

    object_path = sequence / "object_0_largebox.csv"
    with object_path.open("w", newline="") as stream:
        stream.write("# object: largebox\n# sample_rate: 50\n")
        writer = csv.DictWriter(
            stream,
            fieldnames=["time", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"],
        )
        writer.writeheader()
        for frame in range(3):
            writer.writerow(
                {
                    "time": frame * 0.02,
                    "pos_x": frame * 0.02,
                    "pos_y": 0.0,
                    "pos_z": 0.2,
                    "quat_x": 0.0,
                    "quat_y": 0.0,
                    "quat_z": 0.0,
                    "quat_w": -1.0 if frame == 1 else 1.0,
                }
            )


def _episode(motion_id: int, length: int = 4) -> dict:
    return {
        "motion_id": torch.full((length, 1), motion_id, dtype=torch.long),
        "observation": {"state": torch.zeros(length, 2)},
    }


class CarryBoxTest(unittest.TestCase):
    def test_object_observation_contains_only_masked_pose_rotation_and_size(self) -> None:
        batch = 2
        zeros3 = torch.zeros(batch, 3)
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(batch, 1)
        valid = torch.tensor([[1.0], [0.0]])
        frame = object_observation(
            base_pos=zeros3,
            base_quat_xyzw=identity,
            object_pos=torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]),
            object_quat_xyzw=identity,
            valid=valid,
            cfg=CarryBoxConfig(),
        )
        self.assertEqual(tuple(frame.shape), (batch, OBJECT_FRAME_DIM))
        torch.testing.assert_close(frame[0, :3], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(frame[0, -3:], 2.0 * torch.tensor(CarryBoxConfig().half_extents))
        torch.testing.assert_close(frame[1], torch.zeros(OBJECT_FRAME_DIM))

        history = temporal_object_history(frame)
        self.assertEqual(tuple(history.shape), (batch, OBJECT_OBS_DIM))
        torch.testing.assert_close(history[0, :OBJECT_FRAME_DIM], frame[0])
        torch.testing.assert_close(history[0, OBJECT_FRAME_DIM:], torch.zeros(OBJECT_OBS_DIM - OBJECT_FRAME_DIM))

    def test_large_box_approach_distance_uses_surface_not_center(self) -> None:
        cfg = CarryBoxConfig(half_extents=(1.0, 0.5, 0.25), collision_center=(0.0, 0.0, 0.0))
        distance, opposition = hand_box_surface_geometry(
            hand_pos=torch.tensor([[[0.0, 0.51, 0.0], [0.0, -0.51, 0.0]]]),
            object_pos=torch.zeros(1, 3),
            object_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            cfg=cfg,
        )
        torch.testing.assert_close(distance, torch.full((1, 2), 0.01), atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(opposition, torch.ones(1), atol=1.0e-6, rtol=0.0)

    def test_hand_surface_distance_does_not_reward_deep_interior_points(self) -> None:
        cfg = CarryBoxConfig(half_extents=(1.0, 0.5, 0.25), collision_center=(0.0, 0.0, 0.0))
        distance, _opposition = hand_box_surface_geometry(
            hand_pos=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            object_pos=torch.zeros(1, 3),
            object_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            cfg=cfg,
        )
        torch.testing.assert_close(distance, torch.full((1, 2), 0.25), atol=1.0e-6, rtol=0.0)

    def test_dense_pick_and_transport_rewards_unlock_before_binary_lift(self) -> None:
        cfg = CarryBoxConfig(
            half_extents=(0.25, 0.20, 0.20),
            collision_center=(0.0, 0.0, 0.0),
            lift_height=0.12,
        )
        object_pos = torch.tensor([[0.20, 0.0, 0.26]])
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        aux, state = carry_task_terms(
            hand_pos=torch.tensor([[[0.20, 0.20, 0.26], [0.20, -0.20, 0.26]]]),
            bilateral_contact=torch.tensor([False]),
            object_pos=object_pos,
            object_quat_xyzw=identity,
            object_lin_vel=torch.zeros(1, 3),
            object_ang_vel=torch.zeros(1, 3),
            goal_pos=torch.tensor([[1.0, 0.0, 0.20]]),
            valid=torch.ones(1),
            ground_height=torch.zeros(1),
            ever_lifted=torch.tensor([False]),
            prev_hand_distance=torch.tensor([0.10]),
            prev_goal_distance=torch.tensor([1.0]),
            prev_lift_fraction=torch.zeros(1),
            cfg=cfg,
        )
        self.assertGreater(float(aux["carry_pick"].item()), 0.2)
        self.assertGreater(float(aux["carry_transport_progress"].item()), 0.0)
        self.assertGreater(float(state["lift_fraction"].item()), 0.0)
        self.assertLess(float(state["lift_fraction"].item()), 1.0)

    def test_carry_stage_classifier_covers_all_reference_phases(self) -> None:
        half = np.asarray([0.25, 0.20, 0.20], dtype=np.float32)
        bottom = np.asarray([0.0, 0.0, 0.02, 0.08, 0.20, 0.30, 0.30, 0.20, 0.05, 0.0])
        object_pos = np.stack(
            [np.linspace(0.0, 1.0, len(bottom)), np.zeros(len(bottom)), bottom + half[2]],
            axis=-1,
        ).astype(np.float32)
        identity = np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (len(bottom), 1))
        goal = np.tile(np.asarray([[1.0, 0.0, half[2]]], dtype=np.float32), (len(bottom), 1))
        phases = classify_carry_stages(
            object_pos,
            identity,
            goal,
            np.ones((len(bottom), 1), dtype=np.float32),
            fps=10.0,
            collision_center=np.zeros(3, dtype=np.float32),
            half_extents=half,
            lift_height_m=0.12,
            goal_tolerance_m=0.20,
            pickup_lead_seconds=0.2,
        )
        self.assertEqual(set(phases[:, 0].astype(int)), {1, 2, 3, 4})
        self.assertTrue(np.all(np.diff(phases[:, 0]) >= 0.0))

    def test_box_body_mass_is_exactly_500_grams(self) -> None:
        model = make_carry_box_spec(CarryBoxConfig()).compile()
        box_body_id = model.body("carry_box").id
        self.assertAlmostEqual(float(model.body_mass[box_body_id]), 0.5, places=6)

    def test_collision_proxy_is_hidden_and_visual_mesh_matches_its_bounds(self) -> None:
        cfg = CarryBoxConfig()
        model = make_carry_box_spec(cfg).compile()
        collision_id = model.geom("carry_box_collision").id
        visual_id = model.geom("carry_box_visual").id
        self.assertEqual(float(model.geom_rgba[collision_id, 3]), 0.0)
        mesh_id = model.mesh("largebox_visual_mesh").id
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_end = vertex_start + int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vertex_start:vertex_end]
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[visual_id])
        vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[visual_id]
        visual_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        visual_half_extents = 0.5 * (vertices.max(axis=0) - vertices.min(axis=0))
        np.testing.assert_allclose(
            visual_center,
            model.geom_pos[collision_id],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            visual_half_extents,
            model.geom_size[collision_id],
            atol=1.0e-6,
        )

    def test_inactive_box_is_parked_above_ground_and_far_from_robot(self) -> None:
        cfg = CarryBoxConfig()
        parked = parked_box_local_position(cfg)
        min_corner_z = parked[2] + cfg.collision_center[2] - cfg.half_extents[2]
        self.assertAlmostEqual(min_corner_z, cfg.park_ground_clearance, places=7)
        self.assertGreaterEqual(parked[0], 50.0)
        self.assertGreater(parked[2], 0.0)

    def test_json_round_trip_restores_tuple_fields(self) -> None:
        serialized = CarryBoxConfig().model_dump(mode="json")
        self.assertIsInstance(serialized["half_extents"], list)
        self.assertIsInstance(serialized["collision_center"], list)
        self.assertIsInstance(serialized["visual_mesh_scale"], list)
        self.assertIsInstance(serialized["visual_up_axis"], list)
        self.assertIsInstance(serialized["hand_body_names"], list)

        restored = CarryBoxConfig.model_validate(serialized)
        self.assertIsInstance(restored.half_extents, tuple)
        self.assertIsInstance(restored.collision_center, tuple)
        self.assertIsInstance(restored.visual_mesh_scale, tuple)
        self.assertIsInstance(restored.visual_up_axis, tuple)
        self.assertIsInstance(restored.hand_body_names, tuple)

    def test_old_default_box_config_preserves_original_geometry(self) -> None:
        legacy = {
            "enabled": True,
            "half_extents": [0.235577105, 0.229365065, 0.20394774],
            "collision_center": [0.001494335, -0.000715375, 0.00575559],
            "hand_body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        }

        restored = CarryBoxConfig.model_validate(legacy)
        np.testing.assert_allclose(restored.half_extents, legacy["half_extents"], atol=0.0)
        np.testing.assert_allclose(restored.collision_center, legacy["collision_center"], atol=0.0)
        np.testing.assert_allclose(restored.visual_mesh_scale, (1.0, 1.0, 1.0), atol=0.0)

    def test_formal_training_rejects_resized_reference_metadata(self) -> None:
        records = [
            {
                "motion_key": "bad_g1fit",
                "object_valid": np.ones((1, 1), dtype=np.float32),
                "metadata": {"object_geometry_retarget": {"mesh_scale": [0.55, 0.55, 0.55]}},
            }
        ]
        with self.assertRaisesRegex(ValueError, "native, unscaled object trajectories"):
            assert_native_reference_geometry(records)

    def test_collision_bounds_match_the_visual_mesh(self) -> None:
        cfg = CarryBoxConfig()
        center, half_extents = mesh_axis_aligned_bounds(cfg.visual_mesh_path)
        np.testing.assert_allclose(center * cfg.visual_mesh_scale, cfg.collision_center, atol=1.0e-7)
        np.testing.assert_allclose(half_extents * cfg.visual_mesh_scale, cfg.half_extents, atol=1.0e-7)

    def test_place_success_uses_dataset_mesh_minus_z_as_up(self) -> None:
        cfg = CarryBoxConfig(
            half_extents=(0.2, 0.2, 0.2),
            collision_center=(0.0, 0.0, 0.0),
            place_height_tolerance=0.02,
        )
        object_pos = torch.tensor([[1.0, 0.0, 0.201]])
        flipped_about_x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        aux, _state = carry_task_terms(
            hand_pos=torch.tensor([[[1.0, 0.2, 0.201], [1.0, -0.2, 0.201]]]),
            bilateral_contact=torch.tensor([False]),
            object_pos=object_pos,
            object_quat_xyzw=flipped_about_x,
            object_lin_vel=torch.zeros(1, 3),
            object_ang_vel=torch.zeros(1, 3),
            goal_pos=object_pos.clone(),
            valid=torch.ones(1),
            ground_height=torch.zeros(1),
            ever_lifted=torch.tensor([True]),
            prev_hand_distance=torch.zeros(1),
            prev_goal_distance=torch.zeros(1),
            prev_lift_fraction=torch.ones(1),
            cfg=cfg,
        )
        torch.testing.assert_close(aux["carry_success"], torch.ones(1))

    def test_ground_projection_respects_box_orientation_and_recomputes_velocity(self) -> None:
        positions = np.zeros((2, 3), dtype=np.float32)
        quaternions = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)],
            ],
            dtype=np.float32,
        )
        projected, velocity, goal, lift = sanitize_object_ground_trajectory(
            positions,
            quaternions,
            fps=10.0,
            collision_center=np.zeros(3, dtype=np.float32),
            half_extents=np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        )
        np.testing.assert_allclose(lift, [0.251, 1.001], atol=1.0e-5)
        minimum_z = oriented_box_min_corner_z(
            projected,
            quaternions,
            collision_center=np.zeros(3, dtype=np.float32),
            half_extents=np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        )
        np.testing.assert_allclose(minimum_z, np.full(2, 0.001), atol=1.0e-5)
        np.testing.assert_allclose(velocity[:, 2], np.full(2, 7.5), atol=1.0e-5)
        np.testing.assert_allclose(goal[:, 2], np.full(2, 0.626), atol=1.0e-5)

    def test_box_extends_original_ufo_inputs_without_goal_or_custom_networks(self) -> None:
        cfg = build_fb_agent(device="cpu", compile=False, carry_box=True)
        arch = cfg.model.archi
        for network in (arch.actor, arch.f, arch.critic, arch.aux_critic, arch.b):
            self.assertIn("object_obs", network.input_filter.key)
            self.assertNotIn("goal_obs", network.input_filter.key)
        self.assertEqual(arch.discriminator.name, "DiscriminatorArchi")
        self.assertIn("object_obs", arch.discriminator.input_filter.key)

        legacy = build_fb_agent(device="cpu", compile=False, carry_box=False)
        self.assertNotIn("object_obs", legacy.model.archi.actor.input_filter.key)
        self.assertNotIn("object_obs", legacy.model.obs_normalizer.normalizers)

    def test_source_priorities_preserve_lafan_carry_mix(self) -> None:
        buffer = TrajectoryDictBuffer(
            [_episode(0), _episode(1), _episode(2)],
            source_ids=[0, 0, 1],
            source_weights=[0.7, 0.3],
        )
        self.assertAlmostEqual(float(buffer.priorities[buffer.source_ids == 0].sum()), 0.7, places=6)
        self.assertAlmostEqual(float(buffer.priorities[buffer.source_ids == 1].sum()), 0.3, places=6)
        buffer.update_priorities(torch.tensor([100.0, 1.0, 9.0]), torch.arange(3))
        self.assertAlmostEqual(float(buffer.priorities[buffer.source_ids == 0].sum()), 0.7, places=6)
        self.assertAlmostEqual(float(buffer.priorities[buffer.source_ids == 1].sum()), 0.3, places=6)

    def test_checkpoint_expansion_inserts_zero_object_columns_before_tail(self) -> None:
        source = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        destination = torch.full((2, 8), -1.0)
        expanded = _copy_with_inserted_features(
            source,
            destination,
            insert_count=3,
            tail_count=2,
            zero_insert=True,
        )
        self.assertIsNotNone(expanded)
        torch.testing.assert_close(expanded[:, :3], source[:, :3])
        torch.testing.assert_close(expanded[:, 3:6], torch.zeros(2, 3))
        torch.testing.assert_close(expanded[:, 6:], source[:, 3:])

    def test_reward_mode_can_infer_pick_latent_directly_from_carry_replay(self) -> None:
        class Dataset:
            def size(self):
                return 2

            def sample(self, _count):
                return {
                    "aux_rewards": {
                        "carry_approach": torch.tensor([[0.5], [1.0]]),
                        "carry_pick": torch.tensor([[1.0], [2.0]]),
                    },
                    "B": torch.ones(2, 4),
                }

        class Model:
            device = "cpu"

            def __init__(self):
                self.reward = None

            def reward_wr_inference(self, *, reward, B_vect):
                self.reward = reward
                return (reward[:, None] * B_vect).sum(dim=0, keepdim=True)

        model = Model()
        wrapper = RewardWrapperHV(
            model=model,
            inference_dataset=Dataset(),
            num_samples_per_inference=2,
            inference_function="reward_wr_inference",
            max_workers=1,
        )
        z = wrapper.reward_inference("carry-pick")
        torch.testing.assert_close(model.reward, torch.tensor([1.1, 2.2]))
        torch.testing.assert_close(z, torch.full((1, 4), 3.3))

    def test_paired_reader_keeps_robot_and_box_frames_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            robot_config = _write_tiny_robot(root)
            _write_paired_sequence(root)
            data = load_paired_hhtools_csv(
                root,
                source_name="paired_unit",
                robot_spec=load_robot_spec(robot_config),
            )
        record = data["seq_original"]
        self.assertEqual(record["dof_pos"].shape, (3, 2))
        self.assertEqual(record["object_pos"].shape, (3, 3))
        np.testing.assert_allclose(record["object_lin_vel"][:, 0], np.ones(3), atol=1.0e-6)
        np.testing.assert_allclose(record["object_quat"], np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)))
        np.testing.assert_allclose(record["object_goal_pos"], np.tile([0.02, 0.0, 0.2], (3, 1)))
        self.assertEqual(record["metadata"]["reader"], "robot_state_object_csv")

    def test_schema_rejects_partial_object_trajectory(self) -> None:
        motion = {
            "bad": {
                "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
                "pose_aa": np.zeros((2, 1, 3), dtype=np.float32),
                "fps": 30.0,
                "object_pos": np.zeros((2, 3), dtype=np.float32),
            }
        }
        with self.assertRaisesRegex(ValueError, "partial object trajectory"):
            validate_ufo_motion_dict(motion, "unit")

    def test_object_motion_loader_needs_no_reset_certification_fields(self) -> None:
        frame_count = 3
        record = {
            "object_pos": np.zeros((frame_count, 3), dtype=np.float32),
            "object_quat": np.tile([0.0, 0.0, 0.0, 1.0], (frame_count, 1)).astype(np.float32),
            "object_lin_vel": np.zeros((frame_count, 3), dtype=np.float32),
            "object_ang_vel": np.zeros((frame_count, 3), dtype=np.float32),
            "object_valid": np.ones((frame_count, 1), dtype=np.float32),
            "object_goal_pos": np.zeros((frame_count, 3), dtype=np.float32),
        }
        loaded = _optional_object_state_from_motion_file(record, 0, frame_count, torch.float32)
        self.assertNotIn("object_reset_valid", loaded)
        self.assertNotIn("object_stage_reset_valid", loaded)
        torch.testing.assert_close(loaded["object_phase"], torch.zeros(frame_count, 1))


if __name__ == "__main__":
    unittest.main()
