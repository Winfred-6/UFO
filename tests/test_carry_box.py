from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from humanoidverse.agents.buffers.trajectory import TrajectoryDictBuffer
from humanoidverse.agents.envs.carry_box import (
    OBJECT_OBS_DIM,
    CarryBoxConfig,
    make_carry_box_spec,
    object_observation,
)
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.training.workspace import _copy_with_inserted_features
from humanoidverse.utils.motion_data.paired_object_csv import load_paired_hhtools_csv
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
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
    def test_object_observation_is_a_masked_19d_task_switch(self) -> None:
        batch = 2
        zeros3 = torch.zeros(batch, 3)
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(batch, 1)
        valid = torch.tensor([[1.0], [0.0]])
        obs = object_observation(
            base_pos=zeros3,
            base_quat_xyzw=identity,
            base_lin_vel_world=zeros3,
            base_ang_vel_world=zeros3,
            object_pos=torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]),
            object_quat_xyzw=identity,
            object_lin_vel_world=zeros3,
            object_ang_vel_world=zeros3,
            goal_pos=torch.tensor([[2.0, 2.0, 3.0], [8.0, 8.0, 8.0]]),
            valid=valid,
            cfg=CarryBoxConfig(),
        )
        self.assertEqual(tuple(obs.shape), (batch, OBJECT_OBS_DIM))
        self.assertEqual(obs[0, 0].item(), 1.0)
        torch.testing.assert_close(obs[0, 1:4], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(obs[0, -3:], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(obs[1], torch.zeros(OBJECT_OBS_DIM))

    def test_box_body_mass_is_exactly_500_grams(self) -> None:
        model = make_carry_box_spec(CarryBoxConfig()).compile()
        box_body_id = model.body("carry_box").id
        self.assertAlmostEqual(float(model.body_mass[box_body_id]), 0.5, places=6)

    def test_box_features_do_not_enter_backward_or_discriminator(self) -> None:
        cfg = build_fb_agent(device="cpu", compile=False, carry_box=True)
        arch = cfg.model.archi
        self.assertIn("object_obs", arch.actor.input_filter.key)
        self.assertIn("object_obs", arch.f.input_filter.key)
        self.assertIn("object_obs", arch.critic.input_filter.key)
        self.assertIn("object_obs", arch.aux_critic.input_filter.key)
        self.assertNotIn("object_obs", arch.b.input_filter.key)
        self.assertNotIn("object_obs", arch.discriminator.input_filter.key)

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


if __name__ == "__main__":
    unittest.main()
