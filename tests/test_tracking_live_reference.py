from types import SimpleNamespace

import numpy as np
import torch

from humanoidverse.agents.envs.carry_box import CarryBoxConfig
from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig
from humanoidverse.tracking_inference import _apply_reference_frame_to_mjdata, _prepare_tracking_playback_env_cfg


def _entity(*, free_qpos: list[int], joint_qpos: list[int] | None = None, mocap_id: int | None = None):
    return SimpleNamespace(
        indexing=SimpleNamespace(
            free_joint_q_adr=torch.tensor(free_qpos, dtype=torch.long),
            joint_q_adr=torch.tensor(joint_qpos or [], dtype=torch.long),
            mocap_id=mocap_id,
        )
    )


def test_tracking_playback_disables_training_reset_certification() -> None:
    env_cfg = HumanoidVerseMjlabConfig(
        lafan_tail_path="motions.pkl",
        carry_box=CarryBoxConfig(enabled=True, require_safe_reset_mask=True, stage_reset_curriculum=True),
    )

    playback_cfg, changed = _prepare_tracking_playback_env_cfg(env_cfg)

    assert changed
    assert playback_cfg is not env_cfg
    assert playback_cfg.carry_box.stage_reset_curriculum is False
    assert playback_cfg.carry_box.require_safe_reset_mask is False
    assert env_cfg.carry_box.stage_reset_curriculum is True
    assert env_cfg.carry_box.require_safe_reset_mask is True


def test_tracking_playback_restores_legacy_object_width_and_task_adapter() -> None:
    env_cfg = HumanoidVerseMjlabConfig(
        lafan_tail_path="motions.pkl",
        carry_box=CarryBoxConfig(enabled=True),
    )
    legacy_model = SimpleNamespace(
        obs_space=SimpleNamespace(
            spaces={"object_obs": SimpleNamespace(shape=(48,))},
        ),
        cfg=SimpleNamespace(task_latent_dim=3),
    )

    playback_cfg, changed = _prepare_tracking_playback_env_cfg(env_cfg, model=legacy_model)

    assert changed
    assert playback_cfg.carry_box.object_history_steps == 4
    assert playback_cfg.carry_box.emit_legacy_task_command is True
    assert env_cfg.carry_box.object_history_steps == 5
    assert env_cfg.carry_box.emit_legacy_task_command is False


def test_live_reference_writes_synchronized_robot_box_and_target() -> None:
    scene = {
        "robot": _entity(free_qpos=[2, 3, 4, 5, 6, 7, 8], joint_qpos=[0, 1]),
        "carry_box": _entity(free_qpos=[9, 10, 11, 12, 13, 14, 15]),
        "carry_target": _entity(free_qpos=[], mocap_id=0),
    }
    data = SimpleNamespace(
        qpos=np.zeros(16, dtype=np.float64),
        mocap_pos=np.zeros((1, 3), dtype=np.float64),
        mocap_quat=np.zeros((1, 4), dtype=np.float64),
    )
    expert_qpos = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2],
            [2.0, 3.0, 1.2, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5],
        ]
    )
    reference = {
        "object_pos": np.asarray([[0.0, 0.0, 0.2], [2.2, 3.1, 0.5]]),
        "object_quat": np.asarray([[0.0, 0.0, 0.0, 1.0], [0.1, 0.2, 0.3, 0.9]]),
        "object_valid": np.asarray([[1.0], [1.0]]),
        "object_goal_pos": np.asarray([[1.0, 0.0, 0.2], [4.0, 3.0, 0.2]]),
    }

    selected = _apply_reference_frame_to_mjdata(
        data,
        scene,
        expert_qpos,
        reference,
        frame=10,
        position_offset=np.asarray([0.0, 1.5, 0.0]),
    )

    assert selected == 1
    np.testing.assert_allclose(data.qpos[[2, 3, 4]], [2.0, 4.5, 1.2])
    np.testing.assert_allclose(data.qpos[[5, 6, 7, 8]], [0.9, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(data.qpos[[0, 1]], [0.4, 0.5])
    np.testing.assert_allclose(data.qpos[[9, 10, 11]], [2.2, 4.6, 0.5])
    np.testing.assert_allclose(data.qpos[[12, 13, 14, 15]], [0.9, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(data.mocap_pos[0], [4.0, 4.5, 0.2])
    np.testing.assert_allclose(data.mocap_quat[0], [1.0, 0.0, 0.0, 0.0])


def test_live_reference_keeps_inactive_box_in_parked_pose() -> None:
    scene = {
        "robot": _entity(free_qpos=list(range(7)), joint_qpos=[7]),
        "carry_box": _entity(free_qpos=list(range(8, 15))),
        "carry_target": _entity(free_qpos=[], mocap_id=0),
    }
    qpos = np.zeros(15, dtype=np.float64)
    qpos[8:11] = [100.0, 0.0, 0.2]
    data = SimpleNamespace(
        qpos=qpos,
        mocap_pos=np.asarray([[100.0, 0.0, 0.2]], dtype=np.float64),
        mocap_quat=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    expert_qpos = np.asarray([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25]])
    reference = {"object_valid": np.asarray([[0.0]])}

    _apply_reference_frame_to_mjdata(
        data,
        scene,
        expert_qpos,
        reference,
        frame=0,
        position_offset=np.asarray([0.0, 1.5, 0.0]),
    )

    np.testing.assert_allclose(data.qpos[8:11], [100.0, 1.5, 0.2])
    np.testing.assert_allclose(data.mocap_pos[0], [100.0, 1.5, 0.2])
