from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from humanoidverse.agents.buffers.torchrl_replay import TorchRLReplayBuffer


class TorchRLReplayBufferTest(unittest.TestCase):
    @staticmethod
    def _make_buffer(*, prefetch: int = 0) -> TorchRLReplayBuffer:
        return TorchRLReplayBuffer(
            capacity=64,
            batch_size=8,
            sample_device="cpu",
            storage_kind="cpu",
            prefetch=prefetch,
            pin_memory_threads=0,
            trajectory=True,
            num_envs=4,
            end_key="truncated",
            output_key_t=["observation", "action", "terminated", "truncated"],
            output_key_tp1=["observation", "terminated"],
        )

    @staticmethod
    def _fill_two_episodes(buffer: TorchRLReplayBuffer) -> None:
        for step in range(8):
            episode_offset = 0 if step < 4 else 100
            observation = episode_offset + step + torch.arange(4).reshape(1, 4, 1) * 1000
            truncated = torch.zeros(1, 4, 1, dtype=torch.bool)
            if step in (3, 7):
                truncated.fill_(True)
            buffer.extend(
                {
                    "observation": {"state": observation},
                    "action": torch.full((1, 4, 1), step),
                    "terminated": torch.zeros(1, 4, 1, dtype=torch.bool),
                    "truncated": truncated,
                }
            )

    def test_reconstructs_next_without_crossing_trajectory_boundaries(self) -> None:
        buffer = self._make_buffer()
        self._fill_two_episodes(buffer)
        self.assertEqual(len(buffer), 32)

        for _ in range(20):
            sample = buffer.sample(8)
            current = sample["observation", "state"]
            next_state = sample["next", "observation", "state"]
            self.assertTrue(torch.equal(next_state - current, torch.ones_like(current)))
            self.assertEqual(sample.batch_size, torch.Size([8]))
            self.assertEqual(sample.device, torch.device("cpu"))
        buffer.close()

    def test_prefetch_has_fixed_transition_batch_size(self) -> None:
        buffer = self._make_buffer(prefetch=2)
        self._fill_two_episodes(buffer)
        self.assertEqual(buffer.sample(8).batch_size, torch.Size([8]))
        with self.assertRaisesRegex(ValueError, "fixed batch_size=8"):
            buffer.sample(4)
        buffer.close()

    def test_save_and_load_round_trip(self) -> None:
        buffer = self._make_buffer()
        self._fill_two_episodes(buffer)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer"
            buffer.save(path)
            loaded = TorchRLReplayBuffer.load(path, sample_device="cpu", prefetch=0, pin_memory_threads=0)
            self.assertEqual(len(loaded), len(buffer))
            sample = loaded.sample(8)
            delta = sample["next", "observation", "state"] - sample["observation", "state"]
            self.assertTrue(torch.equal(delta, torch.ones_like(delta)))
            loaded.close()
        buffer.close()

    def test_full_buffer_handles_ring_wrap_and_extra_next_fields(self) -> None:
        buffer = TorchRLReplayBuffer(
            capacity=16,
            batch_size=4,
            sample_device="cpu",
            storage_kind="cpu",
            prefetch=0,
            pin_memory_threads=0,
            trajectory=True,
            num_envs=4,
            end_key="truncated",
            output_key_t=["observation", "action", "truncated"],
            output_key_tp1=["observation"],
        )
        for step in range(6):
            buffer.extend(
                {
                    "observation": {"state": torch.full((1, 4, 1), step)},
                    "action": torch.full((1, 4, 1), step),
                    "qpos": torch.full((1, 4, 1), step * 10),
                    "qvel": torch.full((1, 4, 1), step * 100),
                    "truncated": torch.zeros(1, 4, 1, dtype=torch.bool),
                }
            )

        buffer.include_next_keys("qpos", "qvel")
        full = buffer.get_full_buffer()
        self.assertEqual(full.batch_size, torch.Size([12]))
        self.assertTrue(
            torch.equal(
                full["next", "observation", "state"] - full["observation", "state"],
                torch.ones(12, 1, dtype=torch.int64),
            )
        )
        self.assertTrue(torch.equal(full["next", "qpos"], full["next", "observation", "state"] * 10))
        self.assertTrue(torch.equal(full["next", "qvel"], full["next", "observation", "state"] * 100))

        sampled = buffer.sample(4)
        self.assertIn("qpos", sampled["next"])
        self.assertIn("qvel", sampled["next"])
        buffer.close()

    def test_memmap_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffer = TorchRLReplayBuffer(
                capacity=64,
                batch_size=8,
                sample_device="cpu",
                storage_kind="memmap",
                prefetch=0,
                pin_memory_threads=0,
                trajectory=True,
                num_envs=4,
                end_key="truncated",
                output_key_t=["observation", "action", "terminated", "truncated"],
                output_key_tp1=["observation", "terminated"],
                scratch_dir=root / "source_memmap",
            )
            self._fill_two_episodes(buffer)
            path = root / "buffer"
            buffer.save(path)
            buffer.close()

            loaded = TorchRLReplayBuffer.load(
                path,
                sample_device="cpu",
                prefetch=0,
                pin_memory_threads=0,
                scratch_dir=root / "loaded_memmap",
            )
            sample = loaded.sample(8)
            delta = sample["next", "observation", "state"] - sample["observation", "state"]
            self.assertTrue(torch.equal(delta, torch.ones_like(delta)))
            loaded.close()

    def test_reward_inference_loads_torchrl_checkpoint(self) -> None:
        from humanoidverse.reward_inference import _load_replay_buffer

        buffer = self._make_buffer()
        self._fill_two_episodes(buffer)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_folder = Path(tmpdir) / "model"
            checkpoint_path = model_folder / "checkpoint" / "buffers" / "train"
            buffer.save(checkpoint_path)
            loaded, loaded_path = _load_replay_buffer(model_folder, buffer_rank=0, buffer_path=None)
            self.assertIsInstance(loaded, TorchRLReplayBuffer)
            self.assertEqual(loaded_path, checkpoint_path)
            self.assertEqual(len(loaded), len(buffer))
            loaded.close()
        buffer.close()

    def test_reward_inference_maps_memmap_checkpoint_without_training_scratch(self) -> None:
        from humanoidverse.reward_inference import _load_replay_buffer

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scratch_dir = root / "training_scratch"
            buffer = TorchRLReplayBuffer(
                capacity=64,
                batch_size=8,
                sample_device="cpu",
                storage_kind="memmap",
                prefetch=0,
                pin_memory_threads=0,
                trajectory=True,
                num_envs=4,
                end_key="truncated",
                output_key_t=["observation", "action", "terminated", "truncated"],
                output_key_tp1=["observation", "terminated"],
                scratch_dir=scratch_dir,
            )
            self._fill_two_episodes(buffer)
            model_folder = root / "model"
            checkpoint_path = model_folder / "checkpoint" / "buffers" / "train"
            buffer.save(checkpoint_path)
            buffer.close()
            shutil.rmtree(scratch_dir)

            loaded, _ = _load_replay_buffer(model_folder, buffer_rank=0, buffer_path=None)
            sample = loaded.sample(8)
            delta = sample["next", "observation", "state"] - sample["observation", "state"]
            self.assertTrue(torch.equal(delta, torch.ones_like(delta)))
            loaded.close()

    def test_replay_cli_defaults_and_smoke_capacity(self) -> None:
        from humanoidverse.train import parse_args

        with patch("sys.argv", ["train.py", "--smoke", "--num-envs", "7"]):
            args = parse_args()
        self.assertEqual(args.buffer_storage, "cpu")
        self.assertEqual(args.buffer_prefetch, 2)
        self.assertEqual(args.buffer_pin_memory_threads, 2)
        self.assertEqual(args.buffer_size % args.num_envs, 0)
        self.assertLessEqual(args.buffer_size, args.num_env_steps)


if __name__ == "__main__":
    unittest.main()
