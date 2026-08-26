from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.train import _ensure_compile_cache, _ensure_rank_compile_cache, build_ufo_mjlab_config, parse_args
from humanoidverse.training.workspace import _accumulate_metrics, _trajectory_output_keys


class UpdateZCliTest(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["train.py", *args]), patch("sys.stderr", io.StringIO()):
            return parse_args()

    def test_agent_specific_defaults_are_preserved(self) -> None:
        self.assertEqual(self._parse("--agent", "fb").update_z_every_step, 100)
        self.assertEqual(self._parse("--agent", "tldr").update_z_every_step, 10)

    def test_compile_cache_paths_are_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "compile-cache"
            relative_root = os.path.relpath(cache_root, Path.cwd())
            with patch.dict(os.environ, {"UFO_CACHE_DIR": relative_root}, clear=True):
                _ensure_compile_cache()

                self.assertEqual(Path(os.environ["UFO_CACHE_DIR"]), cache_root.resolve())
                for key in (
                    "TMPDIR",
                    "TEMP",
                    "TMP",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                    "WARP_CACHE_PATH",
                ):
                    self.assertTrue(Path(os.environ[key]).is_absolute(), key)

    def test_tldr_cli_value_reaches_agent_config(self) -> None:
        args = self._parse("--agent", "tldr", "--update-z-every-step", "37")
        selected = build_agent_preset(
            agent=args.agent,
            device="cpu",
            compile=False,
            update_z_every_step=args.update_z_every_step,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        self.assertEqual(selected["agent_cfg"].train.update_z_every_step, 37)

    def test_gpu_native_and_runtime_timing_cli(self) -> None:
        defaults = self._parse()
        self.assertTrue(defaults.gpu_native_rollout)
        self.assertEqual(defaults.runtime_timing_every, 0)
        self.assertIsNone(defaults.compile_agent)

        configured = self._parse("--no-gpu-native-rollout", "--runtime-timing-every", "25", "--compile")
        self.assertFalse(configured.gpu_native_rollout)
        self.assertEqual(configured.runtime_timing_every, 25)
        self.assertTrue(configured.compile_agent)

    def test_distributed_compile_cache_is_rank_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "compile-cache"
            with patch.dict(os.environ, {}, clear=True):
                _ensure_compile_cache(cache_root)
                _ensure_rank_compile_cache(rank=3, world_size=8)
                expected = cache_root.resolve() / "distributed" / "rank_3"
                self.assertEqual(Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]), expected / "torchinductor")
                self.assertEqual(Path(os.environ["TRITON_CACHE_DIR"]), expected / "triton")
                self.assertEqual(Path(os.environ["TMPDIR"]), expected / "tmp")

    def test_h200_distributed_profile_uses_local_cuda_replay_and_stable_compile_default(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cuda:3",
            work_dir="/tmp/ufo_h200_profile_test",
            num_envs=1024,
            num_env_steps=192_000_000,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            distributed_rank=3,
            distributed_world_size=8,
            agent="fb",
            buffer_size=5_120_000,
            buffer_storage="cuda",
            buffer_prefetch=0,
            buffer_pin_memory_threads=0,
            gpu_native_rollout=True,
            task="carry_box",
        )

        self.assertEqual(cfg.buffer_device, "cuda:3")
        self.assertEqual(cfg.buffer_sample_device, "cuda:3")
        self.assertEqual(cfg.buffer_size, 5_120_000)
        self.assertEqual(cfg.buffer_prefetch, 0)
        self.assertEqual(cfg.buffer_pin_memory_threads, 0)
        self.assertTrue(cfg.gpu_native_rollout)
        self.assertTrue(cfg.distributed_sync)
        self.assertEqual(cfg.distributed_world_size, 8)
        self.assertFalse(cfg.agent.compile)

    def test_programmatic_tldr_default_remains_ten(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_update_z_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="tldr",
        )
        self.assertEqual(cfg.agent.train.update_z_every_step, 10)

    def test_tldr_trajectory_buffer_keeps_aux_rewards(self) -> None:
        selected = build_agent_preset(
            agent="tldr",
            device="cpu",
            compile=False,
            update_z_every_step=10,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        self.assertIn("aux_rewards", _trajectory_output_keys(selected["agent_cfg"]))

    def test_metric_accumulation_accepts_tldr_phase_changes(self) -> None:
        totals, counts = _accumulate_metrics(
            None,
            {},
            {"tldr_te_loss": torch.tensor(2.0)},
        )
        totals, counts = _accumulate_metrics(
            totals,
            counts,
            {
                "tldr_te_loss": torch.tensor(4.0),
                "disc_wgan_gp_loss": torch.tensor(6.0),
            },
        )
        self.assertEqual(counts, {"tldr_te_loss": 2, "disc_wgan_gp_loss": 1})
        self.assertEqual((totals["tldr_te_loss"] / counts["tldr_te_loss"]).item(), 3.0)
        self.assertEqual((totals["disc_wgan_gp_loss"] / counts["disc_wgan_gp_loss"]).item(), 6.0)


if __name__ == "__main__":
    unittest.main()
