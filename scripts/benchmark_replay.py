"""Benchmark UFO's TorchRL replay storage and GPU staging paths."""

from __future__ import annotations

import argparse
import gc
import resource
import tempfile
import time
from pathlib import Path

import torch
from tensordict import TensorDictBase

from humanoidverse.agents.buffers.torchrl_replay import TorchRLReplayBuffer

OUTPUT_KEY_T = [
    "observation",
    "action",
    "z",
    "terminated",
    "truncated",
    "step_count",
    "reward",
    "aux_rewards",
]
OUTPUT_KEY_TP1 = ["observation", "terminated"]


def _make_chunk(steps: int, num_envs: int, time_offset: int) -> dict:
    shape = (steps, num_envs)
    truncated = torch.zeros(*shape, 1, dtype=torch.bool)
    episode_ends = (torch.arange(time_offset, time_offset + steps) + 1) % 64 == 0
    truncated[episode_ends] = True
    return {
        "observation": {
            "state": torch.randn(*shape, 64),
            "privileged_state": torch.randn(*shape, 462),
            "last_action": torch.randn(*shape, 29),
            "history_actor": torch.randn(*shape, 372),
        },
        "action": torch.randn(*shape, 29),
        "z": torch.randn(*shape, 256),
        "terminated": torch.zeros(*shape, 1, dtype=torch.bool),
        "truncated": truncated,
        "step_count": torch.arange(time_offset, time_offset + steps, dtype=torch.int64)[:, None, None].expand(*shape, 1),
        "reward": torch.randn(*shape, 1),
        "aux_rewards": {f"reward_{index}": torch.randn(*shape, 1) for index in range(8)},
        # Stored for reward relabeling, but omitted from training samples.
        "qpos": torch.randn(*shape, 36),
        "qvel": torch.randn(*shape, 35),
    }


def _batch_nbytes(batch: TensorDictBase) -> int:
    return sum(value.numel() * value.element_size() for value in batch.values(True, True) if isinstance(value, torch.Tensor))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(args: argparse.Namespace, storage: str, scratch_root: Path) -> dict[str, float | str]:
    device = torch.device(args.device)
    scratch_dir = scratch_root / storage if storage == "memmap" else None
    buffer = TorchRLReplayBuffer(
        capacity=args.capacity,
        batch_size=args.batch_size,
        sample_device=str(device),
        storage_kind=storage,
        prefetch=args.prefetch,
        pin_memory_threads=args.pin_memory_threads,
        trajectory=True,
        num_envs=args.num_envs,
        end_key="truncated",
        output_key_t=OUTPUT_KEY_T,
        output_key_tp1=OUTPUT_KEY_TP1,
        scratch_dir=scratch_dir,
    )
    time_steps = args.capacity // args.num_envs
    for offset in range(0, time_steps, args.fill_chunk_steps):
        steps = min(args.fill_chunk_steps, time_steps - offset)
        buffer.extend(_make_chunk(steps, args.num_envs, offset))

    for _ in range(args.warmup):
        batch = buffer.sample(args.batch_size)
        batch["observation", "state"].sum()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(args.iterations):
        batch = buffer.sample(args.batch_size)
        batch["observation", "state"].sum()
    _sync(device)
    elapsed = time.perf_counter() - start
    batch_bytes = _batch_nbytes(batch)
    peak_cuda_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    result = {
        "storage": storage,
        "ms_per_batch": elapsed * 1000 / args.iterations,
        "batches_per_second": args.iterations / elapsed,
        "gib_per_second": batch_bytes * args.iterations / elapsed / 2**30,
        "batch_mib": batch_bytes / 2**20,
        "peak_cuda_mib": peak_cuda_mib,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    buffer.close()
    del batch, buffer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=["cpu", "cuda", "memmap", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--pin-memory-threads", type=int, default=2)
    parser.add_argument("--fill-chunk-steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.capacity <= 0 or args.capacity % args.num_envs:
        parser.error("--capacity must be positive and divisible by --num-envs")
    if args.batch_size <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("batch size and iterations must be positive; warmup must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    storages = ["cpu", "cuda", "memmap"] if args.storage == "all" else [args.storage]
    with tempfile.TemporaryDirectory(prefix="ufo_replay_benchmark_") as scratch:
        results = [benchmark(args, storage, Path(scratch)) for storage in storages]

    print(f"capacity={args.capacity} batch_size={args.batch_size} prefetch={args.prefetch} device={args.device}")
    print("storage   ms/batch   batch/s   staged GiB/s   batch MiB   peak CUDA MiB   max RSS MiB")
    for result in results:
        print(
            f"{result['storage']:<8} "
            f"{result['ms_per_batch']:>9.3f} "
            f"{result['batches_per_second']:>9.2f} "
            f"{result['gib_per_second']:>14.2f} "
            f"{result['batch_mib']:>11.2f} "
            f"{result['peak_cuda_mib']:>15.1f} "
            f"{result['max_rss_mib']:>13.1f}"
        )


if __name__ == "__main__":
    main()
