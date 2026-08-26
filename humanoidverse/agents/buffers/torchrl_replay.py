"""TorchRL-backed replay buffers with host storage and prefetched device batches.

The large replay allocation lives in CPU RAM (or a memory map).  Sampling,
trajectory reconstruction, page locking, and host-to-device copies run in the
TorchRL prefetch workers, so the training thread receives a ready-to-use batch
on the agent device.
"""

from __future__ import annotations

import json
import numbers
import threading
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch.utils._pytree import tree_map
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ReplayBuffer, SliceSampler


def _first_tensor(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, TensorDictBase):
        for value in data.values(True, True):
            if isinstance(value, torch.Tensor):
                return value
    elif isinstance(data, Mapping):
        for value in data.values():
            try:
                return _first_tensor(value)
            except ValueError:
                pass
    raise ValueError("Replay data must contain at least one tensor-like leaf")


def _as_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (np.generic, numbers.Number)):
        return torch.as_tensor(value)
    return value


def _as_tensordict(data: Mapping[str, Any] | TensorDictBase, *, batch_ndim: int) -> TensorDictBase:
    if isinstance(data, TensorDictBase):
        return data.detach()
    tensor_data = tree_map(_as_tensor, data)
    first = _first_tensor(tensor_data)
    if first.ndim < batch_ndim:
        raise ValueError(f"Replay data needs {batch_ndim} batch dimensions, got shape={tuple(first.shape)}")
    return TensorDict.from_dict(tensor_data, batch_size=first.shape[:batch_ndim])


class _ReconstructNextAndTransfer:
    """Turn sampled ``[t, t+1]`` slices into transitions and stage them."""

    def __init__(
        self,
        *,
        output_key_t: list[str],
        output_key_tp1: list[str],
        sample_device: str,
        pin_memory_threads: int,
    ) -> None:
        # Keep the lists by reference. Reward inference adds qpos/qvel after
        # loading a checkpoint, and the sampling transform must see that
        # update without rebuilding the ReplayBuffer.
        self.output_key_t = output_key_t
        self.output_key_tp1 = output_key_tp1
        self.sample_device = torch.device(sample_device)
        self.pin_memory_threads = int(pin_memory_threads)

    def __call__(self, sample: TensorDictBase) -> TensorDictBase:
        if sample.shape[0] % 2:
            raise RuntimeError(f"Expected paired trajectory samples, got batch_size={sample.shape[0]}")
        paired = sample.reshape(-1, 2)
        current = paired[:, 0].select(*self.output_key_t, strict=False)
        next_step = paired[:, 1].select(*self.output_key_tp1, strict=False)
        current.set("next", next_step)
        return _stage_batch(
            current,
            device=self.sample_device,
            pin_memory_threads=self.pin_memory_threads,
        )


class _TransferOnly:
    def __init__(self, *, sample_device: str, pin_memory_threads: int) -> None:
        self.sample_device = torch.device(sample_device)
        self.pin_memory_threads = int(pin_memory_threads)

    def __call__(self, sample: TensorDictBase) -> TensorDictBase:
        return _stage_batch(
            sample,
            device=self.sample_device,
            pin_memory_threads=self.pin_memory_threads,
        )


def _stage_batch(batch: TensorDictBase, *, device: torch.device, pin_memory_threads: int) -> TensorDictBase:
    leaf_devices = {
        value.device
        for value in batch.values(True, True)
        if isinstance(value, torch.Tensor)
    }
    if leaf_devices == {device}:
        return batch
    if device.type == "cuda" and leaf_devices and all(source.type == "cpu" for source in leaf_devices):
        # TensorDict pins leaves concurrently, then enqueues non-blocking H2D
        # copies.  TorchRL calls this transform in its prefetch worker.
        return batch.to(
            device,
            non_blocking=True,
            non_blocking_pin=True,
            num_threads=pin_memory_threads,
        )
    return batch.to(device)


class TorchRLReplayBuffer:
    """Small compatibility adapter around TorchRL's composable ReplayBuffer.

    ``capacity`` and ``batch_size`` are measured in transitions.  For a
    trajectory buffer, TorchRL samples pairs of frames internally and this
    adapter reconstructs the public ``next`` mapping without storing a second
    copy of every observation.
    """

    CONFIG_NAME = "ufo_torchrl_buffer.json"
    STATE_DIR = "torchrl_state"

    def __init__(
        self,
        *,
        capacity: int,
        batch_size: int,
        sample_device: str,
        storage_kind: str = "cpu",
        prefetch: int = 2,
        pin_memory_threads: int = 2,
        trajectory: bool = True,
        num_envs: int = 1,
        end_key: str = "truncated",
        output_key_t: list[str] | None = None,
        output_key_tp1: list[str] | None = None,
        scratch_dir: str | Path | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        if batch_size <= 0:
            raise ValueError("Replay batch size must be positive")
        if prefetch < 0:
            raise ValueError("Replay prefetch depth must be non-negative")
        if pin_memory_threads < 0:
            raise ValueError("pin_memory_threads must be non-negative")
        if trajectory and num_envs <= 0:
            raise ValueError("num_envs must be positive for trajectory replay")
        if trajectory and capacity % num_envs:
            raise ValueError(f"Trajectory replay capacity ({capacity}) must be divisible by num_envs ({num_envs})")
        if storage_kind not in {"cpu", "cuda", "memmap"}:
            raise ValueError(f"Unsupported replay storage kind: {storage_kind}")

        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.sample_device = str(sample_device)
        self.storage_kind = storage_kind
        self.prefetch = int(prefetch)
        self.pin_memory_threads = int(pin_memory_threads)
        self.trajectory = bool(trajectory)
        self.num_envs = int(num_envs)
        self.end_key = end_key
        self.output_key_t = list(output_key_t or [])
        self.output_key_tp1 = list(output_key_tp1 or [])
        self.scratch_dir = str(scratch_dir) if scratch_dir is not None else None

        ndim = 2 if self.trajectory else 1
        if storage_kind == "memmap":
            if scratch_dir is None:
                raise ValueError("scratch_dir is required for memmap replay storage")
            Path(scratch_dir).mkdir(parents=True, exist_ok=True)
            storage = LazyMemmapStorage(
                self.capacity,
                scratch_dir=str(scratch_dir),
                device="cpu",
                ndim=ndim,
                existsok=True,
            )
        else:
            storage_device = self.sample_device if storage_kind == "cuda" else "cpu"
            storage = LazyTensorStorage(
                self.capacity,
                device=torch.device(storage_device),
                ndim=ndim,
                consolidated=True,
            )

        if self.trajectory:
            if not self.output_key_t or not self.output_key_tp1:
                raise ValueError("Trajectory replay requires output_key_t and output_key_tp1")
            sampler = SliceSampler(
                slice_len=2,
                end_key=self.end_key,
                truncated_key=None,
                cache_values=True,
            )
            transform = _ReconstructNextAndTransfer(
                output_key_t=self.output_key_t,
                output_key_tp1=self.output_key_tp1,
                sample_device=self.sample_device,
                pin_memory_threads=self.pin_memory_threads,
            )
            torchrl_batch_size = self.batch_size * 2
            dim_extend = 0
        else:
            sampler = None
            transform = _TransferOnly(
                sample_device=self.sample_device,
                pin_memory_threads=self.pin_memory_threads,
            )
            torchrl_batch_size = self.batch_size
            dim_extend = 0

        self._buffer = ReplayBuffer(
            storage=storage,
            sampler=sampler,
            transform=transform,
            batch_size=torchrl_batch_size,
            dim_extend=dim_extend,
            prefetch=self.prefetch or None,
        )

    @property
    def device(self) -> str:
        return self.sample_device if self.storage_kind == "cuda" else "cpu"

    def __len__(self) -> int:
        return len(self._buffer)

    def size(self) -> int:
        return len(self)

    def empty(self) -> bool:
        return len(self) == 0

    @torch.no_grad()
    def extend(self, data: Mapping[str, Any] | TensorDictBase) -> None:
        batch_ndim = 2 if self.trajectory else 1
        tensordict = _as_tensordict(data, batch_ndim=batch_ndim)
        if self.trajectory and tensordict.shape[1] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} environments, got replay batch shape={tuple(tensordict.shape)}")
        self._buffer.extend(tensordict)

    @torch.no_grad()
    def sample(self, batch_size: int = 1, seq_length: int | None = None) -> TensorDictBase:
        if self.trajectory and seq_length not in (None, 1):
            raise ValueError("Online TorchRL replay currently samples one-step transitions only")
        if batch_size != self.batch_size:
            if self.prefetch:
                raise ValueError(
                    f"Prefetched replay has fixed batch_size={self.batch_size}, got sample batch_size={batch_size}"
                )
            raw_batch_size = batch_size * 2 if self.trajectory else batch_size
            return self._buffer.sample(raw_batch_size)
        return self._buffer.sample()

    def include_next_keys(self, *keys: str) -> None:
        """Include additional stored fields in subsequent ``next`` samples."""

        if not self.trajectory:
            raise ValueError("Additional next keys are only supported for trajectory replay")
        self._drain_prefetch()
        for key in keys:
            if key not in self.output_key_tp1:
                self.output_key_tp1.append(key)

    @torch.no_grad()
    def get_full_buffer(self) -> TensorDictBase:
        """Return every valid one-step transition in trajectory order.

        TorchRL stores frames as a time-by-environment ring. This reconstructs
        the same transition view as ``sample`` while excluding episode ends and
        the newest frame, whose successor has not been written yet.
        """

        if not self.trajectory:
            return self._buffer.storage[:]

        frames = self._buffer.storage[:]
        if frames.ndim != 2:
            raise RuntimeError(f"Expected time-by-environment replay storage, got shape={tuple(frames.shape)}")
        time_steps, num_envs = frames.shape
        if time_steps < 2:
            return TensorDict({}, batch_size=[0], device=frames.device)

        storage = self._buffer.storage
        if storage._is_full:
            cursor = int(self._buffer.writer._cursor)
            current_time = torch.arange(time_steps, device=frames.device)
            next_time = (current_time + 1) % time_steps
            has_successor = current_time != ((cursor - 1) % time_steps)
        else:
            current_time = torch.arange(time_steps - 1, device=frames.device)
            next_time = current_time + 1
            has_successor = torch.ones_like(current_time, dtype=torch.bool)

        current_time = current_time[:, None].expand(-1, num_envs)
        next_time = next_time[:, None].expand(-1, num_envs)
        env_index = torch.arange(num_envs, device=frames.device)[None, :].expand_as(current_time)
        has_successor = has_successor[:, None].expand_as(current_time)

        end = frames.get(self.end_key)[current_time, env_index]
        end = end.reshape(*current_time.shape, -1).any(dim=-1)
        valid = has_successor & ~end
        current_index = (current_time[valid], env_index[valid])
        next_index = (next_time[valid], env_index[valid])

        current = frames[current_index].select(*self.output_key_t, strict=False)
        next_step = frames[next_index].select(*self.output_key_tp1, strict=False)
        current.set("next", next_step)
        return _stage_batch(
            current,
            device=torch.device(self.sample_device),
            pin_memory_threads=self.pin_memory_threads,
        )

    def save(self, folder: str | Path) -> None:
        self._drain_prefetch()
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        with (folder / self.CONFIG_NAME).open("w") as file:
            json.dump(self._config_dict(), file, indent=2)
        self._buffer.dumps(folder / self.STATE_DIR)

    def _drain_prefetch(self) -> None:
        queue = getattr(self._buffer, "_prefetch_queue", ())
        while queue:
            future = queue.popleft()
            if not future.cancel():
                future.result()

    def close(self) -> None:
        self._drain_prefetch()
        executor = getattr(self._buffer, "_prefetch_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _config_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "sample_device": self.sample_device,
            "storage_kind": self.storage_kind,
            "prefetch": self.prefetch,
            "pin_memory_threads": self.pin_memory_threads,
            "trajectory": self.trajectory,
            "num_envs": self.num_envs,
            "end_key": self.end_key,
            "output_key_t": self.output_key_t,
            "output_key_tp1": self.output_key_tp1,
            "scratch_dir": self.scratch_dir,
        }

    @classmethod
    def load(
        cls,
        folder: str | Path,
        *,
        sample_device: str | None = None,
        prefetch: int | None = None,
        pin_memory_threads: int | None = None,
        scratch_dir: str | Path | None = None,
    ) -> "TorchRLReplayBuffer":
        folder = Path(folder)
        with (folder / cls.CONFIG_NAME).open() as file:
            config = json.load(file)
        if sample_device is not None:
            config["sample_device"] = sample_device
        if prefetch is not None:
            config["prefetch"] = prefetch
        if pin_memory_threads is not None:
            config["pin_memory_threads"] = pin_memory_threads
        if scratch_dir is not None:
            config["scratch_dir"] = str(scratch_dir)
        buffer = cls(**config)
        buffer._buffer.loads(folder / cls.STATE_DIR)
        return buffer


class PrefetchedDeviceBuffer:
    """Prefetch samples from an existing immutable buffer onto a device.

    This is used for the expert trajectory slicer, whose motion-level priority
    semantics are kept intact while CPU sampling and H2D staging overlap the
    preceding optimizer update.
    """

    def __init__(
        self,
        buffer: Any,
        *,
        batch_size: int,
        sample_device: str,
        prefetch: int = 2,
        pin_memory_threads: int = 2,
    ) -> None:
        self._buffer = buffer
        self.batch_size = int(batch_size)
        self.sample_device = torch.device(sample_device)
        self.prefetch = int(prefetch)
        self.pin_memory_threads = int(pin_memory_threads)
        self._lock = threading.RLock()
        self._queue: deque[Future] = deque()
        self._executor = ThreadPoolExecutor(max_workers=max(self.prefetch, 1), thread_name_prefix="ufo-expert-prefetch")

    def __len__(self) -> int:
        return len(self._buffer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._buffer, name)

    def _sample_and_stage(self, batch_size: int, seq_length: int | None) -> TensorDictBase:
        batch = self._buffer.sample(batch_size, seq_length=seq_length)
        tensordict = _as_tensordict(batch, batch_ndim=1)
        return _stage_batch(
            tensordict,
            device=self.sample_device,
            pin_memory_threads=self.pin_memory_threads,
        )

    def _fill_default_queue(self) -> None:
        while len(self._queue) < self.prefetch:
            self._queue.append(self._executor.submit(self._sample_and_stage, self.batch_size, None))

    @torch.no_grad()
    def sample(self, batch_size: int = 1, seq_length: int | None = None) -> TensorDictBase:
        if not self.prefetch or batch_size != self.batch_size or seq_length is not None:
            return self._sample_and_stage(batch_size, seq_length)
        with self._lock:
            self._fill_default_queue()
            result = self._queue.popleft().result()
            self._fill_default_queue()
            return result

    def _drain_prefetch(self) -> None:
        with self._lock:
            while self._queue:
                future = self._queue.popleft()
                if not future.cancel():
                    future.result()

    def update_priorities(self, *args, **kwargs) -> None:
        self._drain_prefetch()
        self._buffer.update_priorities(*args, **kwargs)

    def close(self) -> None:
        self._drain_prefetch()
        self._executor.shutdown(wait=True, cancel_futures=True)
