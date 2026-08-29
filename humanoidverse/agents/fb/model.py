# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import typing as tp

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..base_model import BaseModel, BaseModelConfig
from ..nn_filter_models import (
    ActorFilterArchiConfig,
    BackwardFilterArchiConfig,
    ForwardFilterArchiConfig,
    ResidualActorFilterArchiConfig,
    SimpleActorFilterArchiConfig,
)
from ..nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    ForwardArchiConfig,
    ResidualActorArchiConfig,
    SimpleActorArchiConfig,
    eval_mode,
)
from ..normalizers import ObsNormalizerConfig
from ..pytree_utils import tree_get_batch_size


class FBModelArchiConfig(BaseConfig):
    z_dim: int = 100
    norm_z: bool = True
    f: ForwardArchiConfig | ForwardFilterArchiConfig = pydantic.Field(ForwardArchiConfig(), discriminator="name")
    b: BackwardArchiConfig | BackwardFilterArchiConfig = pydantic.Field(BackwardArchiConfig(), discriminator="name")
    # Because of the "name" attribute, these two can be chosen between via strings easily
    actor: (
        ActorArchiConfig
        | ActorFilterArchiConfig
        | SimpleActorArchiConfig
        | ResidualActorArchiConfig
        | SimpleActorFilterArchiConfig
        | ResidualActorFilterArchiConfig
    ) = pydantic.Field(SimpleActorArchiConfig(), discriminator="name")


class FBModelConfig(BaseModelConfig):
    name: tp.Literal["FBModel"] = "FBModel"

    archi: FBModelArchiConfig = FBModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = ObsNormalizerConfig()
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    amp: bool = False
    # Deprecated inference compatibility for pre-goal_obs carry checkpoints.
    # New presets keep these values unset so project_z normalizes the complete
    # latent and no physical command is assigned to arbitrary FB coordinates.
    task_latent_key: str | None = None
    task_latent_dim: int = 0
    task_latent_scale: float = 1.0

    def build(self, obs_space, action_dim) -> "FBModel":
        return self.object_class(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBModel


class FBModel(BaseModel):
    config_class = FBModelConfig

    def __init__(self, obs_space, action_dim, cfg: FBModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg: FBModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device
        self.amp_dtype = torch.bfloat16
        if self.cfg.task_latent_dim < 0 or self.cfg.task_latent_dim >= arch.z_dim:
            raise ValueError(
                f"task_latent_dim must be in [0, {arch.z_dim}), got {self.cfg.task_latent_dim}"
            )
        if (self.cfg.task_latent_dim > 0) != (self.cfg.task_latent_key is not None):
            raise ValueError("task_latent_key and a positive task_latent_dim must be configured together")

        # create networks
        self._backward_map = arch.b.build(obs_space, arch.z_dim)
        self._forward_map = arch.f.build(obs_space, arch.z_dim, action_dim)
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            z = self.condition_task_latent(z, obs)
            return self._actor(self._normalize(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        task_dim = int(self.cfg.task_latent_dim)
        if task_dim > 0:
            style_dim = z.shape[-1] - task_dim
            style = z[..., :style_dim]
            if self.cfg.archi.norm_z:
                style = math.sqrt(style_dim) * F.normalize(style, dim=-1)
            task = torch.zeros_like(z[..., style_dim:])
            return torch.cat([style, task], dim=-1)
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def condition_task_latent(
        self,
        z: torch.Tensor,
        obs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the deprecated z-tail adapter for legacy inference only."""

        task_dim = int(self.cfg.task_latent_dim)
        if task_dim <= 0:
            return z
        if not isinstance(obs, dict) or self.cfg.task_latent_key not in obs:
            raise KeyError(f"Missing task-latent observation key {self.cfg.task_latent_key!r}")
        command = obs[self.cfg.task_latent_key]
        if command.shape[-1] != task_dim:
            raise ValueError(f"Expected task command dim={task_dim}, got {command.shape[-1]}")
        if z.shape[0] == 1 and command.shape[0] != 1:
            z = z.expand(command.shape[0], -1)
        if z.shape[0] != command.shape[0]:
            raise ValueError(f"Task command batch={command.shape[0]} does not match z batch={z.shape[0]}")
        base = self.project_z(z)
        command = command.to(device=z.device, dtype=z.dtype).clamp(-1.0, 1.0)
        command = command * float(self.cfg.task_latent_scale)
        return torch.cat([base[..., :-task_dim], command], dim=-1)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()  # TODO we upcast to float32 to make sure the action can be converted to numpy later

    def reward_inference(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
            batch_size = tree_get_batch_size(next_obs)
            num_batches = int(np.ceil(batch_size / self.cfg.inference_batch_size))
            z = 0
            wr = reward if weight is None else reward * weight
            for i in range(num_batches):
                start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
                next_obs_slice = tree_map(lambda x: x[start_idx:end_idx].to(self.device), next_obs)
                B = self.backward_map(next_obs_slice)
                z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
