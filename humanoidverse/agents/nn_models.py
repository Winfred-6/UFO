# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import numbers
import typing as tp

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal

from .base import BaseConfig
from .nn_filters import IdentityInputFilterConfig, NNFilter

##########################
# Initialization utils
##########################


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Update utils
##########################


def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


def soft_update_params(net, target_net, tau) -> None:
    tau = float(min(max(tau, 0), 1))
    net_params = tuple(x.data for x in net.parameters())
    target_net_params = tuple(x.data for x in target_net.parameters())
    _soft_update_params(net_params, target_net_params, tau)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Creation utils
##########################


class ForwardArchiConfig(BaseConfig):
    name: tp.Literal["ForwardArchi"] = "ForwardArchi"
    hidden_dim: int = 1024
    model: tp.Literal["simple", "residual"] = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: tp.Literal["batch", "seq", "vmap"] = "batch"  # {'batch', 'seq', 'vmap'}
    input_filter: NNFilter = IdentityInputFilterConfig()

    def model_post_init(self, context):
        if self.num_parallel > 1 and self.ensemble_mode == "seq":
            raise ValueError("seq ensemble mode is not compatible with num_parallel > 1. Use 'batch' or 'vmap' instead.")

    def build(self, obs_space, z_dim: int, action_dim, output_dim=None) -> torch.nn.Module:
        """Note: Forward model is also used for critics"""
        if self.ensemble_mode == "seq":
            return SequetialFMap(obs_space, z_dim, action_dim, self)
        elif self.ensemble_mode == "vmap":
            raise NotImplementedError("vmap ensemble mode is currently not supported")

        assert self.ensemble_mode == "batch", "Invalid value for ensemble_mode. Use {'batch', 'seq', 'vmap'}"
        return _build_batch_forward(self, obs_space, z_dim, action_dim, output_dim)


def _build_batch_forward(cfg, obs_space, z_dim, action_dim, output_dim=None):
    if cfg.model == "residual":
        forward_cls = ResidualForwardMap
    elif cfg.model == "simple":
        forward_cls = ForwardMap
    else:
        raise ValueError(f"Unsupported forward_map model {cfg.model}")
    return forward_cls(obs_space, z_dim, action_dim, cfg, output_dim=output_dim)


class ActorArchiConfig(BaseConfig):
    name: tp.Literal["actor"] = "actor"
    model: tp.Literal["simple", "residual"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim, action_dim):
        if self.model == "residual":
            return ResidualActor(obs_space, z_dim, action_dim, self)
        elif self.model == "simple":
            return Actor(obs_space, z_dim, action_dim, self)
        else:
            raise ValueError(f"Unsupported actor model {self.model}. Define 'model' or use other configs explicitely")


class DiscriminatorArchiConfig(BaseConfig):
    name: tp.Literal["DiscriminatorArchi"] = "DiscriminatorArchi"
    hidden_dim: int = 1024
    hidden_layers: int = 2
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim) -> "Discriminator":
        return Discriminator(obs_space, z_dim, self)


class ConditionalTemporalDiscriminatorArchiConfig(BaseConfig):
    """One gated temporal style critic for both walk and carry data."""

    name: tp.Literal["ConditionalTemporalDiscriminatorArchi"] = "ConditionalTemporalDiscriminatorArchi"
    hidden_dim: int = 1024
    hidden_layers: int = 3
    history_steps: int = 4
    object_history_steps: int = 4
    object_frame_dim: int = 12
    state_key: str = "state"
    privileged_key: str = "privileged_state"
    history_key: str = "history_actor"
    action_key: str = "last_action"
    object_key: str = "object_obs"
    # Carry goals are encoded by B for FB/reward inference, but the style
    # discriminator must not learn target-coordinate shortcuts through z.
    condition_on_z: bool = True

    def build(self, obs_space, z_dim) -> "ConditionalTemporalDiscriminator":
        return ConditionalTemporalDiscriminator(obs_space, z_dim, self)


def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################


class BackwardArchiConfig(BaseConfig):
    name: tp.Literal["BackwardArchi"] = "BackwardArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim: int):
        return BackwardMap(obs_space, z_dim, self)


class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        seq = [nn.Linear(filtered_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, z_dim)]
        if cfg.norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.input_filter(x)
        return self.net(x)


def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class ForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        action_dim,
        cfg: ForwardArchiConfig,
        output_dim=None,
    ) -> None:
        super().__init__()

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.cfg = cfg
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor):
        obs = self.input_filter(obs)
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class SequetialFMap(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg, output_dim=None):
        super().__init__()
        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.models = nn.ModuleList(
            [_build_batch_forward(cfg, obs_dim, z_dim, action_dim, cfg, output_dim, parallel=False) for _ in range(cfg.num_parallel)]
        )

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        predictions = [model(obs, z, action) for model in self.models]
        return torch.stack(predictions)


class SimpleActorArchiConfig(ActorArchiConfig):
    name: tp.Literal["simple"] = "simple"
    model: tp.Literal["simple"] = "simple"

    def build(self, obs_space, z_dim: int, action_dim: int) -> "Actor":
        return Actor(obs_space, z_dim, action_dim, self)


class Actor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: SimpleActorArchiConfig) -> None:
        super().__init__()

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.cfg: SimpleActorArchiConfig = cfg
        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z, std):
        obs = self.input_filter(obs)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class Discriminator(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: DiscriminatorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        seq = [nn.Linear(obs_dim + z_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        s = self.compute_logits(obs, z)
        return torch.sigmoid(s)

    def compute_logits(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        obs = self.input_filter(obs)
        x = torch.cat([z, obs], dim=1)
        logits = self.trunk(x)
        return logits

    def compute_reward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        s = self.forward(obs, z)
        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward


class ConditionalTemporalDiscriminator(nn.Module):
    """Conditional robot/object temporal discriminator with one scalar output.

    The robot branch is always active.  The box interaction residual is
    multiplied by a non-learned gate inferred from the current size fields:
    walk observations are exactly zero and therefore bypass the entire object
    branch, while carry observations jointly score robot and object motion.
    """

    def __init__(self, obs_space, z_dim: int, cfg: ConditionalTemporalDiscriminatorArchiConfig) -> None:
        super().__init__()
        if not isinstance(obs_space, gymnasium.spaces.Dict):
            raise TypeError("ConditionalTemporalDiscriminator requires a Dict observation space")
        required = (cfg.state_key, cfg.privileged_key, cfg.history_key, cfg.action_key, cfg.object_key)
        missing = [key for key in required if key not in obs_space.spaces]
        if missing:
            raise ValueError(f"Conditional temporal discriminator is missing observation keys: {missing}")

        self.cfg = cfg
        self.state_dim = int(obs_space[cfg.state_key].shape[0])
        self.privileged_dim = int(obs_space[cfg.privileged_key].shape[0])
        self.action_dim = int(obs_space[cfg.action_key].shape[0])
        self.history_dim = int(obs_space[cfg.history_key].shape[0])
        self.object_dim = int(obs_space[cfg.object_key].shape[0])
        if self.state_dim != 2 * self.action_dim + 6:
            raise ValueError(
                "Expected state=[dof_pos,dof_vel,gravity,base_ang_vel], got "
                f"state_dim={self.state_dim}, action_dim={self.action_dim}"
            )
        expected_history_dim = cfg.history_steps * (self.state_dim + self.action_dim)
        if self.history_dim != expected_history_dim:
            raise ValueError(
                f"Expected history_actor dim={expected_history_dim}, got {self.history_dim}"
            )
        expected_object_dim = cfg.object_history_steps * cfg.object_frame_dim
        if self.object_dim != expected_object_dim:
            raise ValueError(f"Expected object_obs dim={expected_object_dim}, got {self.object_dim}")
        if cfg.object_frame_dim < 3:
            raise ValueError("object_frame_dim must reserve its final three values for box size")

        temporal_dim = max(cfg.hidden_dim // 4, 64)
        self.robot_temporal = nn.Sequential(
            nn.Conv1d(self.state_dim, temporal_dim, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.privileged_encoder = nn.Sequential(
            nn.Linear(self.privileged_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.Tanh(),
        )
        robot_input_dim = 2 * temporal_dim + (z_dim if cfg.condition_on_z else 0)
        self.robot_encoder = nn.Sequential(
            nn.Linear(robot_input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Tanh(),
        )
        self.object_temporal = nn.Sequential(
            nn.Conv1d(cfg.object_frame_dim, temporal_dim, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(temporal_dim, cfg.hidden_dim),
            nn.Tanh(),
        )
        interaction_layers: list[nn.Module] = [
            nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Tanh(),
        ]
        for _ in range(max(cfg.hidden_layers - 2, 0)):
            interaction_layers.extend([nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()])
        self.interaction = nn.Sequential(*interaction_layers)
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def task_gate(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        object_seq = obs[self.cfg.object_key].reshape(
            -1,
            self.cfg.object_history_steps,
            self.cfg.object_frame_dim,
        )
        current_size = object_seq[:, 0, -3:]
        return (current_size.abs().sum(dim=-1, keepdim=True) > 1.0e-6).to(current_size.dtype)

    def _robot_sequence(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        history = obs[self.cfg.history_key]
        steps = self.cfg.history_steps
        dof_dim = self.action_dim
        offset = steps * dof_dim  # sorted history begins with past actions; AMP ignores them.
        base_ang = history[:, offset : offset + steps * 3].reshape(-1, steps, 3)
        offset += steps * 3
        dof_pos = history[:, offset : offset + steps * dof_dim].reshape(-1, steps, dof_dim)
        offset += steps * dof_dim
        dof_vel = history[:, offset : offset + steps * dof_dim].reshape(-1, steps, dof_dim)
        offset += steps * dof_dim
        gravity = history[:, offset : offset + steps * 3].reshape(-1, steps, 3)
        past_state = torch.cat([dof_pos, dof_vel, gravity, base_ang], dim=-1)
        return torch.cat([obs[self.cfg.state_key].unsqueeze(1), past_state], dim=1)

    def compute_logits(self, obs: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        robot_sequence = self._robot_sequence(obs)
        temporal_robot = self.robot_temporal(robot_sequence.transpose(1, 2)).squeeze(-1)
        privileged = self.privileged_encoder(obs[self.cfg.privileged_key])
        robot_inputs = [temporal_robot, privileged]
        if self.cfg.condition_on_z:
            robot_inputs.insert(0, z)
        robot = self.robot_encoder(torch.cat(robot_inputs, dim=-1))

        object_sequence = obs[self.cfg.object_key].reshape(
            -1,
            self.cfg.object_history_steps,
            self.cfg.object_frame_dim,
        )
        object_embedding = self.object_temporal(object_sequence.transpose(1, 2))
        interaction = self.interaction(torch.cat([robot, object_embedding, robot * object_embedding], dim=-1))
        fused = robot + self.task_gate(obs) * interaction
        return self.head(fused)

    def forward(self, obs: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.compute_logits(obs, z))

    def compute_reward(self, obs: dict[str, torch.Tensor], z: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
        score = self.forward(obs, z).clamp(eps, 1.0 - eps)
        return score.log() - (1.0 - score).log()


class VForwardArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim: int, output_dim=None) -> torch.nn.Module:
        return VForwardMap(obs_space, z_dim, output_dim, self)


class VForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        output_dim=None,
        cfg: VForwardArchiConfig = VForwardArchiConfig(),
    ) -> None:
        super().__init__()

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        obs = self.input_filter(obs)
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([s_embedding, z_embedding], dim=-1))


##########################
# Residual models
##########################


class ResidualBlock(nn.Module):
    def __init__(self, dim, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(dim, num_parallel)
        lin = linear(dim, dim, num_parallel)
        self.mlp = nn.Sequential(ln, lin, nn.Mish())

    def forward(self, x):
        return x + self.mlp(x)


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(input_dim, num_parallel)
        lin = linear(input_dim, output_dim, num_parallel)
        seq = [ln, lin] + ([nn.Mish()] if activation else [])
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        return self.mlp(x)


def residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [Block(input_dim, hidden_dim, True, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]
    return nn.Sequential(*seq)


class ResidualForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        action_dim,
        cfg: ForwardArchiConfig,
        output_dim=None,
    ) -> None:
        super().__init__()

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = residual_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.num_parallel)
        self.embed_sa = residual_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.num_parallel)

        seq = [ResidualBlock(cfg.hidden_dim, cfg.num_parallel) for _ in range(cfg.hidden_layers)]
        seq += [Block(cfg.hidden_dim, output_dim if output_dim else z_dim, False, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = self.input_filter(obs)
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class ResidualActorArchiConfig(ActorArchiConfig):
    name: tp.Literal["residual"] = "residual"
    model: tp.Literal["residual"] = "residual"

    def build(self, obs_space, z_dim, action_dim) -> "Actor":
        return ResidualActor(obs_space, z_dim, action_dim, self)


class ResidualActor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: ResidualActorArchiConfig) -> None:
        super().__init__()

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.cfg: ResidualActorArchiConfig = cfg
        self.embed_z = residual_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = residual_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = [ResidualBlock(cfg.hidden_dim) for _ in range(cfg.hidden_layers)] + [Block(cfg.hidden_dim, action_dim, False)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z, std):
        obs = self.input_filter(obs)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    # TODO why do these layers have their own reset scheme?
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, elementwise_affine=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [
                normalized_shape,
            ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([*self.normalized_shape], **factory_kwargs))
            else:
                self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
                self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class EMA(nn.Module):
    """exponential moving average"""

    def __init__(self, tau=0.99, epsilon=1e-8, shape=(1,), translate=False, scale=False) -> None:
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("mean_square", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("counter", torch.LongTensor([0]))
        self.translate = translate
        self.scale = scale

    def forward(self, x):
        m = x.mean()
        sm = x.pow(2).mean()
        self.mean.data = self.tau * self.mean + (1 - self.tau) * m
        self.mean_square.data = self.tau * self.mean_square + (1 - self.tau) * sm
        self.counter += 1  # type: ignore
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm
        ema_mean_square = self.mean_square / norm
        var = torch.clamp(ema_mean_square - ema_mean**2, min=self.epsilon)

        translate_mean = ema_mean if self.translate else 0
        scale_std = torch.sqrt(var) if self.scale else 1
        return (x - translate_mean) / scale_std

    @property
    def S(self):
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm
        ema_mean_square = self.mean_square / norm
        var = torch.clamp(ema_mean_square - ema_mean**2, self.epsilon)
        return var

    @property
    def M(self):
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm
        return ema_mean


class RewardNormalizerConfig(BaseConfig):
    name: tp.Literal["RewardNormalizer"] = "RewardNormalizer"
    translate: bool = False
    scale: bool = False

    def build(self) -> nn.Module:
        return EMA(translate=self.translate, scale=self.scale)
