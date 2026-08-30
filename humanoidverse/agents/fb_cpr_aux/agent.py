# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Dict

import pydantic
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ...distributed import average_gradients
from ..base import BaseConfig
from ..fb_cpr.agent import FBcprAgent, FBcprAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from .model import FBcprAuxModelConfig


class FBcprAuxAgentTrainConfig(FBcprAgentTrainConfig):
    lr_aux_critic: float = 1e-4
    reg_coeff_aux: float = 1.0
    aux_critic_pessimism_penalty: float = 0.5


class FBcprAuxAgentConfig(BaseConfig):
    name: tp.Literal["FBcprAuxAgent"] = "FBcprAuxAgent"

    model: FBcprAuxModelConfig = FBcprAuxModelConfig()
    train: FBcprAuxAgentTrainConfig = FBcprAuxAgentTrainConfig()
    aux_rewards: list[str] = pydantic.Field(default_factory=list)
    aux_rewards_scaling: dict[str, float] = pydantic.Field(default_factory=dict)
    cudagraphs: bool = False
    compile: bool = False
    fail_fast_diagnostics: bool = False

    def build(self, obs_space, action_dim: int) -> "FBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        return FBcprAuxAgent


class FBcprAuxAgent(FBcprAgent):
    config_class = FBcprAuxAgentConfig

    @staticmethod
    def _iter_diagnostic_tensors(value: Any, prefix: str = ""):
        if isinstance(value, torch.Tensor):
            yield prefix or "<tensor>", value
        elif isinstance(value, Mapping):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from FBcprAuxAgent._iter_diagnostic_tensors(child, child_prefix)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                yield from FBcprAuxAgent._iter_diagnostic_tensors(child, f"{prefix}[{index}]")

    def _diagnostic_assert_finite(self, value: Any, label: str, step: int) -> None:
        if not self.cfg.fail_fast_diagnostics:
            return
        failures = []
        for name, tensor in self._iter_diagnostic_tensors(value):
            if not tensor.is_floating_point():
                continue
            finite = torch.isfinite(tensor)
            if bool(torch.all(finite).item()):
                continue
            bad = (~finite).nonzero(as_tuple=False)
            finite_values = tensor[finite]
            failures.append(
                {
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "bad_count": int((~finite).sum().item()),
                    "first_bad_index": bad[0].detach().cpu().tolist(),
                    "first_bad_value": tensor[tuple(bad[0].tolist())].detach().cpu().item(),
                    "finite_min": float(finite_values.min().item()) if finite_values.numel() else None,
                    "finite_max": float(finite_values.max().item()) if finite_values.numel() else None,
                    "finite_max_abs": float(finite_values.abs().max().item()) if finite_values.numel() else None,
                }
            )
        if failures:
            raise FloatingPointError(f"FB raw-update fail-fast: step={step}, label={label}, failures={failures}")

    def _diagnostic_assert_module(self, module: torch.nn.Module, label: str, step: int) -> None:
        if not self.cfg.fail_fast_diagnostics:
            return
        state = {f"parameter.{name}": value for name, value in module.named_parameters()}
        state.update({f"buffer.{name}": value for name, value in module.named_buffers()})
        floating = {name: value for name, value in state.items() if value.is_floating_point()}
        if not floating:
            return
        # Keep the healthy path to one synchronization per module.  Detailed
        # tensor inspection is only paid for after a failure is detected.
        bad_flags = torch.stack([~torch.isfinite(value).all() for value in floating.values()])
        if not bool(torch.any(bad_flags).item()):
            return
        bad_state = {
            name: value
            for (name, value), is_bad in zip(floating.items(), bad_flags.tolist())
            if is_bad
        }
        self._diagnostic_assert_finite(bad_state, label, step)

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(),
            lr=self.cfg.train.lr_aux_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer.state_dict()
        return optimizers

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_aux_critic = torch.compile(self.update_aux_critic, mode=mode)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_aux_critic = CudaGraphModule(self.update_aux_critic, warmup=5)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        profiler = getattr(self, "_runtime_profiler", None)

        def stage(name: str, *, cuda: bool = False):
            return profiler.stage(name, cuda=cuda) if profiler is not None else nullcontext()

        with stage("replay_sample"):
            expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
            train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        self._diagnostic_assert_finite(expert_batch, "raw_expert_batch", step)
        self._diagnostic_assert_finite(train_batch, "raw_train_batch", step)
        self._diagnostic_assert_module(self._model._obs_normalizer, "obs_normalizer_before_update", step)

        with stage("update_prepare", cuda=True):
            train_obs, train_action, train_next_obs = (
                tree_map(lambda x: x.to(self.device), train_batch["observation"]),
                train_batch["action"].to(self.device),
                tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
            )
            discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
            expert_obs, expert_next_obs = (
                tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
                tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
            )

            self._model._obs_normalizer(train_obs)
            self._model._obs_normalizer(train_next_obs)
            self._diagnostic_assert_module(self._model._obs_normalizer, "obs_normalizer_after_update", step)

            with torch.no_grad(), eval_mode(self._model._obs_normalizer):
                train_obs, train_next_obs = (
                    self._model._obs_normalizer(train_obs),
                    self._model._obs_normalizer(train_next_obs),
                )
                expert_obs, expert_next_obs = (
                    self._model._obs_normalizer(expert_obs),
                    self._model._obs_normalizer(expert_next_obs),
                )
            self._diagnostic_assert_finite(
                {
                    "train_obs": train_obs,
                    "train_next_obs": train_next_obs,
                    "expert_obs": expert_obs,
                    "expert_next_obs": expert_next_obs,
                },
                "normalized_observations",
                step,
            )

            torch.compiler.cudagraph_mark_step_begin()
            expert_z = self.encode_expert(next_obs=expert_next_obs)
            train_z = train_batch["z"].to(self.device)
            self._diagnostic_assert_finite({"expert_z": expert_z, "train_z": train_z}, "initial_latents", step)

        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        with stage("discriminator", cuda=True):
            metrics = self.update_discriminator(
                expert_obs=expert_obs,
                expert_z=expert_z,
                train_obs=train_obs,
                train_z=train_z,
                grad_penalty=grad_penalty,
            )
        self._diagnostic_assert_finite(metrics, "discriminator_metrics", step)
        self._diagnostic_assert_module(self._model._discriminator, "discriminator_after_update", step)

        with stage("latent_prepare", cuda=True):
            z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
            self.z_buffer.add(z)
            self._diagnostic_assert_finite(z, "mixed_latent", step)

            if self.cfg.train.relabel_ratio is not None:
                mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
                train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        with stage("fb", cuda=True):
            fb_metrics = self.update_fb(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
                    next_obs=train_next_obs,
                    goal=train_next_obs,
                    z=train_z,
                    q_loss_coef=q_loss_coef,
                    clip_grad_norm=clip_grad_norm,
                )
            self._diagnostic_assert_finite(fb_metrics, "fb_metrics", step)
            metrics.update(fb_metrics)
        self._diagnostic_assert_module(self._model._forward_map, "forward_map_after_update", step)
        self._diagnostic_assert_module(self._model._backward_map, "backward_map_after_update", step)
        with stage("critic", cuda=True):
            critic_metrics = self.update_critic(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
                    next_obs=train_next_obs,
                    z=train_z,
                )
            self._diagnostic_assert_finite(critic_metrics, "critic_metrics", step)
            metrics.update(critic_metrics)
        self._diagnostic_assert_module(self._model._critic, "critic_after_update", step)
        # compute scalar auxiliary reward as a weighted sum of the auxiliary rewards
        with stage("aux_reward_prepare", cuda=True):
            aux_reward = torch.zeros(
                (self.cfg.train.batch_size, 1),
                device=self.device,
                dtype=torch.float32,
            )
            for aux_reward_name in self.cfg.aux_rewards:
                # let's log even this information
                metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
                aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)

            self._diagnostic_assert_finite(aux_reward, "raw_weighted_aux_reward", step)
            aux_reward = self._model._aux_reward_normalizer(aux_reward)
            self._diagnostic_assert_finite(aux_reward, "normalized_aux_reward", step)
            self._diagnostic_assert_module(self._model._aux_reward_normalizer, "aux_reward_normalizer", step)

        with stage("aux_critic", cuda=True):
            aux_critic_metrics = self.update_aux_critic(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
                    aux_reward=aux_reward,
                    next_obs=train_next_obs,
                    z=train_z,
                )
            self._diagnostic_assert_finite(aux_critic_metrics, "aux_critic_metrics", step)
            metrics.update(aux_critic_metrics)
        self._diagnostic_assert_module(self._model._aux_critic, "aux_critic_after_update", step)
        with stage("actor", cuda=True):
            actor_metrics = self.update_actor(
                    obs=train_obs,
                    action=train_action,
                    z=train_z,
                    clip_grad_norm=clip_grad_norm,
                )
            self._diagnostic_assert_finite(actor_metrics, "actor_metrics", step)
            metrics.update(actor_metrics)
        self._diagnostic_assert_module(self._model._actor, "actor_after_update", step)

        with stage("target_update", cuda=True), torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            _soft_update_params(
                self._aux_critic_map_paramlist,
                self._aux_target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            num_parallel = self.cfg.model.archi.aux_critic.num_parallel
            # compute target critic
            with torch.no_grad():
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_aux_critic(next_obs, z, next_action)  # num_parallel x batch x 1
                # TODO AL: should we have aux_critic parameters here?
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.aux_critic_pessimism_penalty)
                target_Q = aux_reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            # compute critic loss
            Qs = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        average_gradients(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_Q.mean().detach(),
                "auxQ1": Qs.mean().detach(),
                "mean_next_auxQ": Q_mean.mean().detach(),
                "unc_auxQ": Q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)

            # compute discriminator reward loss
            Qs_discriminator = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute auxiliary reward loss
            Qs_aux = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
            _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

            # compute fb reward loss
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
            Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = (
                -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
                - Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
                - Q_fb.mean()
            )

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        average_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
        return output_metrics
