"""FB agent preset for UFO training."""

from __future__ import annotations

from humanoidverse.agents.envs.carry_box import OBJECT_FRAME_DIM, OBJECT_HISTORY_STEPS, TASK_COMMAND_DIM
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from humanoidverse.agents.nn_filters import DictInputFilterConfig
from humanoidverse.agents.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    ConditionalTemporalDiscriminatorArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from humanoidverse.agents.normalizers import BatchNormNormalizerConfig, IdentityNormalizerConfig, ObsNormalizerConfig


TRAIN_RUNTIME = {
    "log_every_updates": 384000,
    "update_agent_every": 1024,
    "num_seed_steps": 10240,
    "num_agent_updates": 16,
    "checkpoint_buffer": True,
    "use_trajectory_buffer": True,
    "buffer_size": 5120000,
    "eval_every_steps": 3200000,
}


def build_fb_agent(
    device: str = "cuda",
    compile: bool = True,
    update_z_every_step: int = 100,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    cartwheel_aux_safe: bool = False,
    carry_box: bool = False,
) -> FBcprAuxAgentConfig:
    if cartwheel_aux_safe:
        aux_rewards = [
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
        ]
        aux_rewards_scaling = {
            "penalty_action_rate": -0.03,
            "limits_dof_pos": -10.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }
    else:
        aux_rewards = [
            "penalty_torques",
            "penalty_action_rate",
            "limits_dof_pos",
            "limits_torque",
            "penalty_undesired_contact",
            "penalty_feet_ori",
            "penalty_ankle_roll",
            "penalty_slippage",
        ]
        aux_rewards_scaling = {
            "penalty_action_rate": -0.1,
            "penalty_feet_ori": -0.4,
            "penalty_ankle_roll": -4.0,
            "limits_dof_pos": -10.0,
            "penalty_slippage": -2.0,
            "penalty_undesired_contact": -1.0,
            "penalty_torques": 0.0,
            "limits_torque": 0.0,
        }

    if carry_box:
        aux_rewards.extend(
            [
                "carry_approach",
                "carry_approach_progress",
                "carry_pick",
                "carry_transport_progress",
                "carry_success",
                "carry_recovery_progress",
                "carry_drop_penalty",
                "box_overspeed_penalty",
            ]
        )
        aux_rewards_scaling.update(
            {
                "carry_approach": 0.2,
                "carry_approach_progress": 0.5,
                "carry_pick": 1.0,
                "carry_transport_progress": 2.0,
                "carry_success": 5.0,
                "carry_recovery_progress": 1.0,
                "carry_drop_penalty": -1.0,
                "box_overspeed_penalty": -0.05,
            }
        )

    forward_keys = ["state", "privileged_state", "last_action", "history_actor"]
    actor_keys = ["state", "last_action", "history_actor"]
    if carry_box:
        forward_keys.append("object_obs")
        actor_keys.append("object_obs")

    return FBcprAuxAgentConfig(
        name="FBcprAuxAgent",
        model=FBcprAuxModelConfig(
            name="FBcprAuxModel",
            device=device,
            task_latent_key="task_command" if carry_box else None,
            task_latent_dim=TASK_COMMAND_DIM if carry_box else 0,
            # task_command is normalized by the environment; this restores
            # target offsets to metres in the reserved z tail.
            task_latent_scale=5.0 if carry_box else 1.0,
            archi=FBcprAuxModelArchiConfig(
                name="FBcprAuxModelArchiConfig",
                z_dim=256,
                norm_z=True,
                f=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=forward_keys
                    ),
                ),
                b=BackwardArchiConfig(
                    name="BackwardArchi",
                    hidden_dim=256,
                    hidden_layers=1,
                    norm=True,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
                ),
                actor=ActorArchiConfig(
                    name="actor",
                    model="residual",
                    hidden_dim=2048,
                    hidden_layers=6,
                    embedding_layers=2,
                    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=actor_keys),
                ),
                critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=forward_keys
                    ),
                ),
                discriminator=(
                    ConditionalTemporalDiscriminatorArchiConfig(
                        name="ConditionalTemporalDiscriminatorArchi",
                        hidden_dim=1024,
                        hidden_layers=3,
                        history_steps=4,
                        object_history_steps=OBJECT_HISTORY_STEPS,
                        object_frame_dim=OBJECT_FRAME_DIM,
                    )
                    if carry_box
                    else DiscriminatorArchiConfig(
                        name="DiscriminatorArchi",
                        hidden_dim=1024,
                        hidden_layers=3,
                        input_filter=DictInputFilterConfig(
                            name="DictInputFilterConfig", key=["state", "privileged_state"]
                        ),
                    )
                ),
                aux_critic=ForwardArchiConfig(
                    name="ForwardArchi",
                    hidden_dim=2048,
                    model="residual",
                    hidden_layers=6,
                    embedding_layers=2,
                    num_parallel=2,
                    ensemble_mode="batch",
                    input_filter=DictInputFilterConfig(
                        name="DictInputFilterConfig", key=forward_keys
                    ),
                ),
            ),
            obs_normalizer=ObsNormalizerConfig(
                name="ObsNormalizerConfig",
                normalizers={
                    "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
                    **(
                        {"object_obs": IdentityNormalizerConfig(name="IdentityNormalizerConfig")}
                        if carry_box
                        else {}
                    ),
                    **(
                        {"task_command": IdentityNormalizerConfig(name="IdentityNormalizerConfig")}
                        if carry_box
                        else {}
                    ),
                },
                allow_mismatching_keys=True,
            ),
            inference_batch_size=500000,
            seq_length=8,
            actor_std=0.05,
            amp=False,
            norm_aux_reward=RewardNormalizerConfig(name="RewardNormalizer", translate=False, scale=True),
        ),
        train=FBcprAuxAgentTrainConfig(
            name="FBcprAuxAgentTrainConfig",
            lr_f=0.0003 * lr_scale,
            lr_b=1e-05 * lr_scale,
            lr_actor=0.0003 * lr_scale,
            weight_decay=0.0,
            clip_grad_norm=clip_grad_norm,
            fb_target_tau=0.01,
            ortho_coef=100.0,
            train_goal_ratio=0.2,
            fb_pessimism_penalty=0.0,
            actor_pessimism_penalty=0.5,
            stddev_clip=0.3,
            q_loss_coef=0.0,
            batch_size=1024,
            discount=0.98,
            use_mix_rollout=True,
            update_z_every_step=int(update_z_every_step),
            z_buffer_size=8192,
            rollout_expert_trajectories=True,
            rollout_expert_trajectories_length=250,
            rollout_expert_trajectories_percentage=0.5,
            lr_discriminator=1e-05 * lr_scale,
            lr_critic=0.0003 * lr_scale,
            critic_target_tau=0.005,
            critic_pessimism_penalty=0.5,
            reg_coeff=0.05,
            scale_reg=True,
            expert_asm_ratio=0.6,
            relabel_ratio=0.8,
            grad_penalty_discriminator=10.0,
            weight_decay_discriminator=0.0,
            balanced_object_discriminator=carry_box,
            discriminator_mismatch_coef=0.5 if carry_box else 0.0,
            lr_aux_critic=0.0003 * lr_scale,
            reg_coeff_aux=0.02,
            aux_critic_pessimism_penalty=0.5,
        ),
        aux_rewards=aux_rewards,
        aux_rewards_scaling=aux_rewards_scaling,
        cudagraphs=False,
        compile=compile,
    )
