# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import mtrl
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import os
from mtrl.agent import utils as agent_utils
from mtrl.agent.abstract import Agent as AbstractAgent
from mtrl.agent.ds.mt_obs import MTObs
from mtrl.agent.ds.task_info import TaskInfo
from mtrl.env.types import ObsType
from mtrl.logger import Logger
from mtrl.replay_buffer import ReplayBuffer, ReplayBufferSample
from mtrl.utils.types import ConfigType, ModelType, ParameterType, TensorType
import pudb
from mtrl.ppo_org.gail_airl_ppo.network.disc import AIRLDiscrim
from torch.optim import Adam

class Agent(AbstractAgent):
    """SAC algorithm."""

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        actor_cfg: ConfigType,
        critic_cfg: ConfigType,
        alpha_optimizer_cfg: ConfigType,
        actor_optimizer_cfg: ConfigType,
        critic_optimizer_cfg: ConfigType,
        multitask_cfg: ConfigType,
        global_config: ConfigType,
        discount: float,
        init_temperature: float,
        actor_update_freq: int,
        critic_tau: float,
        critic_target_update_freq: int,
        encoder_tau: float,
        loss_reduction: str = "mean",
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
        expert_file = None,
        demo_actor_pth = None
    ):
        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            device=device,
        )
        self.should_use_task_encoder = self.multitask_cfg.should_use_task_encoder
        self.global_config = global_config
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.actor = hydra.utils.instantiate(
            actor_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)
        self.demo_actor_pth = demo_actor_pth

        self.critic = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)

        self.critic_target = hydra.utils.instantiate(
            critic_cfg, env_obs_shape=env_obs_shape, action_shape=action_shape
        ).to(self.device)
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(
                [
                    np.log(init_temperature, dtype=np.float32)
                    for _ in range(self.num_envs)
                ]
            ).to(self.device)
        )
        # self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self._components = {
            "actor": self.actor,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "log_alpha": self.log_alpha,  # type: ignore[dict-item]
        }
        # optimizers
        self.actor_optimizer = hydra.utils.instantiate(
            actor_optimizer_cfg, params=self.get_parameters(name="actor")
        )
        self.critic_optimizer = hydra.utils.instantiate(
            critic_optimizer_cfg, params=self.get_parameters(name="critic")
        )
        self.log_alpha_optimizer = hydra.utils.instantiate(
            alpha_optimizer_cfg, params=self.get_parameters(name="log_alpha")
        )
        if loss_reduction not in ["mean", "none"]:
            raise ValueError(
                f"{loss_reduction} is not a supported value for `loss_reduction`."
            )

        self.loss_reduction = loss_reduction
        self.expert_file = expert_file

        self._optimizers = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
            "log_alpha": self.log_alpha_optimizer,
        }

        if self.should_use_task_encoder:
            try:
                self.task_encoder = hydra.utils.instantiate(
                    self.multitask_cfg.task_encoder_cfg.model_cfg,
                ).to(self.device)
            except:
                self.task_encoder = mtrl.agent.components.task_encoder.TaskEncoder(
                    pretrained_embedding_cfg= {'should_use': True, 'path_to_load_from': '/home/ssinha97/mtrl/metadata/task_embedding/roberta_small/'+self.global_config.env.name+'.json', 'ordered_task_list': self.global_config.env.ordered_task_list},
                    num_embeddings= self.global_config.env.num_envs,
                    embedding_dim= 50,
                    hidden_dim= 50,
                    num_layers= 2,
                    output_dim= 50,
                )
            name = "task_encoder"
            self._components[name] = self.task_encoder
            self.task_encoder_optimizer = hydra.utils.instantiate(
                self.multitask_cfg.task_encoder_cfg.optimizer_cfg,
                params=self.get_parameters(name=name),
            )
            self._optimizers[name] = self.task_encoder_optimizer

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)
        if self.demo_actor_pth is not None:
            self.actor.load_state_dict(torch.load("../../../"+self.demo_actor_pth))
        
        if self.expert_file != "None":
            # Discriminator.
            self.disc = AIRLDiscrim(
                state_shape=env_obs_shape,
                gamma=0.995
                # hidden_units_r=units_disc_r,
                # hidden_units_v=units_disc_v,
                # hidden_activation_r=nn.ReLU(inplace=True),
                # hidden_activation_v=nn.ReLU(inplace=True)
            ).to(device)

            self.learning_steps_disc = 0
            self.optim_disc = Adam(self.disc.parameters(), lr=3e-4)
            self.epoch_disc = 10
            self.expert_buffer_dict = torch.load(self.expert_file)
            self.expert_buffer = hydra.utils.instantiate(
                self.global_config.replay_buffer,
                device=self.device,
                env_obs_shape=env_obs_shape,
                task_obs_shape=(1,),
                action_shape=action_shape,
            )
            for bidx in range(len(self.expert_buffer_dict["done"])):
                for env_id in range(self.expert_buffer_dict["state"][bidx].shape[0]):
                    state = self.expert_buffer_dict["state"][bidx][env_id,:]
                    action = self.expert_buffer_dict["action"][bidx][env_id,:]
                    reward = self.expert_buffer_dict["reward"][bidx][env_id]
                    done = self.expert_buffer_dict["done"][bidx][env_id]
                    next_state = self.expert_buffer_dict["next_state"][bidx][env_id,:]
                    try:
                        self.expert_buffer.add(state, action, reward, next_state, done > 0, [env_id])
                    except: pu.db

                # state = self.expert_buffer_dict["state"][bidx]
                # action = self.expert_buffer_dict["action"][bidx]
                # reward = self.expert_buffer_dict["reward"][bidx]
                # done = self.expert_buffer_dict["done"][bidx]
                # next_state = self.expert_buffer_dict["next_state"][bidx]
                # try:
                #     self.expert_buffer.add(state, action, reward, next_state, done > 0, [0])
                # except: pu.db



    def complete_init(self, cfg_to_load_model: Optional[ConfigType]):
        if cfg_to_load_model:
            self.load(**cfg_to_load_model)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
        self.train()

    def train(self, training: bool = True) -> None:
        self.training = training
        for name, component in self._components.items():
            if name != "log_alpha":
                component.train(training)

    def get_alpha(self, env_index: TensorType) -> TensorType:
        """Get the alpha value for the given environments.

        Args:
            env_index (TensorType): environment index.

        Returns:
            TensorType: alpha values.
        """
        if self.multitask_cfg.should_use_disentangled_alpha:
            return self.log_alpha[env_index].exp()
        else:
            return self.log_alpha[0].exp()

    def get_task_encoding(
        self, env_index: TensorType, modes: List[str], disable_grad: bool
    ) -> TensorType:
        """Get the task encoding for the different environments.

        Args:
            env_index (TensorType): environment index.
            modes (List[str]):
            disable_grad (bool): should disable tracking gradient.

        Returns:
            TensorType: task encodings.
        """
        if disable_grad:
            with torch.no_grad():
                return self.task_encoder(env_index.to(self.device))
        return self.task_encoder(env_index.to(self.device))

    def act(
        self,
        multitask_obs: ObsType,
        # obs, env_index: TensorType,
        modes: List[str],
        sample: bool,
    ) -> np.ndarray:
        """Select/sample the action to perform.

        Args:
            multitask_obs (ObsType): Observation from the multitask environment.
            mode (List[str]): mode in which to select the action.
            sample (bool): sample (if `True`) or select (if `False`) an action.

        Returns:
            np.ndarray: selected/sample action.

        """
        env_obs = multitask_obs["env_obs"]
        env_index = multitask_obs["task_obs"]
        env_index = env_index.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.should_use_task_encoder:
                task_encoding = self.get_task_encoding(
                    env_index=env_index, modes=modes, disable_grad=True
                )
            else:
                task_encoding = None  # type: ignore[assignment]
            task_info = self.get_task_info(
                task_encoding=task_encoding, component_name="", env_index=env_index
            )
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Make a batch
            mtobs = MTObs(env_obs=obs, task_obs=env_index, task_info=task_info)
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            if sample:
                action = pi
            else:
                action = mu
            action = action.clamp(*self.action_range)
            # assert action.ndim == 2 and action.shape[0] == 1
            return action.detach().cpu().numpy()

    def select_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=False)

    def sample_action(self, multitask_obs: ObsType, modes: List[str]) -> np.ndarray:
        return self.act(multitask_obs=multitask_obs, modes=modes, sample=True)

    def get_last_shared_layers(self, component_name: str) -> Optional[List[ModelType]]:  # type: ignore[return]

        if component_name in [
            "actor",
            "critic",
            "transition_model",
            "reward_decoder",
            "decoder",
        ]:
            return self._components[component_name].get_last_shared_layers()  # type: ignore[operator]
            # The mypy error is because self._components can contain a tensor as well.
        if component_name in ["log_alpha", "encoder", "task_encoder"]:
            return None
        if component_name not in self._components:
            raise ValueError(f"""Component named {component_name} does not exist""")

    def _compute_gradient(
            self,
            loss: TensorType,
            parameters: List[ParameterType],
            step: int,
            component_names: List[str],
            type = None,
            retain_graph: bool = False,
        ):
            """Method to override the gradient computation.

                Useful for algorithms like PCGrad and GradNorm.

            Args:
                loss (TensorType):
                parameters (List[ParameterType]):
                step (int): step for tracking the training of the agent.
                component_names (List[str]):
                retain_graph (bool, optional): if it should retain graph. Defaults to False.
            """
            if self.demo_actor_pth is None or type is not None:
                loss.backward(retain_graph=retain_graph)

    # def _compute_gradient(
    #     self,
    #     loss: TensorType,
    #     parameters: List[ParameterType],
    #     step: int,
    #     component_names: List[str],
    #     retain_graph: bool = False,
    # ):
    #     """Method to override the gradient computation.

    #         Useful for algorithms like PCGrad and GradNorm.

    #     Args:
    #         loss (TensorType):
    #         parameters (List[ParameterType]):
    #         step (int): step for tracking the training of the agent.
    #         component_names (List[str]):
    #         retain_graph (bool, optional): if it should retain graph. Defaults to False.
    #     """
    #     loss.backward(retain_graph=retain_graph)

    def _get_target_V(
        self, batch: ReplayBufferSample, task_info: TaskInfo
    ) -> TensorType:
        """Compute the target values.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.

        Returns:
            TensorType: target values.
        """
        mtobs = MTObs(env_obs=batch.next_env_obs, task_obs=None, task_info=task_info)
        _, policy_action, log_pi, _ = self.actor(mtobs=mtobs)
        target_Q1, target_Q2 = self.critic_target(mtobs=mtobs, action=policy_action)
        return (
            torch.min(target_Q1, target_Q2)
            - self.get_alpha(env_index=batch.task_obs).detach() * log_pi
        )

    def update_critic(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the critic component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        # pu.db
        # print("")
        if self.expert_file is not "None":
            # print("")
            with torch.no_grad():
                target_V = self._get_target_V(batch=self.expert_batch, task_info=task_info)
                target_Q = self.expert_batch.reward + (self.expert_batch.not_done * self.discount * target_V)
        else:
            with torch.no_grad():
                target_V = self._get_target_V(batch=batch, task_info=task_info)
                target_Q = batch.reward + (batch.not_done * self.discount * target_V)

        # get current Q estimates
        mtobs = MTObs(env_obs=batch.env_obs, task_obs=None, task_info=task_info)
        current_Q1, current_Q2 = self.critic(
            mtobs=mtobs,
            action=batch.action,
            detach_encoder=False,
        )
        critic_loss = F.mse_loss(
            current_Q1, target_Q, reduction=self.loss_reduction
        ) + F.mse_loss(current_Q2, target_Q, reduction=self.loss_reduction)

        loss_to_log = critic_loss
        if self.loss_reduction == "none":
            loss_to_log = loss_to_log.mean()
        logger.log("train/critic_loss", loss_to_log, step)

        if loss_to_log > 1e8:
            raise RuntimeError(
                f"critic_loss = {loss_to_log} is too high. Stopping training."
            )

        component_names = ["critic"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=critic_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            **kwargs_to_compute_gradient,
        )

        # Optimize the critic
        self.critic_optimizer.step()

    def update_actor_and_alpha(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the actor and alpha component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """

        # detach encoder, so we don't update it with the actor loss
        mtobs = MTObs(
            env_obs=batch.env_obs,
            task_obs=None,
            task_info=task_info,
        )
        _, pi, log_pi, log_std = self.actor(mtobs=mtobs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(mtobs=mtobs, action=pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        if self.loss_reduction == "mean":
            actor_loss = (
                self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            ).mean()
            logger.log("train/actor_loss", actor_loss, step)

        elif self.loss_reduction == "none":
            actor_loss = self.get_alpha(batch.task_obs).detach() * log_pi - actor_Q
            logger.log("train/actor_loss", actor_loss.mean(), step)

        logger.log("train/actor_target_entropy", self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )

        logger.log("train/actor_entropy", entropy.mean(), step)

        # optimize the actor
        component_names = ["actor"]
        parameters: List[ParameterType] = []
        for name in component_names:
            self._optimizers[name].zero_grad()
            parameters += self.get_parameters(name)
        if task_info.compute_grad:
            component_names.append("task_encoder")
            kwargs_to_compute_gradient["retain_graph"] = True
            parameters += self.get_parameters("task_encoder")

        self._compute_gradient(
            loss=actor_loss,
            parameters=parameters,
            step=step,
            component_names=component_names,
            type="actor",
            **kwargs_to_compute_gradient,
        )
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        if self.loss_reduction == "mean":
            alpha_loss = (
                self.get_alpha(batch.task_obs)
                * (-log_pi - self.target_entropy).detach()
            ).mean()
            logger.log("train/alpha_loss", alpha_loss, step)
        elif self.loss_reduction == "none":
            alpha_loss = (
                self.get_alpha(batch.task_obs)
                * (-log_pi - self.target_entropy).detach()
            )
            logger.log("train/alpha_loss", alpha_loss.mean(), step)
        # breakpoint()
        # logger.log("train/alpha_value", self.get_alpha(batch.task_obs), step)
        self._compute_gradient(
            loss=alpha_loss,
            parameters=self.get_parameters(name="log_alpha"),
            step=step,
            component_names=["log_alpha"],
            **kwargs_to_compute_gradient,
        )
        if self.demo_actor_pth is None:
            torch.save(self.actor.state_dict(), "actor.pth")
        self.log_alpha_optimizer.step()

    def get_task_info(
        self, task_encoding: TensorType, component_name: str, env_index: TensorType
    ) -> TaskInfo:
        """Encode task encoding into task info.

        Args:
            task_encoding (TensorType): encoding of the task.
            component_name (str): name of the component.
            env_index (TensorType): index of the environment.

        Returns:
            TaskInfo: TaskInfo object.
        """
        if self.should_use_task_encoder:
            if component_name in self.multitask_cfg.task_encoder_cfg.losses_to_train:
                task_info = TaskInfo(
                    encoding=task_encoding, compute_grad=True, env_index=env_index
                )
            else:
                task_info = TaskInfo(
                    encoding=task_encoding.detach(),
                    compute_grad=False,
                    env_index=env_index,
                )
        else:
            task_info = TaskInfo(
                encoding=task_encoding, compute_grad=False, env_index=env_index
            )
        return task_info

    def update_transition_reward_model(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the transition model and reward decoder.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update_task_encoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the task encoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        self.task_encoder_optimizer.step()

    def update_decoder(
        self,
        batch: ReplayBufferSample,
        task_info: TaskInfo,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Dict[str, Any],
    ) -> None:
        """Update the decoder component.

        Args:
            batch (ReplayBufferSample): batch from the replay buffer.
            task_info (TaskInfo): task_info object.
            logger ([Logger]): logger object.
            step (int): step for tracking the training of the agent.
            kwargs_to_compute_gradient (Dict[str, Any]):

        """
        raise NotImplementedError("This method is not implemented for SAC agent.")

    def update(
        self,
        replay_buffer: ReplayBuffer,
        logger: Logger,
        step: int,
        kwargs_to_compute_gradient: Optional[Dict[str, Any]] = None,
        buffer_index_to_sample: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update the agent.

        Args:
            replay_buffer (ReplayBuffer): replay buffer to sample the data.
            logger (Logger): logger for logging.
            step (int): step for tracking the training progress.
            kwargs_to_compute_gradient (Optional[Dict[str, Any]], optional): Defaults
                to None.
            buffer_index_to_sample (Optional[np.ndarray], optional): if this parameter
                is specified, use these indices instead of sampling from the replay
                buffer. If this is set to `None`, sample from the replay buffer.
                buffer_index_to_sample Defaults to None.

        Returns:
            np.ndarray: index sampled (from the replay buffer) to train the model. If
                buffer_index_to_sample is not set to None, return buffer_index_to_sample.

        """

        if kwargs_to_compute_gradient is None:
            kwargs_to_compute_gradient = {}

        if buffer_index_to_sample is None:
            batch = replay_buffer.sample()
        else:
            batch = replay_buffer.sample(buffer_index_to_sample)
        # pu.db
        # print("")
        self.expert_batch = self.expert_buffer.sample()
        # pass
        logger.log("train/batch_reward", batch.reward.mean(), step)
        if self.should_use_task_encoder:
            self.task_encoder_optimizer.zero_grad()
            task_encoding = self.get_task_encoding(
                env_index=batch.task_obs.squeeze(1),
                disable_grad=False,
                modes=["train"],
            )
        else:
            task_encoding = None  # type: ignore[assignment]

        task_info = self.get_task_info(
            task_encoding=task_encoding,
            component_name="critic",
            env_index=batch.task_obs,
        )
        self.update_critic(
            batch=batch,
            task_info=task_info,
            logger=logger,
            step=step,
            kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
        )
        if step % self.actor_update_freq == 0:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="actor",
                env_index=batch.task_obs,
            )
            self.update_actor_and_alpha(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        if (
            "transition_model" in self._components
            and "reward_decoder" in self._components
        ):
            # some of the logic is a bit sketchy here. We will get to it soon.
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="transition_reward",
                env_index=batch.task_obs,
            )
            self.update_transition_reward_model(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )
        if (
            "decoder" in self._components  # should_update_decoder
            and self.decoder is not None  # type: ignore[attr-defined]
            and step % self.decoder_update_freq == 0  # type: ignore[attr-defined]
        ):
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="decoder",
                env_index=batch.task_obs,
            )
            self.update_decoder(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        if self.should_use_task_encoder:
            task_info = self.get_task_info(
                task_encoding=task_encoding,
                component_name="task_encoder",
                env_index=batch.task_obs,
            )
            self.update_task_encoder(
                batch=batch,
                task_info=task_info,
                logger=logger,
                step=step,
                kwargs_to_compute_gradient=deepcopy(kwargs_to_compute_gradient),
            )

        return batch.buffer_index

    def get_parameters(self, name: str) -> List[torch.nn.parameter.Parameter]:
        """Get parameters corresponding to a given component.

        Args:
            name (str): name of the component.

        Returns:
            List[torch.nn.parameter.Parameter]: list of parameters.
        """
        if name == "actor":
            return list(self.actor.model.parameters())
        elif name in ["log_alpha", "alpha"]:
            return [self.log_alpha]
        elif name == "encoder":
            return list(self.critic.encoder.parameters())
        else:
            return list(self._components[name].parameters())

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
