from collections import OrderedDict
import copy
import numpy as np
import torch

from gym.spaces.box import Box
from gym.spaces.dict import Dict
from training.utils.pytorch import to_tensor, center_crop
from training.utils.info_dict import Info
from training.utils.logger import logger
from training.utils.general import join_dicts, join_spaces

from training.agents.base_agent import (
    BaseMultiStageAgent,
    CrossEnvAgentWrapper,
)
from training.agents.gaifo_agent import GAIfOAgent
from training.agents.sac_agent import SACAgent
from training.agents.asymmetric_sac_agent import ASACAgent
from training.agents.got_agent import GOTAgent


class IDAPTAgent(BaseMultiStageAgent):
    def __init__(
        self,
        config,
        source_ob_space,
        source_ac_space,
        source_env_ob_space,
        target_ob_space,
        target_ac_space,
        target_env_ob_space,
    ):
        super().__init__(config, source_ob_space, target_ob_space)

        assert source_ob_space == target_ob_space
        self._ob_space = source_ob_space
        self._state_space = source_env_ob_space
        self._source_ac_space = source_ac_space
        self._target_ac_space = target_ac_space

        ### Initializing RL Agent ###

        if config.encoder_type == "cnn":
            if config.run_rl_from_state:
                sconfig = copy.copy(config)
                sconfig.encoder_type = "mlp"
                self._agent = SACAgent(
                    sconfig,
                    source_env_ob_space,
                    source_ac_space,
                    source_env_ob_space,
                )
            else:
                self._agent = ASACAgent(
                    config, source_ob_space, source_ac_space, source_env_ob_space
                )
        else:
            self._agent = SACAgent(
                config,
                source_ob_space,
                source_ac_space,
                source_env_ob_space,
            )

        self._encode_fn = lambda o, e: self._agent._actor.encoder.forward(o)
        encoded_ob_space = Dict(
            {
                "ob": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._agent._actor.encoder.output_dim,),
                )
            }
        )

        ### Initializing Observaation Transformation Model ###
        if "visual" in config.mode:
            self._encoder = GOTAgent(
                config,
                source_ob_space,
                source_ac_space,
                target_ob_space,
                source_env_ob_space,
            )
            self.translate = self._encoder._translate
            self.record_video = self._encoder._record_video
            self.record_image = self._encoder._record_image

        ### Initializing Action Transformation Model ###
        at_ob_space = join_spaces(encoded_ob_space, target_ac_space)
        at_env_ob_space = encoded_ob_space
        at_ac_space = target_ac_space
        self._AT = GAIfOAgent(
            config,
            at_ob_space,
            at_ac_space,
            at_env_ob_space,
        )

        if config.mode != "cross_visual":
            self.evaluate_AT = self._evaluate_AT

        ### Initialize variables for tracking stage ###
        self._grounded = False
        if "supervised_pretraining" in config.max_stage_steps.keys():
            if config.grounding_steps > 0:
                self.stages = [
                    "supervised_pretraining",
                    "policy_init",
                ] + config.grounding_steps * [
                    "supervised_training",
                    "grounding",
                    "policy_training",
                ]
            else:
                self.stages = ["supervised_pretraining", "policy_init"]
            self._supervised_training_step = 0
        else:
            self.stages = ["policy_init"] + config.grounding_steps * [
                "grounding",
                "policy_training",
            ]

        self._grounding_step = 0
        self._AT_initialized = False

        self._log_creation()

    def store_episode(self, rollouts, env_type, stage):
        if stage == "grounding":
            rollouts["ob"] = rollouts["encoded_ob"]
            rollouts["ob_next"] = rollouts["encoded_ob_next"]
            self._AT.store_episode(rollouts, env_type, stage)

        elif (
            stage == "policy_training" or stage == "policy_init"
        ) and env_type == "source":
            if stage == "policy_training":
                rollouts["ac"] = rollouts["target_ac"]
                rollouts["ac_before_activation"] = rollouts[
                    "target_ac_before_activation"
                ]
            if "visual" in self._config.mode:
                rollouts["ob"] = rollouts["got_ob"]
                rollouts["ob_next"] = rollouts["got_ob_next"]
            if "state" in rollouts.keys():
                rollouts["state"] = [state["state"] for state in rollouts["state"]]
                rollouts["state_next"] = [
                    state["state"] for state in rollouts["state_next"]
                ]
            self._agent.store_episode(rollouts)

        elif "supervised" in stage:
            self._encoder.store_episode(rollouts, env_type)
            if (
                self._config.mode == "cross_visual_physics"
                and env_type == "target"
                and stage == "supervised_training"
            ):
                ### Store target trajectories for AT training later
                rollouts["ob"] = rollouts["encoded_ob"]
                rollouts["ob_next"] = rollouts["encoded_ob_next"]
                self._AT.store_episode(rollouts, env_type, stage)

    def encode(self, ob, domain=None):
        """
        Encodes observation with self._encode_fn.
        """
        ob = ob.copy()
        for k, v in ob.items():
            if self._config.encoder_type == "cnn" and len(v.shape) == 3:
                ob[k] = center_crop(v, self._config.encoder_image_size)
            else:
                ob[k] = np.expand_dims(ob[k], axis=0)

        with torch.no_grad():
            ob = to_tensor(ob, self._config.device)
            encoded_ob = self._encode_fn(ob, domain).detach().cpu().numpy().reshape(-1)
        return {"ob": encoded_ob}

    def act(self, ob, env_type, stage, is_train=True, **kwargs):

        is_train = env_type != "grounding" and is_train

        if stage == "supervised_pretraining" and is_train:
            if self._config.gather_random_rollouts:
                if env_type == "source":
                    action = self._source_ac_space.sample()
                elif env_type == "target":
                    action = self._target_ac_space.sample()
            else:
                with torch.no_grad():
                    action, _ = self._agent.act(ob, is_train=is_train)
            return (
                action,
                action,
                {"target_ac": action, "target_ac_before_activation": action},
            )

        with torch.no_grad():
            if "visual" in self._config.mode:
                ob_in = self.translate(ob, domain=env_type)
            else:
                ob_in = ob
            target_ac, target_activation = self._agent.act(ob_in, is_train=is_train)

            if stage != "grounding" and not self._AT_initialized:
                return (
                    target_ac,
                    target_activation,
                    {
                        "target_ac": target_ac,
                        "target_ac_before_activation": target_activation,
                    },
                )
            else:
                encoded_ob = self.encode(ob_in, env_type)
                AT_in = self._generate_AT_input(encoded_ob, target_ac)

                AT_ac, activation = self._AT.act(AT_in, is_train=is_train)
                if self._config.AT_acdiff:
                    ac = OrderedDict()
                    for k, _ in self._source_ac_space.spaces.items():
                        ac[k] = target_ac[k] + AT_ac[k]
                    AT_ac = ac

                ac = self._smooth_action(AT_ac, target_ac)

                if env_type == "source":
                    return (
                        ac,
                        activation,
                        {
                            "AT_ac": AT_ac,
                            "target_ac": target_ac,
                            "target_ac_before_activation": target_activation,
                            "encoded_ob": encoded_ob,
                        },
                    )
                elif env_type == "target":
                    return (
                        target_ac,
                        target_activation,
                        {
                            "AT_ac": AT_ac,
                            "source_ac": ac,
                            "source_ac_before_activation": activation,
                            "encoded_ob": encoded_ob,
                        },
                    )

    def train(self, env_types, curr_info):
        curr_stage = curr_info["stage"]
        train_info = Info()

        ### Train Observation Model during Pretraining or Grounding Step ###
        if "supervised" in curr_stage:
            self._supervised_training_step += 1
            encoder_train_info = self._encoder.train(
                self._supervised_training_step, self._grounding_step
            )
            train_info.add(encoder_train_info)

            if (
                self._supervised_training_step
                == self._config.max_stage_steps[curr_stage]
            ):
                if not self._config.accumulate_data:
                    self._encoder.clear_dataset()
                self._supervised_training_step = 0
                if "visual" in self._config.mode:
                    ### reset parameters for next step
                    self._config.gather_random_rollouts = False
                    self._config.source_encoder_train_epoch = (
                        self._config.source_encoder_finetune_epoch
                    )
                if "pretraining" not in curr_stage:
                    self._grounding_step += 1

        ### Regular RL for Agent ###
        if curr_stage == "policy_training" or curr_stage == "policy_init":
            agent_train_info = self._agent.train()
            train_info.add(agent_train_info)
            self._grounded = False

        ### Train AT during Grounding Step ###
        if curr_stage == "grounding":
            if self._grounded == False:
                ### First iteration in new grounding step
                if (
                    self._config.mode != "cross_visual_physics"
                    or self._grounding_step == 0
                ):
                    self._grounding_step += 1
                self._grounded = True

                if self._config.encoder_type == "cnn" and "visual" in self._config.mode:
                    self._agent.set_encoder_requires_grad(False)
                    self._agent.clear_buffer()
                    self._encoder.requires_grad = False

            train_info["grounding_step"] = self._grounding_step
            AT_train_info = self._AT.train()
            self._AT_initialized = True
            train_info.add(AT_train_info)

        return train_info.get_dict(only_scalar=True)

    ### GARAT Helper Functions

    def _smooth_action(self, ac_pred, ac):
        """
        smoothed action = ac_pred * self._config.smoothing_factor + ac * (1 - self._config.smoothing_factor)
        """
        out = OrderedDict()
        for k, _ in self._source_ac_space.spaces.items():
            out[k] = ac_pred[k] * self._config.smoothing_factor
            out[k] += ac[k] * (1 - self._config.smoothing_factor)
        return out

    def _generate_AT_input(self, ob, ac):
        """
        Generates input to AT by concatenating observation and action
        """

        return join_dicts(ob, ac)

    def _evaluate_AT(self, transitions, env_type, other_env, stage):
        """
        Calculates AT error.  This is only called in cross_physics mode.
        """

        qpos = transitions["qpos"]
        qvel = transitions["qvel"]
        qpos_next = transitions["qpos"]
        qvel_next = transitions["qvel"]
        acs = transitions["ac"]
        if stage == "policy_init":
            other_acs = transitions["ac"]
        else:
            if env_type == "source":
                other_acs = transitions["target_ac"]
            elif env_type == "target":
                other_acs = transitions["source_ac"]

        errors = []
        ac_diff = []
        ac_ratio = []
        for qpos, qvel, qpos_next, qvel_next, ac, other_ac in zip(
            qpos, qvel, qpos_next, qvel_next, acs, other_acs
        ):
            if env_type == "source":
                ac_diff.append(ac["ac"] - other_ac["ac"])
                ac_ratio.append(ac["ac"] / other_ac["ac"])
            elif env_type == "target":
                ac_diff.append(other_ac["ac"] - ac["ac"])
                ac_ratio.append(other_ac["ac"] / ac["ac"])
            if self._config.tanh_policy:
                rescaled_ac = OrderedDict()
                rescaled_ac["ac"] = other_env.action_space["ac"].low + 0.5 * (
                    other_ac["ac"] + 1
                ) * (
                    other_env.action_space["ac"].high - other_env.action_space["ac"].low
                )
            other_qpos_next, other_qvel_next = other_env.step_from_state(
                qpos, qvel, rescaled_ac
            )

            error = np.linalg.norm(
                (qpos_next - other_qpos_next), ord=1
            ) + np.linalg.norm((qvel_next - other_qvel_next), ord=1)
            errors.append(error)
        return np.mean(errors), np.mean(ac_diff), np.mean(ac_ratio)

    ### Misc. ###

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a GAT Agent")

    def decide_runner_type(self, curr_info):
        if "supervised" in curr_info["stage"]:
            return "supervised"

        if curr_info["stage"] == "grounding":
            return "on_policy"
        else:
            return "off_policy"

    def decide_env_type(self, curr_info):
        if (
            curr_info["stage"] == "policy_init"
            or curr_info["stage"] == "policy_training"
        ):
            return ["source"]
        elif curr_info["stage"] == "grounding":
            if self._config.mode == "cross_physics" and not self._grounded:
                return ["target", "source"]
            else:
                return ["source"]
        elif "supervised" in curr_info["stage"]:
            if self._supervised_training_step == 0:
                return ["source", "target", "source_eval", "target_eval"]
            else:
                return []

    def sync_networks(self):
        self._agent.sync_networks()
        self._AT.sync_networks()
        if "visual" in self._config.mode:
            self._encoder.sync_networks()

    def state_dict(self, stage=None):
        if "supervised" in stage:
            return {"encoder": self._encoder.state_dict()}
        elif "training" in stage or "grounding" in stage:
            return {
                "agent_state_dict": self._agent.state_dict(),
                "AT_state_dict": self._AT.state_dict(),
            }
        else:
            return {
                "encoder": self._encoder.state_dict(),
                "agent_state_dict": self._agent.state_dict(),
                "AT_state_dict": self._AT.state_dict(),
            }

    def load_state_dict(self, ckpt):
        if "agent_state_dict" in ckpt.keys():
            try:
                if self._config.run_rl_from_state and self._config.encoder_type == "cnn":
                    ckpt["agent_state_dict"]["ob_norm_state_dict"] = {"state": ckpt["agent_state_dict"]["ob_norm_state_dict"]['ob']}
                self._agent.load_state_dict(ckpt["agent_state_dict"])
                print("Agent loaded from ckpt.")
            except:
                print("Agent not loaded.")

        if self._config.load_AT:
            try:
                self._AT.load_state_dict(ckpt["AT_state_dict"])
                print("AT loaded from ckpt.")
            except:
                print("AT not loaded.")

        if "encoder" in ckpt.keys():
            try:
                self._encoder.load_state_dict(ckpt["encoder"])
                print("Encoder loaded from ckpt.")
            except:
                print("Encoder not loaded.")

    def update_normalizer(self, obs, env_type, stage):
        if self._config.ob_norm:
            if stage == "grounding" and env_type == "source":
                self._AT.update_normalizer(obs)
            elif stage == "policy_init" or stage == "policy_training":
                self._agent.update_normalizer(obs)

    def run_in(self, env_type, **kwargs):
        if kwargs["stage"] == "grounding" and env_type == "source":
            return IDAPTAgentILWrapper(self, env_type, **kwargs)
        else:
            return CrossEnvAgentWrapper(self, env_type, **kwargs)


class IDAPTAgentILWrapper:
    """hack : wrapper so GARAT can run with Imitation Loss reward"""

    def __init__(self, agent, env_type, **kwargs):
        self._agent = agent
        self._config = agent._config
        self._env_type = env_type
        self._action_args = kwargs

    def act(self, ob, is_train=True):
        return self._agent.act(
            ob, self._env_type, is_train=is_train, **self._action_args
        )

    def predict_reward(self, ob, ac, next_ob):
        il_ob = join_dicts(ob, ac)
        return self._agent._AT.predict_reward(il_ob, ac, next_ob)
