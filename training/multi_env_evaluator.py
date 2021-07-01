"""
Evaluates Policy in multiple environments.
"""
import copy
import gym
import gzip
import h5py
import numpy as np
import os
import pickle
import wandb
from collections import defaultdict, OrderedDict
from time import time

# import environments
from training.agents import get_multi_stage_agent_by_name
from training.rollouts import RolloutRunner
from training.utils.env import make_env
from training.utils.general import dict_add_prefix
from training.utils.info_dict import Info
from training.utils.logger import logger
from training.multi_stage_trainer import MultiStageTrainer


class MultiEnvEvaluator(MultiStageTrainer):
    def __init__(self, config):
        """
        Initializes class with the configuration.
        """

        self._config = config
        self._is_chef = config.is_chef

        # create source and target environment
        self._envs = {}
        for env_name in config.envs:
            env = make_env(env_name, copy.copy(config), "target")
            self._envs[env_name] = env
            ob_space = env.observation_space
            env_ob_space = env.env_observation_space
            ac_space = env.action_space
        logger.info("Observation space: " + str(ob_space))
        logger.info("Action space: " + str(ac_space))

        # create a new observation space after data augmentation (random crop)
        if config.encoder_type == "cnn":
            assert (
                not config.ob_norm
            ), "Turn off the observation norm (--ob_norm False) for pixel inputs"
            # env
            ob_space = gym.spaces.Dict(spaces=dict(ob_space.spaces))
            for k in ob_space.spaces.keys():
                if len(ob_space.spaces[k].shape) == 3:
                    shape = [
                        ob_space.spaces[k].shape[0],
                        config.encoder_image_size,
                        config.encoder_image_size,
                    ]
                    ob_space.spaces[k] = gym.spaces.Box(
                        low=0, high=255, shape=shape, dtype=np.uint8
                    )

        # build agent and networks for algorithm
        self._agent = get_multi_stage_agent_by_name(config.algo)(
            config, ob_space, ac_space, env_ob_space, ob_space, ac_space, env_ob_space,
        )

        # build rollout runner
        self._runners = {}
        for k, env in self._envs.items():
            self._runners[k] = RolloutRunner(
                config,
                env,
                env,
                self._agent.run_in("target", stage="policy_init"),
                "target",
            )

    def _evaluate(self, record_video=False):
        """
        Runs one rollout if in eval mode (@idx is not None).
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            step: the number of environment steps.
            record_video: whether to record video or not.
        """
        assert record_video == False
        num_eval = self._config.num_eval
        rollouts = {}
        info_history = Info()

        for k, runner in self._runners.items():
            logger.warn("Run %d evaluations for %s", num_eval, k)
            for i in range(num_eval):
                logger.warn("Evaluate run %d", i + 1)
                rollout, info, frames = runner.run_episode(is_train=False)
                info = dict_add_prefix(info, k)

                rollouts[k] = rollout
                logger.info(
                    "rollout: %s", {k: v for k, v in info.items() if not "qpos" in k},
                )

                info_history.add(info)

        avg_stats = {}
        for k, v in info_history.items():
            if isinstance(v, list) and not isinstance(v[0], wandb.Video):
                avg_stats[k] = np.mean(v)

        std_stats = {}
        for k, v in info_history.items():
            if isinstance(v, list) and not isinstance(v[0], wandb.Video):
                std_stats[k] = np.std(v)

        logger.info(
            "Evaluation Average stats: %s", {k: v for k, v in avg_stats.items()},
        )
        logger.info(
            "Evaluation Standard Deviation stats: %s",
            {k: v for k, v in std_stats.items()},
        )

        return rollouts, info_history

    def evaluate(self):
        """ Evaluates an agent in multiple environments """
        self._load_ckpt(self._config.init_ckpt_path, self._config.ckpt_num)

        logger.info(
            "Run %d evaluations for %d environments",
            self._config.num_eval,
            len(self._envs),
        )
        rollouts, info = self._evaluate(record_video=self._config.record_video)

        info_stat = info.get_stat()
        os.makedirs("result", exist_ok=True)
        with h5py.File("result/{}.hdf5".format(self._config.run_name), "w") as hf:
            for k, v in info.items():
                hf.create_dataset(k, data=info[k])
        with open("result/{}.txt".format(self._config.run_name), "w") as f:
            for k, v in info_stat.items():
                f.write("{}\t{:.03f} $\\pm$ {:.03f}\n".format(k, v[0], v[1]))
