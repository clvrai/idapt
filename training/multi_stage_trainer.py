"""
Base code for Multi stage policy transfer training
Collect rollouts from two environments and update policy networks
"""
import copy
import gym
import gzip
import h5py
import json
import moviepy.editor as mpy
import numpy as np
import os
import pickle
import psutil
import torch
import wandb
from collections import defaultdict, OrderedDict
from time import time, sleep
from tqdm import tqdm, trange

# import environments
from training.agents import get_multi_stage_agent_by_name, run_in
from training.rollouts import RolloutRunner
from training.utils.env import make_basic_env
from training.utils.env import make_env
from training.utils.general import dict_add_prefix, get_gpu_memory
from training.utils.info_dict import Info
from training.utils.logger import logger
from training.utils.mpi import mpi_sum, mpi_gather_average
from training.utils.pytorch import get_ckpt_path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # for running on macOS


def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_percent()
    return mem


def check_memory_kill_switch(avail_thresh=1.0):
    """Kills program if available memory is below threshold to avoid memory overflows."""
    try:
        if (
            psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
            < avail_thresh
        ):
            print(
                "Current memory usage of {}% surpasses threshold, killing program...".format(
                    psutil.virtual_memory().percent
                )
            )
            sleep(
                10 * np.random.rand()
            )  # avoid that all processes get killed at once
            if (
                psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
                < avail_thresh
            ):
                exit(0)
    except FileNotFoundError:  # seems to happen infrequently
        pass


class MultiStageTrainer(object):
    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config
        self._is_chef = config.is_chef

        if hasattr(self._config, "buffer_size"):
            self._config.buffer_size = (
                self._config.buffer_size // self._config.num_workers
            )
        if hasattr(self._config, "rollout_length"):
            self._config.rollout_length = (
                self._config.rollout_length // self._config.num_workers
            )

        # create source and target environment
        self._source_env = make_env(config.source_env, config, "source")
        self._target_env = make_env(config.target_env, config, "target")
        source_ob_space = self._source_env.observation_space
        source_env_ob_space = self._source_env.env_observation_space
        target_ob_space = target_env_ob_space = self._target_env.observation_space
        source_ac_space = self._source_env.action_space
        target_ac_space = self._target_env.action_space
        logger.info("Sim Observation space: " + str(source_ob_space))
        logger.info(
            "Sim Action space: "
            + str(source_ac_space)
            + str(source_ac_space["ac"].high)
            + str(source_ac_space["ac"].low)
        )
        logger.info("target Observation space: " + str(target_ob_space))
        logger.info("target Action space: " + str(target_ac_space))

        config_eval = copy.copy(config)
        if hasattr(config_eval, "port"):
            config_eval.port += 1
        sconfig_eval = copy.copy(config_eval)
        if config.env_ob_source:
            sconfig_eval.encoder_type = "mlp"
        self._source_env_eval = self._source_env
        self._target_env_eval = self._target_env

        # create a new observation space after data augmentation (random crop)
        if config.encoder_type == "cnn":
            assert (
                not config.ob_norm
            ), "Turn off the observation norm (--ob_norm False) for pixel inputs"
            # source env
            source_ob_space = gym.spaces.Dict(spaces=dict(source_ob_space.spaces))
            for k in source_ob_space.spaces.keys():
                if len(source_ob_space.spaces[k].shape) == 3:
                    shape = [
                        source_ob_space.spaces[k].shape[0],
                        config.encoder_image_size,
                        config.encoder_image_size,
                    ]
                    source_ob_space.spaces[k] = gym.spaces.Box(
                        low=0, high=255, shape=shape, dtype=np.uint8
                    )
            # target env
            target_ob_space = gym.spaces.Dict(spaces=dict(target_ob_space.spaces))
            for k in target_ob_space.spaces.keys():
                if len(target_ob_space.spaces[k].shape) == 3:
                    shape = [
                        target_ob_space.spaces[k].shape[0],
                        config.encoder_image_size,
                        config.encoder_image_size,
                    ]
                    target_ob_space.spaces[k] = gym.spaces.Box(
                        low=0, high=255, shape=shape, dtype=np.uint8
                    )

        # evalutate rollout will mix source and target demo so record_demo need to be stopped
        if config.record_demo:
            raise Exception("Record_demo not currently supported")

        # build agent and networks for algorithm
        self._agent = get_multi_stage_agent_by_name(config.algo)(
            config,
            source_ob_space,
            source_ac_space,
            source_env_ob_space,
            target_ob_space,
            target_ac_space,
            target_env_ob_space,
        )

        # build rollout runner
        self._source_runner, self._target_runner = {}, {}
        for s in self._agent.stages:
            self._source_runner[s] = RolloutRunner(
                config,
                self._source_env,
                self._source_env_eval,
                self._agent.run_in("source", stage=s),
                "source",
            )
            self._target_runner[s] = RolloutRunner(
                config,
                self._target_env,
                self._target_env_eval,
                self._agent.run_in("target", stage=s),
                "target",
            )

        # setup log
        if self._is_chef and config.is_train:
            exclude = ["device"]
            if not config.wandb:
                os.environ["WANDB_MODE"] = "dryrun"

            wandb.init(
                name=config.run_name,
                project=config.wandb_project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity=config.wandb_entity,
                notes=config.notes,
            )

            if config.dr:
                with open("idapt/utils/dr_config.json") as f:
                    dr_params = json.load(f)
                    wandb.config.update(dr_params[config.dr_params_set])

    def train(self):
        """ Trains an agent. """
        config = self._config

        # load checkpoint
        curr_info = {"step": 0, "update_iter": 0, "stage": ""}
        self._load_ckpt(config.init_ckpt_path, config.ckpt_num)

        if (
            hasattr(config, "encoder_init_ckpt_path")
            and config.encoder_init_ckpt_path is not None
        ):
            self._load_ckpt(config.encoder_init_ckpt_path, config.ckpt_num)

        # sync the networks across the cpus
        self._agent.sync_networks()

        logger.info("Start training at step=%d", curr_info["step"])

        total_max_stage_step = np.cumsum(
            [self._config.max_stage_steps[stage] for stage in self._agent.stages]
        )

        if self._is_chef:
            pbar = tqdm(
                initial=curr_info["step"],
                total=total_max_stage_step[-1],
                desc=config.run_name,
            )
            ep_info = Info()
            train_info = Info()

        st_time = time()
        st_step = curr_info["step"]

        evaluate_interval, ckpt_interval, log_interval, warm_up_steps = 0, 0, 0, 0

        for stage, max_stage_step in zip(self._agent.stages, total_max_stage_step):
            stage_start_step = curr_info["step"]
            if stage == "policy_init":
                evaluate_interval = (
                    config.policy_init_evaluate_interval // self._config.num_workers
                )
                ckpt_interval = (
                    config.policy_init_ckpt_interval // self._config.num_workers
                )
                log_interval = config.policy_init_log_interval
                warm_up_steps = (
                    config.policy_init_warm_up_steps // self._config.num_workers
                )
            elif stage == "policy_training":
                self._agent._agent._buffer.clear()
                evaluate_interval = (
                    config.policy_training_evaluate_interval // self._config.num_workers
                )
                ckpt_interval = (
                    config.policy_training_ckpt_interval // self._config.num_workers
                )
                log_interval = config.policy_training_log_interval
                warm_up_steps = (
                    config.policy_training_warm_up_steps // self._config.num_workers
                )
            elif stage == "grounding":
                evaluate_interval = (
                    config.grounding_evaluate_interval
                ) = self._config.num_workers
                ckpt_interval = (
                    config.grounding_ckpt_interval
                ) = self._config.num_workers
                log_interval = config.grounding_log_interval
                warm_up_steps = (
                    config.grounding_warm_up_steps // self._config.num_workers
                )
            elif "supervised" in stage:
                evaluate_interval = (
                    config.supervised_evaluate_interval // self._config.num_workers
                )
                ckpt_interval = (
                    config.supervised_ckpt_interval // self._config.num_workers
                )
                log_interval = config.supervised_log_interval
                warm_up_steps = (
                    config.supervised_warm_up_steps // self._config.num_workers
                )

            curr_info["stage"] = stage
            runner = self._get_runner(
                self._agent.decide_runner_type(curr_info),
                curr_info["step"],
                curr_info["stage"],
            )

            if self._is_chef and max_stage_step == 0 and stage == "policy_init":
                logger.info("Evaluate at %d", curr_info["update_iter"])
                record_video = config.record_video
                rollout, info = self._evaluate(
                    stage=curr_info["stage"],
                    step=curr_info["step"],
                    record_video=record_video,
                )
                self._log_test(curr_info["step"], info)

            while (
                runner
                and curr_info["step"] - stage_start_step < warm_up_steps
                and curr_info["step"] < max_stage_step
            ):
                rollout, info = {}, {}
                running_env = self._agent.decide_env_type(curr_info)
                for env in running_env:
                    rollout[env], info[env] = next(runner[env])
                    self._agent.store_episode(rollout[env], env, stage)
                    step_per_batch = mpi_sum(len(rollout[env]["ac"]))
                    curr_info["step"] += step_per_batch
                if runner and curr_info["step"] < config.max_ob_norm_step:
                    self._agent.update_normalizer(rollout[env]["ob"], env, stage)
                if self._is_chef:
                    pbar.update(step_per_batch)

            while curr_info["step"] < max_stage_step:
                # collect rollouts
                rollout, info = {}, {}
                running_env = self._agent.decide_env_type(curr_info)

                step_per_batch = 0
                for env in running_env:
                    if runner[env]:
                        rollout[env], info[env] = next(runner[env])
                        info[env] = dict_add_prefix(
                            info[env],
                            env + "_" + stage + "_" + str(self._agent._grounding_step),
                        )
                        self._agent.store_episode(rollout[env], env, stage)
                        step_per_batch += mpi_sum(len(rollout[env]["ob"]))
                    else:
                        if "supervised" not in stage:
                            step_per_batch += mpi_sum(1)
                        info[env] = {}

                if "supervised" in stage:
                    step_per_batch = 1

                curr_info["step"] += step_per_batch

                _train_info = self._agent.train(running_env, curr_info)
                check_memory_kill_switch()

                for env in running_env:
                    if runner[env] and curr_info["step"] < config.max_ob_norm_step:
                        ### hack : agent needs stage to normalize correctly
                        ### When grounding AT ob is modified in store episode
                        self._agent.update_normalizer(rollout[env]["ob"], env, stage)

                curr_info["update_iter"] += 1

                # log training and episode information or evaluate
                if self._is_chef:

                    pbar.update(step_per_batch)

                    for env in running_env:
                        ep_info.add(info[env])

                    train_info.add(_train_info)

                    if "supervised" in stage and hasattr(self._agent, "record_image"):
                        self._agent.record_image(curr_info["step"], self._target_env)

                    if curr_info["update_iter"] % log_interval == 0:
                        train_info.add(
                            {
                                "sec": (time() - st_time) / log_interval,
                                "steps_per_sec": (curr_info["step"] - st_step)
                                / (time() - st_time),
                                "update_iter": curr_info["update_iter"],
                            }
                        )
                        st_time = time()
                        st_step = curr_info["step"]

                        self._log_train(
                            curr_info["step"],
                            train_info.get_dict(only_scalar=True),
                            ep_info.get_dict(),
                        )
                        ep_info = Info()
                        train_info = Info()

                    if (
                        curr_info["update_iter"] % evaluate_interval == 1
                        or evaluate_interval == 1
                    ):
                        logger.info("Evaluate at %d", curr_info["update_iter"])
                        record_video = config.record_video and (
                            curr_info["update_iter"] % (evaluate_interval * 5) == 1
                        )
                        rollout, info = self._evaluate(
                            stage=curr_info["stage"],
                            step=curr_info["step"],
                            record_video=record_video,
                        )

                        self._log_test(curr_info["step"], info)

                    if (
                        not self._config.ckpt_stage_only
                        and curr_info["update_iter"] % ckpt_interval == 0
                    ):
                        self._save_ckpt(
                            curr_info["step"],
                            curr_info["update_iter"],
                            curr_info["stage"],
                        )
                    if (
                        self._config.ckpt_stage_only
                        and curr_info["step"] == max_stage_step
                    ):
                        self._save_ckpt(
                            curr_info["step"],
                            curr_info["update_iter"],
                            curr_info["stage"],
                        )

        self._save_ckpt(curr_info["step"], curr_info["update_iter"], curr_info["stage"])
        logger.info(
            "Reached %s steps. worker %d stopped.", curr_info["step"], config.rank
        )

    def evaluate(self):
        """ Evaluates an agent stored in chekpoint with @self._config.ckpt_num. """
        step, update_iter = self._load_ckpt(
            self._config.init_ckpt_path, self._config.ckpt_num
        )

        logger.info(
            "Run %d evaluations at step=%d, update_iter=%d",
            self._config.num_eval,
            step,
            update_iter,
        )
        rollouts, info = self._evaluate(
            step=step, record_video=self._config.record_video, stage="policy_init"
        )

        info_stat = info.get_stat()
        os.makedirs("result", exist_ok=True)
        with h5py.File("result/{}.hdf5".format(self._config.run_name), "w") as hf:
            for k, v in info.items():
                if type(v[0]) != wandb.data_types.Video:
                    hf.create_dataset(k, data=info[k])
        with open("result/{}.txt".format(self._config.run_name), "w") as f:
            for k, v in info_stat.items():
                f.write("{}\t{:.03f} $\\pm$ {:.03f}\n".format(k, v[0], v[1]))

        if self._config.record_demo:
            new_rollouts = []
            for rollout in rollouts:
                new_rollout = {
                    "obs": rollout["ob"],
                    "actions": rollout["ac"],
                    "rewards": rollout["rew"],
                    "dones": rollout["done"],
                }
                new_rollouts.append(new_rollout)

            fname = "{}_step_{:011d}_{}_trajs.pkl".format(
                self._config.run_name, step, self._config.num_eval,
            )
            path = os.path.join(self._config.demo_dir, fname)
            logger.warn("[*] Generating demo: {}".format(path))
            with open(path, "wb") as f:
                pickle.dump(new_rollouts, f)

    def _evaluate(self, stage, step=None, record_video=False):
        """
        Runs one rollout if in eval mode (@idx is not None).
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            step: the number of environment steps.
            record_video: whether to record video or not.
        """
        num_eval = self._config.num_eval
        logger.info("Run %d evaluations at step=%d, stage=%s", num_eval, step, stage)
        rollouts = {"target": [], "source": []}
        info_history = Info()
        start = time()
        for i in range(num_eval):
            logger.warn("Evalute run %d", i + 1)
            record_q_state = (
                "supervised" in stage
                or (hasattr(self._agent, "evaluate_AT"))
                or (i == 0 and hasattr(self._agent, "record_video"))
            )
            target_rollout, target_info, target_frames = self._target_runner[
                stage
            ].run_episode(
                is_train=False,
                record_video=(record_video and i == 0),
                record_qpos_qvel=record_q_state,
            )
            source_rollout, source_info, source_frames = self._source_runner[
                stage
            ].run_episode(
                is_train=False,
                record_video=(record_video and i == 0),
                record_qpos_qvel=record_q_state,
            )
            source_info = dict_add_prefix(
                source_info, "source_" + stage + "_" + str(self._agent._grounding_step)
            )
            target_info = dict_add_prefix(
                target_info, "target_" + stage + "_" + str(self._agent._grounding_step)
            )

            if not "supervised" in stage and hasattr(self._agent, "evaluate_AT"):
                (
                    source_info["source_AT_error"],
                    source_info["source_AT_acdiff"],
                    source_info["source_AT_acratio"],
                ) = self._agent.evaluate_AT(
                    source_rollout, "source", self._target_env, stage
                )
                (
                    target_info["target_AT_error"],
                    target_info["target_AT_acdiff"],
                    target_info["target_AT_acratio"],
                ) = self._agent.evaluate_AT(
                    target_rollout, "target", self._source_env, stage
                )

            rollouts["source"].append(source_rollout)
            rollouts["target"].append(target_rollout)
            logger.info(
                "source rollout: %s",
                {k: v for k, v in source_info.items() if not "qpos" in k},
            )
            logger.info(
                "target rollout: %s",
                {k: v for k, v in target_info.items() if not "qpos" in k},
            )

            if record_video and i == 0:
                # source video
                source_ep_rew = source_info[
                    "source_" + stage + "_" + str(self._agent._grounding_step) + "_rew"
                ]
                source_ep_success = (
                    "s"
                    if "source_episode_success" in source_info
                    and source_info[
                        "source_"
                        + stage
                        + "_"
                        + str(self._agent._grounding_step)
                        + "_episode_success"
                    ]
                    else "f"
                )
                fname = "{}_{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    "source",
                    self._config.source_env,
                    step,
                    i,
                    source_ep_rew,
                    source_ep_success,
                )
                source_video_path = self._save_video(fname, source_frames)
                source_info["source_" + stage + "_video"] = wandb.Video(
                    source_video_path, fps=15, format="mp4"
                )

                # target video
                target_ep_rew = target_info[
                    "target_" + stage + "_" + str(self._agent._grounding_step) + "_rew"
                ]
                target_ep_success = (
                    "s"
                    if "target_"
                    + stage
                    + "_"
                    + str(self._agent._grounding_step)
                    + "_episode_success"
                    in target_info
                    and target_info[
                        "target_"
                        + stage
                        + "_"
                        + str(self._agent._grounding_step)
                        + "_episode_success"
                    ]
                    else "f"
                )
                fname = "{}_{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    "target",
                    self._config.target_env,
                    step,
                    i,
                    target_ep_rew,
                    target_ep_success,
                )
                target_video_path = self._save_video(fname, target_frames)
                target_info["target_" + stage + "_video"] = wandb.Video(
                    target_video_path, fps=15, format="mp4"
                )

            # gen video
            if hasattr(self._agent, "record_video") and i == 0:
                gen_frames = self._agent.record_video(
                    source_rollout["ob"],
                    domain="source",
                    qposs=source_rollout["qpos"],
                    qvels=source_rollout["qvel"],
                    target_env=self._target_env,
                )
                source_ep_rew = source_info[
                    "source_" + stage + "_" + str(self._agent._grounding_step) + "_rew"
                ]
                source_ep_success = (
                    "s"
                    if "source_episode_success" in source_info
                    and source_info[
                        "source_"
                        + stage
                        + "_"
                        + str(self._agent._grounding_step)
                        + "_episode_success"
                    ]
                    else "f"
                )
                fname = "{}_{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    "gen",
                    self._config.source_env,
                    step,
                    i,
                    source_ep_rew,
                    source_ep_success,
                )
                gen_video_path = self._save_video(fname, gen_frames)
                source_info["gen_" + stage + "_video"] = wandb.Video(
                    gen_video_path, fps=15, format="mp4"
                )

                gen_target_frames = self._agent.record_video(
                    target_rollout["ob"],
                    domain="target",
                    qposs=target_rollout["qpos"],
                    qvels=target_rollout["qvel"],
                    source_env=self._source_env,
                )
                target_ep_rew = target_info[
                    "target_" + stage + "_" + str(self._agent._grounding_step) + "_rew"
                ]
                target_ep_success = (
                    "s"
                    if "target_"
                    + stage
                    + "_"
                    + str(self._agent._grounding_step)
                    + "_episode_success"
                    in target_info
                    and target_info[
                        "target_"
                        + stage
                        + "_"
                        + str(self._agent._grounding_step)
                        + "_episode_success"
                    ]
                    else "f"
                )
                fname = "{}_{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    "gen_target",
                    self._config.target_env,
                    step,
                    i,
                    target_ep_rew,
                    target_ep_success,
                )
                gen_video_path = self._save_video(fname, gen_target_frames)
                target_info["gen_target_" + stage + "_video"] = wandb.Video(
                    gen_video_path, fps=15, format="mp4"
                )

            info = {**source_info, **target_info}
            info_history.add(info)

        avg_stats = {}
        for k, v in info_history.items():
            if isinstance(v, list) and not isinstance(v[0], wandb.Video):
                avg_stats[k] = np.mean(v)

        end = time()

        logger.info(
            "Evaluation Time: %f, Evaluation stats: %s",
            end - start,
            {k: v for k, v in avg_stats.items()},
        )

        return rollouts, info_history

    def _get_runner(self, runner_type, step, stage):
        """
        decide how many episodes or how long rollout to collect
        'on_policy': Sim (rollout_length steps) + target (target_transitions_count steps)
        'off_policy': Sim (1 step) + target (target_transitions_count steps)
        'supervised': Sim (num_transitions steps) + target (num_transitions steps)
        """
        if runner_type == "on_policy":
            _source_runner = self._source_runner[stage].run(
                every_steps=self._config.rollout_length, log_prefix="source", step=step
            )
            _target_runner = self._target_runner[stage].run(
                every_steps=self._config.target_transitions_count,
                log_prefix="target",
                step=step,
            )
        elif runner_type == "off_policy":
            _source_runner = self._source_runner[stage].run(
                every_steps=1, log_prefix="source", step=step
            )
            _target_runner = self._target_runner[stage].run(
                every_episodes=self._config.target_transitions_count,
                log_prefix="target",
                step=step,
            )
        elif runner_type == "supervised":
            _source_runner = self._source_runner[stage].run(
                every_steps=self._config.num_transitions,
                log_prefix="source",
                step=step,
                record_qpos_qvel=True,
            )
            _target_runner = self._target_runner[stage].run(
                every_steps=self._config.num_transitions,
                log_prefix="target",
                step=step,
                record_qpos_qvel=True,
            )
            _source_eval_runner = self._source_runner[stage].run(
                every_steps=self._config.num_eval_transitions,
                log_prefix="source_eval",
                step=step,
                record_qpos_qvel=True,
            )
            _target_eval_runner = self._target_runner[stage].run(
                every_steps=self._config.num_eval_transitions,
                log_prefix="target_eval",
                step=step,
                record_qpos_qvel=True,
            )
            return {
                "source": _source_runner,
                "target": _target_runner,
                "source_eval": _source_eval_runner,
                "target_eval": _target_eval_runner,
            }

        else:
            raise NotImplementedError

        return {"source": _source_runner, "target": _target_runner}

    def _save_ckpt(self, ckpt_num, update_iter, stage):
        """
        Save model checkpoint to log directory depending on stage.

        Args:
            ckpt_num: number appended to checkpoint name. The number of
                environment step is used in this code.
            update_iter: number of policy update. It will be used for resuming training.
            stage: current stage
        """
        ckpt_path = os.path.join(
            self._config.log_dir, "ckpt_%s_%09d.pt" % (stage, ckpt_num)
        )
        state_dict = {"step": ckpt_num, "update_iter": update_iter}
        state_dict["agent"] = self._agent.state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warn("Save checkpoint: %s", ckpt_path)

    def _load_ckpt(self, ckpt_path, ckpt_num):
        """
        Loads checkpoint with path @ckpt_path or index number @ckpt_num. If @ckpt_num is None,
        it loads and returns the checkpoint with the largest index number.
        """
        if ckpt_path is None:
            ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)
        else:
            ckpt_num = int(ckpt_path.rsplit("_", 1)[-1].split(".")[0])

        if ckpt_path is not None:
            logger.warn("Load checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self._config.device)
            self._agent.load_state_dict(ckpt["agent"])

            if self._config.is_train and self._agent.is_off_policy():
                replay_path = os.path.join(
                    self._config.log_dir, "replay_%08d.pkl" % ckpt_num
                )
                logger.warn("Load replay_buffer %s", replay_path)
                if os.path.exists(replay_path):
                    with gzip.open(replay_path, "rb") as f:
                        replay_buffers = pickle.load(f)
                        self._agent.load_replay_buffer(replay_buffers["replay"])
                else:
                    logger.warn("Replay buffer not exists at %s", replay_path)

            if (
                self._config.init_ckpt_path is not None
                and "bc" in self._config.init_ckpt_path
            ):
                return 0, 0
            else:
                return ckpt["step"], ckpt["update_iter"]
        logger.warn("Randomly initialize models")
        return 0, 0

    def _log_train(self, step, train_info, ep_info):
        """
        Logs training and episode information to wandb.
        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
            ep_info: episode information to log, such as reward, episode time.
        """
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, "shape") and np.prod(v.shape) == 1):
                wandb.log({"train_rl/%s" % k: v}, step=step)
            else:
                wandb.log({"train_rl/%s" % k: [wandb.Image(v)]}, step=step)

        for k, v in ep_info.items():
            wandb.log({"train_ep/%s" % k: np.mean(v)}, step=step)
            wandb.log({"train_ep_max/%s" % k: np.max(v)}, step=step)

    def _log_test(self, step, ep_info):
        """
        Logs episode information during testing to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        if self._config.is_train:
            for k, v in ep_info.items():
                if isinstance(v, wandb.Video):
                    wandb.log({"test_ep/%s" % k: v}, step=step)
                elif isinstance(v, list) and isinstance(v[0], wandb.Video):
                    for i, video in enumerate(v):
                        wandb.log({"test_ep/%s_%d" % (k, i): video}, step=step)
                else:
                    wandb.log({"test_ep/%s" % k: np.mean(v)}, step=step)

    def _save_video(self, fname, frames, fps=15.0):
        """ Saves @frames into a video with file name @fname. """
        path = os.path.join(self._config.record_dir, fname)
        logger.warn("[*] Generating video: {}".format(path))

        def f(t):
            frame_length = len(frames)
            new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames) / fps + 2)

        video.write_videofile(path, fps, verbose=False)
        logger.warn("[*] Video saved: {}".format(path))
        return path

    def _update_normalizer(self, rollout, env_type):
        """ Updates normalizer with @rollout. """
        if self._config.ob_norm:
            self._agent.update_normalizer(rollout["ob"], env_type)
