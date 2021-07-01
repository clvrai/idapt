import cv2
import numpy as np
import pickle
import random
from collections import defaultdict, OrderedDict
from copy import deepcopy

import numpy as np

from training.utils.gym_env import zero_value
from training.utils.info_dict import Info


class Rollout(object):
    def __init__(self):      
        """ Initialize buffer. """
        self._history = defaultdict(list)

    def add(self, data):
        """ Add a transition @data to rollout buffer. """
        for key, value in data.items():
            self._history[key].append(value)

    def get(self):
        """ Returns rollout buffer and clears buffer. """
        batch = {}

        for key, value in self._history.items():
            batch[key] = value

        self._history = defaultdict(list)
        return batch


class RolloutRunner(object):
    def __init__(self, config, env, env_eval, pi, env_type):
        self._config = config
        self._env = env
        self._env_eval = env_eval
        self._pi = pi
        self._env_type = env_type

    def run(
        self,
        is_train=True,
        every_steps=None,
        every_episodes=None,
        log_prefix="",
        step=0,
        record_qpos_qvel=False,
    ):
        """
        Collects trajectories and yield every @every_steps/@every_episodes.

        Args:
            is_train: whether rollout is for training or evaluation.
            every_steps: if not None, returns rollouts @every_steps
            every_episodes: if not None, returns rollouts @every_epiosdes
            log_prefix: log as @log_prefix rollout: %s


        Changes:
            Expects ac to return ac_info in addition to ac, activation
            Pass ob_next to predict reward
        """
        if every_steps is None and every_episodes is None:
            raise ValueError("Both every_steps and every_episodes cannot be None")

        config = self._config
        device = config.device
        env = self._env  # if is_train else self._env_eval
        pi = self._pi
        recurrent = config.recurrent
        il = hasattr(pi, "predict_reward")
        translate = hasattr(pi._agent, "translate")
        enc = hasattr(pi._agent, "encode")

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = Info()
        ep_info = Info()
        episode = 0

        while True:

            done = False
            ep_len = 0
            ep_rew = 0
            ep_rew_rl = 0
            if il:
                ep_rew_il = 0

            ob = env.reset()

            if recurrent:
                ac_prev = np.zeros_like(env.action_space.sample()['ac'], dtype=np.float64)
                if len(ac_prev.shape) < 1:
                    ac_prev = np.expand_dims(ac_prev, 0)
                rnn_state_in = np.zeros((2, config.rnn_dim,))
                c_rnn_state_in = np.zeros((2, config.rnn_dim,))


            # run rollout
            state = None
            if "state" in ob.keys():
                state = ob.pop("state", None)

            while not done:
                # sample action from policy

                if step < config.warm_up_steps:
                    ac = env.action_space.sample()
                    ac_before_activation = ac
                    ac_info = {
                        "target_ac": ac,
                        "target_ac_before_activation": ac_before_activation,
                    }
                    if recurrent:
                        _, _, temp = pi.act(
                                ob, ac_prev, rnn_state_in, is_train=is_train
                        )
                        
                        ac_info["rnn_state_in"] = rnn_state_in
                        ac_info["c_rnn_state_in"] = c_rnn_state_in
                        c_rnn_state_in = pi._agent.get_c_rnn_state(state, ac_prev, c_rnn_state_in)
                        ac_info["rnn_state_out"] = temp["rnn_state_out"]
                        rnn_state_in = ac_info["rnn_state_out"]
                        ac_info["ac_prev"] = ac_prev
                        ac_prev = ac["ac"].astype(np.float64)
                else:
                    if self._config.run_rl_from_state:
                        ac, ac_before_activation, ac_info = pi.act(
                            state, is_train=is_train
                        )
                    else:
                        if recurrent:
                            ac, ac_before_activation, ac_info = pi.act(
                                ob, ac_prev, rnn_state_in, is_train=is_train
                            )
                            ac_info["ac_prev"] = ac_prev
                            ac_info["rnn_state_in"] = rnn_state_in
                            ac_info["c_rnn_state_in"] = c_rnn_state_in
                            c_rnn_state_in = pi._agent.get_c_rnn_state(state, ac_prev, c_rnn_state_in)
                            ac_prev = ac["ac"]
                            rnn_state_in = ac_info["rnn_state_out"]
                        else:
                            ac, ac_before_activation, ac_info = pi.act(
                                ob, is_train=is_train
                            )

                if state is not None:
                    rollout.add({"state": state})
                    # rollout.add({"state": {"state": state}})
                    state = None

                if len(ac["ac"].shape) != 1 or (
                    recurrent and (len(rnn_state_in.shape) != 2)
                ):
                    import ipdb

                    ipdb.set_trace()

                ac_info.update(
                    {
                        "ob": deepcopy(ob),
                        "ac": ac,
                        "ac_before_activation": ac_before_activation,
                    }
                )

                if translate and not "got_ob" in ac_info.keys():
                    translated_ob = pi._agent.translate(ob, self._env_type)
                    ac_info.update({"got_ob": deepcopy(translated_ob)})

                if enc and not self._config.run_rl_from_state:
                    if translate:
                        encoded_ob = pi._agent.encode(ac_info["got_ob"], self._env_type)
                    else:
                        encoded_ob = pi._agent.encode(ob, self._env_type)
                    ac_info.update({"encoded_ob": encoded_ob})
                    if not "got_ob" in ac_info.keys():
                        ac_info.update({"got_ob": encoded_ob})

                rollout.add(ac_info)

                if record_qpos_qvel:
                    rollout.add(
                        {
                            "qpos": env.sim.data.qpos.copy(),
                            "qvel": env.sim.data.qvel.copy(),
                        }
                    )

                ob_next, reward, done, info = env.step(ac)

                if "state" in ob_next.keys():
                    state = ob_next.pop("state", None)
                    # rollout.add({"state_next": {"state": state}})
                    rollout.add({"state_next": state})

                rollout.add({"ob_next": ob_next})

                if record_qpos_qvel:
                    rollout.add(
                        {
                            "qpos_next": env.sim.data.qpos.copy(),
                            "qvel_next": env.sim.data.qvel.copy(),
                        }
                    )

                # replace reward
                if il:
                    if enc and not self._config.run_rl_from_state:
                        encoded_ob = ac_info["encoded_ob"]
                        if translate:
                            translated_ob_next = pi._agent.translate(
                                ob_next, self._env_type
                            )
                            encoded_ob_next = pi._agent.encode(
                                translated_ob_next, self._env_type
                            )
                        else:
                            encoded_ob_next = pi._agent.encode(ob_next, self._env_type)
                    reward_il = pi.predict_reward(encoded_ob, ac, encoded_ob_next)
                    reward_rl = (
                        1 - config.gaifo_env_reward
                    ) * reward_il + config.gaifo_env_reward * reward
                else:
                    reward_rl = reward

                ob = ob_next

                step += 1
                ep_len += 1
                ep_rew += reward
                ep_rew_rl += reward_rl

                early_term_done = done
                if "success_reward" in info.keys() and info["success_reward"] == 1.0:
                    early_term_done = False
                if done and config.early_term and ep_len == env.max_episode_steps:
                    early_term_done = False

                rollout.add({"done": early_term_done, "rew": reward})

                if il:
                    ep_rew_il += reward_il

                if enc and not self._config.run_rl_from_state:
                    if translate:
                        translated_ob_next = pi._agent.translate(
                            ob_next, self._env_type
                        )
                        rollout.add({"got_ob_next": deepcopy(translated_ob_next)})
                        encoded_ob_next = pi._agent.encode(
                            translated_ob_next, self._env_type
                        )
                    else:
                        encoded_ob_next = pi._agent.encode(ob_next, self._env_type)
                    rollout.add({"encoded_ob_next": encoded_ob_next})
                    if not translate:
                        rollout.add({"got_ob_next": encoded_ob})

                if done and ep_len < env.max_episode_steps:
                    done_mask = 0  # -1 absorbing, 0 done, 1 not done

                else:
                    done_mask = 1

                rollout.add(
                    {"done_mask": done_mask}
                )  # -1 absorbing, 0 done, 1 not done

                reward_info.add(info)

                if config.absorbing_state and done_mask == 0:
                    absorbing_state = env.get_absorbing_state()
                    absorbing_action = zero_value(env.action_space)
                    rollout._history["ob_next"][-1] = absorbing_state
                    rollout.add(
                        {
                            "ob": absorbing_state,
                            "ob_next": absorbing_state,
                            "ac": absorbing_action,
                            "ac_before_activation": absorbing_action,
                            "rew": 0.0,
                            "done": 0,
                            "done_mask": -1,  # -1 absorbing, 0 done, 1 not done
                        }
                    )

                if every_steps is not None and step % every_steps == 0:
                    yield rollout.get(), ep_info.get_dict(only_scalar=True)

            # compute average/sum of information
            ep_info.add({"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl})
            if il:
                ep_info.add({"rew_il": ep_rew_il})
            reward_info_dict = reward_info.get_dict(reduction="sum", only_scalar=True)
            ep_info.add(reward_info_dict)
            reward_info_dict.update({"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl})

            if il:
                reward_info_dict.update({"rew_il": ep_rew_il})

            episode += 1
            if every_episodes is not None and episode % every_episodes == 0:
                yield rollout.get(), ep_info.get_dict(only_scalar=True)

    def run_episode(
        self, max_step=10000, is_train=True, record_video=False, record_qpos_qvel=False
    ):
        """
        Runs one episode and returns the rollout (mainly for evaluation).

        Args:
            max_step: maximum number of steps of the rollout.
            is_train: whether rollout is for training or evaluation.
            record_video: record video of rollout if True.

        Changes:
            Expects ac to return ac_info in addition to ac, activation
            Record action difference between real and transformed
            Pass ob_next to predict reward
        """
        config = self._config
        device = config.device
        env = self._env  # if is_train else self._env_eval
        pi = self._pi
        recurrent = config.recurrent
        il = hasattr(pi, "predict_reward")
        enc = hasattr(pi._agent, "encode")
        translate = hasattr(pi._agent, "translate")

        # initialize rollout buffer
        rollout = Rollout()
        reward_info = Info()
        action_info = Info()

        done = False
        ep_len = 0
        ep_rew = 0
        ep_rew_rl = 0
        if il:
            ep_rew_il = 0

        ob = env.reset()

        if recurrent:
            ac_prev = np.zeros_like(env.action_space.sample()['ac'], dtype=np.float64)
            if len(ac_prev.shape) < 1:
                ac_prev = np.expand_dims(ac_prev, 0)
            rnn_state_in = np.zeros((2, config.rnn_dim,))

        self._record_frames = []
        if record_video:
            self._store_frame(env, ep_len, ep_rew)

        state = None
        if "state" in ob.keys():
            state = ob.pop("state", None)
        # run rollout

        while not done and ep_len < max_step:
            # sample action from policy

            if self._config.run_rl_from_state:
                ac, ac_before_activation, ac_info = pi.act(state, is_train=is_train)
            else:
                if recurrent:
                    ac, ac_before_activation, ac_info = pi.act(
                        ob, ac_prev, rnn_state_in, is_train=is_train
                    )
                    ac_prev = ac["ac"]
                    rnn_state_in = ac_info["rnn_state_out"]
                else:
                    ac, ac_before_activation, ac_info = pi.act(ob, is_train=is_train)

            if state is not None:
                rollout.add({"state": {"state": state}})
                state = None

            ac_info.update(
                {"ob": ob, "ac": ac, "ac_before_activation": ac_before_activation,}
            )

            rollout.add(ac_info)

            if record_qpos_qvel:
                rollout.add(
                    {"qpos": env.sim.data.qpos.copy(), "qvel": env.sim.data.qvel.copy()}
                )

            ob_next, reward, done, info = env.step(ac)

            if "state" in ob_next.keys():
                state = ob_next.pop("state", None)
                rollout.add({"state_next": {"state": state}})

            rollout.add({"ob_next": ob_next})

            if record_qpos_qvel:
                rollout.add(
                    {
                        "qpos_next": env.sim.data.qpos.copy(),
                        "qvel_next": env.sim.data.qvel.copy(),
                    }
                )

            # replace reward
            if il:
                if enc and not self._config.run_rl_from_state:
                    if translate:
                        translated_ob = pi._agent.translate(ob, self._env_type)
                        translated_ob_next = pi._agent.translate(
                            ob_next, self._env_type
                        )
                        encoded_ob = pi._agent.encode(translated_ob, self._env_type)
                        encoded_ob_next = pi._agent.encode(
                            translated_ob_next, self._env_type
                        )

                    else:
                        encoded_ob = pi._agent.encode(ob, self._env_type)
                        encoded_ob_next = pi._agent.encode(ob_next, self._env_type)
                reward_il = pi.predict_reward(encoded_ob, ac, encoded_ob_next)
                reward_rl = (
                    1 - config.gaifo_env_reward
                ) * reward_il + config.gaifo_env_reward * reward
            else:
                reward_rl = reward

            ob = ob_next

            ep_len += 1
            ep_rew += reward
            ep_rew_rl += reward_rl

            early_term_done = done
            if "success_reward" in info.keys() and info["success_reward"] == 1.0:
                early_term_done = False
            if done and config.early_term and ep_len == env.max_episode_steps:
                early_term_done = False

            rollout.add({"done": early_term_done, "rew": reward})

            if il:
                ep_rew_il += reward_il

            reward_info.add(info)

            if record_video:
                frame_info = info.copy()
                if "target_ac" in ac_info and config.source_env == "SawyerPush-v0":
                    frame_info.update(
                        {
                            "x_at": ac["ac"][0] - ac_info["target_ac"]["ac"][0],
                            "y_at": ac["ac"][1] - ac_info["target_ac"]["ac"][1],
                        }
                    )
                frame_info.update({"action": ac["ac"]})
                if il:
                    frame_info.update(
                        {
                            "ep_rew_il": ep_rew_il,
                            "rew_il": reward_il,
                            "rew_rl": reward_rl,
                        }
                    )
                self._store_frame(env, ep_len, ep_rew, frame_info)

        # add last observation
        if "state" in ob.keys():
            ob.pop("state")
        rollout.add({"ob": ob})

        if enc and not self._config.run_rl_from_state:
            rollout.add({"encoded_ob": pi._agent.encode(ob, self._env_type)})

        # compute average/sum of information
        ep_info = {"len": ep_len, "rew": ep_rew, "rew_rl": ep_rew_rl}
        if il:
            ep_info["rew_il"] = ep_rew_il

        ep_info.update(reward_info.get_dict(reduction="sum", only_scalar=True))
        ep_info.update(action_info.get_dict(reduction="mean", only_scalar=True))

        return rollout.get(), ep_info, self._record_frames

    def _store_frame(self, env, ep_len, ep_rew, info={}):
        """ Renders a frame and stores in @self._record_frames. """
        color = (200, 200, 200)

        # render video frame
        frame = env.render("rgb_array")
        if len(frame.shape) == 4:
            frame = frame[0]
        if np.max(frame) <= 1.0:
            frame *= 255.0

        h, w = frame.shape[:2]
        if h < 500:
            h, w = 500, 500
            frame = cv2.resize(frame, (h, w))
        frame = np.concatenate([frame, np.zeros((h, w, 3))], 0)
        scale = h / 500

        # add caption to video frame
        if self._config.record_video_caption:
            text = "{:4} {}".format(ep_len, ep_rew)
            font_size = 0.4 * scale
            thickness = 1
            offset = int(12 * scale)
            x, y = int(5 * scale), h + int(10 * scale)
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
            for i, k in enumerate(info.keys()):
                v = info[k]
                key_text = "{}: ".format(k)
                (key_width, _), _ = cv2.getTextSize(
                    key_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness
                )

                cv2.putText(
                    frame,
                    key_text,
                    (x, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (66, 133, 244),
                    thickness,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    str(v),
                    (x + key_width, y + offset * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_size,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        self._record_frames.append(frame)
