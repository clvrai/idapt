import gym
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv


class GymInvertedPendulumRNN(InvertedPendulumEnv):
    """Color change, weight"""

    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/inverted_pendulum_easy.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done =  not (np.isfinite(ob).all() and (np.abs(ob[1]) <= 1.5))
        return ob, reward, done, {}
    
    def reset_model(self):
        return super().reset_model()

class GymInvertedPendulumEasy(InvertedPendulumEnv):
    """Color change, weight"""

    def __init__(self):
        utils.EzPickle.__init__(self)
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/inverted_pendulum_easy.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

class GymInvertedPendulumDM(InvertedPendulumEnv):
    """Color change, viewpoint, background, weight"""

    def __init__(self):
        gym.utils.EzPickle.__init__(self)
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/inverted_pendulum_dm.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

    def viewer_setup(self):
        # DEFAULT_CAMERA_CONFIG = {
        #     'elevation': -55.0,
        #     'lookat': np.array([0.05, 0.0, 0.0]),
        # }
        DEFAULT_CAMERA_CONFIG = {
            "elevation": -55.0,
            "azimuth": 100.0,
        }
        # import ipdb
        # ipdb.set_trace()
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class GymInvertedPendulumDMVisual(GymInvertedPendulumDM):
    """Color change, viewpoint, background, weight"""

    def __init__(self):
        gym.utils.EzPickle.__init__(self)
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/inverted_pendulum_dm_visual.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)
