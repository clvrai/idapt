import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.walker2d import Walker2dEnv


class GymWalker(Walker2dEnv):
    def __init__(self):
        super().__init__()

class GymWalkerRNN(Walker2dEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_easy.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)
    
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        reward *= (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        done = not (height > 0.2 and height < 3.0 and
                    ang > -2.7 and ang < 2.7)
        ob = self._get_obs()
        return ob, reward, done, {}
    
    def reset_model(self):
        return super().reset_model()

class GymWalkerDM(Walker2dEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_dm.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)


class GymWalkerDMVisual(Walker2dEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/walker2d_dm_visual.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)


class GymWalkerEasy(Walker2dEnv):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "assets/walker2d_easy.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 4)
        utils.EzPickle.__init__(self)


class GymWalkerBackwards(Walker2dEnv):
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_sourceulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = -1.0 * ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()

        return ob, reward, done, {}
