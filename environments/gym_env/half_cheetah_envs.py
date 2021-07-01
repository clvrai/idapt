import os

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class GymHalfCheetahDM(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/half_cheetah_dm.xml"
        )
        super().__init__(xml_file=model_path)


class GymHalfCheetahDMVisual(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/half_cheetah_dm_visual.xml"
        )
        super().__init__(xml_file=model_path)


class GymHalfCheetahEasy(HalfCheetahEnv):
    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "assets/half_cheetah_easy.xml"
        )
        super().__init__(xml_file=model_path)
