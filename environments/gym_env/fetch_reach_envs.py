import gym
import mujoco_py
import numpy as np
import os

from environments.gym_env.fetch import fetch_env, utils
from environments.unity_interface import UnityInterface


class GymFetchReachEnv(fetch_env.FetchEnv, gym.utils.EzPickle):
    """Unity"""

    def __init__(self, xml_path="reach_target.xml", **kwargs):
        self._screen_width = kwargs["screen_width"]
        self._screen_height = kwargs["screen_height"]
        self._background = "Interior"
        self._unity = None  # kwargs["unity"]
        self._unity_updated = None
        self._prev_unity_qpos = []
        self._curr_unity_qpos = []
        if kwargs["unity"]:
            self._unity = UnityInterface(
                kwargs["port"], kwargs["unity_editor"], kwargs["virtual_display"]
            )
            self._unity.set_quality(4)
            self._unity.set_background(self._background)

        theta = kwargs["action_rotation_degrees"] * np.pi / 180  # pi/2 = 90 deg
        c, s = np.cos(theta), np.sin(theta)
        self.rot = np.array(((c, -s), (s, c)))
        self.bias = np.array([0, 0, kwargs["action_z_bias"], 0])
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }

        full_xml_path = os.path.join(
            os.path.dirname(__file__), "assets", "fetch", xml_path
        )
        fetch_env.FetchEnv.__init__(
            self,
            full_xml_path,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type="dense",
        )
        gym.utils.EzPickle.__init__(self)
        print("cameras: ", self.sim.model.camera_names)
        # Camera
        self._camera_id = self.sim.model.camera_names.index("frontview")

        # Unity
        if self._unity:
            self._unity.change_model(
                # xml=self.model.get_xml(),
                xml_path=full_xml_path,
                camera_id=self._camera_id,
                screen_width=self._screen_width,
                screen_height=self._screen_height,
            )
            self._unity.set_background(self._background)

    def set_state(self, qpos, qvel, goal=None):
        # assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self._unity_updated = False
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        if goal:
            self.goal = goal
        self.sim.forward()

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link_body")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.0
        self.viewer.cam.azimuth = 177  # 132.
        self.viewer.cam.elevation = -32  # 14.

    def reset(self):
        obs = super().reset()
        self._unity_updated = False

        return obs

    def step(self, action):
        self._unity_updated = False
        obs = self._get_obs()
        obs, rew, done, info = super().step(action)
        return obs, rew, done, info

    def render(self, mode="human", width=500, height=500, camera_id=None):
        # self._render_callback()
        if self._unity and not self._unity_updated:
            self._update_unity()
            self._unity_updated = True

        if mode == "rgb_array":
            if self._unity:
                if camera_id is not None:
                    camera_obs, _ = self._unity.get_images([camera_id])
                else:
                    camera_obs, _ = self._unity.get_images()
                camera_obs = camera_obs[0, :, :, :]
            else:
                camera_obs = super().render(mode, width, height)
            # camera_obs = camera_obs[::-1, :, :] #/ 255.0
            assert np.sum(camera_obs) > 0, "rendering image is blank"

            return camera_obs

        raise NotImplementedError

    def close(self):
        if self._unity:
            self._unity.disconnect_to_unity()
        super().close()

    def _update_unity(self):
        """
        Updates unity rendering with qpos. Call this after you change qpos
        """
        unity_qpos = self.sim.data.qpos.copy()
        ### HACK HACK HACK
        unity_qpos[6] = -unity_qpos[6]
        self._unity.set_qpos(unity_qpos)
        geom_id = self.sim.model.geom_name2id("target0")
        gripper_id = self.sim.model.geom_name2id("robot0:gripper_link_geom")
        self.sim.model.geom_pos[geom_id] = self.goal  # + np.array([-0.792, -0.749, 0])
        unity_goal = self.goal.copy()
        unity_goal[1] = self.initial_gripper_xpos[1] - (
            unity_goal[1] - self.initial_gripper_xpos[1]
        )
        self._unity.set_geom_pos(
            "target0", unity_goal
        )  # np.array([-0.258, 0.749, 0.535])) #self.sim.model.geom_pos[geom_id])
        self._unity.set_background(self._background)

    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        ### action bias
        action[:2] = self.rot.dot(action[:2])
        action = action + self.bias

        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)


class GymFetchReach2Env(GymFetchReachEnv):
    def __init__(self, xml_path="reach_target_easy.xml", **kwargs):
        super().__init__(xml_path=xml_path, **kwargs)

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link_body")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.2
        self.viewer.cam.azimuth = 165  # 132.
        self.viewer.cam.elevation = -35  # 14.
