import gym
import json
import mujoco_py
import numpy as np
import os
import wandb
from collections import OrderedDict
from copy import deepcopy
from gym.envs.mujoco import mujoco_env
from mujoco_py.modder import CameraModder, LightModder, TextureModder

from training.utils.domain_randomization import (
    sample,
    sample_light_dir,
    look_at,
    jitter_angle,
    ImgTextureModder,
)
from training.utils.gym_env import (
    DictWrapper,
    FrameStackWrapper,
    GymWrapper,
    AbsorbingWrapper,
)
from training.utils.inverse_kinematics import (
    qpos_from_site_pose_sampling,
    qpos_from_site_pose,
)
from training.utils.logger import logger
from training.utils.xml_mod import XML_mod


class ActionNoiseWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        noise_level=0.0,
        noise_bias=0.0,
        ob_noise_level=0.0,
        height=100,
        width=100,
        camera_id=None,
    ):
        super().__init__(env)
        self.env = env
        self.noise_level = noise_level
        self.noise_bias = noise_bias
        self.ob_noise_level = ob_noise_level
        self._height = height
        self._width = width
        self._camera_id = camera_id

    def step(self, ac):
        new_ac = OrderedDict()
        for k, v in self.env.action_space.spaces.items():
            noise = (
                np.random.randn(gym.spaces.flatdim(v)) * self.noise_level
                + self.noise_bias
            )
            new_ac[k] = ac[k] + noise
        ob, rew, done, _ = self.env.step(new_ac)

        new_ob = OrderedDict()
        for k, v in ob.items():
            if k == "state" or self.ob_noise_level == 0:
                new_ob[k] = ob[k]
            else:
                noise = np.random.rand(*v.shape) * self.ob_noise_level
                new_ob[k] = ob[k] + noise

        return new_ob, rew, done, _


class SawyerECWrapper(gym.Wrapper):
    """
    only use on sawyer !!!
    """

    def __init__(self, env, config):
        super().__init__(env)
        self.env = env
        self._config = config
        ik_config = deepcopy(config)
        ik_config.end_effector = False
        self.ik_env = make_ik_env(ik_config.source_env, config, "source")
        self.action_space = gym.spaces.Dict(
            [("ac", gym.spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32))]
        )

    def reset(self):
        self.ik_env.reset()
        return self.env.reset()

    def step(self, ac):
        converted_ac = self._cart2joint_ac(self.env, self.ik_env, ac)
        return self.env.step(converted_ac)

    def _cart2joint_ac(self, env, ik_env, ac):
        config = self._config
        ac = np.concatenate([ac["ac"], np.array([0.0, 0.0])])

        curr_qpos = env.sim.data.qpos.copy()

        ### limit action to tabletop surface
        min_tabletop_size = [0.26, -0.4, 0]
        max_tabletop_size = [1.06, 0.4, 0]
        target_cart = np.clip(
            env.sim.data.get_site_xpos(config.ik_target)[: len(env.min_world_size)]
            + config.action_range * ac[:3],
            min_tabletop_size,
            max_tabletop_size,
        )

        # target_cart = np.clip(
        #     env.sim.data.get_site_xpos(config.ik_target)[: len(env.min_world_size)]
        #     + config.action_range * ac[:3],
        #     env.min_world_size,
        #     env.max_world_size,
        # )
        target_cart[2] = 0.95
        if len(env.min_world_size) == 2:
            target_cart = np.concatenate(
                (
                    target_cart,
                    np.array([env.sim.data.get_site_xpos(config.ik_target)[2]]),
                )
            )
        else:
            target_quat = None
        ik_env.set_state(curr_qpos.copy(), env.data.qvel.copy())
        result = qpos_from_site_pose(
            ik_env,
            config.ik_target,
            target_pos=target_cart,
            target_quat=target_quat,
            rot_weight=2.0,
            joint_names=env.robot_joints,
            max_steps=100,
            tol=1e-2,
        )
        target_qpos = env.sim.data.qpos.copy()
        target_qpos[env.ref_joint_pos_indexes] = result.qpos[
            env.ref_joint_pos_indexes
        ].copy()
        pre_converted_ac = (
            target_qpos[env.ref_joint_pos_indexes]
            - curr_qpos[env.ref_joint_pos_indexes]
        ) / 0.05
        pre_converted_ac = np.concatenate((pre_converted_ac, ac[-1:]))
        converted_ac = OrderedDict([("ac", pre_converted_ac)])
        return converted_ac


class DRWrapper(gym.Wrapper):
    def __init__(self, env, config, env_type):
        super().__init__(env)
        self.env = env
        self._config = config
        self.sim = env.sim
        self.model = self.sim.model
        with open("idapt/utils/dr_config.json") as f:
            self._params = json.load(f)[config.dr_params_set]

        # init for each modder
        if self._params["tex_mod"]["active"]:
            self.tex_modder = ImgTextureModder(
                self.sim, modes=self._params["tex_mod"]["modes"]
            )
            if self.model.mat_rgba is not None:
                self.tex_modder.whiten_materials()

        if self._params["camera_mod"]["active"]:
            self.cam_modder = CameraModder(self.sim)
            if self._params["camera_mod"]["use_default"]:
                self.unwrapped._get_viewer(mode="rgb_array")
                self.original_lookat = self.viewer.cam.lookat.copy()
            else:
                self.original_pos = self.cam_modder.get_pos(
                    self._params["camera_mod"]["cam_name"]
                ).copy()
                self.original_quat = self.cam_modder.get_quat(
                    self._params["camera_mod"]["cam_name"]
                ).copy()
                self._cam_name = self._params["camera_mod"]["cam_name"]

        if self._params["light_mod"]["active"]:
            self.light_modder = LightModder(self.sim)

        if self._params["dynamics_mod"]["active"]:
            dynamics_params = self._params["dynamics_mod"]
            if "body_mass_multiplier" in dynamics_params:
                self._original_intertia = self.model.body_inertia.copy()
                self._original_mass = self.model.body_mass.copy()
            if "armature_multiplier" in dynamics_params:
                self._original_armature = self.model.dof_armature.copy()
            if "mass_targets" in dynamics_params:
                self.inertia_mass_one = np.zeros_like(self.model.body_inertia)
                for body, param in dynamics_params["mass_targets"].items():
                    body_id = self.model.body_name2id(body)
                    # body inertia when mass is 1
                    self.inertia_mass_one[body_id, :] = (
                        self.model.body_inertia[body_id] / self.model.body_mass[body_id]
                    )

    def _rand_textures(self):
        """Rand
        omize all the textures in the scene, including the skybox"""
        for name in self._params["tex_mod"]["geoms"]:
            self.tex_modder.rand_all(name)

    def _rand_colors(self):
        if "partial_body_change" in self._params["color_mod"]:
            for part in self._params["color_mod"]["partial_body_change"]["parts"]:
                self.model.geom_rgba[part, 0] = np.random.uniform(0, 1)
                self.model.geom_rgba[part, 1] = np.random.uniform(0, 1)
                self.model.geom_rgba[part, 2] = np.random.uniform(0, 1)
        elif "geoms_change" in self._params["color_mod"]:
            for geom in self._params["color_mod"]["geoms_change"]:
                geom_id = self.model.geom_name2id(geom)
                self.model.geom_rgba[geom_id, :3] = sample([[0, 1]] * 3)
        elif "full_body_change" in self._params["color_mod"] and self._params["color_mod"]["full_body_change"]:
            self.model.geom_rgba[:, 0] = np.random.uniform(0, 1)
            self.model.geom_rgba[:, 1] = np.random.uniform(0, 1)
            self.model.geom_rgba[:, 2] = np.random.uniform(0, 1)
        else:
            a = self.model.geom_rgba.T[3, :].copy()
            self.model.geom_rgba[:] = np.random.rand(*self.model.geom_rgba.shape)
            self.model.geom_rgba.T[3, :] = a

    def _rand_camera(self):
        # Params
        cam_params = self._params["camera_mod"]
        if cam_params["use_default"]:
            # when default free camera is used
            # only work with rgb_array mode
            self.model.vis.global_.fovy = np.random.uniform(*cam_params["fovy_range"])
            self.unwrapped._viewers["rgb_array"].__init__(self.sim, -1)
            try:
                self.viewer_setup()
            except:
                self.unwrapped._viewer_setup()
            for key, value in cam_params["veiwer_cam_param_targets"].items():
                if isinstance(value, np.ndarray):
                    getattr(self.viewer.cam, key)[:] = sample(value)
                else:
                    setattr(self.viewer.cam, key, np.random.uniform(*value))
        else:
            # when camera is defined
            # Look approximately at the robot, but then randomize the orientation around that
            cam_pos = self.original_pos + sample(cam_params["pos_change_range"])
            self.cam_modder.set_pos(cam_params["cam_name"], cam_pos)
            self.sim.set_constants()

            quat = self.original_quat
            if "camera_focus" in cam_params:
                cam_id = self.cam_modder.get_camid(cam_params["cam_name"])
                target_id = self.model.body_name2id(cam_params["camera_focus"])
                quat = look_at(
                    self.model.cam_pos[cam_id], self.sim.data.body_xpos[target_id]
                )
            if "ang_jitter_range" in cam_params:
                quat = jitter_angle(quat, cam_params["ang_jitter_range"])

            self.cam_modder.set_quat(cam_params["cam_name"], quat)

            self.cam_modder.set_fovy(
                cam_params["cam_name"], np.random.uniform(*cam_params["fovy_range"])
            )

    def _rand_lights(self):
        """Randomize pos, direction, and lights"""
        # adjesting user defined lights
        light_params = self._params["light_mod"]
        if self.model.light_pos is not None:
            # pick light that is guaranteed to be on
            # other lights has 20% chance to be turned off
            always_on = np.random.choice(len(self.model.light_pos))
            for lightid in range(len(self.model.light_pos)):
                self.model.light_dir[lightid] = sample_light_dir()
                self.model.light_pos[lightid] = sample(light_params["pos_range"])

                if "color_range" in light_params:
                    color = np.array(sample(light_params["color_range"]))
                else:
                    color = np.ones(3)
                spec = np.random.uniform(*light_params["spec_range"])
                diffuse = np.random.uniform(*light_params["diffuse_range"])
                ambient = np.random.uniform(*light_params["ambient_range"])

                self.model.light_specular[lightid] = spec * color
                self.model.light_diffuse[lightid] = diffuse * color
                self.model.light_ambient[lightid] = ambient * color
                self.model.light_castshadow[lightid] = np.random.uniform(0, 1) < 0.5
        if light_params["head_light"]:
            if "color_range" in light_params:
                color = np.array(sample(light_params["color_range"]))
            else:
                color = np.ones(3)
            spec = np.random.uniform(*light_params["spec_range"])
            diffuse = np.random.uniform(*light_params["diffuse_range"])
            ambient = np.random.uniform(*light_params["ambient_range"])
            # adjust headlight
            self.model.vis.headlight.diffuse[:] = spec * color
            self.model.vis.headlight.ambient[:] = diffuse * color
            self.model.vis.headlight.specular[:] = diffuse * color

    def _rand_dynamics(self):
        dynamics_params = self._params["dynamics_mod"]
        if "action_mod" in dynamics_params:
            theta_degree = np.random.uniform(*dynamics_params["action_mod"]["theta"])
            theta = theta_degree * np.pi / 180  # pi/2 = 90 deg
            c, s = np.cos(theta), np.sin(theta)
            self.unwrapped.rot = np.array(((c, -s), (s, c)))
            self.unwrapped.bias = np.zeros(4)
            self.unwrapped.bias[2] = np.random.uniform(
                *dynamics_params["action_mod"]["bias"]
            )

        if "armature_multiplier" in dynamics_params:
            multiplier = np.random.uniform(*dynamics_params["armature_multiplier"])
            self.model.dof_armature[:] = self._original_armature * multiplier

        if "friction_targets" in dynamics_params:
            for geom, param in dynamics_params["friction_targets"].items():
                new_friction = param["range"][0] + np.random.rand() * (
                    param["range"][1] - param["range"][0]
                )
                geom_id = self.model.geom_name2id(geom)
                self.model.geom_friction[geom_id][0] = new_friction

        if "mass_targets" in dynamics_params:
            for body, param in dynamics_params["mass_targets"].items():
                new_mass = param["range"][0] + np.random.rand() * (
                    param["range"][1] - param["range"][0]
                )
                body_id = self.model.body_name2id(body)
                self.model.body_mass[body_id] = new_mass
                if param["inerta_reset"]:
                    self.model.body_inertia[body_id, :] = (
                        self.inertia_mass_one[body_id] * new_mass
                    )

        self.sim.set_constants()

    def reset(self):
        if self._params["tex_mod"]["active"]:
            self._rand_textures()
        if self._params["camera_mod"]["active"]:
            self._rand_camera()
        if self._params["light_mod"]["active"]:
            self._rand_lights()
        if self._params["color_mod"]["active"]:
            self._rand_colors()
        if self._params["dynamics_mod"]["active"]:
            self._rand_dynamics()
        return self.env.reset()


class SettableStateWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        from_pixels=False,
        height=100,
        width=100,
        camera_id=None,
        channels_first=True,
    ):
        super().__init__(env)
        self.env = env
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._channels_first = channels_first

    ### Make state settable for testing image/action translations ###

    def set_state_from_ob(self, qpos, qvel, goal=None, reset=False):
        # InvertedPendulum

        if goal is not None:
            self.set_state(qpos, qvel, goal=goal)
        else:
            self.set_state(qpos, qvel)

        if self._from_pixels:
            ob = self.render(
                mode="rgb_array",
                height=self._height,
                width=self._width,
                camera_id=self._camera_id,
            )
            if reset:
                ob = self.render(
                    mode="rgb_array",
                    height=self._height,
                    width=self._width,
                    camera_id=self._camera_id,
                )
            if self._channels_first:
                ob = ob.transpose(2, 0, 1).copy()
        else:
            raise NotImplementedError
        return ob

    def step_from_state(self, qpos, qvel, action):
        self.set_state(qpos, qvel)
        self.env.step(action)
        return self.env.sim.data.qpos.copy(), self.env.sim.data.qvel.copy()


def set_params(env, config):
    xml = XML_mod(env.model.get_xml())
    xml.mod_xml(config.param_mod_instructions)
    path = os.path.dirname(__file__) + "/temp.xml"
    xml.save_xml(path)
    mujoco_env.MujocoEnv.__init__(env.unwrapped, path, 4)
    os.remove(path)
    return env


def make_basic_env(name, config=None):
    """
    Creates a new environment instance with @name and @config.
    """
    # get default config if not provided
    if config is None:
        from config import create_parser

        config, unparsed = create_parser().parse_args()

    return get_gym_env(name, config)


def get_gym_env(env_id, config):
    env_kwargs = config.__dict__.copy()
    try:
        env = gym.make(env_id, **env_kwargs)
    except Exception as e:
        logger.warn("Failed to launch an environment with config.")
        logger.warn(e)
        logger.warn("Launch an environment without config.")
        env = gym.make(env_id)
    env.seed(config.seed)
    env = GymWrapper(
        env=env,
        from_pixels=(config.encoder_type == "cnn"),
        height=config.screen_height,
        width=config.screen_width,
        channels_first=True,
        frame_skip=config.action_repeat,
        return_state=(config.encoder_type == "cnn"),
    )

    env = DictWrapper(
        env, return_state=(config.encoder_type == "cnn")
    )
    if config.encoder_type == "cnn":
        env = FrameStackWrapper(
            env,
            frame_stack=3,
            return_state=(config.encoder_type == "cnn"),
        )
    if config.absorbing_state:
        env = AbsorbingWrapper(env)

    return env


def make_noisey_env(env, config, env_type):
    if config.algo == "MATL":
        config.env_type = env_type
    out = make_basic_env(env, config)
    if env_type == "source":
        return ActionNoiseWrapper(
            out,
            noise_bias=config.source_noise_bias,
            noise_level=config.source_noise_level,
            ob_noise_level=config.source_ob_noise_level,
        )
    else:
        return ActionNoiseWrapper(
            out,
            noise_bias=config.target_noise_bias,
            noise_level=config.target_noise_level,
            ob_noise_level=config.target_ob_noise_level,
        )


def make_env(env, config, env_type):
    """
    separate source or target specific arg names which look like "[env]_env_[arg_name]"
    and set up corresponding envs accordingly
    Also add wrappers 
    """
    env_config = deepcopy(config)
    arg_identifier = env_type + "_env_"
    for k, v in env_config.__dict__.items():
        if k.startswith(arg_identifier):
            env_config.__dict__[k[len(arg_identifier) :]] = v

    if env_type != "target":
        env_config.__dict__["unity"] = False

    out = make_basic_env(env, env_config)
    if config.mod_env_params and env_type == "target":
        out = set_params(out, config)

    if config.dr and env_type == "source":
        out = DRWrapper(out, config, env_type)

    if "SawyerPush" in env:
        if config.end_effector:
            out = SawyerECWrapper(out, config)

    if env_type == "source":
        out = ActionNoiseWrapper(
            out,
            noise_bias=config.source_noise_bias,
            noise_level=config.source_noise_level,
            ob_noise_level=config.source_ob_noise_level,
        )
    else:
        out = ActionNoiseWrapper(
            out,
            noise_bias=config.target_noise_bias,
            noise_level=config.target_noise_level,
            ob_noise_level=config.target_ob_noise_level,
        )

    out = SettableStateWrapper(
        out,
        from_pixels=(config.encoder_type == "cnn"),
        height=config.screen_height,
        width=config.screen_width,
    )

    return out


def make_ik_env(env, config, env_type):
    """
    ik_env_used in ECWrapper
    """
    env_config = deepcopy(config)
    arg_identifier = env_type + "_env_"
    for k, v in env_config.__dict__.items():
        if k.startswith(arg_identifier):
            env_config.__dict__[k[len(arg_identifier) :]] = v
    env_config.dr = False
    env_config.__dict__["unity"] = False
    out = make_basic_env(env, env_config)

    return out
