import mujoco_py.modder
import numpy as np
import quaternion
from PIL import Image
from mujoco_py import functions
from mujoco_py.modder import Texture

normalize = lambda x: x / np.linalg.norm(x)


# MATH UTILS
def look_at(from_pos, to_pos):
    """
    Compute quaternion to point from `from_pos` to `to_pos`
    We define this ourselves, rather than using Mujoco's body tracker,
    because it makes it easier to randomize without relying on calling forward() 

    Reference: https://stackoverflow.com/questions/10635947/what-exactly-is-the-up-vector-in-opengls-lookat-function
    """
    up = np.array([0, 0, 1])  # I guess this is who Mujoco does it
    n = normalize(from_pos - to_pos)
    u = normalize(np.cross(up, n))
    v = np.cross(n, u)
    mat = np.stack([u, v, n], axis=1).flatten()
    quat = np.zeros(4)
    functions.mju_mat2Quat(
        quat, mat
    )  # this can be replaced with np.quaternion something if we need
    return quat


def sample_quat(angle3):
    """Sample a quaterion from a range of euler angles in degrees"""
    roll = np.random.uniform(*angle3[0]) * np.pi / 180
    pitch = np.random.uniform(*angle3[1]) * np.pi / 180
    yaw = np.random.uniform(*angle3[2]) * np.pi / 180

    quat = quaternion.from_euler_angles(roll, pitch, yaw)
    return quat.normalized().components


def jitter_angle(quat, angle3):
    """Jitter quat with an angle range"""
    sampled = sample_quat(angle3)
    return (np.quaternion(*quat) * np.quaternion(*sampled)).normalized().components


def sample(sample_range):
    """
    @input: tuple of sampling range
    @out: tuple of sampled values
    """
    return tuple(np.random.uniform(*r) for r in sample_range)


def sample_light_dir():
    """Sample a random direction for a light. I don't quite understand light dirs so
    this might be wrong"""
    # Pretty sure light_dir is just the xyz of a quat with w = 0.
    # I random sample -1 to 1 for xyz, normalize the quat, and then set the tuple (xyz) as the dir
    LIGHT_DIR = ((-1, 1), (-1, 1), (-1, 1))
    return np.quaternion(0, *sample(LIGHT_DIR)).normalized().components.tolist()[1:]


class ImgTextureModder(mujoco_py.modder.TextureModder):
    """
    from https://github.com/matwilso/domrand
    """

    def __init__(self, *args, modes=[], **kwargs):
        super().__init__(*args, **kwargs)

        self.textures = [Texture(self.model, i) for i in range(self.model.ntex)]
        self._build_tex_geom_map()
        self._img_paths = [
            "idapt/utils/texutures/wood.jpg",
            "idapt/utils/texutures/wall.jpg",
            "idapt/utils/texutures/earth.jpg",
            "idapt/utils/texutures/metal.jpg",
        ]
        self._imgs = [Image.open(path) for path in self._img_paths]

        name_to_rand_method = {
            "checker": self.rand_checker,
            "imgs": self.rand_imgs,
            "gradient": self.rand_gradient,
            "rgb": self.rand_rgb,
            "noise": self.rand_noise,
        }
        self._choices = []
        for mode in modes:
            self._choices.append(name_to_rand_method[mode])

    def rand_all(self, name):
        choice = self.random_state.randint(len(self._choices))
        # when all images are used then use gradient
        return self._choices[choice](name)

    def rand_imgs(self, name):
        choice = self.random_state.randint(len(self._imgs))
        img_texture = self._imgs[choice]
        return self.set_imgs(name, img_texture)

    def set_imgs(self, name, img_texture):
        bitmap = self.get_texture(name).bitmap
        width, height, _ = bitmap.shape
        img_texture_array = np.array(
            img_texture.resize((height, width), Image.ANTIALIAS)
        )
        bitmap[:, :, :] = img_texture_array

        self.upload_texture(name)
        return bitmap

    def _cache_checker_matrices(self):
        """
        Cache two matrices of the form [[1, 0, 1, ...],
                                        [0, 1, 0, ...],
                                        ...]
        and                            [[0, 1, 0, ...],
                                        [1, 0, 1, ...],
                                        ...]
        for each texture. To use for fast creation of checkerboard patterns
        """
        if self.model.mat_texid is not None:
            self._geom_checker_mats = []
            for geom_id in range(self.model.ngeom):
                mat_id = self.model.geom_matid[geom_id]
                tex_id = self.model.mat_texid[mat_id]
                texture = self.textures[tex_id]
                h, w = texture.bitmap.shape[:2]
                self._geom_checker_mats.append(self._make_checker_matrices(h, w))

        # add skybox
        skybox_tex_id = -1
        for tex_id in range(self.model.ntex):
            skybox_textype = 2
            if self.model.tex_type[tex_id] == skybox_textype:
                skybox_tex_id = tex_id
        if skybox_tex_id >= 0:
            texture = self.textures[skybox_tex_id]
            h, w = texture.bitmap.shape[:2]
            self._skybox_checker_mat = self._make_checker_matrices(h, w)
        else:
            self._skybox_checker_mat = None
