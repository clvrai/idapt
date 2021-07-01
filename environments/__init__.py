""" Define all custom environments. """

from gym.envs.registration import register

# Walker envs
register(
    id="GymWalkerBackwards-v0",
    entry_point="environments.gym_env:GymWalkerBackwards",
    max_episode_steps=1000,
)

register(
    id="GymWalkerRNN-v0",
    entry_point="environments.gym_env:GymWalkerRNN",
    max_episode_steps=1000,
)

register(
    id="GymWalkerEasy-v0",
    entry_point="environments.gym_env:GymWalkerEasy",
    max_episode_steps=1000,
)

register(
    id="GymWalkerDM-v0",
    entry_point="environments.gym_env:GymWalkerDM",
    max_episode_steps=1000,
)

register(
    id="GymWalkerDMVisual-v0",
    entry_point="environments.gym_env:GymWalkerDMVisual",
    max_episode_steps=1000,
)

# Inverted Pendulum envs
register(
    id="GymInvertedPendulumEasy-v0",
    entry_point="environments.gym_env:GymInvertedPendulumEasy",
    max_episode_steps=1000,
)

register(
    id="GymInvertedPendulumRNN-v0",
    entry_point="environments.gym_env:GymInvertedPendulumRNN",
    max_episode_steps=1000,
)

register(
    id="GymInvertedPendulumDM-v0",
    entry_point="environments.gym_env:GymInvertedPendulumDM",
    max_episode_steps=1000,
)

register(
    id="GymInvertedPendulumDMVisual-v0",
    entry_point="environments.gym_env:GymInvertedPendulumDMVisual",
    max_episode_steps=1000,
)

# Half Cheetah Envs
register(
    id="GymHalfCheetahEasy-v0",
    entry_point="environments.gym_env:GymHalfCheetahEasy",
    max_episode_steps=1000,
)

register(
    id="GymHalfCheetahDM-v0",
    entry_point="environments.gym_env:GymHalfCheetahDM",
    max_episode_steps=1000,
)

register(
    id="GymHalfCheetahDMVisual-v0",
    entry_point="environments.gym_env:GymHalfCheetahDMVisual",
    max_episode_steps=1000,
)

# FetchReach Envs
register(
    id="GymFetchReach-v0",
    entry_point="environments.gym_env:GymFetchReachEnv",
    kwargs={},
    max_episode_steps=50,
)

register(
    id="GymFetchReach2-v0",
    entry_point="environments.gym_env:GymFetchReach2Env",
    kwargs={},
    max_episode_steps=50,
)

# Sawyer Envs
register(
    id="SawyerPushShiftViewZoomBackground-v0",
    entry_point="environments.sawyer_env:SawyerPushShiftViewZoomBackground",
    kwargs={},
    max_episode_steps=70,
)

register(
    id="SawyerPushZoom-v0",
    entry_point="environments.sawyer_env:SawyerPushZoom",
    kwargs={},
    max_episode_steps=70,
)

register(
    id="SawyerPushZoomEasy-v0",
    entry_point="environments.sawyer_env:SawyerPushZoomEasy",
    kwargs={},
    max_episode_steps=70,
)
