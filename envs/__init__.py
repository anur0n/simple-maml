from gym.envs.registration import register

register(
    'HalfCheetahVel-v2',
    entry_point='envs.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.half_cheetah:HalfCheetahVelEnv'}
)

register(
    'HalfCheetahDir-v2',
    entry_point='envs.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'envs.half_cheetah:HalfCheetahDirEnv'}
)