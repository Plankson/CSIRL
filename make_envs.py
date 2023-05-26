import gym
import highway_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

from wrappers.atari_wrapper import ScaledFloatFrame, FrameStack, PyTorchFrame
from wrappers.normalize_action_wrapper import check_and_normalize_box_actions
import envs
import numpy as np

# Register all custom envs
envs.register_custom_envs()


def make_atari(env):
    env = AtariWrapper(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    return env


def is_atari(env_name):
    return env_name in ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']


def is_highway(env_name):
    return env_name in ['highway-fast-v0']

def is_merge(env_name):
    return env_name in ['merge-v0']

def is_roundabout(env_name):
    return env_name in ['roundabout-v0','roundabout-v1']

def is_intersection(env_name):
    return env_name in ['intersection-v0']

def is_mujoco(env_name):
    return env_name in ['antmaze-umaze-v0']

class HighwayObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(HighwayObs, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(shape[0] * shape[1],), dtype=np.float32)

    def observation(self, observation):
        return observation.flatten()

def make_env(args, monitor=True):
    print(args.env.name)
    env = gym.make(args.env.name)

    if monitor:
        env = Monitor(env, "gym")

    if is_atari(args.env.name):
        env = make_atari(env)

    if is_highway(args.env.name):
        env = HighwayObs(env)
        if args.env.action_type == 'continues':
            env_config = {
                "action": {
                    "type": "ContinuousAction"
                },
                "is_record": False,
                "total_time": 0.2,
                "simulation_frequency": 1,
                "duration": 150,
                "vehicles_speed": args.env.speed,
                "vehicles_density": args.env.density
            }
            env.configure(env_config)
            env.reset()
    if is_merge(args.env.name):
        env = HighwayObs(env)
        if args.env.action_type == 'continues':
            env_config = {
                "action": {
                    "type": "ContinuousAction"
                },
                "is_record": False,
                "total_time": 0.2,
                "simulation_frequency": 1,
                "duration": 60
            }
            env.configure(env_config)
            env.reset()
    if is_roundabout(args.env.name):
        env = HighwayObs(env)
        if args.env.action_type == 'continues':
            env_config = {
                "action": {
                    "type": "ContinuousAction"
                },
                "is_record": False,
                "total_time": 0.2,
                "simulation_frequency": 1,
                "duration": 55
            }
            env.configure(env_config)
            env.reset()
    if is_intersection(args.env.name):
        env = HighwayObs(env)
        if args.env.action_type == 'continues':
            env_config = {
                "action": {
                    "type": "ContinuousAction"
                },
                "is_record": False,
                "total_time": 0.2,
                "destination": args.env.destination,
                "finish_position":None,
                "simulation_frequency": 1,
                "duration": 65
            }
            if args.env.destination=="o11":
                env_config["finish_position"]=[-45.0, -2.0]
            elif args.env.destination=="o21":
                env_config["finish_position"]=[2.0, -45.0]
            else:
                env_config["finish_position"]=[45.0, 6.0]
            env.configure(env_config)
            env.reset()
    # Normalize box actions to [-1, 1]
    env = check_and_normalize_box_actions(env)
    return env
