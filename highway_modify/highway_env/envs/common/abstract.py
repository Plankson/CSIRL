import copy
import os
from typing import List, Tuple, Optional, Callable
import gym
from gym import Wrapper
from gym.wrappers import RecordVideo
from gym.utils import seeding
import numpy as np

from highway_env import utils
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
import sys
Observation = np.ndarray

class AbstractEnv(gym.Env):

    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    observation_type: ObservationType
    action_type: ActionType
    _record_video_wrapper: Optional[RecordVideo]
    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }

    PERCEPTION_DISTANCE = 5.0 * Vehicle.MAX_SPEED
    """The maximum distance of any vehicle present in the observation [m]"""

    """my modify!"""

    def record_epoch(self):
        if self.is_record_finish():
            self.write_trajectory()
        self.expert_data["states"].append([])
        self.expert_data["actions"].append([])
        self.expert_data["next_states"].append([])
        self.expert_data["rewards"].append([])
        self.expert_data["dones"].append([])
        self.expert_data["lengths"].append(0)
        self.expert_data["d_actions"].append([])
        self.cur_epoch+=1
    def delete_epoch(self):
        del self.expert_data["states"][-1]
        del self.expert_data["actions"][-1]
        del self.expert_data["next_states"][-1]
        del self.expert_data["rewards"][-1]
        del self.expert_data["dones"][-1]
        del self.expert_data["d_actions"][-1]
        del self.expert_data["lengths"][-1]
        self.cur_epoch-=1
    def add_timestep(self,obs,next_obs,action,reward,done,d_action):
        #print(self.expert_data["lengths"][-1])
        for _ in action:
            if action[_]==1.0:
                action[_]=0.999
            elif action[_]==-1.0:
                action[_]=-0.999
        acc=action["acceleration"]
        ste=action["steering"]
        re_act=[acc,ste]
        self.expert_data["states"][-1].append(np.array(obs).flatten())
        self.expert_data["next_states"][-1].append(np.array(next_obs).flatten())
        self.expert_data["actions"][-1].append(re_act)
        self.expert_data["rewards"][-1].append(reward)
        self.expert_data["dones"][-1].append(done)
        self.expert_data["d_actions"][-1].append(d_action)
        self.expert_data["lengths"][-1] += 1
    def write_trajectory(self):
        self.expert_data["states"] = np.array(self.expert_data["states"])
        self.expert_data["actions"] = np.array(self.expert_data["actions"])
        self.expert_data["next_states"] = np.array(self.expert_data["next_states"])
        self.expert_data["rewards"] = np.array(self.expert_data["rewards"])
        self.expert_data["dones"] = np.array(self.expert_data["dones"])
        self.expert_data["lengths"] = np.array(self.expert_data["lengths"])
        self.expert_data["d_actions"] = np.array(self.expert_data["d_actions"])
        np.save(self.config["write_path"], self.expert_data, allow_pickle=True)
        self.config["is_record"] = False
        print("trajectories writing finish!")
        exit(0)

    def is_record_finish(self):
        return True if self.cur_epoch >= self.config["record_num"] else False

    def is_legal_continues(self):
        if len(self.expert_data["lengths"]) == 0:
            return True
        return True if self._legal_terminal() else False
#       return False if self.expert_data["length"][-1] != self.config["duration"] else True

    def is_legal_discrete(self):
        if len(self.expert_data["lengths"]) == 0:
            return True
        return False if self.vehicle.crashed else True
#        return False if self.expert_data["length"][-1] != self.config["duration"] else True

    def write_continues(self):        # judge if the last epoch is legel and move to the next epoch
        if not self.config["is_record"]:
            return
        if self.is_legal_continues():  # a legel trajecory that length equals to the duration
            print("now record "+str(self.cur_epoch)+" epoches")
            if self.is_record_finish():# get required number of trajectories
                self.write_trajectory()
            else:
                self.record_epoch()    #add a new trajectory
        else:
            self.delete_epoch()        # delete the illegel trajectory
            self.record_epoch()

    def __init__(self, config: dict = None) -> None:
        # Configuration
        self.config = self.default_config()
        self.configure(config)
         # Seeding
        self.np_random = None
        self.seed()
        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        #my modify!!!!
        self.cur_epoch=0
        self.expert_data = {"states": [], "actions": [], "d_actions": [], "next_states": [], "rewards": [], "dones": [], "lengths": []}
        if not self.config["is_record"]:
            self.reset()

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "simulation_frequency": 5,  # [Hz]
            "total_time": 1,
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False,
            "is_record": False
        }

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def update_metadata(self, video_real_time_ratio=10):
        frames_freq = self.config["simulation_frequency"] \
            if self._record_video_wrapper else self.config["policy_frequency"]
        self.metadata['video.frames_per_second'] = 10

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _legal_terminal(self)-> bool:
        raise  NotImplementedError
    def dis_tofinal(self) -> float:
        cur_pos = self.controlled_vehicles[0].position
        return np.linalg.norm(cur_pos-np.array(self.config["finish_position"]))
    def _info(self, obs: Observation, action: Action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "dis": self.dis_tofinal()
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass
        return info

    def _cost(self, action: Action) -> float:
        """
        A constraint metric, for budgeted MDP.

        If a constraint is defined, it must be used with an alternate reward that doesn't contain it as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def reset(self) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        if self.config["is_record"]:
            if "record_num" in self.config != False:
                self.write_continues()
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        return self.observation_type.observe()

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()



    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])

        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            # record code
            rec_obs = self.observation_type.observe()
            self.road.act()
            self.road.step(self.config["total_time"]/self.config["simulation_frequency"])
            self.time += 1
            if self.config["is_record"]:
                rec_reward = self._reward(action)
                # rec_reward = self._reward(self.controlled_vehicles[0].action)
                rec_next_obs = self.observation_type.observe()
                rec_action = self.controlled_vehicles[0].action
                rec_action["steering"]=rec_action["steering"]/(np.pi/2)
                rec_action["acceleration"]=rec_action["acceleration"]/15.0
                rec_terminal = self._is_terminal()
                self.add_timestep(rec_obs,rec_next_obs,rec_action,rec_reward,rec_terminal,action)
            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self) -> List[int]:
        return self.action_type.get_available_actions()

    def set_record_video_wrapper(self, wrapper: RecordVideo):
        self._record_video_wrapper = wrapper
        self.update_metadata()

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.
        If a RecordVideo wrapper has been set, use it to capture intermediate frames.
        """
        if self.viewer is not None and self.enable_auto_render:

            if self._record_video_wrapper and self._record_video_wrapper.video_recorder:
                self._record_video_wrapper.video_recorder.capture_frame()
            else:
                self.render(self.rendering_mode)

    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy

    def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
        """
        Change the type of all vehicles on the road

        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behavior(self) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', '_record_video_wrapper']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result


class MultiAgentWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = info["agents_rewards"]
        done = info["agents_dones"]
        return obs, reward, done, info