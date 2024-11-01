import torch
from torch import Tensor
from gymnasium import spaces
import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict, List, OrderedDict
from torch.distributions import Categorical
from pydantic import Field

from agents.configs.configs import BaseConfig

class GridWorldConfig(BaseConfig):
    ''' Grid World Config'''
    grid_size: int = Field(
        default=4,
        description='Size of grid world environment'
    )
    goal_position: Tuple[int, int] = Field(
        default=(1, 1),
        description='tuple position of the end goal in grid world'
    )
    num_orientations: int = Field(
        default=4,
        description='Number of ways that agent can spin'
    )
    context: Any = Field(
        default=None,
        description='Any extra context for an environment'
    )
    render_mode: str = Field(
        default=None,
        description='Render mode for environment'
    )


class VectorizedGridWorld(gym.Env):
    '''Vectorized version of custom grid world with spins'''
    def __init__(self,
                 grid_size: int = 4,
                 goal_position: Tuple[int, int] = (1, 1),
                 num_orientations: int = 4,
                 action_space: Any = None,
                 context: Any = None,
                 render_mode: str = None, 
                 ):
        super().__init__()
        self.grid_size = grid_size
        self.num_orientations = num_orientations
        # action space: move forward or turn left
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                'x': spaces.Discrete(self.grid_size),
                'y': spaces.Discrete(self.grid_size),
                'orientation': spaces.Discrete(self.num_orientations)
            }
        )
        self.goal_position = goal_position
        self.context = context
        self.render_mode = render_mode
        self.state = None
        self._environment_info = {
            'info': f'Grid is size {self.grid_size}, goal position is at {self.goal_position}'}

        self.coordinate_set_probability = np.ones(
            self.grid_size) / self.grid_size
        self.orientation_set_probability = np.ones(
            self.num_orientations) / self.num_orientations

    @property
    def coordinate_space_distro(self) -> Categorical:
        coord_space_prob = torch.from_numpy(self.coordinate_set_probability)
        return torch.distributions.Categorical(coord_space_prob)

    @property
    def orientation_space_distro(self) -> Categorical:
        orientation_space_prob = torch.from_numpy(
            self.orientation_set_probability)
        return torch.distributions.Categorical(orientation_space_prob)

    @property
    def environment_info(self):
        return self._environment_info

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        self.state = {'x': 0, 'y': 0, 'orientation': 0}
        return self.state, {'info': f'Grid is size {self.grid_size}, goal position is at {self.goal_position}'}

    def step(self, actions: Any) -> Tuple[Dict[str, int], float, bool, bool, Dict[str, str]]:
        total_reward = 0
        for action in actions:
            assert action in [
                0, 1, 1000], 'Invalid Action: Must Choose from [0, 1, 1000]'
            x, y, orientation = self.state['x'], self.state['y'], self.state['orientation']
            if action == 0:  # move forward
                if orientation == 0:  # facing up
                    y -= 1
                elif orientation == 1:  # facing right
                    x += 1
                elif orientation == 2:  # facing down
                    y += 1
                elif orientation == 3:  # facing left
                    x -= 1
            elif action == 1:  # turn left
                # a % b = a - floor(a / b) * b
                orientation = (orientation - 1) % 4
            elif action == 1000:  # special action just does nothing FOR ERROR HANDLING
                continue

            # Enforce boundary conditions
            x = max(0, min(x, self.grid_size - 1))
            y = max(0, min(y, self.grid_size - 1))

            self.state = {'x': x, 'y': y, 'orientation': orientation}
            total_reward += self._compute_reward()

        done = (self.state['x'], self.state['y']) == (0, 0)
        truncated = False  # set your own condition for truncated if needed

        return (self.state, 
                total_reward, 
                done, 
                truncated, 
                {'info': f'Grid is size {self.grid_size}, goal position is at {self.goal_position}'})

    @staticmethod
    def create_states_from_sampled_states(sampled_states: Tensor) -> List[Dict[str, int]]:
        '''Takes a sampled state tensor of shape [num_samples, 3] and returns '''
        assert sampled_states.shape[
            1] == 3, f'Sampled states tensor must have (x, y, orientation) per row'
        states = [
            {'x': x, 'y': y, 'orientation': orientation}
            for (x, y, orientation) in sampled_states
        ]
        return states

    @staticmethod
    def process_info(grid_size: int | list[int],
                     goal_position: Tuple[int, int] | list[Tuple[int, int]]
                     ) -> Dict[str, str] | List[Dict[str, str]]:
        '''Helper function that compiles a string prompt from the information coordinates'''

        if isinstance(grid_size, list):
            assert isinstance(
                goal_position, list), f'Goal position must also be a list if grid size is list for batching'
            ...
        else:
            assert isinstance(
                goal_position, Tuple), f'Goal position must be a tuple if grid size is an int'
            ...
        return {'info': f'Grid is size {grid_size}, goal position is at {goal_position}'}

    def _compute_reward(self) -> int:
        if (self.state['x'], self.state['y']) == self.goal_position:
            return 100
        else:
            return -1  # penalty if not finished


def process_environment_outputs(observations: OrderedDict,
                                infos: Dict[str, np.ndarray]
                                ) -> list[Dict[str, Any]]:
    '''Process function that returns a list of observations and infos from environment'''

    obs_list = []
    info_list = []
    for idx, (_, info) in enumerate(zip(observations['orientation'], infos['info'])):
        obs_element = {key: value[idx] for key, value in observations.items()}
        info_element = {'info': info}

        obs_list.append(str(obs_element))
        info_list.append(str(info_element))

    return obs_list, info_list



if __name__ == "__main__":

    from lactchain.models.actor import ActorConfig, LactChain, LoraConfigSettings
    from lactchain.models.critic import ValueFunction, ValueFunctionConfig

   # ACTOR_PATH='/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de'
   # actor_config=ActorConfig()
   # lora_config=LoraConfigSettings()
   # actor = LightningLactChain(ACTOR_PATH, actor_config, lora_config)

    # CRITIC_PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/models/models--Salesforce--SFR-Embedding-Mistral/snapshots/938c560d1c236aa563b2dbdf084f28ab28bccb11"
    # critic_config=ValueFunctionConfig()
    # critic = LightningValueFunction(CRITIC_PATH, critic_config)

    out = VectorizedGridWorld()
    distro = out.orientation_set_distro
    other_distro = out.coordinate_space_distro
    num_envs = 64  # Number of environments to run in parallel
    async_vector_env = gym.vector.AsyncVectorEnv(
        [make_env for _ in range(num_envs)])

    vect_obs, vect_info = async_vector_env.reset()
    obs, info = process_environment_outputs(vect_obs, vect_info)

    for unit in obs:
        print(f'Vector Environment Observations: {unit}')

   # obs, info=process_environment_outputs(vect_obs, vect_info)
   # mapped_actions, actions, contexts=actor.sample_actions(obs, info)
   # next_obs, reward, done, truncated, info = async_vector_env.step(mapped_actions)

    # actions=[np.array([0, 1]), np.array([1, 0, 1, 0, 1, 0]), np.array([0, 1]), np.array([1, 0])]
    # next_obs, reward, done, truncated, info = async_vector_env.step(actions)
