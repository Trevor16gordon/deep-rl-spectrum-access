import gym
import numpy as np
from gym import spaces

class FrequencySpectrumEnv(gym.Env):
    """Multi Agent Environment for spectrum sharing"""

    def __init__(self, num_freq_bands, num_agents, max_iterations=20):
        super().__init__()
        self.num_actions = num_freq_bands + 1
        self.num_freq_bands = num_freq_bands
        self.num_agents = num_agents
        self.agents = [x for x in range(num_agents)]
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_actions)
        self.max_iterations = max_iterations

    def step(self, action_n, verbose=False):
        
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        self.agent_actions = np.zeros((self.num_actions, self.num_agents), dtype=np.int32)
        for i, a in enumerate(action_n):
            self.agent_actions[a, i] = 1

        # Channel states: 0 for no transmission, 1 for successful, 2 for collisions
        self.channel_state = self.agent_actions.sum(axis=1).reshape((-1, 1))
        self.channel_state[self.channel_state > 2] = 2

        # Keep track of collisions and throughput
        self.num_collisions += np.count_nonzero(self.channel_state[1:] == 2)
        throughput_this_time_splot = np.count_nonzero(self.channel_state[1:] == 1)/ self.num_freq_bands
        new_throughput = (self.throughput*(self.iter - 1) + throughput_this_time_splot)/self.iter

        if verbose:
            print(f"self.throughout old is {self.throughput:.2f} this timeslot "
            "{throughput_this_time_splot:.2f} new_throughput {new_throughput:.2f}")
        
        self.throughput = new_throughput
        
        reward_info = self.channel_state.copy()
        reward_info[reward_info >= 2] = 0
        reward_info[0] = 0
        reward_n = reward_info[action_n].reshape(-1)

        # Run episode for set number of iterations
        if self.iter < self.max_iterations:
            done_n = [0 for _ in range(self.num_agents)]
            self.iter += 1
        else:   
            done_n = [1 for _ in range(self.num_agents)]

        obs_n = self.channel_state.reshape(-1)[1:]

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.num_collisions = 0
        self.throughput = 0
        self.iter = 1
        return np.zeros(self.num_freq_bands)

    def render(self, mode='human', close=False):
        print("rendering")


# TODO: Abstract out different types of rewards and different types of observations