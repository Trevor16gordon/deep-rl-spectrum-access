import gym
import pdb
import numpy as np
from gym import spaces

class FrequencySpectrumEnv(gym.Env):
    """Multi Agent Environment for spectrum sharing"""

    def __init__(self, num_freq_bands, num_agents, max_iterations=20, temporal_len=3, observation_type="aggregate"):
        super().__init__()
        self.num_actions = num_freq_bands + 1
        self.num_freq_bands = num_freq_bands
        self.num_agents = num_agents
        self.agents = [x for x in range(num_agents)]
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_actions)
        self.max_iterations = max_iterations
        self.temporal_len = temporal_len
        self.agent_actions_history = np.zeros((self.temporal_len, self.num_actions, self.num_agents), dtype=np.int32)
        self.observation_type = observation_type

    def step(self, action_n, verbose=False):
        
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        self.agent_actions = np.zeros((self.num_actions, self.num_agents), dtype=np.int32)

        for i, a in enumerate(action_n):
            self.agent_actions[a, i] = 1

        self.agent_actions_history = np.concatenate([
            self.agent_actions.reshape((1, self.num_actions, self.num_agents)),
            self.agent_actions_history[:-1, :, :]],
            axis=0)

        obs_n = self.agent_actions_history_to_observations(self.agent_actions_history)

        # Keep track of collisions and throughput
        transmissions = self.agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2
        self.num_collisions += np.count_nonzero(transmissions[1:] == 2)
        throughput_this_time_splot = np.count_nonzero(transmissions[1:] == 1)/ self.num_freq_bands
        new_throughput = (self.throughput*(self.iter - 1) + throughput_this_time_splot)/self.iter

        if verbose:
            print(f"self.throughout old is {self.throughput:.2f} this timeslot "
            "{throughput_this_time_splot:.2f} new_throughput {new_throughput:.2f}")
        
        self.throughput = new_throughput
        
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = -1
        reward_info[0] = 0
        reward_n = reward_info[action_n].reshape(-1)

        # Run episode for set number of iterations
        if self.iter < self.max_iterations:
            done_n = [0 for _ in range(self.num_agents)]
            self.iter += 1
        else:   
            done_n = [1 for _ in range(self.num_agents)]

        return obs_n, reward_n, done_n, info_n

    def agent_actions_history_to_observations(self, agent_actions_history):
        """Get the observation for an agent

        Args:
            agent_actions_history (_type_): Complete information about who attempted communication where. Shape is (num_time, num_freq + 1, num_agents)
        """
        if self.observation_type == "aggregate":
            observations = self._get_observation_aggregate(agent_actions_history)
        else:
            raise KeyError(f"observation type needs to be in [aggregate, ].. got {self.observation_type}")

        return observations

    def _get_observation_aggregate(self, agent_actions_history):
        """Get the observation for an agent. includes everything

        Each agent knows whether there is communication in a channel but they don't know who sent it
        Each agent has the exact same information

        # Channel states: 0 for no transmission, 1 for successful, 2 for collisions

        Args:
            agent_actions_history (_type_): Complete information about who attempted communication where. Shape is (num_time, num_freq + 1, num_agents)
        
        Returns:
            list(states): state for all agents has shape (num_agents, num_in_time, __)
        """
        agent_actions_history = agent_actions_history.sum(axis=2).reshape((self.temporal_len, -1))
        agent_actions_history[agent_actions_history > 2] = 2
        agent_actions_history = agent_actions_history[:, 1:]
        agent_actions_history = np.tile(agent_actions_history, (self.num_agents, 1, 1))
        return agent_actions_history

    def _get_observation_aggregate2(self, agent_actions_history):
        """Like _get_observation_aggregate but agents can't see the state of no_transmit action space"""
        return

    def reset(self):
        self.num_collisions = 0
        self.throughput = 0
        self.iter = 1
        obs_n = self.agent_actions_history_to_observations(self.agent_actions_history)
        return obs_n

    def render(self, mode='human', close=False):
        print("rendering")


# TODO: Abstract out different types of rewards and different types of observations



# TODO: Add ability to save image after some steps showing what agents chose