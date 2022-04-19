import gym
import pdb
import numpy as np
from gym import spaces

class FrequencySpectrumEnv(gym.Env):
    """Multi Agent Environment for spectrum sharing"""

    def __init__(self,
                 num_freq_bands,
                 num_agents,
                 max_iterations=10000,
                 temporal_len=3,
                 observation_type="aggregate",
                 reward_type="transmission1",
                 reward_history_len=100):
        super().__init__()
        self.num_actions = num_freq_bands + 1
        self.num_freq_bands = num_freq_bands
        self.num_agents = num_agents
        self.agents = [x for x in range(num_agents)]
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_actions)
        self.max_iterations = max_iterations
        self.temporal_len = temporal_len
        self.agent_actions_history = np.zeros(
            (self.temporal_len, self.num_actions, self.num_agents), dtype=np.int32)
        self.observation_type = observation_type
        self.reward_type = reward_type
        self.agent_actions_complete_history = []
        self.reward_history_len = reward_history_len
        self.reward_hist = np.zeros((self.reward_history_len, self.num_agents))
        self.reward_hist_pointer = 0

    def step(self, action_n, verbose=False):
        
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        
        self.agent_actions = np.zeros((self.num_actions, self.num_agents), dtype=np.int32)
        self.agent_actions[action_n, np.arange(self.num_agents)] = 1
        

        self.agent_actions_complete_history.append(action_n)

        # for i, a in enumerate(action_n):
        #     self.agent_actions[a, i] = 1

        self.agent_actions_history = np.concatenate([
            self.agent_actions.reshape((1, self.num_actions, self.num_agents)),
            self.agent_actions_history[:-1, :, :]],
            axis=0)

        obs_n = self.agent_actions_history_to_observations(self.agent_actions_history)

        

        # Keep track of collisions and throughput
        transmissions = self.agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2

        # Start keeping track of rewards / collisions for purpose of modifying rewards
        agent_status = np.copy(transmissions)
        agent_status[0] = 0
        self.reward_hist[self.reward_hist_pointer, :] = agent_status[action_n]
        self.reward_hist_pointer += 1
        self.reward_hist_pointer = (self.reward_hist_pointer % self.reward_history_len)
        # End section

        self.num_collisions += np.count_nonzero(transmissions[1:] == 2)
        throughput_this_time_splot = np.count_nonzero(transmissions[1:] == 1)/ self.num_freq_bands
        new_throughput = (self.throughput*(self.iter - 1) + throughput_this_time_splot)/self.iter

        if verbose:
            print(f"self.throughout old is {self.throughput:.2f} this timeslot "
            "{throughput_this_time_splot:.2f} new_throughput {new_throughput:.2f}")
        
        self.throughput = new_throughput

        reward_n = self.agent_actions_to_rewards(self.agent_actions, action_n)

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
        elif self.observation_type == "aggregate2":
            observations = self._get_observation_aggregate2(agent_actions_history)
        elif self.observation_type == "aggregate3":
            observations = self._get_observation_aggregate3(agent_actions_history)
        elif self.observation_type == "own_actions":
            observations = self._get_observation_own_actions(agent_actions_history)
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
        obvs = agent_actions_history.sum(axis=2).reshape((self.temporal_len, -1))
        obvs[obvs > 2] = 2
        obvs = obvs[:, 1:]
        obvs = np.tile(obvs, (self.num_agents, 1, 1))
        return obvs

    def _get_observation_is_channel_available(self, agent_actions_history):
        """Channels available if no transmit or collision

        # Channel states: 1 for no transmission, 0 for channel occupied, 1 for collisions

        Args:
            agent_actions_history (_type_): Complete information about who attempted communication where. Shape is (num_time, num_freq + 1, num_agents)
        
        Returns:
            list(states): state for all agents has shape (num_agents, num_in_time, __)
        """
        obvs = agent_actions_history.sum(axis=2).reshape((self.temporal_len, -1))
        # Results in -1 ACK for collision
        # obvs[obvs >= 2] = 2

        # For collisions we'll just say channel busy
        obvs[obvs >= 2] = 1


        obvs = obvs[:, 1:]
        # obvs = 1 - obvs
        obvs = np.tile(obvs, (self.num_agents, 1, 1))
        return obvs

    def _get_observation_aggregate2(self, agent_actions_history):
        """Like _get_observation_aggregate but agents can't see the state of no_transmit action space"""
        obvs1 = self._get_observation_aggregate(agent_actions_history)
        obvs2 = self._get_observation_own_actions(agent_actions_history)
        obvs = np.concatenate([obvs1, obvs2], axis=2)
        return obvs

    def _get_observation_aggregate3(self, agent_actions_history):
        """Like _get_observation_aggregate but agents can't see the state of no_transmit action space"""
        obvs1 = self._get_observation_is_channel_available(agent_actions_history)
        obvs2 = self._get_observation_own_actions(agent_actions_history)
        
        # Collisions fail so NACK will be -1 and ACK will be 1. If we didn't transmit, leave at 0
        obvs2[obvs2 >= 2] = -1
        obvs = np.concatenate([obvs1, obvs2], axis=2)
        return obvs

    def _get_observation_own_actions(self, agent_actions_history):
        """Get the observation for an agent

        Each agent only knows it's own chosen action and whether it got a reward

        Args:
            agent_actions_history (_type_): Complete information about who attempted communication where. Shape is (num_time, num_freq + 1, num_agents)
        
        Returns:
            list(states): state for all agents has shape (num_agents, num_in_time, __)
        """
        channel_usage = agent_actions_history.sum(axis=2)
        channel_usage = channel_usage[:, :, np.newaxis]
        # Not possible to have collisions in the no transmit state
        channel_usage[:, 0, 0] = 0

        #TODO: Rewards shouldn't be calculated here..
        # agent_actions_history only has a 1 in each row. So elementwise multiplication picks out successful transmissions or collisions
        # and then summing along that access just pulls only that element
        rewards = np.sum(agent_actions_history * channel_usage, axis=1)
        rewards = rewards[:, np.newaxis, :]
        obvs = np.concatenate([agent_actions_history, rewards], axis=1)
        obvs = np.swapaxes(obvs, 0, 1)
        obvs = np.swapaxes(obvs, 0, 2)
        
        return obvs

    def agent_actions_to_rewards(self, agent_actions, action_ints):
        """Get the rewards for an agent

        Args:
            agent_actions (_type_): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
        """
        if self.reward_type == "transmission1":
            observations = self._get_reward_transmission1(agent_actions, action_ints)
        elif self.reward_type == "collisionpenality1":
            observations = self._get_reward_collisionpenality1(agent_actions, action_ints)
        elif self.reward_type == "collisionpenality2":
            observations = self._get_reward_collisionpenality2(agent_actions, action_ints)
        elif self.reward_type == "centralized":
            observations = self._get_reward_centralized(agent_actions, action_ints)
        elif self.reward_type == "transmision_normalized":
            observations = self._get_reward_transmision_normalized(agent_actions, action_ints)
        else:
            raise KeyError(f"observation type needs to be in [aggregate, ].. got {self.reward_type}")

        return observations

    def _get_reward_transmission1(self, agent_actions, action_ints):
        """Get the reward for an agent.

        Positive 1 rewards for a successful transmission

        Args:
            agent_actions (_type_): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (_type_): Int representation of what agents chose Shape is (num_agents)
        
        Returns:
            list(states): rewards for all agents has shape (num_agents, 1)
        """
        transmissions = agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = 0
        reward_info[0] = 0
        reward_n = reward_info[action_ints].reshape(-1)
        return reward_n

    def _get_reward_transmision_normalized(self, agent_actions, action_ints):
        """Get the reward for an agent.

        Rewards are divided by the number of successful transmissions in the last temporal sequence
        Intended to make transmissions more valuable to those who haven't gotten them
        And to make transmissions less valuable to those who have been sending

        Args:
            agent_actions (_type_): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (_type_): Int representation of what agents chose Shape is (num_agents)
        
        Returns:
            list(states): rewards for all agents has shape (num_agents, 1)
        """

        # #TODO: This function uses self.agent_actions_history which might not be clear as it isn't normally passed in.
        # channel_usage = self.agent_actions_history.sum(axis=2)
        # channel_usage = channel_usage[:, :, np.newaxis]
        # # Not possible to have collisions in the no transmit state
        # channel_usage[:, 0, 0] = 0

        # #TODO: Rewards shouldn't be calculated here..
        # # agent_actions_history only has a 1 in each row. So elementwise multiplication picks out successful transmissions or collisions
        # # and then summing along that access just pulls only that element
        # rewards_history = np.sum(self.agent_actions_history * channel_usage, axis=1)
        # rewards_history = rewards_history[:, np.newaxis, :]
        # successful_trans = (rewards_history == 1).sum(axis=0)
        # collisions = (rewards_history == 2).sum(axis=0)

        # successful_trans_norm = successful_trans
        # # Can't divide by zero
        # successful_trans_norm[successful_trans_norm<1] = 1
        # successful_trans_norm = successful_trans_norm.reshape(-1)
        

        transmissions = agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = -1
        reward_info[reward_info == 1] = 2
        reward_info[0] = 0
        reward_n = reward_info[action_ints].reshape(-1)


        successes = (self.reward_hist == 1).sum(axis=0)
        collisions = (self.reward_hist == 2).sum(axis=0)
        no_trans = (self.reward_hist == 0).sum(axis=0)

        # Rewards are multiplied number of (no transmits / 100)
        # Rewards are divided by number of (sucesses / 100)

        reward_n = reward_n.astype(np.float32)
        # Positive reward modification
        pos_mul = (no_trans / self.reward_history_len) / (1 + (successes / self.reward_history_len))
        neg_mul = (successes / self.reward_history_len) / (1 + (no_trans / self.reward_history_len))

        # Pull values from pos or neg multiplier based on if reward is pos or neg
        indexer = np.arange(self.num_agents),(reward_n < 0).astype(np.int)
        multiplier = np.concatenate([pos_mul.reshape((-1, 1)), neg_mul.reshape((-1, 1))], axis=1)[indexer]
        reward_n *= multiplier


        return reward_n

    def _get_reward_collisionpenality1(self, agent_actions, action_ints):
        """Get the reward for an agent.

        +1 rewards for a successful transmission
        0 for no transmission
        -1 rewards for a collision

        Args:
            agent_actions (_type_): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (_type_): Int representation of what agents chose Shape is (num_agents)
        
        Returns:
            list(states): rewards for all agents has shape (num_agents, 1)
        """
        transmissions = agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = -1
        reward_info[0] = 0
        reward_n = reward_info[action_ints].reshape(-1)
        return reward_n

    def _get_reward_collisionpenality2(self, agent_actions, action_ints):
        """Get the reward for an agent.

        +2 rewards for a successful transmission
        0 for no transmission
        -1 rewards for a collision

        Args:
            agent_actions (_type_): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (_type_): Int representation of what agents chose Shape is (num_agents)
        
        Returns:
            list(states): rewards for all agents has shape (num_agents, 1)
        """
        transmissions = agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = -1
        reward_info[reward_info == 1] = 2
        reward_info[0] = 0
        reward_n = reward_info[action_ints].reshape(-1)
        return reward_n

    def _get_reward_centralized(self, agent_actions, action_ints):
        """Get the reward for an agent.

        Rewards for all agents

        Args:
            agent_actions (_type_): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (_type_): Int representation of what agents chose Shape is (num_agents)
        
        Returns:
            list(states): rewards for all agents has shape (num_agents, 1)
        """
        transmissions = agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = 0
        reward_info[0] = 0
        trans_total = reward_info.sum(axis=0)
        reward_n = reward_info[action_ints].reshape(-1)
        reward_n[:] = trans_total
        return reward_n

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