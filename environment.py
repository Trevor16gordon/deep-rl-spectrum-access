""" 
Gym Environment for multi agent dynamics spectrum access.
"""
import gym
import pdb
import numpy as np
from gym import spaces

class FrequencySpectrumEnv(gym.Env):
    """Multi Agent Environment for dynamics spectrum access
    
    - At each timestep, agents choose with frequency band to transmit on, or to not transmit at all.
    - The goal is to maximize channel utilization and make sure all agents have roughly the same number of successful packets.
    """

    def __init__(self,
                 num_freq_bands,
                 num_agents,
                 max_iterations=10000,
                 temporal_len=3,
                 observation_type="aggregate",
                 reward_type="transmission1",
                 reward_history_len=100):
        """Initialize the environment

        Args:
            num_freq_bands (int): The number of available frequency bands
            num_agents (int): The number of agents
            max_iterations (int, optional): Number of iterations before an episode completes. Defaults to 10000.
            temporal_len (int, optional): Number of previous timesteps are available in the obvservation. Defaults to 3.
            observation_type (str, optional): Specifies what the agents can observe. Defaults to "aggregate".
            reward_type (str, optional): Specifies how agents receive rewards. Defaults to "transmission1".
            reward_history_len (int, optional): If reward_type is "transmision_normalized" this is 
                The number of timesteps in the past used for normalizing agent rewards and collisions. Defaults to 100.
        """
        super().__init__()
        self.num_actions = num_freq_bands + 1
        self.num_freq_bands = num_freq_bands
        self.num_agents = num_agents
        self.agents = [x for x in range(num_agents)]
        self.action_space = spaces.Discrete(self.num_actions)
        self.max_iterations = max_iterations
        self.temporal_len = temporal_len
        self.agent_actions_history = np.zeros(
            (self.temporal_len, self.num_actions, self.num_agents), dtype=np.int32)
        self.observation_type = observation_type
        self.observation_space = spaces.Discrete(self.num_actions)
        self.reward_type = reward_type
        self.reward_history_len = reward_history_len
        self.reward_hist = np.zeros((self.reward_history_len, self.num_agents))
        self.reward_hist_pointer = 0
        # this is used by the main function to pull history about what happened and save it
        self.agent_actions_complete_history = []

    def step(self, action_n, verbose=False):
        """Step in environment

        This function generally
        - Takes in the action integers and figures out where collisions occur
        - Collisions occur when agents select the same integer except 0 as 0 is no transmit
        - Gives each agent back an obvservation. See self.agent_actions_history_to_observations() for possibilities
        - Gives each agent back a reward. See self.agent_actions_to_rewards() for possibilities

        Args:
            action_n (np.array): List of ints representing the chosen actions for each agent
            verbose (bool, optional): . Defaults to False.

        Returns:
           (obs_n,
           reward_n,
           done_n,
           info_n
        """
        
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {"n": []}
        
        self.agent_actions = np.zeros((self.num_actions, self.num_agents), dtype=np.int32)
        self.agent_actions[action_n, np.arange(self.num_agents)] = 1
        self.agent_actions_complete_history.append(action_n)

        # This a look of temporal_length into the past
        # Observation types will start with this and convert this into a formatted observation
        self.agent_actions_history = np.concatenate([
            self.agent_actions.reshape((1, self.num_actions, self.num_agents)),
            self.agent_actions_history[:-1, :, :]],
            axis=0)

        obs_n = self.agent_actions_history_to_observations(self.agent_actions_history)

        # Keep track of collisions and throughput
        # SUM - to get collections on most recent
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
            f"{throughput_this_time_splot:.2f} new_throughput {new_throughput:.2f}")
        
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
        elif self.observation_type == "own_actions":
            observations = self._get_observation_own_actions(agent_actions_history)
        elif self.observation_type == "is_channel_available":
            observations = self._get_observation_is_channel_available(agent_actions_history)
        else:
            raise KeyError(f"observation type needs to be in [aggregate, own_actions, is_channel_available].. got {self.observation_type}")

        return observations

    def _get_observation_is_channel_available(self, agent_actions_history):
        """Channels available if no transmit or collision

        Channel states: 0 for no transmission, 1 for channel occupied, 1 for collisions

        The example table below shows a slice along the agent dimension for 1 agent, a temporal length of 5, and 2 frequency bands.

        |              |              | t | t-1 | t-2 | t-3 | t-4 |
        |--------------|--------------|---|-----|-----|-----|-----|
        | Band Status: |  Freq Band 1 | 0 | 1   | 0   | 1   | 1   |
        | Band Status: |  Freq Band 0 | 0 | 1   | 1   | 0   | 0   |

        Args:
            agent_actions_history (np.array): Complete information about who attempted communication where. 
                Shape is (num_time, num_freq + 1, num_agents)
        
        Returns:
           np.array: state for all agents has shape (num_agents, num_in_time, self.num_freq_bands)
        """
        obvs = agent_actions_history.sum(axis=2).reshape((self.temporal_len, -1))
        # For collisions we'll just say channel busy
        obvs[obvs >= 2] = 1
        # Channel 0 is for no transmit. Can't observe that from other agents
        obvs = obvs[:, 1:]
        obvs = np.tile(obvs, (self.num_agents, 1, 1))
        return obvs

    def _get_observation_aggregate(self, agent_actions_history):
        """Observation where agents can see what other agents are doing and channel status

        The example table below shows a slice along the agent dimension for 1 agent, a temporal length of 5, and 2 frequency bands.

        |              |              | t | t-1 | t-2 | t-3 | t-4 |
        |--------------|--------------|---|-----|-----|-----|-----|
        | Band Status: |  Freq Band 1 | 0 | 1   | 0   | 1   | 1   |
        | Band Status: |  Freq Band 0 | 0 | 1   | 1   | 0   | 0   |
        | Action:      |  Freq Band 1 | 0 | 0   | 0   | 1   | 0   |
        | Action:      |  Freq Band 0 | 0 | 1   | 1   | 0   | 0   |
        | Action:      |  No Transmit | 1 | 0   | 0   | 0   | 1   |
        | Success:     |              | 0 | 1   | -1  | -1  | 0   |

        Args:
            agent_actions_history (np.array): Complete information about who attempted communication where. 
                Shape is (num_time, num_freq + 1, num_agents)

        Returns:
            np.array: state for all agents has shape (num_agents, num_in_time, 2*self.num_freq_bands + 2) 
                Note:( self.num_freq_bands + 1) actions, 1 for transmission status, self.num_freq_bands channel status
        """
        obvs1 = self._get_observation_is_channel_available(agent_actions_history)
        obvs2 = self._get_observation_own_actions(agent_actions_history)
        obvs = np.concatenate([obvs1, obvs2], axis=2)
        return obvs

    def _get_observation_own_actions(self, agent_actions_history):
        """Get the observation for an agent

        Each agent knows it's own chosen action (one hot encoded) and,
        The transmision status: 0: No transmit, 1: success, -1: collision

        |                     | t | t-1 | t-2 | t-3 | t-4 |
        |---------------------|---|-----|-----|-----|-----|
        |         Freq Band 1 | 0 | 0   | 0   | 1   | 0   |
        |         Freq Band 0 | 0 | 1   | 1   | 0   | 0   |
        |         No Transmit | 1 | 0   | 0   | 0   | 1   |
        | Transmission Status | 0 | 1   | -1  | -1  | 0   |

        Args:
            agent_actions_history (_type_): Complete information about who attempted communication where. Shape is (num_time, num_freq + 1, num_agents)
        
        Returns:
            np.array: state for all agents has shape (num_agents, num_in_time, self.num_freq_bands + 1 + 1) 
                Note: self.num_freq_bands + 1 actions and 1 for transmission status
        """
        channel_usage = agent_actions_history.sum(axis=2)
        channel_usage = channel_usage[:, :, np.newaxis]
        # Not possible to have collisions in the no transmit state
        channel_usage[:, 0, 0] = 0
        # agent_actions_history only has a 1 in each row. So elementwise multiplication picks out successful transmissions or collisions
        # and then summing along that access just pulls only that element
        # SUM - to get collisions on temporal amount then pick out specifics for agent
        transmission_status = np.sum(agent_actions_history * channel_usage, axis=1)
        transmission_status = transmission_status[:, np.newaxis, :]
        obvs = np.concatenate([agent_actions_history, transmission_status], axis=1)
        obvs = np.swapaxes(obvs, 0, 1)
        obvs = np.swapaxes(obvs, 0, 2)
        # Replace collisions with -1 so NN can learn easier
        obvs[obvs >= 2] = -1
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
        Choosing 0 action (no transmit) can't result in a reward

        Args:
            agent_actions (np.array): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (np.array): Int representation of what agents chose Shape is (num_agents)
        
        Returns:
            list(states): rewards for all agents has shape (num_agents, 1)
        """
        # SUM - to get collisions on most recent
        transmissions = agent_actions.sum(axis=1)
        transmissions[transmissions > 2] = 2
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = 0
        reward_info[0] = 0
        reward_n = reward_info[action_ints].reshape(-1)
        return reward_n

    def _get_reward_transmision_normalized(self, agent_actions, action_ints):
        """Get the reward for an agent.

        Original rewards taken from _get_reward_collisionpenality2()

        Then:

        Rewards are divided by the number of successful transmissions in the reward_hist buffer
        Intended to make transmissions more valuable to those who haven't gotten them
        And to make transmissions less valuable to those who have been sending

        Collisions are divided by the number of no transmits in the reward_hist buffer
        Intended to make collisions less costly to those who haven't been transmitting
        And to make collisions more costly to those who have been successful

        Args:
            agent_actions (_type_): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (_type_): Int representation of what agents chose Shape is (num_agents)
        
        Returns:
            list(states): rewards for all agents has shape (num_agents, 1)
        """

        reward_n = self._get_reward_collisionpenality2(agent_actions=agent_actions, action_ints=action_ints)
        reward_n = reward_n.astype(np.float32)

        successes = (self.reward_hist == 1).sum(axis=0)
        collisions = (self.reward_hist == 2).sum(axis=0)
        no_trans = (self.reward_hist == 0).sum(axis=0)

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
        reward_info = transmissions.copy()
        reward_info[reward_info >= 2] = -1
        reward_info[reward_info == 1] = 3
        reward_info[0] = 0
        reward_n = reward_info[action_ints].reshape(-1)
        return reward_n

    def _get_reward_centralized(self, agent_actions, action_ints):
        """Get the reward for an agent.

        Agents get the sum of all agent rewards

        Args:
            agent_actions (np.array): Complete information about who attempted communication where. Shape is (num_freq + 1, num_agents)
            action_ints (np.array): Int representation of what agents chose Shape is (num_agents)
        
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

    def render(self, mode="human", close=False):
        """Not currently supported
        """
        print("rendering but not really :(")