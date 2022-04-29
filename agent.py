""" 
Agents for dynamic spectrum access problem:

Referenced this webpage: https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
"""

import tensorflow as tf
import numpy as np
from model import DDQN
from visualization import plot_and_save_freq_status_and_network_output
import pdb

class DynamicSpectrumAccessAgentBase():
    """Base class for a dynamic spectrum access agent

    Must subclass this and implement existing function
    """

    def act(self, state):
        """Given the current state give a distribution on actions

        Args:
            state (np.array): shape is variable

        Returns:
            np.array: likelihood on selecting action
        """
        raise NotImplementedError

    def observe_result(self, state, action, reward, next_state, done):
        """Given information about what happened in environment observe what happened

        Typical things to do might be:
        - Save information in memory
        - Train

        Args:
            state (np.array): Varyiable size array for what an agent considers the state input to be
            action (np.array): _description_
            reward np.int32: The reward
            next_state (np.array): Varyiable size array for what an agent considers the state input to be
            done (bool): Whether the agent is done.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

class DynamicSpectrumAccessAgentPeriodic(DynamicSpectrumAccessAgentBase):
    """Dumb agent


    Transmits for on_period, then idle for off_period

    Args:
        DynamicSpectrumAccessAgentBase (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, chosen_band, num_on_periods, num_off_periods) -> None:
        super().__init__()
        self.chosen_band = chosen_band
        self.num_on_periods = num_on_periods
        self.num_off_periods = num_off_periods
        self.trainstep = 0
        #TODO remove beta it's not needed
        self.beta = 0
        self.period = self.num_on_periods + self.num_off_periods
        self.agent_id = 1234

    def act(self, state, save_visualization_filepath=""):
        """Given the current state give a distribution on actions

        Args:
            state (np.array): shape is variable

        Returns:
            np.array: likelihood on selecting action
        """
        time = self.trainstep % self.period
        if time < self.num_off_periods:
            action = self.chosen_band
        else:
            action = 0
        self.trainstep += 1
        return action

    def observe_result(self, state, action, reward, next_state, done):
        return


class DynamicSpectrumAccessAgent1(DynamicSpectrumAccessAgentBase):
    """Dynamic Spectrum Agent with:

    Model: 
    - Dueling DQN with memory replay
    - Target network and memory replay

    Input Data Preparation
    - Uses just the aggregate

    Args:
        DynamicSpectrumAccessAgentBase (_type_): _description_
    """

    def __init__(self, num_bands, obvs_space_dim, gamma=0.9, epsilon=0.02, replace=100, temperature=0.005, lr=0.0001, temporal_length=6, epsilon_decay=1e-3, buffer_size=1000):
        self.num_bands = num_bands
        self.n_action_space = num_bands + 1
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = 1
        self.min_epsilon = 0.01
        self.epsilon_decay = epsilon_decay
        self.replace = replace
        self.trainstep = 0
        self.memory = ExperienceReplay(obvs_space_dim, temporal_length, buffer_size=buffer_size)
        self.batch_size = 6
        self.q_net = DDQN(num_bands, obvs_space_dim, temporal_length)
        self.target_net = DDQN(num_bands, obvs_space_dim, temporal_length)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        # self.q_net.compile(loss='MeanSquaredError', optimizer=opt, run_eagerly=True)
        self.update_target()
        # self.target_net.compile(loss='MeanSquaredError', optimizer=opt, run_eagerly=True)
        self.temporal_length = temporal_length
        self.agent_id = [str(x) for x in np.random.randint(0, high=9, size=10).tolist()]
        self.agent_id = "".join(self.agent_id)

        # High temps cause all actions to be equally probable
        self.temperature = temperature
        
        # Last value function should be the same shape as number of bands + 1
        self.last_value_function = np.zeros(self.num_bands+1)

        # Las prob value will be the softmax temperature resulting action probabilities
        self.last_prob_value = np.zeros(self.num_bands+1)

    def act(self, state, save_visualization_filepath=False):

          
        if np.random.rand() <= self.epsilon:
            self.last_value_function = np.ones(self.num_bands+1)/(self.num_bands+1)
            self.last_prob_value = np.ones(self.num_bands+1)/(self.num_bands+1)
            return np.random.choice([i for i in range(self.n_action_space)])
        else:

            Qs = self.q_net.advantage(state[np.newaxis, :, :])
            self.last_value_function = Qs.numpy()[0].tolist()

            Qs_num = Qs.numpy()
            action = np.random.choice(np.flatnonzero(Qs_num == Qs_num.max()))

            # # Set the minimum value to be 0 so that after exp the minimum is 1
            # Qs_norm = Qs - np.min(Qs)

            # # Changing to try and have temp control exploration
            # actions = np.exp(Qs_norm/self.temperature)

            # # Sometimes large actions can end up as + inf here.
            # # In this case just take the argmax
            # if  np.any(np.isinf(actions)) or np.isinf(np.sum(actions)):
            #     actions = np.zeros(len(Qs[0]))
            #     actions[np.argmax(Qs)] = 1

            # prob = (actions / np.sum(actions)).reshape(-1)
            # self.last_prob_value = prob.tolist()

            # try:
            #     action = np.random.choice(np.arange(len(prob)), p=prob)
            # except:
            #     pdb.set_trace()
            #     print("whatttt")


        #   if np.any(np.isinf(actions)) or (np.sum(actions) == 0):
        #       action = np.argmax(Qs)
        #   else:
        #     actions /= np.sum(actions)
        #     if np.sum(actions) == 0:
        #         action = np.argmax(Qs)
        #     else:
        #         actions = np.nan_to_num(actions, nan=1)
        #         prob = actions.reshape(-1)
        #         try:
        #             action = np.random.choice(np.arange(len(prob)), p=prob)
        #         except:
        #             print("whaaat")
        #             action = np.argmax(Qs)           
            

            if save_visualization_filepath:
                # TODO: Plotting shouldn't be in here
            #   filename = f"plots/trainstep_{self.trainstep}_{self.agent_id}.png"
                filename = save_visualization_filepath
                plot_and_save_freq_status_and_network_output(state.T, prob.T, filepath=filename
                )

            return action

    def observe_result(self, state, action, reward, next_state, done):
        """Given information about what happened in environment observe what happened

        Args:
            state (np.array): Shape should be (num_agents, num_in_time, __)
            action (np.array): _description_
            reward np.int32: The reward
            next_state (np.array): Varyiable size array for what an agent considers the state input to be
            done (bool): Whether the agent is done.

        Raises:
            NotImplementedError: _description_
        """
        self.update_mem(state, action, reward, next_state, done)
        self.train()

    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())     

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self):
        if self.memory.pointer < (self.batch_size + self.temporal_length):
            return 
        
        if self.trainstep % self.replace == 0:
            self.update_target()
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        
        
        next_state_val_target_net = self.target_net.predict(next_states)
        next_state_val_q_net = self.q_net.predict(next_states)

        # Making change the maximum and the value should come from the target network.
        # Actually no for dueling it comes from q
        max_action = np.argmax(next_state_val_q_net, axis=1)
        #max_action = np.argmax(next_state_val_target_net, axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        

        # q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones
        
        # The shapes of actions and everything were messed up so training wasn't occuring properly
        # q_target[batch_index, actions[:, 0]] = rewards[:, 0] + self.gamma * next_state_val[batch_index, max_action]*dones[:, 0]
        # pdb.set_trace()
        # self.q_net.train_on_batch(states, q_target)



        updated_values_these_actions = rewards[:, 0] + self.gamma * next_state_val_target_net[batch_index, max_action]*dones[:, 0]

        with tf.GradientTape() as tape:
            state_values = self.q_net(states, training=True)
            state_values_these_actions = tf.gather_nd(params=state_values, indices=actions, batch_dims=1)
            loss = self.loss_fn(state_values_these_actions, updated_values_these_actions)

        # Compute gradients
        gradients = tape.gradient(loss, self.q_net.trainable_variables)


        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.q_net.trainable_variables))

        self.update_epsilon()
        self.trainstep += 1


# class DynamicSpectrumAccessAgent2(DynamicSpectrumAccessAgent1):
#     """Dynamic Spectrum Agent with:

#     Same as DynamicSpectrumAccessAgent1 but include the chosen action and the reward in the network input space

#     Args:
#         DynamicSpectrumAccessAgentBase (_type_): _description_
#     """
#     def __init__(self, num_bands, gamma=0.9, replace=100, lr=0.0001, temporal_length=6):
#         super().__init__(num_bands, gamma, replace, lr, temporal_length)
#         self.last_action = 0
#         self.last_reward = 0

#     def observe_result(self, state, action, reward, next_state, done):
#         """Given information about what happened in environment observe what happened

#         Same as DynamicSpectrumAccessAgent1 but include the chosen action and the reward in the network input space

#         Args:
#             state (np.array): Varyiable size array for what an agent considers the state input to be
#             action (np.array): _description_
#             reward np.int32: The reward
#             next_state (np.array): Varyiable size array for what an agent considers the state input to be
#             done (bool): Whether the agent is done.

#         Raises:
#             NotImplementedError: _description_
#         """
#         self.update_mem(state, action, reward, next_state, done)

#         self.last_reward = reward
#         self.last_action = action
#         self.train()

class ExperienceReplay():
    
    def __init__(self, obs_dimensions, temporal_length=6, buffer_size=1000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, temporal_length, obs_dimensions), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size, 1), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, temporal_length, obs_dimensions), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size, 1), dtype=np.bool)
        self.obs_dimensions = obs_dimensions
        self.pointer = 0
        self.temporal_length = temporal_length

    def add_exp(self, state, action, reward, next_state, done):
        # if state is None:
        #     print(f"state is None counter is {self.pointer}")
        
        idx  = self.pointer % self.buffer_size 
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones


class ExperienceReplayWindowed():

    def __init__(self, obs_dimensions, temporal_length=6, buffer_size=1000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, obs_dimensions), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size, 1), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, obs_dimensions), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size, 1), dtype=np.bool)
        self.obs_dimensions = obs_dimensions
        self.pointer = 0
        self.temporal_length = temporal_length

    def add_exp(self, state, action, reward, next_state, done):
        # if state is None:
        #     print(f"state is None counter is {self.pointer}")
            
        idx  = self.pointer % self.buffer_size 
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def get_last_several_states(self, num_states):
        
        idx  = self.pointer % self.buffer_size 
        if idx < num_states:
            states = np.zeros((self.temporal_length-1, self.obs_dimensions))
            print("Used zero padding")
        else:
            states = self.next_state_mem[idx-num_states:idx]
        
        return states

    def sample_using_window(self, batch, arr_in, window_len):
        x_indexer = np.arange(len(batch))
        x_indexer = batch[x_indexer].reshape((-1, 1))
        # Y indexers sequentially goes from that batch point to the follow time sequence
        y_indexer = np.arange(window_len).reshape((1, -1))
        indexer = x_indexer + y_indexer 
        new_arr = arr_in[indexer, :]
        return new_arr

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size) - self.temporal_length
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.sample_using_window(batch, self.state_mem, self.temporal_length)
        actions = self.sample_using_window(batch, self.action_mem, self.temporal_length)
        rewards = self.sample_using_window(batch, self.reward_mem, self.temporal_length)
        next_states = self.sample_using_window(batch, self.next_state_mem, self.temporal_length)
        dones = self.sample_using_window(batch, self.done_mem, self.temporal_length)
        return states, actions, rewards, next_states, dones

