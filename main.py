import tensorflow as tf
import numpy as np
import pandas as pd
import pdb
from environment import FrequencySpectrumEnv
from agent import DynamicSpectrumAccessAgent1#, DynamicSpectrumAccessAgent2
from utils import  agent_actions_to_information_table
from visualization import plot_spectrum_usage_over_time


# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


num_agents = 3
num_bands = 2
temporal_length = 5
reward_type = "transmission1"
# reward_type = "collisionpenality1"
# obs_type = "own_actions"
obs_type, obvs_space_dim = "own_actions", num_bands+2
# obs_type, obvs_space_dim = "aggregate", num_bands


env = FrequencySpectrumEnv(num_bands, num_agents, temporal_len=temporal_length, reward_type=reward_type, observation_type=obs_type)

agents = [DynamicSpectrumAccessAgent1(num_bands, obvs_space_dim, temporal_length=temporal_length) for _ in range(num_agents)]
# agents = [DynamicSpectrumAccessAgent1(num_bands, temporal_length=temporal_length) for _ in range(num_agents)]

steps = 101
all_collisions = []
all_throughputs = []

plot_every = 25
for s in range(steps):
    # print(f"s is {s}")
    done = False
    state = env.reset()
    total_reward = 0

    should_plot = (s % plot_every) == 0
    
    counter = 0
    while not done:
        counter += 1
      #env.render()
      # try:
        if should_plot:
            actions = [agents[i].act(state[i], save_visualization_filepath=True) for i in range(num_agents)]
        else:
            actions = [agents[i].act(state[i]) for i in range(num_agents)]
        next_state, rewards, done_i, info = env.step(actions)
        # pdb.set_trace()

          
        done = done_i[0]
        for i in range(num_agents):
          # state, action, reward, next_state, done
          agents[i].observe_result(state[i], actions[i], rewards[i], next_state[i], done_i[i])
          
          state = next_state
          total_reward += sum(rewards)

        if done:
          print(f"total reward after {s} episode is {total_reward} throughput is {env.throughput:.2f} num_collisions is {env.num_collisions} and epsilon is {agents[0].epsilon:.2f}")
          all_collisions.append(env.num_collisions)
          all_throughputs.append(env.throughput)
    
    if should_plot:
        agent_actions = np.array(env.agent_actions_complete_history)
        df_more_info = agent_actions_to_information_table(agent_actions, reward_type=reward_type)
        filename = f"plots/extra_info_trainstep_{s}.png"
        fig= plot_spectrum_usage_over_time(df_more_info, add_collisions=True, filepath=filename)
          
df = pd.DataFrame.from_dict({"collisions":all_collisions, "throughout": all_throughputs})
df.to_csv("output.csv")