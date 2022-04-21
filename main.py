from multiprocessing import shared_memory
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pdb
from datetime import datetime
from environment import FrequencySpectrumEnv
from agent import DynamicSpectrumAccessAgent1, DynamicSpectrumAccessAgentPeriodic#, DynamicSpectrumAccessAgent2
from utils import  agent_actions_to_information_table
from visualization import plot_spectrum_usage_over_time
from utils import create_stacked_csv, complete_df_to_stacked, create_stacked_csv_old
import argparse

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


parser = argparse.ArgumentParser()
parser.add_argument("--eps_decay", "-epsd", default=1e-5)
parser.add_argument("--epsilon", "-eps", default=0.1)
parser.add_argument("--num_agents", "-na", default=2)
parser.add_argument("--num_bands", "-b", default=1)
parser.add_argument("--temporal_length", "-tl", default=5)
parser.add_argument("--reward_type", "-r", default="collisionpenality2")
parser.add_argument("--obs_type", "-o", default="aggregate3")
parser.add_argument("--agents_shared_memory", "-sm", default=1)
parser.add_argument("--buffer_size", "-bs", default=1000)
parser.add_argument("--episode_len", "-el", default=40001)
parser.add_argument("--temperature", "-t", default=0.005)


all_config = parser.parse_args().__dict__


num_bands = int(all_config["num_bands"])
num_agents = int(all_config["num_agents"])

if all_config["obs_type"] == "own_actions":
    all_config["obvs_space_dim"] = num_bands+2
elif all_config["obs_type"] == "aggregate":
    all_config["obvs_space_dim"] = num_bands
elif all_config["obs_type"] == "aggregate2":
    all_config["obvs_space_dim"] = 2*num_bands+2
elif all_config["obs_type"] == "aggregate3":
    all_config["obvs_space_dim"] = 2*num_bands+2

top_level_folder = "plots.nosync/"
all_config["time_folder"] = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
path = top_level_folder + all_config["time_folder"]
os.mkdir(path)

# all_config["agent_homogeneity"] = "one_periodic"
all_config["agent_homogeneity"] = "all_same"

pd.DataFrame.from_dict([all_config]).to_csv(path + "/config.csv")


env = FrequencySpectrumEnv(num_bands, num_agents, temporal_len=int(all_config["temporal_length"]), reward_type=all_config["reward_type"], observation_type=all_config["obs_type"])

agents = [DynamicSpectrumAccessAgent1(num_bands,
                                        all_config["obvs_space_dim"], 
                                        temporal_length=int(all_config["temporal_length"]), 
                                        epsilon=float(all_config["epsilon"]),
                                        temperature=float(all_config["temperature"]),
                                        buffer_size=int(all_config["buffer_size"]),
                                         epsilon_decay=float(all_config["eps_decay"])) for _ in range(num_agents)]

if all_config["agent_homogeneity"] == "one_periodic":
    agents[-1] = DynamicSpectrumAccessAgentPeriodic(1, 1, 1)


if all_config["agents_shared_memory"]:
    for agent_i in agents[1:]:
        agent_i.memory = agents[0].memory

save_checkpoint_csv_every = 500
info_every = 10

state = env.reset()


total_reward = 0
counter = 0
agent_values = []
agent_action_prob = []

while True:
    counter += 1

    if counter > int(all_config["episode_len"]):
        print("Breaking because episode is done")
        break

    actions = [agents[i].act(state[i]) for i in range(num_agents)]

    # Collect agent value neural networks and resulting probabilities for analysis
    agent_values.append([agents[i].last_value_function for i in range(num_agents)])
    agent_action_prob.append([agents[i].last_prob_value for i in range(num_agents)])

    next_state, rewards, done_i, info = env.step(actions)

    for i in range(num_agents):
        agents[i].observe_result(state[i], actions[i], rewards[i], next_state[i], done_i[i])
        
    state = next_state
    total_reward += sum(rewards)

    if (counter % info_every) == 0:
        print(f"total reward after {counter} episode is {total_reward} throughput is {env.throughput:.2f} num_collisions is {env.num_collisions} and epsilon is {agents[0].epsilon:.2f}")
    
    if (counter % save_checkpoint_csv_every) == 0:
        agent_actions = np.array(env.agent_actions_complete_history)
        df_more_info = agent_actions_to_information_table(agent_actions, reward_type=all_config["reward_type"])
        df = create_stacked_csv(agent_actions, agent_values, agent_action_prob)
        merged= pd.merge(df_more_info, df, on="time")
        merged.to_csv(f"{path}/history.csv")