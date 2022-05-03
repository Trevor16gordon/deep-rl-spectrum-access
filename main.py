import os
import pdb
import tqdm
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from environment import FrequencySpectrumEnv
from agent import (DynamicSpectrumAccessAgent1,
                   DynamicSpectrumAccessAgentActorCritic,
                   DynamicSpectrumAccessAgentPeriodic)
from utils import agent_actions_to_information_table, create_stacked_csv
from config_file import TrainingConfig, ReportingConfig


tf.compat.v1.enable_eager_execution()

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


def train(args):

    all_config = args.__dict__

    if all_config["obs_type"] == "own_actions":
        all_config["obvs_space_dim"] = args.num_bands+2
    elif all_config["obs_type"] == "aggregate":
        all_config["obvs_space_dim"] = args.num_bands
    elif all_config["obs_type"] == "aggregate2":
        all_config["obvs_space_dim"] = 2*args.num_bands+2
    elif all_config["obs_type"] == "aggregate3":
        all_config["obvs_space_dim"] = 2*args.num_bands+2

    all_config["time_folder"] = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    path = ReportingConfig.TOP_LEVEL_FOLDER + all_config["time_folder"]
    os.mkdir(path)

    df_config = pd.DataFrame.from_dict([all_config])
    df_config.to_csv(os.path.join(
        path, ReportingConfig.CONFIG_FILE_NAME_TO_SAVE))

    env = FrequencySpectrumEnv(
        args.num_bands,
        args.num_agents,
        temporal_len=args.temporal_length,
        reward_history_len=args.reward_history_len,
        reward_type=args.reward_type,
        observation_type=args.obs_type)

    if args.model_type == "ddqn":
        agents = [DynamicSpectrumAccessAgent1(args.num_bands,
                                              args.obvs_space_dim,
                                              temporal_length=args.temporal_length,
                                              epsilon=args.epsilon,
                                              temperature=args.temperature,
                                              buffer_size=args.buffer_size,
                                              epsilon_decay=args.eps_decay) for _ in range(args.num_agents)]
    else:
        agents = [DynamicSpectrumAccessAgentActorCritic(args.num_bands,
                                                        args.obvs_space_dim,
                                                        temporal_length=args.temporal_length,
                                                        epsilon=args.epsilon,
                                                        epsilon_decay=args.eps_decay) for _ in range(args.num_agents)]

    if args.agent_homogeneity == "one_periodic":
        agents[-1] = DynamicSpectrumAccessAgentPeriodic(1, 1, 1)

    if args.agents_shared_memory:
        for agent_i in agents[1:]:
            agent_i.memory = agents[0].memory

    state = env.reset()

    total_reward = 0
    counter = 0
    agent_values = []
    agent_action_prob = []

    for counter in (pbar := tqdm.tqdm(range(args.episode_len))):
        # counter += 1

        if counter > int(all_config["episode_len"]):
            print("Breaking because episode is done")
            break

        actions = [agents[i].act(state[i]) for i in range(args.num_agents)]

        # Collect agent value neural networks and resulting probabilities for analysis
        agent_values.append(
            [agents[i].last_value_function for i in range(args.num_agents)])
        agent_action_prob.append(
            [agents[i].last_prob_value for i in range(args.num_agents)])

        next_state, rewards, done_i, info = env.step(actions)

        for i in range(args.num_agents):
            agents[i].observe_result(
                state[i], actions[i], rewards[i], next_state[i], done_i[i])

        state = next_state
        total_reward += sum(rewards)

        if (counter % ReportingConfig.PRINT_UPDATE_EVERY) == 0:
            pbar.set_description(
                (f"Episodes: {counter}  Total reward: {total_reward} Throughput: {env.throughput:.2f}"
                f" Collisions: {env.num_collisions} Epsilon: {agents[0].epsilon:.2f}"))

        if (counter % ReportingConfig.SAVE_CHECKPOINT_CSV_EVERY) == 0:
            agent_actions = np.array(env.agent_actions_complete_history)
            df_more_info = agent_actions_to_information_table(
                agent_actions, reward_type=all_config["reward_type"])
            df = create_stacked_csv(
                agent_actions, agent_values, agent_action_prob)
            merged = pd.merge(df_more_info, df, on="time")
            merged.to_csv(f"{path}/history.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eps_decay", "-epsd",
                        default=TrainingConfig.EPS_DECAY, type=float)
    parser.add_argument("--epsilon", "-eps",
                        default=TrainingConfig.EPSILON, type=float)
    parser.add_argument("--num_agents", "-na",
                        default=TrainingConfig.NUM_AGENTS, type=int)
    parser.add_argument("--num_bands", "-b",
                        default=TrainingConfig.NUM_BANDS, type=int)
    parser.add_argument("--temporal_length", "-tl",
                        default=TrainingConfig.TEMPORAL_LENGTH, type=int)
    parser.add_argument("--reward_type", "-r",
                        default=TrainingConfig.REWARD_TYPE, type=str)
    parser.add_argument("--obs_type", "-o",
                        default=TrainingConfig.OBS_TYPE, type=str)
    parser.add_argument("--agents_shared_memory", "-sm",
                        default=TrainingConfig.AGENTS_SHARED_MEMORY, type=int)
    parser.add_argument("--buffer_size", "-bs",
                        default=TrainingConfig.BUFFER_SIZE, type=int)
    parser.add_argument("--episode_len", "-el",
                        default=TrainingConfig.EPISODE_LEN, type=int)
    parser.add_argument("--temperature", "-t",
                        default=TrainingConfig.TEMPERATURE, type=float)
    parser.add_argument("--reward_history_len", "-rhl",
                        default=TrainingConfig.REWARD_HISTORY_LEN, type=int)
    parser.add_argument("--model_type", "-mt",
                        default=TrainingConfig.MODEL_TYPE, type=str)
    parser.add_argument("--agent_homogeneity", "-ah",
                        default=TrainingConfig.AGENT_HOMOGENEITY, type=str)

    args = parser.parse_args()

    train(args)
