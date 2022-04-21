""" 
Contains utily functions
"""
import numpy as np
import pandas as pd
import pdb


def agent_actions_to_information_table(agent_actions, num_bands=None, reward_type="collisionpenality1"):
    """Raw agent actions to more information

    

    Args:
        agent_actions (np.array): Expects agent actions to have columns for each agent. Int for which freq was selected
        optional num_bands (int): If not pass, the max value in agent_actions denotes this
    """
    
    num_time, num_agents = agent_actions.shape
    num_bands = agent_actions.max()

    no_transmit_name = "No Transmit"

    agent_name_cols = [f"agent_{i}" for i in range(num_agents)]
    band_name_cols = [f"band_{i}" for i in range(num_bands)]
    rew_col_names = [f"reward_agent_{i}" for i in range(num_agents)]
    cum_rew_col_names = [f"cum_reward_agent_{i}" for i in range(num_agents)]
    r_col_names = [*rew_col_names, *cum_rew_col_names]
    
    df = pd.DataFrame(agent_actions)
    df.columns = agent_name_cols
    
    
    freq = np.zeros((len(agent_actions), num_bands+1), dtype=np.int32)
    # freq
    for i in range(num_agents):
        freq[range(num_time), agent_actions[:, i]] += 1
    freq[freq>2] = 2
    band_status = pd.DataFrame(freq, columns=[no_transmit_name]+band_name_cols)
    # df = pd.concat([df, band_status], axis=1)

    
    throughput = np.count_nonzero(band_status.iloc[:, 1:].values == 1, axis=1) / num_bands
    collision_dens = np.count_nonzero(band_status.iloc[:, 1:].values >= 2, axis=1) / num_bands
    
    # moving_throughput = np.cumsum(throughput) / np.arange(1, len(throughput)+1)
    N = len(throughput)//10
    pd.Series(throughput).rolling(4).mean()
    # moving_throughput = np.convolve(throughput, np.ones(N)/N, mode='same')
    # moving_collision_dens = np.cumsum(collision_dens) / np.arange(1, len(collision_dens)+1)
    # moving_collision_dens = np.convolve(collision_dens, np.ones(N)/N, mode='same')
    
    collision_dens = collision_dens.reshape((-1, 1))
    throughput = throughput.reshape((-1, 1))
    # moving_throughput = moving_throughput.reshape((-1, 1))
    # moving_collision_dens = moving_collision_dens.reshape((-1, 1))

    # collision_info = np.concatenate([throughput, moving_throughput, collision_dens, moving_collision_dens], axis=1)
    collision_info = np.concatenate([throughput, collision_dens], axis=1)
    collision_stats = pd.DataFrame(collision_info, columns=["throughput","collision_dens"])
    collision_stats["moving_throughput"] = collision_stats["throughput"].rolling(N).mean()
    collision_stats["moving_collision_dens"] = collision_stats["collision_dens"].rolling(N).mean()
    collision_stats["no_transmit"] = 1 - collision_stats["throughput"] - collision_stats["collision_dens"]
    collision_stats["moving_no_transmit"] = 1 - collision_stats["moving_throughput"] - collision_stats["moving_collision_dens"]

    ids = np.tile(np.arange(num_time).reshape(-1, 1), num_agents)
    freq_copy = freq.copy()
    freq_copy[:, 0] = 0
    rewards = freq_copy[ids, agent_actions]

    if reward_type == "collisionpenality1":
        rewards[rewards==2] = -1
    elif reward_type == "collisionpenality2":
        rewards[rewards==2] = -1
        rewards[rewards==1] = 2
    elif reward_type == "transmission1":
        rewards[rewards==2] = 0
    elif reward_type == "centralized":
        rewards[rewards==2] = 0
    elif reward_type == "transmision_normalized":
        rewards[rewards==2] = 0
    else:
        raise KeyError
    rewards = np.concatenate([rewards, rewards.cumsum(axis=0)], axis=1)
    
    rewards = pd.DataFrame(rewards, columns=r_col_names)


    df = pd.concat([df, band_status, rewards, collision_stats], axis=1)

    # df["time"] = df.index
    # df1 = df.melt(id_vars="time", value_vars=["agent_0", "agent_1", "agent_2"], var_name='Agent', value_name='Action')
    # df2 = df.melt(id_vars="time", value_vars=rew_col_names, var_name='Agent', value_name='Reward')
    # df3 = df.melt(id_vars="time", value_vars=cum_rew_col_names, var_name='Agent', value_name='Cumulative Reward')
    # df4 = df.melt(id_vars="time", value_vars=["band_0", "band_1", "band_2"], var_name='Frequency', value_name='Status')

    # df3
    df["time"] = df.index
    return df



def create_stacked_csv_old(agent_actions, agent_values, agent_action_prob):
    """Take in a binary indicator 

    Args:
        agent_actions (np.array): (num_timesteps, num_agents) with both columsn being ints
        agent_values (list(list(float))): (num_timesteps, num_agents)
        agent_action_prob (list(list(float))): (num_timesteps, num_agents)

    Returns:
        df (pd.DataFrame): Columns are ["timestep", "agent", "action", "chosen_action", "value", "prob"]  
            timestep (int): [0, num_timesteps-1]
            agent (int): [0, num_agents-1]
            action (int): [0, num_bands +1 -1]
            chosen_action (bool): Whether this particular action was actually chosen
            value (float): The agent's neural network value function output
            prob (float): The agent's neural network value function as a probability using softmax with temperature selection


            timestep	agent	action	chosen_action	value	    prob
            101	        0	    0	    False        	-0.093850	2.372763e-16
            101	        0	    1	    False        	-0.008329	6.360327e-09
            101	        0	    2	    True         	0.086037	1.000000e+00
            101	        1	    0	    False        	-0.138464	4.739735e-15
            101	        1	    1	    False        	0.023966	6.085078e-01

    """
    agent_actions = agent_actions.copy()
    num_bands = np.max(agent_actions)
    num_actions = num_bands+1
    num_timesteps, num_agents = agent_actions.shape
    agent_actions = agent_actions.reshape((agent_actions.shape[0], agent_actions.shape[1], 1))
    agent_values_np = np.array(agent_values)
    agent_action_prob_np = np.array(agent_action_prob)

    num_timesteps = agent_action_prob_np.shape[0]
    agent_ind = np.tile(np.arange(num_agents), num_timesteps).reshape((num_timesteps, num_agents, 1))
    action_ind = np.tile(np.arange(num_actions).reshape((1, 1, num_actions)), (num_timesteps, num_agents, 1))

    time_steps = (np.ones((num_timesteps, num_agents, 1)) * np.arange(num_timesteps).reshape((num_timesteps, 1, 1)))

    time_steps = np.tile(time_steps, (1, 1, num_actions))
    agent_actions = np.tile(agent_actions, (1, 1, num_actions))
    agent_ind = np.tile(agent_ind, (1, 1, num_actions))
    all_info_together = np.stack([time_steps, agent_ind, action_ind, agent_actions, agent_values_np, agent_action_prob_np], axis=-1)

    df_cols = ["timestep", "agent", "action", "chosen_action", "value", "prob"] 
    df = pd.DataFrame(all_info_together.reshape((num_timesteps*num_agents*(num_actions), -1)), columns=df_cols)
    df["chosen_action"] = df["chosen_action"] == df["action"]
    df["timestep"] = df["timestep"].astype(int)
    df["agent"] = df["agent"].astype(int)
    df["action"] = df["action"].astype(int)
    
    return df


def create_stacked_csv(agent_actions, agent_values, agent_action_prob):
    """Take in a binary indicator 

    Args:
        agent_actions (np.array): (num_timesteps, num_agents) with both columsn being ints
        agent_values (list(list(float))): (num_timesteps, num_agents)
        agent_action_prob (list(list(float))): (num_timesteps, num_agents)

    Returns:
        df (pd.DataFrame): Columns are ["timestep", "agent", "action", "chosen_action", "value", "prob"]  
            timestep (int): [0, num_timesteps-1]
            agent (int): [0, num_agents-1]
            action (int): [0, num_bands +1 -1]
            chosen_action (bool): Whether this particular action was actually chosen
            value (float): The agent's neural network value function output
            prob (float): The agent's neural network value function as a probability using softmax with temperature selection


            timestep	agent	action	chosen_action	value	    prob
            101	        0	    0	    False        	-0.093850	2.372763e-16
            101	        0	    1	    False        	-0.008329	6.360327e-09
            101	        0	    2	    True         	0.086037	1.000000e+00
            101	        1	    0	    False        	-0.138464	4.739735e-15
            101	        1	    1	    False        	0.023966	6.085078e-01

    """
    agent_actions = agent_actions.copy()
    num_bands = np.max(agent_actions)
    num_actions = num_bands+1
    num_timesteps, num_agents = agent_actions.shape
   
    agent_action_prob = np.array(agent_action_prob).reshape((num_timesteps, num_actions*num_agents))
    agent_values = np.array(agent_values).reshape((num_timesteps, num_actions*num_agents))

    #df_cols = ["timestep", "agent", "action", "chosen_action", "value", "prob"] 

    df_cols = [f"value_agent_{i}_action_{j}" for i in range(num_agents) for j in range(num_actions)]
    df_cols += [f"prob_agent_{i}_action_{j}" for i in range(num_agents) for j in range(num_actions)]
    df = pd.DataFrame(np.concatenate([agent_values, agent_action_prob], axis=1), columns=df_cols)
    
    df["time"] = range(num_timesteps)
    df["time"] = df["time"].astype(int)

    return df


def complete_df_to_stacked(merged):


    chosen_action_cols = [x for x in merged.columns if ("agent" in x) and (len(x.split("_")) == 2)]
    num_agents = len(chosen_action_cols)
    agent_names = [f"agent_{i}" for i in range(num_agents)]
    agent_cols = [[x for x in merged.columns if (agent in x) or "time" in x] for agent in agent_names]


    sub_dfs = [merged.loc[:, agent_cols_i] for agent_cols_i in agent_cols]

    for i, sub_df_i in enumerate(sub_dfs):
        new_cols = sub_df_i.columns
        sub_df_i.rename({agent_names[i]: "chosen_action"},axis=1, inplace=True)
        sub_df_i.columns = [x.replace(agent_names[i], "")  for x in sub_df_i.columns]
        sub_df_i["agent"] = i

    stacked_agents = pd.concat(sub_dfs)

    id_cols = [x for x in stacked_agents.columns if ("value" not in x) and ("prob" not in x)]
    val_cols = [x for x in stacked_agents.columns if ("value" in x) or ("prob" in x)]

    stacked_actions = pd.melt(stacked_agents, id_vars=id_cols, value_vars=val_cols)
    stacked_actions["value_or_prob"] = stacked_actions["variable"].apply(lambda x: x.split("_")[0])
    stacked_actions["action"] = stacked_actions["variable"].apply(lambda x: x.split("_")[-1])
    stacked_actions["action"] = stacked_actions["action"].astype(int)

    stacked_actions.pivot(columns="value_or_prob", values="value")
    stacked_actions = stacked_actions.drop("variable", axis=1)

    group_cols = [x for x in stacked_actions.columns if x not in ["value_or_prob", "value"]]


    def groupby_pic(sub): 
        val = sub.loc[sub["value_or_prob"] == "value", "value"].iloc[0]
        prob = sub.loc[sub["value_or_prob"] == "prob", "value"].iloc[0]
        return pd.Series({"value": val, "prob":prob})

    stacked_actions = stacked_actions.groupby(group_cols, as_index=False).apply(groupby_pic)
    stacked_actions["chosen_action"] = stacked_actions["chosen_action"] == stacked_actions["action"]

    stacked_actions = stacked_actions.sort_values(["time", "agent", "action"])

    return stacked_actions