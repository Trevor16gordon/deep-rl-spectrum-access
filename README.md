# deep-rl-spectrum-access

![Model](static/images/problem.jpg)

In this project I'm looking to solve the problem of uncoordinated spectrum access. As described in the [DARPA Collaboration Challenge](<https://www.darpa.mil/program/spectrum-collaboration-challenge>), the number of devices communicating using the RF spectrum continues to increase. Historically, there have been centralized rigid rules for what devices can access specific frequency bands at any given time. As the number of devices increases, we need to find ways to share the frequency spectrum more efficiently. Deep Reinforcement Learning (DRL) is a promising framework for autonomous agents to learn usage patterns in the frequency spectrum and dynamically adapt to changing environments.

See [this report](https://github.com/Trevor16gordon/deep-rl-spectrum-access/blob/trevor_develop/Report.pdf) for in depth analysis on the problem and results!

# Installation
```pip install -r requirements.txt```

# Training
```
usage: train.py [-h] [--learning_rate LEARNING_RATE] [--eps_decay EPS_DECAY] [--epsilon EPSILON] [--num_agents NUM_AGENTS] [--num_bands NUM_BANDS] [--temporal_length TEMPORAL_LENGTH]
                [--reward_type REWARD_TYPE] [--obs_type OBS_TYPE] [--agents_shared_memory AGENTS_SHARED_MEMORY] [--buffer_size BUFFER_SIZE] [--episode_len EPISODE_LEN]
                [--temperature TEMPERATURE] [--reward_history_len REWARD_HISTORY_LEN] [--model_type MODEL_TYPE] [--agent_homogeneity AGENT_HOMOGENEITY]

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --eps_decay EPS_DECAY, -epsd EPS_DECAY
  --epsilon EPSILON, -eps EPSILON
  --num_agents NUM_AGENTS, -na NUM_AGENTS
  --num_bands NUM_BANDS, -b NUM_BANDS
  --temporal_length TEMPORAL_LENGTH, -tl TEMPORAL_LENGTH
  --reward_type REWARD_TYPE, -r REWARD_TYPE
  --obs_type OBS_TYPE, -o OBS_TYPE
  --agents_shared_memory AGENTS_SHARED_MEMORY, -sm AGENTS_SHARED_MEMORY
  --buffer_size BUFFER_SIZE, -bs BUFFER_SIZE
  --episode_len EPISODE_LEN, -el EPISODE_LEN
  --temperature TEMPERATURE, -t TEMPERATURE
  --reward_history_len REWARD_HISTORY_LEN, -rhl REWARD_HISTORY_LEN
  --model_type MODEL_TYPE, -mt MODEL_TYPE
  --agent_homogeneity AGENT_HOMOGENEITY, -ah AGENT_HOMOGENEITY
  ```

# To Visualize After / During Training
During training, data is periodically saved. To look at the model performance and interactively look at what the agent's value neural networks are doing, use the interactive plotter with:
```python serve_dash_plotting.py -p your_local_folder/static/example_output_data/history.csv```

look for the link "Dash is running on http://127.0.0.1:8051/" and open that link in your browser. Use the slider to zoom in on a specific section

![Alt Text](https://github.com/Trevor16gordon/deep-rl-spectrum-access/blob/main/static/images/interactive_plotting.gif)


## Overview of Repo Structure

- **environment.py**: A custom openai gym environment to represent the multi agent dynamic spectrum access problem. This file contains the logic for keeping track of agent's action and return rewards and observations to the agents. This environment can be configured for different types of reward and observation situations.
- **agent.py**: Contains different types of agents with their own policies (learned or static) for determining how they should act in their environment.
- **model.py**: Contains the tensorflow deep learning models
- **serve_dash_plotting.py**: This file is used for interactive visualization of a training run. Instructions for using are in the Visualize Section of this readme.
- **run_all_experiments.py**: This file contains calls to run specific experiments
- **utils.py**: This file contains utility functions mostly for reshaping array and preparing data to save


# Results

Here is an example of 3 agents learning to coordinate and share 2 available spectrum bands. See [this report](https://github.com/Trevor16gordon/deep-rl-spectrum-access/blob/trevor_develop/Report.pdf) for in depth analysis on the problem and results!

![image](https://github.com/Trevor16gordon/deep-rl-spectrum-access/blob/trevor_develop/static/images/three_agents_two_bands.jpg)


## Problem Formulation

At every time step time step, each autonomous agent decides between N + 1 actions where N is the number of frequency bands and the +1 corresponds with choosing not to transmit. It is assumed that all agents always have packets to transmit. If only one agent choosing to transmit on a specific band, there is a successful transmission. If two agents choose to transmit on the same bands, there is a collision. There are two goals in this scenario: 
1) Maximize the number of successful transmissions
2) Minimize the difference between successful transmissions of all agents


# References

- Deep Multi-User RL for Distributed Dynamic Spectrum Access
Oshri Naparstek and Kobi Cohen
- The Application of Deep Reinforcement Learning to Distributed Spectrum Access in Dynamic Heterogeneous Environments With Partial Observations
- ACTOR-CRITIC DEEP REINFORCEMENT LEARNING FOR DYNAMIC MULTICHANNEL ACCESS
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8646405

