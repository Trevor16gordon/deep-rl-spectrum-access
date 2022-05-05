# -*- coding: utf-8 -*-
"""Single place for all config"""


class TrainingConfig():
    """
    Config for training Info GAN
    """
    learning_rate=0.0001
    eps_decay = 1e-5
    epsilon = 0.1
    num_agents = 2
    num_bands = 1
    temporal_length = 5
    reward_type = "transmision_normalized"
    obs_type = "aggregate"
    agents_shared_memory = True
    buffer_size = 1000
    episode_len = 6001
    temperature = 0.005
    reward_history_len = 100
    model_type = "ddqn"
    agent_homogeneity = "all_same" # Can also be "one_periodic"


class ReportingConfig():
    """
    Config for reporting / visulalization
    """
    TOP_LEVEL_FOLDER = "/Users/trevorgordon/Library/Mobile Documents/com~apple~CloudDocs/Documents/root/Columbia/Spring2022/Research/deep-rl-spectrum-access/plots.nosync/"
    IMAGES_TO_SAVE = 10
    SAVE_CHECKPOINT_CSV_EVERY = 50
    PRINT_UPDATE_EVERY = 10
    CONFIG_FILE_NAME_TO_SAVE = "config.csv"