# -*- coding: utf-8 -*-
"""Single place for all config"""


class TrainingConfig():
    """
    Config for training Info GAN
    """
    EPS_DECAY = 1e-5
    EPSILON = 0.1
    NUM_AGENTS = 2
    NUM_BANDS = 1
    TEMPORAL_LENGTH = 5
    REWARD_TYPE = "transmision_normalized"
    OBS_TYPE = "aggregate3"
    AGENTS_SHARED_MEMORY = True
    BUFFER_SIZE = 1000
    EPISODE_LEN = 40001
    TEMPERATURE = 0.005
    REWARD_HISTORY_LEN = 100
    MODEL_TYPE = "ddqn"
    AGENT_HOMOGENEITY = "all_same" # Can also be "one_periodic"


class ReportingConfig():
    """
    Config for reporting / visulalization
    """
    TOP_LEVEL_FOLDER = "plots.nosync/"
    IMAGES_TO_SAVE = 10
    SAVE_CHECKPOINT_CSV_EVERY = 500
    PRINT_UPDATE_EVERY = 10
    CONFIG_FILE_NAME_TO_SAVE = "config.csv"