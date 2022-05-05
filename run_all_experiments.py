""" 
File to run all experiments and generate figures and tables
"""
from train import train_agents

def run_experiment_basic_coordination_one_primary_user():
    """Basic learning experiment


    1 Agent is a primary user that simply transmits periodically
    1 Agent is a learning agent
    """

    class TrainingConfigOnePrimaryUser():
        """
        Config for training Info GAN
        """
        learning_rate=0.0001
        eps_decay = 1e-5
        epsilon = 0.1
        num_agents = 2
        num_bands = 1
        temporal_length = 10
        reward_type = "transmision_normalized"
        obs_type = "aggregate"
        agents_shared_memory = False
        buffer_size = 1000
        episode_len = 2001
        temperature = 0.005
        reward_history_len = 100
        model_type = "ddqn"
        agent_homogeneity = "one_periodic"

    train_agents(TrainingConfigOnePrimaryUser)

def run_experiment_basic_coordination_one_primary_user_actor_critic():
    """Basic learning experiment

    Using policy gradient

    1 Agent is a primary user that simply transmits periodically
    1 Agent is a learning agent
    """

    class TrainingConfigOnePrimaryUser():
        """
        Config for training Info GAN
        """
        learning_rate=0.0001
        eps_decay = 1e-5
        epsilon = 0.1
        num_agents = 2
        num_bands = 1
        temporal_length = 10
        reward_type = "transmision_normalized"
        obs_type = "aggregate"
        agents_shared_memory = False
        buffer_size = 1000
        episode_len = 10001
        temperature = 0.005
        reward_history_len = 100
        model_type = "actorcritic"
        agent_homogeneity = "one_periodic"

    train_agents(TrainingConfigOnePrimaryUser)

def run_experiment_coordination_two_agents():
    """Basic learning experiment


    1 Agent is a primary user that simply transmits periodically
    1 Agent is a learning agent
    """

    class TrainingConfigTwoAgents():
        """
        Config for training Info GAN
        """
        learning_rate=0.0001
        eps_decay = 1e-5
        epsilon = 0.1
        num_agents = 2
        num_bands = 1
        temporal_length = 10
        reward_type = "transmision_normalized"
        obs_type = "aggregate"
        agents_shared_memory = False
        buffer_size = 1000
        episode_len = 6001
        temperature = 0.005
        reward_history_len = 5
        model_type = "ddqn"
        agent_homogeneity = "all_same"

    train_agents(TrainingConfigTwoAgents)

def run_experiment_coordination_three_agents():
    """Basic learning experiment


    1 Agent is a primary user that simply transmits periodically
    1 Agent is a learning agent
    """

    class TrainingConfigTwoAgents():
        """
        Config for training Info GAN
        """
        learning_rate=0.0001
        eps_decay = 1e-5
        epsilon = 0.1
        num_agents = 3
        num_bands = 2
        temporal_length = 10
        reward_type = "transmision_normalized"
        obs_type = "aggregate"
        agents_shared_memory = False
        buffer_size = 1000
        episode_len = 15001
        temperature = 0.005
        reward_history_len = 100
        model_type = "ddqn"
        agent_homogeneity = "all_same"

    train_agents(TrainingConfigTwoAgents)

if __name__ == "__main__":

    # run_experiment_basic_coordination_one_primary_user()
    # run_experiment_basic_coordination_one_primary_user_actor_critic()
    # run_experiment_coordination_two_agents()
    # run_experiment_coordination_two_agents_actor_critic()
    run_experiment_coordination_three_agents()