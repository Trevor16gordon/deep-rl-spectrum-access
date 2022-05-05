import time
import ray
import copy
from ray import tune
from config_file import TrainingConfig
from train import train_agents


def evaluation_fn(step, width, height):
    time.sleep(0.1)
    return (0.1 + width * step / 100)**(-1) + height * 0.1


def easy_objective(config):

    train_config = copy.deepcopy(TrainingConfig)

    # override argparse with ray tune params
    for k, v in config.items():
        setattr(train_config, k, v)
    
    train_agents(train_config, report_tune=True)


if __name__ == "__main__":
    
    ray.init(configure_logging=False)

    analysis = tune.run(
        easy_objective,
        metric="episode_reward_mean",
        mode="max",
        num_samples=10,
        config={
            "learning_rate": tune.uniform(0.0001, 0.001),
            "eps_decay": tune.uniform(1e-5, 1e-6),
            "eps": tune.uniform(0.1, 0.9),
            "temporal_length": tune.choice([5,8,10,12,15,18,20]),
        })

    print("Best hyperparameters found were: ", analysis.best_config)



    # should save the df results here ..