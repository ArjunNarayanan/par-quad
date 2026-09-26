import numpy as np
import yaml
from envs.environment_maker import get_env_feature_size
from src.feature_extractor import feature_extractor_initializer
from stable_baselines3 import PPO


def load_yaml_config(config_fn):
    print("\nLOADING CONFIG FILE AT : ", config_fn)
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def load_model_from_checkpoint(checkpoint_file, config_file):
    config = load_yaml_config(config_file)
    features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config)
    num_input_features = get_env_feature_size(config["environment"])
    features_extractor_kwargs.update({"input_features": num_input_features})
    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
    )
    ppo_config = dict(policy_kwargs=policy_kwargs)
    model = PPO.load(checkpoint_file, custom_objects=ppo_config)
    return model


def learning_rate_schedule(value):
    """A float, or `"lin_<rate>"` for a rate that decays linearly to zero.

    Best-of-5 scores wander between neighbouring checkpoints at a fixed rate --
    the metric's own sd is 1.6 domains, and a policy still taking full-size steps
    at the end of a run adds to that. Decaying the rate lets the last checkpoints
    settle, and it is the last checkpoint that gets packaged.

    Lives here because every entry point that builds a PPO has to apply it:
    `train_warm_ppo` did and `behaviour_cloning.build_model` did not, so a config
    carrying `lin_` trained fine under one and asserted under the other.
    """
    if isinstance(value, str) and value.startswith("lin_"):
        rate = float(value[4:])
        return lambda progress_remaining: progress_remaining * rate
    return float(value)
