import src.filter_deprecationwarnings

import os

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from src.environment import Game2048
from src.agent import register_agent

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
register_agent()


def register_wrapped_env(env_cls, name):
    """We wrap around an environment with the Monitor to create nice videos."""
    from time import time
    from gym.wrappers import Monitor
    from ray.tune.registry import register_env

    def env_creator(env_config):
        env = env_cls(env_config)
        wrapped_env = Monitor(
            env, f"./videos/{time()}/", video_callable=lambda ep_id: True, force=True
        )
        return wrapped_env

    register_env(name, env_creator)
    print(f"Registered environment: {name}")


env_config = {"board_shape": (4, 4)}
register_wrapped_env(Game2048, "Game2048_monitored")


def on_episode_end(info):
    env = info["env"].get_unwrapped()[0].unwrapped
    info["episode"].user_data["board"] = env.board.values
    info["episode"].user_data["score"] = env.board.score
    info["episode"].custom_metrics["score"] = env.board.score
    info["episode"].custom_metrics["highest_tile"] = env.board.values.max()


callbacks = {"on_episode_end": tune.function(on_episode_end)}

tune.run(
    "PPO",
    config={
        "monitor": True,  # episode stats and dir to ~/ray_results
        "env": "Game2048_monitored",
        "env_config": env_config,
        "num_workers": 15,
        "train_batch_size": 15 * 256,
        "sample_batch_size": 256,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 2,
        "env_config": env_config,
        "model": {"custom_model": "custom_model"},
        "entropy_coeff": 0.00,  # consider setting to .001
        "vf_clip_param": 10,
        "gamma": 0.9,
        "lambda": 0.9,
        "num_gpus": len(os.environ.get("CUDA_VISIBLE_DEVICES", [])),
        "callbacks": callbacks,
        "lr": 1e-4,
        "vf_share_layers": True,
        "vf_loss_coeff": 0.005 / 10,  # tuned to specific experiment.
    },
)
# for debugging, use `local_mode=True`
