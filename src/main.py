import src.filter_deprecationwarnings

import os

import tensorflow.keras.backend as K

K.set_image_data_format("channels_last")
import ray
from ray import tune, remote
from ray.rllib.agents.ppo import PPOTrainer
from ray.experimental import named_actors
from ray.tune.registry import register_env

from src.environment import (
    Game2048,
    LogRewardWrapper,
    ObservationWrapper,
    MonitorWrapper,
)
from src.agent import register_agent

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
register_agent()


def register_wrapped_env(env_cls, env_name, wrappers):
    """Wrap `env_cls` with a bunch of different wrappers and regster as `env_name`.

    Args:
        env_cls: The Environment we would like to register
        env_name: str, the environment name
        wrappers: list of `gym.Wrapper`
            Wrappers we want to wrap around the base environment
    """

    def env_creator(env_config):
        env = env_cls(env_config)
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)
        return env

    register_env(env_name, env_creator)
    print(f"Registered environment: {env_name}")


env_config = {"board_shape": (4, 4), "max_tile_value": 8192}
register_wrapped_env(
    Game2048, "Game2048", [LogRewardWrapper, ObservationWrapper, MonitorWrapper]
)


def on_episode_end(info):
    env = info["env"].get_unwrapped()[0].unwrapped
    info["episode"].user_data["board"] = env.board.values
    info["episode"].user_data["score"] = env.board.score
    info["episode"].custom_metrics["score"] = env.board.score
    info["episode"].custom_metrics["highest_tile"] = env.board.values.max()


callbacks = {"on_episode_end": tune.function(on_episode_end)}

config = {
    "monitor": True,  # episode stats and dir to ~/ray_results
    "env": "Game2048",
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
}
ray.init(object_store_memory=10 ** 9, ignore_reinit_error=True)
trainer = PPOTrainer(config=config)

row_format = "{:>20}" * 4
for it in range(globals().get("it", 0), globals().get("it", 0) + 200):
    if it == 0 or it % 50 == 0:
        print(row_format.format("iteration", "mean reward", "score max", "max tile"))
    r = trainer.train()
    print(
        row_format.format(
            it,
            r["episode_reward_mean"],
            r["custom_metrics"]["score_max"],
            r["custom_metrics"]["highest_tile_max"],
        )
    )
