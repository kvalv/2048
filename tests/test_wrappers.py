import numpy as np

from src.environment import Game2048, ObservationWrapper


def test_observation_wrapper():
    board_shape, max_tile_value = (4, 4), 256
    env = Game2048(
        env_config={"board_shape": board_shape, "max_tile_value": max_tile_value}
    )

    wrapped_env = ObservationWrapper(env)

    obs = wrapped_env.reset()
    for k, v in env.observation_space.spaces.items():
        assert obs[k].shape == v.shape

    # we also don't want to put in observations that are not used.
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())

    # one-hot encoding --> must sum to 1 on all channels.
    np.testing.assert_array_equal(obs["board"].sum(axis=2), np.ones(board_shape))

    # obs.board.values = np.array([[2, 2, 4], [8, 0, 0], [8, 4, 2]])

    # 0 / 0
