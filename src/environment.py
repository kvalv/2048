import gym
from time import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from gym.wrappers import Monitor
from functools import partial

from src.game import Board, Action
from src.spaces import EnumSpace, BoardSpace

K.set_image_data_format("channels_last")  # workers probably have channels_first


class Game2048(gym.Env):
    metadata = {"render.modes": Board.modes}
    reward_range = (-1, 1)
    board: Board
    t: int = 0

    action_space = gym.spaces.Discrete(4)

    def __init__(self, env_config: dict):
        """
        Args:
            env_config: a dictionary with the following keys:
                'board_shape': int, int
                'max_tile_value': int

        """
        self.env_config = env_config
        # self.observation_space = gym.spaces.Box(
        #     0, 2 ** 16, shape=env_config["board_shape"], dtype=np.int32
        # )
        self._max_tile_value = env_config["max_tile_value"]
        n_channels = int(np.log2(self._max_tile_value) + 1)
        shape = (
            (n_channels, 4, 4)
            if K.image_data_format() == "channels_first"
            else (4, 4, n_channels)
        )
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(0, high=1.0, shape=shape, dtype=np.float32),
                "valid_action_mask": gym.spaces.Box(
                    0.0, 1.0, shape=(4,), dtype=np.float32
                ),
            }
        )
        self.reset()

    def render(self, mode):
        return self.board.render(mode)

    def reset(self):
        self.t = 0
        self._step_counter = 0
        self.board = Board.init_random(shape=self.env_config["board_shape"], n_tiles=2)
        return self.board

    def step(self, action: int):
        """A single step in the game

        rewards: the natural logarithm of difference of the 2048 scoring.
        We also add some penalty in case of no-op
        """
        self._step_counter += 1

        done = False
        if self._step_counter > 100 and self._step_counter / (self.t + 1) > 5:
            # add 1 to avoid DivisionByZero for `self.t`. Yes, it happenend.
            done = True

        action = Action(1 + action)  # discrete -> enum (which is 1-indexed)
        modified_board = Board.apply_action_on_board(self.board, action)
        info = {"step": self.t}

        if Board.is_game_over(modified_board):
            done = True
            reward = 0
        else:

            # An action is invalid if it doesn't change the board.
            valid_action = modified_board != self.board
            if not valid_action:
                # We penalize the agent for doing no-op moves!!! >:(
                penalty = -0.1
                info["no-op"] = True
            else:
                modified_board = Board.spawn_random_tile(modified_board)
                penalty = 0
                self.t += 1
                info["no-op"] = False

            diff = modified_board.score - self.board.score

            reward = np.log(1 + diff) + penalty
            reward = np.clip(reward, -11, 10)  # TODO: move to a wrapper.

        self.board = modified_board
        return self.board, reward, done, info


class LogRewardWrapper(gym.RewardWrapper):
    def reward(self, value):
        return np.log(1 + value)


class ObservationWrapper(gym.ObservationWrapper):
    def observation(self, board):
        """Convert observation to numpy array with a unique channel for each tile.

        A `Board` cannot be used as an observaton. RLlib will complain and crash because
        RLlib expects arrays as observations. Therefore, we convert the `Board` to a
        numpy array, where the first channel has value 1 if it's empty.  The second
        channel correspond to tiles with value 2, the third with value 3 and so on.

        The number of channels in the observation will be `1 + log2(max_tile_value)`.
        For example, `max_tile_value == 256` --> we have 9 tile values.

        Note:
            We assume all tiles are a multiple of 2!

        Returns:
            A dict with the following keys and values:
                - 'valid_action_mask': np.ndarray(4, float)
                    The available actions
                - 'board': np.ndarray((n_rows, n_cols, n_channels), float)
                    The board (in one-hot format).
        """
        channel_indices = np.log2(np.where(board.values == 0, 1, board.values))

        frac_values, _ = np.modf(channel_indices)
        if not frac_values.max() == 0:
            raise ValueError(
                "Unexpected input: got a tile that was not a power of 2. Can't "
                "safely convert observation."
            )
        channel_indices = channel_indices.astype(int)

        yy, xx = np.meshgrid(*[range(dim) for dim in channel_indices.shape])

        one_hot_board = np.zeros(self.env.observation_space["board"].shape)
        if K.image_data_format() == "channels_first":
            one_hot_board[channel_indices.ravel(), yy.ravel(), xx.ravel()] = 1.0
        else:
            one_hot_board[yy.ravel(), xx.ravel(), channel_indices.ravel()] = 1.0

        valid_action_mask = np.zeros(4)
        for action in Board.get_available_actions(board):
            index = action.value - 1  # enums are 1-indexed, so we subtract by 1.
            valid_action_mask[index] = 1.0

        processed_obs = {"valid_action_mask": valid_action_mask, "board": one_hot_board}
        return processed_obs


MonitorWrapper = partial(
    Monitor,
    directory=f"./videos/{time()}/",
    video_callable=lambda ep_id: True,
    force=True,
)
