import gym
import numpy as np

from src.game import Board, Action
from src.spaces import EnumSpace, BoardSpace


class Game2048(gym.Env):
    metadata = {"render.modes": Board.modes}
    reward_range = (-1, 1)
    board: Board
    t: int = 0

    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box

    def __init__(self, env_config: dict):
        """
        Args:
            env_config: a dictionary with the following keys:
                'board_shape': int, int

        """
        self.env_config = env_config
        self.observation_space = gym.spaces.Box(
            0, 2 ** 16, shape=env_config["board_shape"], dtype=np.int32
        )
        self.reset()

    def render(self, mode):
        return self.board.render(mode)

    def reset(self):
        self.t = 0
        self._step_counter = 0
        self.board = Board.init_random(shape=self.env_config["board_shape"], n_tiles=2)
        return self.board.as_observation()

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
            # if np.random.rand() > 0.975:
            #     print(f"Score: {modified_board.score}")
            #     modified_board.display()
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

        self.board = modified_board
        return modified_board.as_observation(), reward, done, info
