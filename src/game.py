from __future__ import annotations

from typing import List, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import gym
import numpy as np

from functools import partial
from src import utils
from src.spaces import EnumSpace

# other ideas: add a bomb tile or something like that which will blow other
# tiles up.


Action = Enum("Action", "UP DOWN LEFT RIGHT")

NO_TILE_VALUE = 0


@dataclass
class Tile:
    """Dataclass to generate different tiles

    Args:
        value: the value numerical value of the tile
    """

    modes: ["terminal", "rgb_array"]  # allowed `render` modes
    value: int

    @staticmethod
    def get_color_code(value):
        """Get a hex representation of the given tile"""
        return {
            0: "#DDDDDD",  # some white color
            2: "#FAE7E0",
            4: "#F5E5CE",
            8: "#FEB17D",
            16: "#EB8E53",
            32: "#F87A63",
            64: "#E95839",
            128: "#F3D96B",
            256: "#F1D04B",
            512: "#E4C02A",
            1024: "#ECC400",
            2048: "#F46575",
            4096: "#F34B5C",
        }[value]

    @staticmethod
    def as_rgb(value):
        h = Tile.get_color_code(value)
        return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


class Board:
    modes = ["rgb_array", "terminal"]

    def __init__(self, n_rows: int, n_cols: int):
        """Container class for holding tiles.

        This class also have static methods that are made to modify boards.

        Args:
            n_rows: the number of rows in the layout
            n_cols: the number of columns in the layout

        """

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.score: int = 0
        self.values = np.full((self.n_rows, self.n_cols), NO_TILE_VALUE, np.int32)

    def __eq__(self, other):
        if not other.n_rows == self.n_rows and other.n_cols == self.n_cols:
            return False
        return np.all(self.as_array() == other.as_array())

    @classmethod
    def from_array(cls, arr):
        n_rows, n_cols = arr.shape
        board = Board(n_rows, n_cols)
        board.values = arr
        return board

    @classmethod
    def init_random(cls, shape, n_tiles):
        arr = np.zeros(shape, np.int32)
        board = cls.from_array(arr)
        for _ in range(n_tiles):
            board.spawn_random_tile(board)
        return board

    def as_array(self):
        return self.values

    def as_observation(self):
        transformed_obs = np.log(self.values + 1) / np.log(2)
        valid_action_mask = np.zeros(4)
        for action in Board.get_available_actions(self):
            index = action.value - 1  # enums are 1-indexed, so we subtract by 1.
            valid_action_mask[index] = 1.0

        return {"obs": transformed_obs, "valid_action_mask": valid_action_mask}

    def render(self, mode="terminal"):
        assert mode in self.modes, "Mode not supported."
        if mode == "terminal":
            print(self.values)
            return None
        elif mode == "rgb_array":
            """RGB representation, except enhanced `N` times"""
            N = 50
            arr = np.empty((self.n_rows * N, self.n_cols * N, 3), dtype=np.uint8)
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    rgb_val = Tile.as_rgb(self.values[i, j])
                    x = np.tile(np.reshape(rgb_val, (1, 1, 3)), (N, N, 1))
                    arr[N * i : N * (i + 1), N * j : N * (j + 1)] = x
            return arr
        else:
            raise NotImplementedError

    @staticmethod
    def get_available_actions(board: Board) -> Set[Action]:
        all_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        return set(
            a for a in all_actions if Board.apply_action_on_board(board, a) != board
        )

    @staticmethod
    def apply_action_on_board(board: Board, action: Action) -> Board:
        """Apply an action on a board and get a new board, along with its score

        NOTE: this will not add new tiles to the board! We only do the action, i.e. we
        only do a swipe.

        """

        transform_op = {
            action.LEFT: lambda x: x,
            action.RIGHT: np.fliplr,
            action.UP: partial(np.rot90, k=+1),
            action.DOWN: partial(np.rot90, k=-1),
        }[action]
        inverse_op = {
            action.LEFT: lambda x: x,
            action.RIGHT: np.fliplr,
            action.UP: partial(np.rot90, k=-1),
            action.DOWN: partial(np.rot90, k=+1),
        }[action]

        values = np.array(board.as_array())
        transformed_board = transform_op(values)
        processed_rows, new_points = [], 0
        for row in transformed_board:
            processed_row, points = Board.process_row(row)
            processed_rows.append(processed_row)
            new_points += points
        transformed_new_board = np.vstack(processed_rows)
        new_board = inverse_op(transformed_new_board)
        new_board = Board.from_array(new_board)
        new_board.score = board.score + new_points
        return new_board

    @staticmethod
    def spawn_random_tile(board: Board, tile_value=None):
        """Spawn a random tile on the board. Optionally specify `tile_value`"""
        values = board.as_array()
        potential_coordinates = np.argwhere(values == NO_TILE_VALUE)
        if len(potential_coordinates) == 0:
            raise RuntimeError("Could not spawn a random tile. Board is full!")

        loc = utils.random_choice_along_axis(potential_coordinates, axis=0)
        values[tuple(loc)] = tile_value or np.random.choice([2, 4], p=[0.8, 0.2])
        modified_board = Board.from_array(values)
        modified_board.score = board.score
        return modified_board

    @staticmethod
    def is_game_over(board: Board,) -> bool:
        # First we check if there exists at least one free tile. If it does, then it's not
        # game over.
        if (board.as_array() == NO_TILE_VALUE).any():
            return False

        return len(Board.get_available_actions(board)) == 0

    @staticmethod
    def process_row(row: np.ndarray,) -> Tuple(np.ndarray, int):
        """Process a single row in the 2048 game, simulating a left-swipe.

        Returns:
            np.ndarray: the potentially modified row as a result of a left-swipe
            int: the additional points received if the row is modified. The score will
                 be the sum of merged tiles.

        Examples:
            >>> process_row(np.array([32, 32, 32, 32]))
            array([64, 64,  0,  0]), 128
            >>> process_row(np.array([2, 4, 2, 4]))
            array([2, 4, 2, 4]), 0
            >>> process_row(np.array([2, 2, 2, 4]))
            array([4, 2, 4, 0]), 4
            >>> process_row(np.array([32, 32, 8, 32]))
            array([64,  8, 32,  0]), 64
            >>> process_row(np.array([0, 0, 0, 2]))
            array([2, 0, 0, 0]), 0
            >>> process_row(np.array([0, 0, 4, 0]))
            array([4, 0, 0, 0]), 0
            >>> process_row(np.array([0, 0, 0, 0]))
            array([0, 0, 0, 0]), 0
            >>> process_row(np.array([128, 128, 128, 64, 64]))
            array([256, 128, 128,   0,   0]), 384

        """
        row = np.array(row)  # let's make a copy
        already_merged = np.full_like(row, False, dtype=bool)
        score = 0

        for i in range(1, row.size):
            value = row[i]
            preceding_elems = row[:i]
            mask = preceding_elems != NO_TILE_VALUE
            row[i] = NO_TILE_VALUE
            if mask.any():  # we have at least one element to the left of us
                neighbouring_value = preceding_elems[mask][-1]
                neighbour_idx = np.argwhere(mask)[-1]
                if neighbouring_value == value and not already_merged[neighbour_idx]:
                    # we can merge
                    row[neighbour_idx] = value * 2
                    already_merged[neighbour_idx] = True
                    score += value * 2
                else:  # we can't merge. Need to "move" to the adjacent tile
                    row[neighbour_idx + 1] = value
            else:  # no elements to the left of us, so we move to the leftmost position
                row[0] = value

        return row, score
