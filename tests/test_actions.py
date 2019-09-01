import pytest
import numpy as np
from itertools import repeat
import re
from src.game import Board, Action


def board_from_string(s, shape=None, dtype=np.int32) -> Board:
    """Utility function to create boards that are visually easy to see + validate

    Args:
        s: the string we want to make a board from
        shape: if None, we try to make a square array of `s`
    """
    sep = " "
    arr = np.fromstring(re.sub(r"\s+", sep, s.strip("\n")), sep=sep)
    shape = repeat(int(np.sqrt(arr.size)), 2) if shape is None else shape
    arr = arr.astype(dtype).reshape(*shape)
    return Board.from_array(arr)


class TestSwiping:
    """Test class to see if some simple swiping works as intended"""

    def _test_swipe(self, before, left, right, up, down):
        """Helper function. All arguments are strings"""
        bfs = board_from_string
        actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
        for s, a in zip([left, right, up, down], actions):
            B0 = bfs(before)
            actual = Board.apply_action_on_board(B0, a)
            expected = board_from_string(s)

            err_msg = f"Action {a}: Got {actual.render()}, expected {expected.render()}"
            assert actual == expected, err_msg

    def test_simple_swiping_2x2(self):
        before = """1 1
                    2 2"""
        self._test_swipe(before, "2 0 4 0", "0 2 0 4", "1 1 2 2", "1 1 2 2")

    def test_simple_swiping_3x3(self):
        before = """2 1 1
                    2 1 4
                    2 3 4"""

        self._test_swipe(
            before,
            "2 2 0 2 1 4 2 3 4",
            "0 2 2 2 1 4 2 3 4",
            "4 2 1 2 3 8 0 0 0",
            "0 0 0 2 2 1 4 3 8",
        )

    def test_simple_swiping_4x4(self):
        before = """  4   2   0   0
                      8   4   2   0
                     16   8   4   2
                    512 128  64  32"""

        self._test_swipe(
            before,
            before,
            "0 0 4 2\n0 8 4 2\n16 8 4 2\n512 128 64 32\n",
            "4 2 2 2\n8 4 4 32\n16 8 64 0\n512 128 0 0\n",
            before,
        )

    def test_another_simple_swiping_4x4(self):
        before = """  2   0   4   2
                      0   8   4   2
                     16   8   4   2
                    512 128  64  32"""

        self._test_swipe(
            before,
            "2 4 2 0\n8 4 2 0\n16 8 4 2\n512 128 64 32",
            "0 2 4 2\n0 8 4 2\n16 8 4 2\n512 128 64 32",
            "2 16 8 4\n16 128 4 2\n512 0 64 32\n0 0 0 0",
            "0 0 0 0\n2 0 4 2\n16 16 8 4\n512 128 64 32",
        )


def test_game_over_checking_works_as_expected():
    not_game_over_strings = ["1 1 1 1", "0 0 0 0", "0 1 0 0", "3 2 1 3 4 5 6 7 8"]
    for s in not_game_over_strings:
        board = board_from_string(s)
        assert not Board.is_game_over(board)

    game_over_strings = ["1 2 3 4", "1 2 3 4 5 6 7 8 9", "4 3 7 2"]
    for s in game_over_strings:
        board = board_from_string(s)
        assert Board.is_game_over(board)


def test_spawning_random_boards():
    board0 = Board.init_random((100, 100), 5)
    assert np.argwhere(board0.as_array()).shape == (5, 2)

    board1 = Board.init_random((4, 4), 2)
    assert np.argwhere(board1.as_array()).shape == (2, 2)

    board2 = Board.init_random((5, 5), 0)
    assert np.argwhere(board2.as_array()).shape == (0, 2)


def test_available_actions_work_as_expected():
    s = """1 2 3
           1 4 5
           6 7 8"""
    board = board_from_string(s)
    available_actions = Board.get_available_actions(board)
    assert available_actions == set([Action.DOWN, Action.UP])

    s = """ 1  2  3  4
            5  6  7  8
            9 10 11 12
           13 14 15 16"""
    board = board_from_string(s)
    available_actions = Board.get_available_actions(board)
    assert available_actions == set([])

    s = """ 1  2  2  4
            5  2  7  8
            9 10 11 12
           13 14 15 16"""
    board = board_from_string(s)
    available_actions = Board.get_available_actions(board)
    assert available_actions == set([Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN])
