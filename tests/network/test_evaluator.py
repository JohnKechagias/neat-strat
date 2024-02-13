import time

import numpy as np

from src.constants import BOARD_SIZE, MAX_TROOPS, Board
from src.network.evaluator import (
    compute_state_hash,
    get_default_state,
    get_piece_index,
    get_possible_moves,
    initialize_zobri_table,
    select_best_move,
)


def get_random_state() -> Board:
    return np.random.randint(-20, 20, size=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def test_select_best_move():
    def activate(var: np.ndarray) -> list[float]:
        return [float(var[3])]

    state = np.asarray(
        [
            [16, 0, -15, 14, -3],
            [6, -8, 6, -19, -13],
            [-1, 7, 13, 6, -12],
            [5, -13, -5, -15, 7],
            [-5, 15, 11, 17, -17],
        ],
        dtype=np.int8,
    )

    best_move, _ = select_best_move(state, activate, 3)
    assert best_move == ((0, 3), (1, 3), 14)


def test_zobrist_hash():
    table = initialize_zobri_table()
    state = get_default_state()
    hash = compute_state_hash(state, table)
    print(hash)


def test_get_piece_index():
    indexes = set()
    for value in range(-MAX_TROOPS, MAX_TROOPS + 1):
        if value == 0:
            continue

        index = get_piece_index(value)
        assert index not in indexes
        indexes.add(index)
