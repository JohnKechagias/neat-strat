import numpy as np

from src.constants import BOARD_SIZE, State
from src.network.evaluator import select_best_move


def get_random_state() -> State:
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
