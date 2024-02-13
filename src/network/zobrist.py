import random

from .constants import BOARD_SIZE, MAX_TROOPS, Board, Player


def get_random_int() -> int:
    return random.randint(0, pow(2, 64))


def initialize_zobri_table() -> list[list[list[int]]]:
    return [
        [[get_random_int() for _ in range(2 * MAX_TROOPS)] for _ in range(BOARD_SIZE)]
        for _ in range(BOARD_SIZE)
    ]


def get_piece_index(piece: int) -> int:
    if piece > 0:
        return piece - 1

    return -1 * piece + MAX_TROOPS - 1


TABLE = initialize_zobri_table()
SIDE_TO_MOVE = get_random_int()


def compute_zobri_hash(board: Board, player_to_move: Player) -> int:
    h = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                piece = get_piece_index(board[i][j])
                h ^= TABLE[i][j][piece]

    if player_to_move == Player.RED:
        h ^= SIDE_TO_MOVE

    return h
