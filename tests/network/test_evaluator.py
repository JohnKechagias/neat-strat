import numpy as np

from src.constants import MAX_TROOPS
from src.network.constants import Player
from src.network.game_state import (
    GameState,
    make_move,
    switch_player_to_move,
    undo_move,
)
from src.network.search import Searcher, get_default_state, get_possible_moves
from src.network.zobrist import compute_zobri_hash, get_hash, get_piece_index


def a_test_select_best_move():
    def evaluator(var: np.ndarray) -> list[float]:
        return [float(var[3])]

    searcher = Searcher(1024, 3)
    board = np.asarray(
        [
            [9, 0, -8, 10, -3],
            [6, -8, 6, -10, -8],
            [-1, 7, 8, 6, -6],
            [5, -6, -5, -5, 7],
            [-5, 8, 5, 6, -6],
        ],
        dtype=np.int8,
    )
    state = GameState(board, compute_zobri_hash(board, Player.BLUE), Player.BLUE)
    best_move = searcher.search(evaluator, state)
    assert best_move == ((0, 4), (0, 4))


def test_get_possible_moves():
    state = get_default_state()
    moves = get_possible_moves(state.board, state.player_to_move)
    for move in moves:
        if len(move) == 2:
            assert move[0] == move[1]
            assert state.board[move[0][0], move[0][1]] > 0
        elif len(move) == 3:
            assert state.board[move[0][0], move[0][1]] > 0
            assert move[2] in range(0, MAX_TROOPS + 1)
        else:
            raise ValueError("Move length should be either 2 or 3.")

    switch_player_to_move(state)
    moves = get_possible_moves(state.board, state.player_to_move)
    for move in moves:
        if len(move) == 2:
            assert move[0] == move[1]
            assert state.board[move[0][0], move[0][1]] < 0
        elif len(move) == 3:
            assert state.board[move[0][0], move[0][1]] < 0
            assert move[2] in range(-MAX_TROOPS, 0)
        else:
            raise ValueError("Move length should be either 2 or 3.")


def test_zobrist_hash():
    state = get_default_state()
    moves = [((4, 0), (4, 0)), ((4, 0), (4, 1), 9)]
    for move in moves:
        original_hash = state.hash
        make_move(state, move)
        new_hash = state.hash
        undo_move(state)
        assert state.hash == original_hash
        assert new_hash != original_hash


def test_switch_current_player():
    state = get_default_state()
    original_hash = state.hash
    switch_player_to_move(state)
    assert state.hash != original_hash
    assert state.player_to_move == Player.RED
    switch_player_to_move(state)
    assert state.hash == original_hash
    assert state.player_to_move == Player.BLUE


def test_make_and_undo_move():
    state = get_default_state()
    reproduction_move = ((4, 0), (4, 0))
    make_move(state, reproduction_move)
    assert state.board[4][0] == 11
    undo_move(state)
    assert state.board[4][0] == 10

    reposition_move = ((4, 0), (4, 1), 9)
    make_move(state, reposition_move)
    assert state.board[4][0] == 1
    assert state.board[4][1] == 9
    undo_move(state)
    assert state.board[4][0] == 10
    assert state.board[4][1] == 0

    reproduction_move = ((0, 4), (0, 4))
    make_move(state, reproduction_move)
    assert state.board[0][4] == -11
    undo_move(state)
    assert state.board[0][4] == -10

    reposition_move = ((0, 4), (1, 4), -9)
    make_move(state, reposition_move)
    assert state.board[0][4] == -1
    assert state.board[1][4] == -9
    undo_move(state)
    assert state.board[0][4] == -10
    assert state.board[1][4] == 0


def test_make_and_undo_move_to_enemy_tile():
    state = get_default_state()
    state.board[1][4] = 10
    state.board[4][0] = 0
    state.player_to_move = Player.RED
    state.hash = compute_zobri_hash(state.board, Player.RED)

    original_hash = state.hash
    reposition_move_to_enemy_tile = ((0, 4), (1, 4), -9)
    make_move(state, reposition_move_to_enemy_tile)

    assert state.player_to_move == Player.BLUE
    assert state.board[0][4] == -1
    assert state.board[1][4] == 1
    undo_move(state)
    assert state.hash == original_hash
    assert state.player_to_move == Player.RED
    assert state.board[0][4] == -10
    assert state.board[1][4] == 10

    reposition_move_to_enemy_tile = ((1, 4), (0, 4), 9)
    state.player_to_move = Player.BLUE
    state.hash = compute_zobri_hash(state.board, Player.BLUE)
    original_hash = state.hash

    make_move(state, reposition_move_to_enemy_tile)
    assert state.player_to_move == Player.RED
    assert state.board[0][4] == -1
    assert state.board[1][4] == 1
    undo_move(state)
    assert state.hash == original_hash
    assert state.player_to_move == Player.BLUE
    assert state.board[0][4] == -10
    assert state.board[1][4] == 10


def test_get_piece_index():
    indexes = set()
    for value in range(-MAX_TROOPS, MAX_TROOPS + 1):
        if value == 0:
            continue

        index = get_piece_index(value)
        assert index not in indexes
        get_hash(0, 0, value)
        indexes.add(index)
