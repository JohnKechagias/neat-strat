from dataclasses import dataclass, field

from . import zobrist
from .constants import Board, Move, Player


@dataclass
class GameState:
    board: Board
    hash: int
    player_to_move: Player
    history: list[Move] = field(default_factory=list)


def make_move(state: GameState, move: Move):
    temp = state.board.copy()
    # If its a production move.
    if len(move) == 2:
        x, y = move[0]

        previous_value = state.board[x][y]
        state.hash ^= zobrist.get_hash(x, y, previous_value)

        if previous_value > 0:
            new_value = previous_value + 1
        else:
            new_value = previous_value - 1

        state.board[x][y] = new_value
        state.hash ^= zobrist.get_hash(x, y, new_value)

    # If its a reposition move.
    elif len(move) == 3:
        source_x, source_y = move[0]
        target_x, target_y = move[1]
        troops = move[2]

        source_prev_value = state.board[source_x][source_y]
        state.hash ^= zobrist.get_hash(source_x, source_y, source_prev_value)

        source_new_value = source_prev_value - troops
        state.board[source_x][source_y] = source_new_value
        if source_new_value != 0:
            state.hash ^= zobrist.get_hash(source_x, source_y, source_new_value)

        target_prev_value = state.board[target_x][target_y]
        if target_prev_value != 0:
            state.hash ^= zobrist.get_hash(target_x, target_y, target_prev_value)

        target_new_value = target_prev_value + troops
        state.board[target_x][target_y] = target_new_value
        try:
            state.hash ^= zobrist.get_hash(target_x, target_y, target_new_value)
        except IndexError as e:
            print("Before Move")
            print(temp)
            print("After Move")
            print(state.board)
            print(source_prev_value)
            print(source_new_value)
            print(target_prev_value)
            print(target_new_value)
            print(move)
            print(state.history)
            raise e

    state.history.append(move)
    switch_player_to_move(state)


def undo_move(state: GameState):
    move = state.history.pop()

    # If its a production move.
    if len(move) == 2:
        x, y = move[0]

        previous_value = state.board[x][y]
        state.hash ^= zobrist.get_hash(x, y, previous_value)

        if previous_value > 0:
            new_value = previous_value - 1
        else:
            new_value = previous_value + 1

        state.board[x][y] = new_value
        state.hash ^= zobrist.get_hash(x, y, new_value)

    # If its a reposition move.
    elif len(move) == 3:
        source_x, source_y = move[0]
        target_x, target_y = move[1]
        troops = move[2]

        source_prev_value = state.board[source_x][source_y]
        if source_prev_value != 0:
            state.hash ^= zobrist.get_hash(source_x, source_y, source_prev_value)

        source_new_value = source_prev_value + troops
        state.board[source_x][source_y] = source_new_value
        state.hash ^= zobrist.get_hash(source_x, source_y, source_new_value)

        target_prev_value = state.board[target_x][target_y]
        state.hash ^= zobrist.get_hash(target_x, target_y, target_prev_value)

        target_new_value = target_prev_value - troops
        state.board[target_x][target_y] = target_new_value
        if target_new_value != 0:
            state.hash ^= zobrist.get_hash(target_x, target_y, target_new_value)

    switch_player_to_move(state)


def switch_player_to_move(state: GameState):
    state.hash ^= zobrist.SIDE_TO_MOVE
    if state.player_to_move == Player.RED:
        state.player_to_move = Player.BLUE
    else:
        state.player_to_move = Player.RED
