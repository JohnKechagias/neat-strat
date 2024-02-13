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
    zobrist.SIDE_TO_MOVE ^= state.hash

    # If its a production move.
    if len(move) == 2:
        x, y = move[0]

        previous_value = state.board[x][y]
        state.hash ^= zobrist.TABLE[x][y][previous_value]

        if previous_value > 0:
            new_value = previous_value + 1
        else:
            new_value = previous_value - 1

        state.board[x][y] = new_value
        state.hash ^= zobrist.TABLE[x][y][new_value]

    # If its a reposition move.
    elif len(move) == 3:
        source_x, source_y = move[0]
        target_x, target_y = move[1]
        troops = move[2]

        source_prev_value = state.board[source_x][source_y]
        state.hash ^= zobrist.TABLE[source_x][source_y][source_prev_value]
        source_new_value = source_prev_value - troops
        state.board[source_x][source_y] = source_new_value
        state.hash ^= zobrist.TABLE[source_x][source_y][source_new_value]

        target_prev_value = state.board[target_x][target_y]
        state.hash ^= zobrist.TABLE[target_x][target_y][target_prev_value]
        target_new_value = target_prev_value + troops
        state.board[target_x][target_y] = target_new_value
        state.hash ^= zobrist.TABLE[target_x][target_y][target_new_value]

    state.history.append(move)

    if state.player_to_move == Player.RED:
        state.player_to_move = Player.BLUE
    else:
        state.player_to_move = Player.RED


def undo_move(state: GameState):
    zobrist.SIDE_TO_MOVE ^= state.hash
    move = state.history.pop()

    # If its a production move.
    if len(move) == 2:
        x, y = move[0]

        previous_value = state.board[x][y]
        state.hash ^= zobrist.TABLE[x][y][previous_value]

        if previous_value > 0:
            new_value = previous_value - 1
        else:
            new_value = previous_value + 1

        state.board[x][y] = new_value
        state.hash ^= zobrist.TABLE[x][y][new_value]

    # If its a reposition move.
    elif len(move) == 3:
        source_x, source_y = move[0]
        target_x, target_y = move[1]
        troops = move[2]

        source_prev_value = state.board[source_x][source_y]
        state.hash ^= zobrist.TABLE[source_x][source_y][source_prev_value]
        source_new_value = source_prev_value + troops
        state.board[source_x][source_y] = source_new_value
        state.hash ^= zobrist.TABLE[source_x][source_y][source_new_value]

        target_prev_value = state.board[target_x][target_y]
        state.hash ^= zobrist.TABLE[target_x][target_y][target_prev_value]
        target_new_value = target_prev_value - troops
        state.board[target_x][target_y] = target_new_value
        state.hash ^= zobrist.TABLE[target_x][target_y][target_new_value]

    if state.player_to_move == Player.RED:
        state.player_to_move = Player.BLUE
    else:
        state.player_to_move = Player.RED
