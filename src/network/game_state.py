from dataclasses import dataclass, field

from . import zobrist
from .constants import Board, Move, Player


@dataclass
class GameState:
    board: Board
    hash: int
    player_to_move: Player
    history: list[Move] = field(default_factory=list)
    captures: dict[Player, int] = field(
        default_factory=lambda: {
            Player.RED: 0,
            Player.BLUE: 0,
        }
    )


def make_move(state: GameState, move: Move):
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

        if target_prev_value * troops < 0:
            state.captures[state.player_to_move] += 1

        target_new_value = target_prev_value + troops
        state.board[target_x][target_y] = target_new_value
        state.hash ^= zobrist.get_hash(target_x, target_y, target_new_value)

    state.history.append(move)
    switch_player_to_move(state)


def undo_move(state: GameState):
    move = state.history.pop()
    switch_player_to_move(state)

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
        if target_new_value * troops < 0:
            state.captures[state.player_to_move] -= 1

        state.board[target_x][target_y] = target_new_value
        if target_new_value != 0:
            state.hash ^= zobrist.get_hash(target_x, target_y, target_new_value)


def switch_player_to_move(state: GameState):
    state.hash ^= zobrist.SIDE_TO_MOVE
    if state.player_to_move == Player.RED:
        state.player_to_move = Player.BLUE
    else:
        state.player_to_move = Player.RED
