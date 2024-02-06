import math
from typing import Literal

import numpy as np
from beartype import beartype

from ..constants import (
    BOARD_SIZE,
    MAX_TROOPS,
    Coords,
    Evaluator,
    GameState,
    Move,
    State,
)

neighbors_mappers: dict[Coords, list[Coords]] = {
    (0, 0): [(1, 0), (0, 1)],
    (0, 1): [(1, 1), (0, 2), (0, 0), (1, 2), (1, 0)],
    (0, 2): [(1, 2), (0, 3), (0, 1)],
    (0, 3): [(1, 3), (0, 4), (0, 2), (1, 4), (1, 2)],
    (0, 4): [(1, 4), (0, 3)],
    (1, 0): [(2, 0), (0, 0), (1, 1), (0, 1)],
    (1, 1): [(2, 1), (0, 1), (1, 2), (1, 0), (2, 2), (2, 0)],
    (1, 2): [(2, 2), (0, 2), (1, 3), (1, 1), (0, 3), (0, 1)],
    (1, 3): [(2, 3), (0, 3), (1, 4), (1, 2), (2, 4), (2, 2)],
    (1, 4): [(2, 4), (0, 4), (1, 3), (0, 3)],
    (2, 0): [(3, 0), (1, 0), (2, 1), (1, 1)],
    (2, 1): [(3, 1), (1, 1), (2, 2), (2, 0), (3, 2), (3, 0)],
    (2, 2): [(3, 2), (1, 2), (2, 3), (2, 1), (1, 3), (1, 1)],
    (2, 3): [(3, 3), (1, 3), (2, 4), (2, 2), (3, 4), (3, 2)],
    (2, 4): [(3, 4), (1, 4), (2, 3), (1, 3)],
    (3, 0): [(4, 0), (2, 0), (3, 1), (2, 1)],
    (3, 1): [(4, 1), (2, 1), (3, 2), (3, 0), (4, 2), (4, 0)],
    (3, 2): [(4, 2), (2, 2), (3, 3), (3, 1), (2, 3), (2, 1)],
    (3, 3): [(4, 3), (2, 3), (3, 4), (3, 2), (4, 4), (4, 2)],
    (3, 4): [(4, 4), (2, 4), (3, 3), (2, 3)],
    (4, 0): [(3, 0), (4, 1), (3, 1)],
    (4, 1): [(3, 1), (4, 2), (4, 0)],
    (4, 2): [(3, 2), (4, 3), (4, 1), (3, 3), (3, 1)],
    (4, 3): [(3, 3), (4, 4), (4, 2)],
    (4, 4): [(3, 4), (4, 3), (3, 3)],
}


def get_neighbors(x: int, y: int) -> list[Coords]:
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    if not y % 2:
        neighbors.append((x - 1, y + 1))
        neighbors.append((x - 1, y - 1))
    else:
        neighbors.append((x + 1, y + 1))
        neighbors.append((x + 1, y - 1))

    def is_neighbor_valid(coords: Coords) -> bool:
        x, y = coords
        return x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE

    return list(filter(is_neighbor_valid, neighbors))


def lookup_neighbors(x: int, y: int) -> list[Coords]:
    return neighbors_mappers[(x, y)]


def get_children(state: State, player: Literal[-1, 1]) -> list[State]:
    children: list[State] = []

    for index, value in np.ndenumerate(state * player):
        if value <= 0:
            continue

        if value < MAX_TROOPS:
            new_state = state.copy()
            new_state[index] = min(value + 1, MAX_TROOPS)
            children.append(new_state)

        for neighbor in lookup_neighbors(*index):
            new_state = state.copy()
            troops_capacity = MAX_TROOPS - new_state[neighbor]
            troops_to_move = min(troops_capacity, new_state[index])
            new_state[index] -= troops_to_move
            new_state[neighbor] += troops_to_move
            children.append(new_state * player)

    return children


def get_possible_moves(state: State) -> dict[tuple, State]:
    moves: dict[tuple, State] = {}

    for index, value in np.ndenumerate(state):
        if value <= 0:
            continue

        if value < MAX_TROOPS:
            new_state = state.copy()
            new_state[index] = min(value + 1, MAX_TROOPS)
            moves[(Coords(index), Coords(index))] = new_state

        for neighbor in lookup_neighbors(*index):
            new_state = state.copy()
            troops_capacity = MAX_TROOPS - new_state[neighbor]
            troops_to_move = min(troops_capacity, new_state[index])
            new_state[index] -= troops_to_move
            new_state[neighbor] += troops_to_move
            moves[(Coords(index), neighbor, troops_to_move)] = new_state

    return moves


def get_possible_states(state: State) -> list[State]:
    states_after_first_move = get_children(state, 1)

    states_after_opposing_player_responds = []
    for state in states_after_first_move:
        children_states = get_children(state, -1)
        states_after_opposing_player_responds.extend(children_states)

    return states_after_opposing_player_responds


@beartype
def min_max(
    state,
    evaluator: Evaluator,
    depth: int,
    maximizes_player: bool,
    alpha: float,
    beta: float,
    max_depth: int,
) -> float:
    if depth == max_depth:
        return float(sum(evaluator(state.flatten())))

    if maximizes_player:
        best_value = -math.inf
        for child in get_children(state, 1):
            value = min_max(child, evaluator, depth + 1, False, alpha, beta, max_depth)
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)

            if beta <= alpha:
                break

        return best_value
    else:
        best_value = +math.inf
        for child in get_children(state, -1):
            value = min_max(child, evaluator, depth + 1, True, alpha, beta, max_depth)
            best_value = min(best_value, value)
            beta = min(beta, best_value)

            if beta <= alpha:
                break

        return best_value


def select_best_move(
    state: State,
    evaluator: Evaluator,
    depth: int,
) -> tuple[tuple, State]:
    possible_moves = get_possible_moves(state)

    best_move = None
    best_state = None
    best_value = -math.inf
    values: list[float] = []
    for move, p_state in possible_moves.items():
        value = min_max(p_state, evaluator, 0, False, -math.inf, math.inf, depth)
        values.append(value)
        if value > best_value:
            best_move = move
            best_value = value
            best_state = p_state

    # if best_move is None or best_state is None:
    #     print(f"Number of available moves: {len(possible_moves)}")
    #     print(f"Best move: {best_move}")
    #     print(f"Best state: {best_state}")
    #     print(f"Best value: {best_value}")
    #     print(f"State: {state}")
    #     print(f"Moves: {possible_moves.keys()}")
    #     print(f"States: {possible_moves.values()}")
    #     print(f"Values: {values}")

    # If the network detects a lossing position the produces value could be -inf.
    # This result in it not picking a move. To avoid that force it to play a move.
    if best_move is None or best_state is None:
        best_move, best_state = list(possible_moves.items())[0]

    assert best_move is not None
    assert best_state is not None
    return best_move, best_state


def play_move(state: State, move: Move) -> State:
    # If its a production move.
    if len(move) == 2:
        state[move[0][0]][move[0][1]] += 1
    # If its a reposition move.
    elif len(move) == 3:
        state[move[0][0]][move[0][1]] -= move[2]
        state[move[1][0]][move[1][1]] += move[2]

    return state


def play(
    player: Evaluator, opponent: Evaluator, rounds: int
) -> tuple[GameState, State, list[tuple[Literal[0, 1], Move]]]:
    depth = 3
    state = np.zeros((5, 5), dtype=np.int8)
    state[0][4] = -10
    state[4][0] = 10

    game_record = []
    for round in range(rounds):
        if round % 2 == 0:
            move, state = select_best_move(state, player, depth)
            game_record.append((0, move))
        else:
            move, state = select_best_move(state * -1, opponent, depth)
            state *= -1
            game_record.append((1, move))

        game_state = get_game_state(state)

        if game_state != GameState.ONGOING:
            return game_state, state, game_record

    return GameState.DRAW, state, game_record


def get_game_state(state: State) -> GameState:
    flattened_state = state.flatten()
    if np.all(flattened_state <= 0):
        return GameState.LOSS
    elif np.all(flattened_state >= 0):
        return GameState.WIN
    elif np.all(flattened_state == 0):
        return GameState.DRAW
    else:
        return GameState.ONGOING
