import math
from typing import Callable, Literal
import numpy as np
from constants import BOARD_SIZE, Coords, State, MAX_TROOPS
from beartype import beartype


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
    (4, 4): [(3, 4), (4, 3), (3, 3)]
}


def get_random_state() -> State:
    return np.random.randint(-20, 20, size=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


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


@beartype
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
            moves[(tuple(index), tuple(index))] = new_state

        for neighbor in lookup_neighbors(*index):
            new_state = state.copy()
            troops_capacity = MAX_TROOPS - new_state[neighbor]
            troops_to_move = min(troops_capacity, new_state[index])
            new_state[index] -= troops_to_move
            new_state[neighbor] += troops_to_move
            moves[(tuple(index), neighbor, troops_to_move)] = new_state

    return moves


@beartype
def get_possible_states(state: State) -> list[State]:
    states_after_first_move = get_children(state, 1)

    states_after_opposing_player_responds = []
    for state in states_after_first_move:
        children_states = get_children(state, -1)
        states_after_opposing_player_responds.extend(children_states)

    return states_after_opposing_player_responds


def activate(state: State) -> float:
    return 1.0


def min_max(state: State, depth: int, maximizes_player: bool, alpha: float, beta: float, max_depth: int) -> float:
    if depth == max_depth:
        return activate(state)

    if maximizes_player:
        best_value = -math.inf
        for child in get_children(state, -1):
            value = min_max(child, depth+1, False, alpha, beta, max_depth)
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)

            if beta <= alpha:
                break

        return best_value
    else:
        best_value = +math.inf
        for child in get_children(state, 1):
            value = min_max(child, depth+1, True, alpha, beta, max_depth)
            best_value = min(best_value, value)
            beta = min(beta, best_value)

            if beta <= alpha:
                break

        return best_value



def select_best_move(state: State, activate: Callable[[State], float], depth: int):
    possible_moves = get_possible_moves(state)

    best_move = None
    best_value = -math.inf
    for move, state in possible_moves.items():
        value = min_max(state, 0, False, -math.inf, math.inf, depth)

        if value > best_value:
            best_move = move
            best_value = value

    return best_move

state = get_random_state()
print(select_best_move(state, lambda x: 1.0, 1))


def test_select_best_move():
    def activate(var: list[int]) -> float:
        if var == [-1, 1]:


