from enum import IntEnum
from typing import Callable

from nptyping import Int, NDArray, Shape


class Player(IntEnum):
    NATURE = 0
    RED = -1
    BLUE = 1


class EndgameState(IntEnum):
    DRAW = 0
    RED_WON = 1
    BLUE_WON = 2
    ONGOING = 3


# Board is square shaped so, size = 5 corresponds to a 5x5 board
BOARD_SIZE = 5
MAX_TROOPS = 10

# Type definitions
Coords = tuple[int, int]
Point = tuple[float, float]
Board = NDArray[Shape[f"{BOARD_SIZE}, {BOARD_SIZE}"], Int]
Move = tuple[Coords, Coords] | tuple[Coords, Coords, int]
MovesRecord = list[Move]
Evaluator = Callable[[NDArray], list[float]]
