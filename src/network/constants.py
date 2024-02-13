from enum import IntEnum

from beartype.typing import Callable
from nptyping import Int, NDArray, Shape


class Player(IntEnum):
    RED = -1
    BLUE = 1


class EndgameState(IntEnum):
    LOSS = -1
    DRAW = 0
    WIN = 1
    ONGOING = 2


# Board is square shaped so, size = 6 corresponds to a 6x6 board
BOARD_SIZE = 5
MAX_TROOPS = 10

# Type definitions
Coords = tuple[int, int]
Point = tuple[float, float]
Board = NDArray[Shape[f"{BOARD_SIZE}, {BOARD_SIZE}"], Int]
Move = tuple[Coords, Coords] | tuple[Coords, Coords, int]
MovesRecord = list[Move]
Evaluator = Callable[[NDArray], list[float]]
