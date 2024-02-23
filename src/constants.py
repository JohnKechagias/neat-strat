import math
from enum import IntEnum

import arcade.color as colors
from arcade.types import Color

from .network.constants import Player

STORAGE_SIZE_MB = 1024
DEPTH = 3


class Action(IntEnum):
    MOVE = 0
    PRODUCE = 1


PLAYER_COLOR: dict[Player, Color] = {
    Player.NATURE: colors.LIGHT_GRAY,
    Player.RED: colors.RED_VIOLET,
    Player.BLUE: colors.EMERALD,
}


class Theme(IntEnum):
    LIGHT = 0
    DARK = 1


# Board is square shaped so, size = 6 corresponds to a 6x6 board
BOARD_SIZE = 5
# Radius of the circle outside of the hexagon
HEX_RADIUS = 50
# Padding between the hexagons
HEX_PADDING = 10
# Padding between the board and the window
BOARD_PADDING = 50
# Padding between the menu and the window
MENU_PADDING = 10
# Menu height, where all the stats are displayed
MENU_HEIGHT = 40
# Menu width, where all the stats are displayed
MENU_WIDTH = 600
# Color theme of the program
COLOR_THEME = Theme.DARK
# Title of the window
TITLE = "Sreak"
MAX_TROOPS = 10


# Half the radius of a hex
HALF_HEX_RADIUS = HEX_RADIUS / 2
# Radius of the circle inside of the hexagon
HEX_S_RADIUS = HEX_RADIUS * math.sin(math.radians(60))

WINDOW_WIDTH = math.ceil(
    2 * (BOARD_SIZE + 0.5) * HEX_S_RADIUS
    + (BOARD_SIZE + 1) * HEX_PADDING
    + 2 * BOARD_PADDING
)
WINDOW_HEIGHT = math.ceil(
    2 * BOARD_SIZE * HEX_S_RADIUS + 2 * BOARD_PADDING + MENU_HEIGHT + MENU_PADDING
)

MENU_WIDTH = min(MENU_WIDTH, math.ceil((8 / 10) * WINDOW_WIDTH - 2 * MENU_PADDING))

X_STEP = 2 * HEX_S_RADIUS + HEX_PADDING
Y_STEP = (3 / 2) * HEX_RADIUS + HEX_PADDING
X_OFFSET = X_STEP / 2

NUM_OF_HEXS = BOARD_SIZE**2
R_1 = BOARD_PADDING + HEX_S_RADIUS - HEX_RADIUS

# Type definitions
Coords = tuple[int, int]
Point = tuple[float, float]
Move = tuple[Coords, Coords] | tuple[Coords, Coords, int]
