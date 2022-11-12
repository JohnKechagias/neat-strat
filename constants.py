import math
from enum import IntEnum

import arcade.color as color
from arcade.arcade_types import *



class Player(IntEnum):
    NATURE = 0,
    PLAYER1 = 1,
    PLAYER2 = 2


PLAYER_COLOR: dict[Player, Color] = {
    Player.NATURE: color.LIGHT_GRAY,
    Player.PLAYER1: color.RED_VIOLET,
    Player.PLAYER2: color.EMERALD
}


class Theme(IntEnum):
    LIGHT = 0,
    DARK = 1


# Board is square shaped so, size = 6 corresponds to a 6x6 board
BOARD_SIZE = 6
# Radius of the circle outside of the hexagon
HEX_RADIUS = 50
# Padding between the hexagons
HEX_PADDING = 10
# Padding between the board and the window
BOARD_PADDING = 50
# Color theme of the program
COLOR_THEME = Theme.DARK
# Title of window
TITLE = 'Sreak'


# Half the radius of a hex
HALF_HEX_RADIUS = HEX_RADIUS / 2
# Radius of the circle inside of the hexagon
HEX_S_RADIUS = HEX_RADIUS * math.sin(math.radians(60))

SCREEN_WIDTH = math.ceil(2 * (BOARD_SIZE + 0.5) * HEX_S_RADIUS +\
            (BOARD_SIZE + 1) * HEX_PADDING + 2 * BOARD_PADDING)
SCREEN_HEIGHT = math.ceil(2 * BOARD_SIZE * HEX_S_RADIUS + 2 * BOARD_PADDING)

X_STEP = 2 * HEX_S_RADIUS + HEX_PADDING
Y_STEP = (3 / 2) * HEX_RADIUS + HEX_PADDING
X_OFFSET = X_STEP / 2

NUM_OF_HEXS = BOARD_SIZE ** 2
R_1 = BOARD_PADDING + HEX_S_RADIUS - HEX_RADIUS

# Type definitions
Coords = tuple[int, int] | list[int]
