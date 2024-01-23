from typing import Optional

import arcade
from arcade.shape_list import create_polygon
from arcade.types import Color

from constants import *


class Tile:
    def __init__(self, coords: Coords, owner: Player = Player.NATURE, troops: int = 0):
        self.coords = coords
        self.troops = troops
        self.data_coords = self._precompute_data_coords(coords)
        self.vertices = self._precompute_vertices(self.data_coords)

        self._owner = owner
        self._color = PLAYER_COLOR[owner]
        self._neighbors = self._pre_compute_neighbors(coords)
        self._highlight_color = self._compute_highlight_color(self._color)
        self._shape = create_polygon(self.vertices, self._color)

        self.troops_text = arcade.Text(
            str(self.troops),
            self.data_coords[0],
            self.data_coords[1],
            colors.BLACK,
            16,
            width=20,
            align="center",
            anchor_x="center",
        )

    @property
    def neighbors(self) -> list[int]:
        return self._neighbors

    @property
    def owner(self) -> Player:
        return self._owner

    @owner.setter
    def owner(self, value: Player):
        self._owner = value
        self._color = PLAYER_COLOR[value]
        self._highlight_color = self._compute_highlight_color(self._color)
        self._shape = create_polygon(self.vertices, self._color)

    @property
    def data(self) -> tuple[int, int, int, int]:
        return (*self.coords, self.owner.value, self.troops)

    @property
    def shape(self):
        return self._shape

    def render(self, color: Optional[arcade.types.Color] = None):
        color = self._color if color is None else color

        arcade.draw_polygon_filled(self.vertices, color)

        if self.owner != Player.NATURE:
            self.troops_text.text = str(self.troops)
            self.troops_text.draw()

    def render_highlighted(self):
        self.render(self._highlight_color)

    def render_border(self, color: Color):
        arcade.draw_polygon_outline(self.vertices, color, 4)

    @staticmethod
    def _pre_compute_neighbors(coords: Coords) -> list[int]:
        x, y = coords
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        if not y % 2:
            neighbors.append((x - 1, y + 1))
            neighbors.append((x - 1, y - 1))
        else:
            neighbors.append((x + 1, y + 1))
            neighbors.append((x + 1, y - 1))

        def is_neighbor_valid(coord: Coords) -> bool:
            x, y = coord
            return x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE

        neighbors = list(filter(is_neighbor_valid, neighbors))
        return [Tile.hex_index_from_grid_coords(i) for i in neighbors]

    @staticmethod
    def _precompute_data_coords(coords: Coords) -> Coords:
        x, y = coords
        # Offset each hex (in the x axis) every 2nd row
        # to the positive side by half its width.
        x_offset = 0 if not y % 2 else X_OFFSET

        x_coord = int(x * X_STEP + x_offset + HEX_RADIUS + BOARD_PADDING)
        y_coord = int(y * Y_STEP + HEX_RADIUS + BOARD_PADDING)
        return (x_coord, y_coord)

    @staticmethod
    def _precompute_vertices(data_coords: Coords) -> list[Point]:
        x, y = data_coords

        return [
            (x - HEX_S_RADIUS, y - HALF_HEX_RADIUS),
            (x, y - HEX_RADIUS),
            (x + HEX_S_RADIUS, y - HALF_HEX_RADIUS),
            (x + HEX_S_RADIUS, y + HALF_HEX_RADIUS),
            (x, y + HEX_RADIUS),
            (x - HEX_S_RADIUS, y + HALF_HEX_RADIUS),
        ]

    def _compute_highlight_color(self, color: Color) -> Color:
        return self.brighten_color(color, 40)

    @staticmethod
    def brighten_color(color: Color, offset: int) -> Color:
        r = min(color.r + offset, 255)
        g = min(color.g + offset, 255)
        b = min(color.b + offset, 255)
        return Color(r, g, b)

    @staticmethod
    def hex_index_from_grid_coords(coords: Coords) -> int:
        x, y = coords
        return x * BOARD_SIZE + y if x != -1 else -1
