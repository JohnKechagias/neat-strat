from typing import Optional

import arcade
from constants import *


class Hexagon:
    def __init__(self, coords: Coords, owner: Player = Player.NATURE, troops: int = 0):
        self.coords = coords
        self.troops = troops
        self.data_coords = self._compute_data_coords()
        self.vertices = self._compute_vertices()

        self._owner = owner
        self.color = PLAYER_COLOR[self._owner]
        self._highlight_color = self._compute_highlight_color()
        self._shape = arcade.create_polygon(self.vertices, self.color)

    @property
    def owner(self) -> Player:
        return self._owner

    @owner.setter
    def owner(self, value: Player):
        self._owner = value
        self.color = PLAYER_COLOR[self.owner]
        self._highlight_color = self._compute_highlight_color()
        self._shape = arcade.create_polygon(self.vertices, self.color)

    @property
    def data(self) -> tuple[int, int, int, int]:
        return (*self.coords, self.owner.value, self.troops)

    @property
    def shape(self) -> arcade.Shape:
        return self._shape

    def render(self, color: Optional[RGB] = None):
        color = self.color if color is None else color

        arcade.draw_polygon_filled(self.vertices, color)

        if self.owner != Player.NATURE:
            arcade.draw_text(
                str(self.troops),
                self.data_coords[0] - 9,
                self.data_coords[1] - 8,
                colors.BLACK,
                16,
                width=20,
                align="center",
            )

    def render_highlighted(self):
        self.render(self._highlight_color)

    def render_border(self, color: RGB):
        arcade.draw_polygon_outline(self.vertices, color, 4)

    def _compute_data_coords(self) -> tuple[float, float]:
        x, y = self.coords
        # offset each hex (in the x axis) every 2nd row
        # to the positive side by half its width
        x_offset = 0 if not y % 2 else X_OFFSET
        return (
            x * X_STEP + x_offset + HEX_RADIUS + BOARD_PADDING,
            y * Y_STEP + HEX_RADIUS + BOARD_PADDING,
        )

    def _compute_vertices(self) -> list[arcade.Point]:
        x, y = self.data_coords

        return [
            (x - HEX_S_RADIUS, y - HALF_HEX_RADIUS),
            (x, y - HEX_RADIUS),
            (x + HEX_S_RADIUS, y - HALF_HEX_RADIUS),
            (x + HEX_S_RADIUS, y + HALF_HEX_RADIUS),
            (x, y + HEX_RADIUS),
            (x - HEX_S_RADIUS, y + HALF_HEX_RADIUS),
        ]

    def _compute_highlight_color(self) -> RGB:
        return self._brighten_color(self.color, 40)

    @staticmethod
    def _brighten_color(color: RGB, offset: int) -> RGB:
        return RGB(min(c + offset, 255) for c in color)
