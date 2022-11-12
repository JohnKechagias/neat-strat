import arcade
from constants import *



class Hexagon:
    def __init__(
        self,
        coords: Point,
        owner: Player = Player.NATURE,
        troops: int = 0
    ):
        self.coords = coords
        self.owner = owner
        self.troops = troops
        self.data_coords = self._compute_data_coords()
        self.vertices = self._compute_vertices()

    @property
    def owner(self) -> Player:
        return self._owner

    @owner.setter
    def owner(self, value: Player):
        self._owner = value
        self.color = PLAYER_COLOR[self.owner]
        self.highlight_color = self._compute_highlight_color()

    @property
    def data(self) -> tuple[int, int, int, int]:
        return(self.coords[0], self.coords[1], self.owner.value, self.troops)

    @property
    def shape(self) -> arcade.Shape:
        return arcade.create_polygon(self.vertices, self.color)

    def render(self, color: Color = None):
        color = self.color if color is None else color
        arcade.draw_polygon_filled(self.vertices, color)

        if self.owner != Player.NATURE:
            arcade.draw_text(
                str(self.troops),
                self.data_coords[0] - 9,
                self.data_coords[1] - 8,
                arcade.color.BLACK,
                16,
                width=20,
                align='center'
            )

    def render_highlighted(self):
        self.render(self.highlight_color)

    def render_border(self, color: Color):
        arcade.draw_polygon_outline(self.vertices, color, 4)

    def _compute_data_coords(self) -> tuple[float, float]:
        x, y = self.coords
        # offset each hex (in the x axis) every 2nd row
        # to the positive side by half its width
        x_offset = 0 if not y % 2 else X_OFFSET
        return (
            x * X_STEP + x_offset + HEX_RADIUS + BOARD_PADDING,
            y * Y_STEP + HEX_RADIUS + BOARD_PADDING
        )

    def _compute_vertices(self) -> PointList:
        x, y = self.data_coords

        return [
            (x - HEX_S_RADIUS, y - HALF_HEX_RADIUS),
            (x, y - HEX_RADIUS),
            (x + HEX_S_RADIUS, y - HALF_HEX_RADIUS),
            (x + HEX_S_RADIUS, y + HALF_HEX_RADIUS),
            (x, y + HEX_RADIUS),
            (x - HEX_S_RADIUS, y + HALF_HEX_RADIUS),
        ]

    def _compute_highlight_color(self) -> Color:
        offset = 40
        brighten = lambda x, y: x + y if x + y < 255 else 255
        return tuple(brighten(x, offset) for x in self.color)
