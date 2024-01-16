import arcade
from constants import *
from arcade.types import Color
from arcade.shape_list import Shape, create_rectangle_outline


class Menu:
    def __init__(self, point: Point, color: Color, width: int, height: int):
        self.point = point
        self.color = color
        self.width = width
        self.height = height
        self.label_font_size = 16
        self.label_width = 20
        self.label_color = colors.WHITE_SMOKE

    @property
    def shape(self) -> Shape:
        return create_rectangle_outline(
            *self.point, self.width, self.height, self.color, 1
        )

    def render(
        self,
        curr_player: Player,
        action: Action,
        round: int,
        start: Coords,
        end: Coords,
    ):
        arcade.draw_text(
            text=curr_player.name,
            start_x=int(self.point[0] - (1 / 2) * MENU_WIDTH),
            start_y=int(self.point[1]),
            color=self.label_color,
            font_size=self.label_font_size,
            width=self.label_width,
        )

        arcade.draw_text(
            text=f"Round {round}",
            start_x=int(self.point[0] - (1 / 10) * MENU_WIDTH),
            start_y=int(self.point[1]),
            color=self.label_color,
            font_size=self.label_font_size,
            width=self.label_width,
        )

        arcade.draw_text(
            text=f"{action.name} {start} -> {end}",
            start_x=int(self.point[0] + (2 / 10) * MENU_WIDTH),
            start_y=int(self.point[1]),
            color=self.label_color,
            font_size=self.label_font_size,
            width=self.label_width,
        )
