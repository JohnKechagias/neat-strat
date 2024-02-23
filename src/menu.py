import arcade
from arcade.shape_list import Shape, create_rectangle_outline
from arcade.types import Color

from .constants import MENU_WIDTH, Action, Move, Point, colors


class Menu:
    def __init__(self, point: Point, color: Color, width: int, height: int):
        self.point = point
        self.color = color
        self.width = width
        self.height = height
        self.label_font_size = 16
        self.label_width = 20
        self.label_color = colors.WHITE_SMOKE
        self.text_objects: list[arcade.Text] = []

        self.curr_player_text = arcade.Text(
            text="",
            start_x=int(self.point[0] - 0.6 * MENU_WIDTH),
            start_y=int(self.point[1]),
            color=self.label_color,
            font_size=self.label_font_size,
            width=self.label_width,
            anchor_x="left",
        )

        self.round_text = arcade.Text(
            text="",
            start_x=int(self.point[0] - 0.4 * MENU_WIDTH),
            start_y=int(self.point[1]),
            color=self.label_color,
            font_size=self.label_font_size,
            width=self.label_width,
        )

        self.action_text = arcade.Text(
            text="",
            start_x=int(self.point[0] - 0.1 * MENU_WIDTH),
            start_y=int(self.point[1]),
            color=self.label_color,
            font_size=self.label_font_size,
            width=self.label_width,
        )

    @property
    def shape(self) -> Shape:
        return create_rectangle_outline(
            *self.point, self.width, self.height, self.color, 1
        )

    def render(
        self,
        player_name: str,
        action: Action,
        move: Move,
        round: int,
    ):
        self.curr_player_text.text = player_name
        self.curr_player_text.draw()
        self.round_text.text = f"Round {round}"
        self.round_text.draw()

        action_text = f"{action.name} {move[0]} -> {move[1]}"

        if len(move) == 3:
            action_text += f" ({abs(move[2])})"

        self.action_text.text = action_text
        self.action_text.draw()
