import math


class Params:
    board_title: str = "Neat Strat"
    board_padding: int = 50
    board_size: int = 5
    menu_height: int = 20
    menu_padding: int = 40
    hex_radius: int = 50
    hex_padding: int = 10

    hex_half_radius = hex_radius / 2
    hex_s_radius = hex_radius * math.sin(math.radians(60))
    board_width = math.ceil(
        2 * (board_size + 0.5) * hex_s_radius
        + (board_size + 1) * hex_padding
        + 2 * board_padding
    )
    board_height = math.ceil(
        2 * board_size * hex_s_radius
        + 2 * board_padding + menu_height
        + menu_padding
    )
    menu_width = math.ceil((8 / 10) * board_width - 2 * menu_padding)
    x_step = 2 * hex_s_radius + hex_padding
    y_step = (3 / 2) * hex_radius + hex_padding
    x_offset = x_step / 2
    r_1 = board_padding + hex_s_radius - hex_radius
