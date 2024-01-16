import timeit
from itertools import product
from typing import Optional

import arcade
from constants import *
from hexagon import Hexagon
from menu import Menu
from parameters import Params


class Board(arcade.Window):
    def __init__(self, width: int, height: int, title: str, size: int, samples: int):
        super().__init__(
            width=width,
            height=height,
            title=title,
            samples=samples,
        )
        self.board_size = size
        self.background_color = colors.BLACK
        self.selected_tile_color = colors.GRAPE
        self.max_troops = 20
        self.setup()

    def setup(self):
        self.round = 1
        self.start_tile_index: Optional[int] = None
        self.curr_player = Player.PLAYER1
        self.action = Action.MOVE
        self.origin_tile_coords = (-1, -1)
        self.destination_tile_coords = (-1, -1)

        # The tile index that the mouse is hovering over.
        # If curr_tile is None, it means that the mouse isn't
        # currently hovering over a valid tile.
        self.curr_tile_index: Optional[int] = None
        # The tiles that need to be rendered meaning the tiles 
        # that have an owner that isn't the default one (nature).
        self.tiles_to_render: list[int] = []
        # The neighbouring tiles of the currently selected tile.
        self.tiles_to_highlight: list[int] = []
        # A list with all the default tile shapes and the menu shape. Its
        # used to speed up the rendering of board in its default state,
        # meaning when all the tiles are owned by nature. So when a tile
        # is owned by nature, we don't have to explicitly render it by
        # adding it to the tiles_to_render list.
        self.shapes_list = arcade.shape_list.ShapeElementList()
        self.text_list: list[arcade.Text] = []

        menu_center = (
            self.width / 2,
            self.height - Params.menu_padding - Params.menu_height / 2
        )
        menu_width = Params.menu_width
        menu_heigh = Params.menu_height
        self.menu = Menu(menu_center, colors.LIGHT_GRAY, menu_width, menu_heigh)

        # A list with all the valid tile coordinates, meaning from (0, 0),
        # (0, 1) ... (BOARD_SIZE, BOARD_SIZE)
        tile_coords = [Coords(i) for i in product(range(self.board_size), repeat=2)]
        # A list that countains all the actual tile instances. In this case, hexagons.
        self.tiles = [Hexagon(i) for i in tile_coords]

        # Populate the shape list with the default tile shapes.
        for tile in self.tiles:
            self.shapes_list.append(tile.shape)
            text = arcade.Text(
                f"{tile.coords[0]},{tile.coords[1]}",
                tile.data_coords[0],
                tile.data_coords[1] - 30,
                colors.BLACK,
                14,
                width=70,
                align="left",
                anchor_x="center",
            )
            self.text_list.append(text)

        # Initialize the starting tile for each player.
        player_1_starting_tile = 0
        player_2_starting_tile = len(self.tiles) - 1
        self.tiles[player_1_starting_tile].owner = Player.PLAYER1
        self.tiles[player_1_starting_tile].troops = 10
        self.tiles[player_2_starting_tile].owner = Player.PLAYER2
        self.tiles[player_2_starting_tile].troops = 10

        # Add the starting tiles to the rendering list.
        self.tiles_to_render.append(player_1_starting_tile)
        self.tiles_to_render.append(player_2_starting_tile)

    def end_round(self):
        self.round += 1

        if self.curr_player == Player.PLAYER1:
            self.curr_player = Player.PLAYER2
        else:
            self.curr_player = Player.PLAYER1

        # Start tile needs to be reset so that the next player
        # can't use the starting tile of the previous player.
        self.start_tile_index = None

    def on_draw(self):
        draw_start_time = timeit.default_timer()

        self.clear()
        self.shapes_list.draw()
        self.menu.render(
            self.curr_player,
            self.action,
            self.round,
            self.origin_tile_coords,
            self.destination_tile_coords,
        )

        for coords in self.tiles_to_render:
            self.tiles[coords].render()

        if self.curr_tile_index is not None:
            self.tiles[self.curr_tile_index].render(self.selected_tile_color)

            for coords in self.tiles_to_highlight:
                self.tiles[coords].render_highlighted()

        for text in self.text_list:
            text.draw()

        self.draw_time = timeit.default_timer() - draw_start_time
        # print(f'{self.draw_time:.4f}')

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        new_curr_tile = self.get_tile_index_from_data_coords(x, y)

        if self.curr_tile_index != new_curr_tile:
            self.curr_tile_index = new_curr_tile
            self.tiles_to_highlight.clear()

            if new_curr_tile is not None:
                tile = self.tiles[new_curr_tile]
                self.tiles_to_highlight.extend(tile.neighbors)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        # Only continue if the mouse pressed a tile.
        if self.curr_tile_index is None:
            return

        tile = self.tiles[self.curr_tile_index]

        match button:
            case 1:
                # Player is only allowed to select his own tiles.
                if tile.owner == self.curr_player:
                    self.start_tile_index = self.curr_tile_index
            case 4:
                # If the starting and the current tiles are valid tiles and the
                # current tile is a valid neighbour of the starting tile and the
                # starting tile isn't the same as the current tile, allow the move
                # action.
                if self.start_tile_index is not None:
                    if self.curr_tile_index in self.tiles[self.start_tile_index].neighbors:
                        self.move(self.start_tile_index, self.curr_tile_index, 2)
                    elif self.curr_tile_index == self.start_tile_index:
                        self.produce(self.curr_tile_index)

    def move(self, start: int, end: int, troops: int):
        start_tile = self.tiles[start]
        end_tile = self.tiles[end]

        start_tile.troops -= troops
        if start_tile.owner == end_tile.owner:
            end_tile.troops += troops
        else:
            if end_tile.troops < troops:
                end_tile.owner = start_tile.owner
                self.tiles_to_render.append(end)
            elif end_tile.troops == troops:
                end_tile.owner = Player.NATURE
                self.tiles_to_render.remove(end)
            end_tile.troops = abs(end_tile.troops - troops)

        if start_tile.troops == 0:
            start_tile.owner = Player.NATURE
            self.tiles_to_render.remove(start)

        self.action = Action.MOVE
        self.origin_tile_coords = start_tile.coords
        self.destination_tile_coords = end_tile.coords
        self.end_round()

    def produce(self, tile_index: int):
        tile = self.tiles[tile_index]
        tile.troops = min(tile.troops + 1, self.max_troops)
        self.action = Action.PRODUCE
        self.origin_tile_coords = tile.coords
        self.destination_tile_coords = tile.coords
        self.end_round()

    def get_tile_from_coords(self, x: int, y: int) -> Hexagon:
        return self.tiles[self.get_tile_index_from_grid_coords(x, y)]

    def get_tile_coords_from_data_coords(self, x: int, y: int) -> Optional[Coords]:
        y_index = math.floor((y - R_1 - HEX_PADDING) / Y_STEP)
        offset = 0 if not y_index % 2 else X_OFFSET
        x_index = math.floor((x - R_1 - offset - HEX_PADDING / 2) / X_STEP)

        if not self.are_grid_coords_valid(x_index, y_index):
            return None

        return x_index, y_index

    def get_tile_index_from_data_coords(self, x: int, y: int) -> Optional[int]:
        if tile_coords := self.get_tile_coords_from_data_coords(x, y):
            return self.get_tile_index_from_grid_coords(*tile_coords)

    def get_tile_index_from_grid_coords(self, x: int, y: int) -> int:
        return x * self.board_size + y if x != -1 else -1

    def are_grid_coords_valid(self, x: int, y: int) -> bool:
        return x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE


def main():
    board = Board(
        Params.board_width,
        Params.board_height,
        Params.board_title,
        Params.board_size,
        16
    )
    board.run()


if __name__ == "__main__":
    main()
