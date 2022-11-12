import timeit
from itertools import product

import arcade
from constants import *
from hexagon import Hexagon



class Board(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, title=TITLE, samples=16)
        self.setup()

    def setup(self):
        self.round = 1
        self.curr_player = Player.PLAYER1
        self.start_tile = None

        # The coords of the tile that the mouse is hovering over.
        # If curr_tile = (-1, -1), it means that the mouse isn't
        # currently hovering over a valid tile.
        self.curr_tile = (-1, -1)
        # The tiles that need to be rendered. That includes all the
        # tiles that have an owner that isn't the default one (nature).
        self.tiles_to_render: list[int] = []
        # The neighbouring tiles of the current tile that the current
        # player can select.
        self.tile_to_highlight: list[int] = []
        # A list with all the default tile shapes. Its used to speed up
        # the rendering of board in its default state, meaning when all
        # the tiles are owned by nature. So when a tile is owned by nature,
        # we don't have to explicitly render it by adding it to the
        # tiles_to_render list.
        self.tile_shapes_list = arcade.ShapeElementList()

        # A list with all the valid tile coordinates, meaning from (0, 0),
        # (0, 1) ... (BOARD_SIZE, BOARD_SIZE)
        tile_coords = [i for i in product(range(BOARD_SIZE), repeat=2)]
        # A list that countains all the actual tile instances. In this
        # case, hexagons.
        self.tiles = [Hexagon(i) for i in tile_coords]

        # Populate the shape list with the default tile shapes.
        for hex in self.tiles:
            self.tile_shapes_list.append(hex.shape)

        # Initialize the starting tile for each player.
        self.tiles[0].owner = Player.PLAYER1
        self.tiles[0].troops = 10
        self.tiles[-1].owner = Player.PLAYER2
        self.tiles[-1].troops = 10

        # Add the starting tiles to the rendering list.
        self.tiles_to_render.append(0)
        self.tiles_to_render.append(len(self.tiles) - 1)

        arcade.set_background_color(arcade.color.BLACK)

    def end_round(self):
        self.round += 1

        if self.curr_player == Player.PLAYER1:
            self.curr_player = Player.PLAYER2
        else:
            self.curr_player = Player.PLAYER1

        self.start_tile = None


    def tile(self, coords: Coords) -> Hexagon:
        return self.tiles[self.hex_index_from_grid_coords(coords)]

    def on_update(self, dt: float): ...

    def on_draw(self):
        draw_start_time = timeit.default_timer()

        arcade.start_render()
        self.tile_shapes_list.draw()

        for coords in self.tiles_to_render:
            self.tiles[coords].render()

        if self.curr_tile[0] != -1:
            self.tile(self.curr_tile).render(arcade.color.GRAPE)

            for coords in self.tile_to_highlight:
                self.tile(coords).render_highlighted()

        self.draw_time = timeit.default_timer() - draw_start_time
        # print(f'{self.draw_time:4f}')

    def on_mouse_motion(self, x, y, dx, dy):
        new_curr_tile = self.grid_coords_from_data_coords(x, y)

        if self.curr_tile != new_curr_tile:
            self.curr_tile = new_curr_tile
            self.tile_to_highlight.clear()

            for coords in self.get_neightbours(new_curr_tile):
                self.tile_to_highlight.append(coords)

    def on_mouse_press(self, x, y, button, modifiers):
        # Only continue if the mouse pressed a tile.
        if self.curr_tile[0] == -1: return

        match button:
            case 1:
                # Player is only allowed to select his own tiles.
                if self.tile(self.curr_tile).owner == self.curr_player:
                    self.start_tile = self.curr_tile
            case 4:
                # If the starting and the current tiles are valid tiles and the
                # current tile is a valid neighbour of the starting tile and the
                # starting tile isn't the same as the current tile, allow the move
                # action.
                if self.start_tile is not None and self.start_tile[0] != -1 and\
                    self.curr_tile[0] != -1 and self.start_tile != self.curr_tile\
                    and self.curr_tile in self.get_neightbours(self.start_tile):
                    self.move(self.start_tile, self.curr_tile, 2)

    def move(self, start: Coords, end: Coords, troops: int):
        start_tile = self.tile(start)
        end_tile = self.tile(end)

        start_tile.troops -= troops
        if start_tile.owner == end_tile.owner:
            end_tile.troops += troops
        else:
            if end_tile.troops < troops:
                end_tile.owner = start_tile.owner
                self.tiles_to_render.append(self.hex_index_from_grid_coords(end))
            elif end_tile.troops == troops:
                end_tile.owner = Player.NATURE
                self.tiles_to_render.remove(self.hex_index_from_grid_coords(end))
            end_tile.troops = abs(end_tile.troops - troops)

        if start_tile.troops == 0:
            start_tile.owner = Player.NATURE
            self.tiles_to_render.remove(self.hex_index_from_grid_coords(start))

        self.end_round()

    def produce(self, coords: Coords):
        self.tile(coords).troops += 1
        self.end_round()

    def grid_coords_from_data_coords(self, x: int, y: int) -> tuple[int, int]:
        y_index = math.floor((y - R_1 - HEX_PADDING) / Y_STEP)
        offset = 0 if not y_index % 2 else X_OFFSET
        x_index = math.floor((x - R_1 - offset - HEX_PADDING / 2) / X_STEP)

        if not self.are_grid_coords_valid((x_index, y_index)):
            return (-1, -1)

        return x_index, y_index

    def hex_index_from_grid_coords(self, coords: Coords) -> int:
        x, y = coords
        return x * BOARD_SIZE + y if x != -1 else -1

    def get_neightbours(self, coords: Coords) -> list[Coords]:
        x, y = coords
        neightbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        if not y % 2:
            neightbours.append((x - 1, y + 1))
            neightbours.append((x - 1, y - 1))
        else:
            neightbours.append((x + 1, y + 1))
            neightbours.append((x + 1, y - 1))

        return list(filter(self.are_grid_coords_valid, neightbours))

    def are_grid_coords_valid(self, coords: Coords) -> bool:
        x, y = coords
        return x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE

def main():
    board = Board()
    arcade.run()


if __name__ == "__main__":
    main()
