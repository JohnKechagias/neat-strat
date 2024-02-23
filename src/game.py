import time
import timeit
from itertools import product
from typing import Optional

import arcade
import numpy as np

from src.network.constants import Evaluator

from .constants import *
from .menu import Menu
from .network.game_state import make_move
from .network.search import EndgameState, Searcher, get_default_state, get_endgame_state
from .parameters import Params
from .tile import Tile


class Game(arcade.Window):
    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        size: int,
        samples: int,
        evaluator: Optional[Evaluator] = None,
        player_starts: Optional[bool] = None,
        model_plays_itself: Optional[bool] = None,
    ):
        super().__init__(
            width=width,
            height=height,
            title=title,
            samples=samples,
        )
        self.board_size = size
        self.background_color = colors.BLACK
        self.selected_tile_color = colors.GRAPE
        self.evaluator = evaluator
        self.player_starts = player_starts
        self.model_plays_itself = model_plays_itself

        if self.evaluator is not None:
            self.searcher = Searcher(STORAGE_SIZE_MB, DEPTH)

    def setup(self):
        if self.evaluator is not None:
            self.searcher.reset()

        self.round = 1
        self.state = get_default_state()
        self.start_tile_index: Optional[int] = None
        self.action = Action.MOVE
        self.player_move = ((0, 0), (0, 0))
        self.origin_tile_coords = (0, 0)
        self.destination_tile_coords = (0, 0)

        # Helper variable that is used when a model plays with itself. Measures
        # the time elapsed since the last time a move was mode. Used to play moves
        # at intervals in order to have time to see what the model plays (else it
        # whould go too fast to see).
        self.timer: float = 0
        # Interval between moves played by the model. Used when a model plays with
        # itself.
        self.interval = 0.05
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
        self.text_objects: list[arcade.Text] = []

        menu_center = (
            self.width / 2,
            self.height - Params.menu_padding - Params.menu_height / 2,
        )
        menu_width = Params.menu_width
        menu_heigh = Params.menu_height
        self.menu = Menu(menu_center, colors.LIGHT_GRAY, menu_width, menu_heigh)

        # A list with all the valid tile coordinates, meaning from (0, 0),
        # (0, 1) ... (BOARD_SIZE, BOARD_SIZE)
        tile_coords = [Coords(i) for i in product(range(self.board_size), repeat=2)]
        # A list that countains all of the tile instances. In this case, hexagons.
        self.tiles = [Tile(i) for i in tile_coords]

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
            self.text_objects.append(text)

        # Initialize the starting tile for each player.
        for coords, troops in np.ndenumerate(self.state.board):
            tile_index = self.get_tile_index_from_grid_coords(*coords)

            if troops > 0:
                tile_index = self.get_tile_index_from_grid_coords(*coords)
                tile = self.tiles[tile_index]
                tile.owner = Player.BLUE
                tile.troops = int(troops)
                self.tiles_to_render.append(tile_index)
            elif troops < 0:
                tile_index = self.get_tile_index_from_grid_coords(*coords)
                tile = self.tiles[tile_index]
                tile.owner = Player.RED
                tile.troops = int(-troops)
                self.tiles_to_render.append(tile_index)

        self.selected_troops: dict[Player, int] = {
            Player.BLUE: 0,
            Player.RED: 0,
        }

        if self.evaluator is None:
            return

        if not self.model_plays_itself and not self.player_starts:
            move = self.searcher.search(self.evaluator, self.state)
            self.make_move(move)

    def on_draw(self):
        draw_start_time = timeit.default_timer()

        self.clear()
        self.shapes_list.draw()
        self.menu.render(
            self.state.player_to_move.name,
            self.action,
            self.player_move,
            self.round,
        )

        for coords in self.tiles_to_render:
            self.tiles[coords].render()

        if self.curr_tile_index is not None:
            self.tiles[self.curr_tile_index].render(self.selected_tile_color)

            for coords in self.tiles_to_highlight:
                self.tiles[coords].render_highlighted()

        for text in self.text_objects:
            text.draw()

        self.draw_time = timeit.default_timer() - draw_start_time

    def on_update(self, delta_time: float):
        if not self.model_plays_itself or self.evaluator is None:
            return

        if time.time() - self.timer < self.interval:
            return

        move = self.searcher.search(self.evaluator, self.state)
        self.make_move(move)
        self.timer = time.time()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        if self.model_plays_itself:
            return

        new_curr_tile = self.get_tile_index_from_data_coords(x, y)

        if self.curr_tile_index != new_curr_tile:
            self.curr_tile_index = new_curr_tile

            if coords := self.get_tile_coords_from_data_coords(x, y):
                self.curr_tile_coords = coords

            self.tiles_to_highlight.clear()

            if new_curr_tile is not None:
                tile = self.tiles[new_curr_tile]
                self.tiles_to_highlight.extend(tile.neighbours)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        if self.model_plays_itself:
            return

        # Only continue if the mouse pressed a tile.
        if self.curr_tile_index is None:
            return

        tile = self.tiles[self.curr_tile_index]
        match button:
            case arcade.MOUSE_BUTTON_LEFT:
                # Player is only allowed to select his own tiles.
                if tile.owner == self.state.player_to_move:
                    self.reset_selected_troops()
                    self.start_tile_index = self.curr_tile_index
                    self.start_tile_coords = self.curr_tile_coords
            case arcade.MOUSE_BUTTON_RIGHT:
                # If the starting and the current tiles are valid tiles and the
                # current tile is a valid neighbour of the starting tile and the
                # starting tile isn't the same as the current tile, allow the move
                # action.
                if self.start_tile_index is not None:
                    start_tile = self.tiles[self.start_tile_index]
                    neighbours = start_tile.neighbours
                    if self.curr_tile_index in neighbours:
                        player_to_move = self.state.player_to_move
                        troops = self.selected_troops[player_to_move] * player_to_move
                        move = (self.start_tile_coords, self.curr_tile_coords, troops)
                        self.make_move(move)

                        if self.evaluator is not None:
                            move = self.searcher.search(self.evaluator, self.state)
                            self.make_move(move)
                    elif (
                        self.curr_tile_index == self.start_tile_index
                        and start_tile.troops < MAX_TROOPS
                    ):
                        self.make_move((self.start_tile_coords, self.curr_tile_coords))

                        if self.evaluator is not None:
                            move = self.searcher.search(self.evaluator, self.state)
                            self.make_move(move)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        if self.model_plays_itself:
            return

        if self.start_tile_index is None:
            return

        if scroll_y > 0:
            start_tile = self.tiles[self.start_tile_index]
            if self.selected_troops[self.state.player_to_move] < start_tile.troops:
                self.selected_troops[self.state.player_to_move] += 1
        elif scroll_y < 0:
            if self.selected_troops[self.state.player_to_move] > 1:
                self.selected_troops[self.state.player_to_move] -= 1

    def make_move(self, move: Move):
        self.player_move = move

        start_index = self.get_tile_index_from_grid_coords(move[0][0], move[0][1])
        if len(move) == 2:
            self.produce(start_index)
        elif len(move) == 3:
            end_index = self.get_tile_index_from_grid_coords(move[1][0], move[1][1])
            self.move(start_index, end_index, move[2] * self.state.player_to_move)

        make_move(self.state, move)
        endgame_state = get_endgame_state(self.state.board)
        if endgame_state != EndgameState.ONGOING:
            self.setup()

        self.end_round()

    def end_round(self):
        self.round += 1
        # Start tile needs to be reset so that the next player
        # can't use the starting tile of the previous player.
        self.start_tile_index = None

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

    def produce(self, tile_index: int):
        tile = self.tiles[tile_index]
        tile.troops = min(tile.troops + 1, MAX_TROOPS)
        self.action = Action.PRODUCE
        self.origin_tile_coords = tile.coords
        self.destination_tile_coords = tile.coords

    def get_tile_from_coords(self, x: int, y: int) -> Tile:
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

    def reset_selected_troops(self):
        self.selected_troops[self.state.player_to_move] = 1
