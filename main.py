import pickle
from enum import StrEnum, auto

from neat import FeedForwardNetwork

from neat_strat.game import Game
from neat_strat.network.neural_network import train
from neat_strat.parameters import Params


class GameVariation(StrEnum):
    MODEL_VS_MODEL = auto()
    PLAYER_VS_MODEL = auto()
    PLAYER_VS_PLAYER = auto()


TRAIN = False
ITERATIONS = 200
PLAY = True
PLAYER_STARTS = True
GAME_VARIATION = GameVariation.PLAYER_VS_MODEL


def main():
    genome = train(ITERATIONS) if TRAIN else None

    if not PLAY:
        return

    evaluator = None
    player_starts = PLAYER_STARTS
    model_plays_itself = GAME_VARIATION == GameVariation.MODEL_VS_MODEL
    if GAME_VARIATION != GameVariation.PLAYER_VS_PLAYER:
        if genome is None:
            with open("aggressive/generation_116_winner.pkl", "rb") as f:
                genome = pickle.load(f)

        evaluator = FeedForwardNetwork.from_genome(genome).activate

    game = Game(
        Params.board_width,
        Params.board_height,
        Params.board_title,
        Params.board_size,
        10,
        evaluator,
        player_starts,
        model_plays_itself,
    )
    game.setup()
    game.run()


if __name__ == "__main__":
    main()
