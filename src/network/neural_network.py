import multiprocessing
import pickle
from pathlib import Path
from typing import Literal

import neat
import numpy as np
from neat import FeedForwardNetwork, Genome
from neat.utils import mean

from constants import GameState, State
from network.evaluator import play
from network.parallel_evaluator import ConcurrentEvaluator


def eval_genome(genome: Genome, opponent_genomes: list[Genome]) -> float:
    opponents = [FeedForwardNetwork.from_genome(g).activate for g in opponent_genomes]
    player = FeedForwardNetwork.from_genome(genome).activate

    fitnesses = []
    max_total_moves = 50
    for opponent in opponents:
        game_state, state, moves_record = play(player, opponent, max_total_moves)
        fitness = get_fitness(game_state, state, moves_record)
        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return mean(fitnesses)


def get_fitness(
    game_state: GameState,
    board_state: State,
    moves_record: list[tuple[Literal[0, 1], tuple]],
) -> float:
    max_fitness = 60.0
    max_rounds = 50
    rounds = len(moves_record)
    players_population = int(np.sum(board_state))

    fitness = 0
    if game_state == GameState.ONGOING:
        if players_population > 0:
            fitness = min(players_population * 0.2, max_fitness * 3 / 4)
    elif game_state == GameState.WIN:
        print("ITS A WIN")
        fitness = (max_rounds - rounds) * 0.2
    elif game_state == GameState.LOSS:
        print("ITS A LOSS")
        fitness = min(rounds * 0.01, 20)

    return fitness


def run():
    local_path = Path(__file__).parent
    config_path = local_path / "config.ini"

    params = neat.Parameters(config_path)
    population = neat.Population(params)

    evaluator = ConcurrentEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(evaluator.evaluate, times=50)

    with open("winner", "wb") as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == "__main__":
    run()
