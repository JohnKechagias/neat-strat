import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from typing import Literal

import neat
import numpy as np
from neat.ctrnn import CTRNN
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.math_util import mean

from constants import GameState, State
from network.evaluator import play
from network.parallel_evaluator import ConcurrentEvaluator
from network.population import Population

from . import visualize

TIME_CONST = 0.01


def eval_genome(
    genome: DefaultGenome,
    config: DefaultGenomeConfig,
    genome_opponents: list[DefaultGenome],
) -> float:
    opponents = [
        partial(
            CTRNN.create(g, config, TIME_CONST).advance,
            advance_time=TIME_CONST,
            time_step=TIME_CONST,
        )
        for g in genome_opponents
    ]

    network = neat.ctrnn.CTRNN.create(genome, config, TIME_CONST)
    player = partial(network.advance, advance_time=TIME_CONST, time_step=TIME_CONST)

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

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    evaluator = ConcurrentEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(evaluator.evaluate, n=50)

    with open("winner", "wb") as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")

    node_names = {-1: "x", -2: "dx", -3: "theta", -4: "dtheta", 0: "control"}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="winner.gv",
    )
    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="winner-ctrnn-pruned.gv",
        prune_unused=True,
    )


if __name__ == "__main__":
    run()
