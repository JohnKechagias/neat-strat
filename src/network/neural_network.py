import pickle
from pathlib import Path

import neat
import numpy as np
from neat import FeedForwardNetwork, Genome
from neat.utils import mean

from .constants import Board, EndgameState, MovesRecord
from .search import play


def print_moves_record(moves_record: MovesRecord):
    strings = []

    for record in moves_record:
        strings.append(f"{record[1]}")

    print("\n".join(strings))


def eval_genome(genome: Genome, opponent_genomes: list[Genome]) -> float:
    player = FeedForwardNetwork.from_genome(genome).activate
    opponents = [FeedForwardNetwork.from_genome(g).activate for g in opponent_genomes]

    max_moves = 20
    max_fitness = 60
    depth = 3

    fitnesses = []
    for opponent in opponents:
        endgame_state, state, history = play(player, opponent, max_moves, depth)
        fitness = get_fitness(endgame_state, state, history, max_moves, max_fitness)
        fitnesses.append(fitness)

    fitness = mean(fitnesses)
    print(f"Assigned fitness: {fitness}")
    return fitness


def get_fitness(
    game_state: EndgameState,
    board_state: Board,
    moves_record: MovesRecord,
    max_moves: int,
    fitness_threshold: float,
) -> float:
    moves = len(moves_record)
    players_population = int(np.sum(board_state))

    fitness = 0
    if game_state == EndgameState.ONGOING:
        fitness = min(players_population * 0.2, fitness_threshold * 3 / 4)
    elif game_state == EndgameState.WIN:
        fitness = (max_moves - moves) * 0.2
    elif game_state == EndgameState.LOSS:
        fitness = min(moves * 0.01, 20)

    print(f"Status: {game_state.name.title()}")
    return fitness


def run():
    local_path = Path(__file__).parent
    config_path = local_path / "config.ini"

    params = neat.Parameters(config_path)
    population = neat.Population(params)
    winner = population.run(eval_genome, times=50)

    with open("winner", "wb") as f:
        pickle.dump(winner, f)
