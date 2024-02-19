import multiprocessing
import pickle
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import neat
import numpy as np
from neat import FeedForwardNetwork, Genome
from neat.utils import mean

from .constants import Board, EndgameState, MovesRecord
from .search import play


def eval_genome(genome: Genome, opponent_genomes: list[Genome]) -> float:
    player = FeedForwardNetwork.from_genome(genome).activate
    opponents = [FeedForwardNetwork.from_genome(g).activate for g in opponent_genomes]

    max_moves = 20
    max_fitness = 400
    depth = 3

    fitnesses: list[float] = []
    for opponent in opponents:
        opponent_starts = random.random() < 0.5
        endgame_state, state, history = play(
            player, opponent, opponent_starts, max_moves, depth
        )
        fitness = get_fitness(
            endgame_state, state, history, opponent_starts, max_moves, max_fitness
        )
        fitnesses.append(fitness)

    fitness = mean(fitnesses)
    print(f"Assigned fitness: {fitness}")
    return fitness


def get_fitness(
    game_state: EndgameState,
    board_state: Board,
    moves_record: MovesRecord,
    opponent_started: bool,
    max_moves: int,
    fitness_threshold: float,
) -> float:
    won_status = EndgameState.BLUE_WON
    lost_status = EndgameState.RED_WON
    if opponent_started:
        won_status = EndgameState.RED_WON
        lost_status = EndgameState.BLUE_WON

    moves = len(moves_record)
    players_population = int(np.sum(board_state))
    win_bonus = fitness_threshold / 3
    max_score_gained_from_move = 5

    fitness = 0.0
    if game_state == EndgameState.ONGOING:
        fitness += min(players_population * 2, fitness_threshold / 2)
    elif game_state == won_status:
        print(f"Status: {game_state.name.title()}")
        fitness += win_bonus
        fitness += (max_moves - moves) * 5
    elif game_state == lost_status:
        print(f"Status: {game_state.name.title()}")
        fitness += min(moves * 3, fitness_threshold / 3)

    consider_move = lambda i: i % 2 if opponent_started else not i % 2
    moves_to_consider = [m for i, m in enumerate(moves_record) if consider_move(i)]
    target = 0 if opponent_started else 10
    for move in moves_to_consider:
        score_lost_due_to_position = target - move[1][0] - move[1][1]
        fitness += max_score_gained_from_move - score_lost_due_to_position

    return fitness


def run():
    local_path = Path(__file__).parent
    config_path = local_path / "config.ini"

    params = neat.Parameters(config_path)
    population = neat.Population(params)

    def evaluate(genomes: list[Genome], opponents: list[Genome]):
        workers = multiprocessing.cpu_count()
        pool = Pool(workers)
        x = partial(eval_genome, opponent_genomes=opponents)
        fitnesses = pool.map(x, genomes)

        for fitness, genome in zip(fitnesses, genomes):
            genome.fitness = fitness

    winner = population.run(evaluate, times=50)

    with open("winner", "wb") as f:
        pickle.dump(winner, f)
