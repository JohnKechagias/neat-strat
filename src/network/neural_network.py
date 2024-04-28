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

from .constants import EndgameState, Player
from .game_state import GameState
from .search import play


def fitness_func(
    genome: Genome,
    opponent_genomes: list[Genome],
    min_opponents_count: int,
) -> float:
    player = FeedForwardNetwork.from_genome(genome).activate
    opponents = [FeedForwardNetwork.from_genome(g).activate for g in opponent_genomes]

    opponents_to_add = min_opponents_count - len(opponents)
    if opponents_to_add > 0:
        times_to_duplicate_opponents = opponents_to_add // len(opponents)
        extra_opponents_to_add = opponents_to_add % len(opponents)
        opponents.extend(opponents[:] * times_to_duplicate_opponents)

        if extra_opponents_to_add:
            opponents.extend(opponents[:extra_opponents_to_add])

    max_moves = 20
    max_fitness = 400
    depth = 3

    fitnesses: list[float] = []
    for opponent in opponents:
        opponent_starts = random.random() < 0.5
        endstate, game_state = play(player, opponent, opponent_starts, max_moves, depth)
        fitness = get_fitness(
            endstate, game_state, opponent_starts, max_moves, max_fitness
        )
        fitnesses.append(fitness)

    fitness = mean(fitnesses)
    print(f"Assigned fitness: {fitness}")
    return fitness


def get_fitness(
    endstate: EndgameState,
    game_state: GameState,
    opponent_started: bool,
    max_moves: int,
    fitness_threshold: float,
) -> float:
    move_record = game_state.history
    won_status = EndgameState.BLUE_WON
    lost_status = EndgameState.RED_WON
    if opponent_started:
        won_status = EndgameState.RED_WON
        lost_status = EndgameState.BLUE_WON

    moves = len(move_record)
    players_population = int(np.sum(game_state.board))
    win_bonus = fitness_threshold / 10
    max_score_gained_from_move = 5.0
    capture_bonus = 8.0

    fitness = 0.0
    player = Player.RED if opponent_started else Player.BLUE
    fitness += game_state.captures[player] * capture_bonus

    if endstate == EndgameState.ONGOING:
        fitness += min(players_population * 2, fitness_threshold / 2)
    elif endstate == won_status:
        print(f"Status: {endstate.name.title()}")
        fitness += win_bonus
        fitness += (max_moves - moves) * 5
    elif endstate == lost_status:
        print(f"Status: {endstate.name.title()}")
        fitness += min(moves * 3, fitness_threshold / 3)

    consider_move = lambda i: i % 2 if opponent_started else not i % 2
    moves_to_consider = [m for i, m in enumerate(move_record) if consider_move(i)]

    # Aggressive settings. Player will rush to the center of the board.
    # target = (2, 2)
    # Defensive settings. Each player will try to remain on his own side of the board.
    target = (0, 4) if player == Player.RED else (4, 0)
    for move in moves_to_consider:
        score_lost_due_to_position = target[0] - move[1][0] + target[1] - move[1][1]
        fitness += min(
            max_score_gained_from_move - abs(score_lost_due_to_position), -1.0
        )

    return fitness


def train(iterations: int) -> Genome:
    local_path = Path(__file__).parent
    config_path = local_path / "config.ini"

    params = neat.Parameters(config_path)
    population = neat.Population(params)

    def evaluate(genomes: list[Genome], opponents: list[Genome]):
        workers = multiprocessing.cpu_count()
        pool = Pool(workers)
        min_opponents_count = 5

        evaluator = partial(
            fitness_func,
            opponent_genomes=opponents,
            min_opponents_count=min_opponents_count,
        )
        fitnesses = pool.map(evaluator, genomes)

        for fitness, genome in zip(fitnesses, genomes):
            genome.fitness = fitness

    winner, statistical_data = population.run(evaluate, times=iterations)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner
