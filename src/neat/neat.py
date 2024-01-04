from typing import Callable, Optional

from pool import Pool

from neat.innovation import InnovationRecord
from neat.parameters import Parameters
from neat.genomes.genome import Genome


class Neat:
    def __init__(self, params: Parameters):
        self.params = params

    def run(self, fitness_function: Callable, input: list[float], times: Optional[int] = None):
        found_optimal_network = False
        pool = self.initialize_pool()

        itterations = 0
        while not found_optimal_network:
            if times and itterations > times:
                break

            pool.evolve()
            self.evaluate(pool.genomes, fitness_function, input)

    def evaluate(self, genomes: list[Genome], fitness_function: Callable, input: list[float]):
        for genome in genomes:
            fitness = fitness_function(genome, input)
            genome.fitness = fitness

    def initialize_pool(self) -> Pool:
        self.innovation_record = InnovationRecord()
        return Pool(self.params, self.innovation_record)
