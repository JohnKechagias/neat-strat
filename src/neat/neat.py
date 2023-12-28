import logging
from typing import Callable, Optional

from pool import Pool

from neat.innovation import InnovationRecord
from neat.network import Network
from neat.parameters import Parameters


class Neat:
    def __init__(self, params: Parameters):
        self.logger = logging.getLogger(__name__)
        self.params = params

    def run(self, fitness_function: Callable, times: Optional[int] = None):
        found_optimal_network = False
        pool = self.initialize_pool()

        while not found_optimal_network:
            pool.evolve()

    def evaluate(self, pool: Pool, input: list[float]):
        for genome in pool.genomes:
            network = Network(genome)

    def initialize_pool(self) -> Pool:
        self.innovation_record = InnovationRecord()
        return Pool(self.params, self.innovation_record, self.logger)
