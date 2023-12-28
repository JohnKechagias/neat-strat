import random
from dataclasses import dataclass
from enum import Enum, auto
from math import floor

from genome import Genome


class FitnessFuncs(Enum):
    MEAN = auto()


class PopulationConfig:
    initial_population_size: int


@dataclass
class SpeciesInfo:
    id: int
    representative: Genome
    age: int = 0
    previous_fitness: float = 0
    stagnant: int = 0

    def add_age(self):
        self.age += 1


class Species:
    def __init__(self, info: SpeciesInfo):
        self.info = info
        # A sorted list with all the genomes in the Species.
        # The list is sorted from most fit to least fit genome.
        self._genomes = [info.representative]

    @property
    def genome_count(self) -> int:
        return len(self._genomes)

    @property
    def age(self) -> int:
        return self.info.age

    @property
    def id(self) -> int:
        return self.info.id

    @property
    def stagnant(self) -> int:
        return self.info.stagnant

    def try_assign_genome(self, genome: Genome) -> bool:
        is_compatible = self.info.representative.is_compatible(genome)

        if is_compatible:
            self._genomes.append(genome)

        return is_compatible

    def force_assign_genome(self, genome: Genome):
        self._genomes.append(genome)

    def kill_worst(self, survival_rate: float):
        remaining = max(floor(len(self._genomes) * survival_rate), 1)
        self._genomes = self._genomes[:remaining]

    def update_adjusted_fitness(self) -> float:
        """Updates the species fitness value and returns it.

        Returns:
            The fitness of the species.
        """
        fitness_sum = 0.0
        for genome in self._genomes:
            fitness_sum += genome.fitness

        num_of_genomes = len(self._genomes)
        fitness = 0.0 if not self._genomes else fitness_sum / num_of_genomes**2

        if fitness < self.info.previous_fitness:
            self.info.stagnant += 1
        else:
            self.info.stagnant = 0

        self.info.previous_fitness = fitness
        return fitness

    def mate(self) -> Genome:
        parent1 = random.choice(self._genomes)
        parent2 = random.choice(self._genomes)
        return parent1.crossover(parent2)

    def elites(self, count: int) -> list[Genome]:
        return self._genomes[:count]
