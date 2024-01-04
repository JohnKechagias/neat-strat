from math import ceil

from neat.genomes.genome import Genome
from neat.innovation import InnovationRecord
from neat.logging import LOGGER
from neat.parameters import Parameters
from neat.species import Species, SpeciesInfo
from neat.utils import get_genome_id, mean


class Pool:
    def __init__(
        self,
        params: Parameters,
        innov_record: InnovationRecord,
    ):
        self.genomes: list[Genome] = []
        self.species: list[Species] = []
        self.params = params
        self.innov_record = innov_record
        self.generation: int = 0
        self.prev_species_info: list[SpeciesInfo] = []

        self.initialize()

    def initialize(self):
        for _ in range(self.params.population):
            self.genomes.append(self._get_new_genome())

        for genome in self.genomes:
            self._assign_genome_to_species(genome)

    def evolve(self):
        remaining_species = self._speciate()

        for species in remaining_species:
            species.kill_worst(self.params.speciation.survival_rate)

        is_stagnant = lambda s: s.stagnant > self.params.speciation.max_stagnation
        remaining_species = [s for s in remaining_species if not is_stagnant(s)]

        if len(remaining_species) == 0:
            LOGGER.warning(
                "There are no remaining species after evolution. "
                "Consider increasing the compatibility threshold."
            )

        offsprings: list[Genome] = []

        for species in remaining_species:
            offsprings.extend(species.elites(self.params.speciation.elitism))

        # The number of genomes that need to be generated to fill out the population.
        extra_genomes_needed = self.params.population - len(offsprings)

        adjusted_fitnesses = [f.update_adjusted_fitness() for f in remaining_species]
        adj_fintess_sum = sum(adjusted_fitnesses)
        avg_adjusted_fitness = mean(adjusted_fitnesses)
        LOGGER.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Contains the number of genomes that each species must generate
        # to fill out the population.
        genomes_to_spawn: list[int] = []
        for fitness in adjusted_fitnesses:
            normalized_fitness = fitness / adj_fintess_sum
            genome_spawn_amount = ceil(extra_genomes_needed * normalized_fitness)
            genomes_to_spawn.append(genome_spawn_amount)

        extra_genomes_generated = sum(genomes_to_spawn)

        # Remove extra genomes that will be generated so that the sum of
        # genomes is the same as the needed population.
        for i in range(extra_genomes_generated - extra_genomes_needed):
            genomes_to_spawn[i % len(remaining_species)] -= 1

        for species, spawn_amount in zip(remaining_species, genomes_to_spawn):
            for _ in range(spawn_amount):
                # TODO add crossover rate.
                offsprint = species.mate()
                offsprint.mutate()
                offsprings.append(offsprint)

        self.prev_species_info = [species.info for species in remaining_species]
        self.genomes = offsprings
        self.generation += 1

    def _assign_genome_to_species(self, genome: Genome):
        for species in self.species:
            if species.try_assign_genome(genome):
                return

        # If genome could not be assigned to a species create a new one.
        species = self._get_new_species(genome)
        self.species.append(species)

    def _speciate(self) -> list[Species]:
        new_species: list[Species] = []

        for info in self.prev_species_info:
            info.add_age()
            new_species.append(Species(info))

        for genome in self.genomes:
            found = False

            for species in new_species:
                if species.try_assign_genome(genome):
                    break

            if not found:
                species = self._get_new_species(genome)
                new_species.append(species)

        return new_species

    def _get_new_genome(self) -> Genome:
        id = get_genome_id()
        return Genome(id, self.innov_record)

    def _get_new_species(self, representative: Genome) -> Species:
        info = SpeciesInfo(self.innov_record.new_species(), representative)
        return Species(info)
