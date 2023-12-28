from dataclasses import dataclass
from typing import Annotated

from utils import ValueRange

from neat.activations import ActivationFuncs
from neat.aggregations import AggregationFuncs
from neat.pool import FitnessFuncs


@dataclass
class SpeciationParameters:
    compatibility_disjoint_coefficient: float
    compatibility_weight_coefficient: float
    compatibility_threshold: float
    fitness_func: FitnessFuncs

    survival_rate: Annotated[float, ValueRange(0.0, 1.0)]

    elitism: int  # Basically the minimum size of a species.
    max_stagnation: int


@dataclass
class ReproductionParameters:
    corssover_rate: Annotated[float, ValueRange(0.0, 1.0)]


@dataclass
class Parameters:
    population: int

    number_of_inputs: int
    number_of_outputs: int
    number_of_hidden_nodes: int

    aggregation_default_value: AggregationFuncs
    aggregation_options: list[AggregationFuncs]
    activation_default_value: ActivationFuncs
    activation_options: list[ActivationFuncs]

    link_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_addition_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_deletion_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_enablement_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_disablement_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_toggle_enable_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_addition_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_deletion_chance: Annotated[float, ValueRange(0.0, 1.0)]

    bias_default_value: float
    bias_min_value: float
    bias_max_value: float
    bias_mutate_power: float
    bias_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]

    response_default_value: float
    response_max_value: float
    response_min_value: float
    response_mutate_change: Annotated[float, ValueRange(0.0, 1.0)]
    response_mutate_power: float

    weight_min_value: float
    weight_max_value: float
    weight_mutate_power: float
    weight_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    weight_severe_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]

    speciation: SpeciationParameters
    reproduction: ReproductionParameters
