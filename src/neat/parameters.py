import re
from pathlib import Path
from inspect import get_annotations
from configparser import ConfigParser

from typing import Annotated, Any, Type
import collections

from neat.fitness import LossFuncs, FitnessCriterionFuncs

from neat.utils import ValueRange

from neat.activations import ActivationFuncs
from neat.aggregations import AggregationFuncs

from configparser import ConfigParser


class CaseInsensitiveDict(collections.UserDict):
    """Ordered case insensitive mutable mapping class."""

    def _convert_keys(self, dictionary: dict) -> dict[str, Any]:
        keys = list(dictionary.keys())
        for key in keys:
            value = dictionary.pop(key)
            dictionary[self.format_key(key)] = value

        return dictionary

    @staticmethod
    def format_key(key: str) -> str:
        """Converts the key to lower case and replaces camel case and
        spaces with snake case.

        >>> format_key("ThisIs ASentence")
        "this_is_a_sentence"
        """
        key_words = re.sub( r"([A-Z][a-z])", r" \1", key).split()
        key_words = [word.lower() for word in key_words]
        return "_".join(key_words)

    def __setitem__(self, key: str, value: Any):
        key = self.format_key(key)
        super().__setitem__(key, value)


class NEATParameters:
    population: int
    reset_on_extinction: bool


class GenomeParameters:
    number_of_inputs: int
    number_of_outputs: int
    number_of_hidden_nodes: int

    aggregation_default: AggregationFuncs
    aggregation_options: list[AggregationFuncs]
    aggregation_mutation_change: float

    activation_default: ActivationFuncs
    activation_options: list[ActivationFuncs]
    activation_mutation_change: float

    link_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_addition_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_deletion_chance: Annotated[float, ValueRange(0.0, 1.0)]
    link_toggle_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_addition_chance: Annotated[float, ValueRange(0.0, 1.0)]
    node_deletion_chance: Annotated[float, ValueRange(0.0, 1.0)]

    bias_init_mean: float
    bias_init_stdev: float
    bias_min_value: float
    bias_max_value: float
    bias_mutation_power: float
    bias_mutation_chance: Annotated[float, ValueRange(0.0, 1.0)]
    bias_replace_rate: float

    response_init_mean: float
    response_init_stdev: float
    response_min_value: float
    response_max_value: float
    response_mutation_change: Annotated[float, ValueRange(0.0, 1.0)]
    response_mutation_power: float
    response_replace_rate: float

    weight_init_mean: float
    weight_init_stdev: float
    weight_min_value: float
    weight_max_value: float
    weight_mutation_chance: float
    weight_mutation_power: float
    weight_severe_mutation_chance: float
    weight_replace_rate: float


class SpeciationParameters:
    compatibility_disjoint_coefficient: float
    compatibility_weight_coefficient: float
    compatibility_threshold: float

    survival_rate: Annotated[float, ValueRange(0.0, 1.0)]

    elitism: int  # Basically the minimum size of a species.
    max_stagnation: int


class EvaluationParameters:
    fitness_criterion: FitnessCriterionFuncs
    fitness_threshold: float
    initial_fitness: float
    loss_function: LossFuncs


class ReproductionParameters:
    corssover_rate: Annotated[float, ValueRange(0.0, 1.0)]


class Parameters:
    def __init__(self, config_file: Path):
        params = ConfigParser(dict_type=CaseInsensitiveDict)
        params.read(config_file)
        
        for attr, attr_type in get_annotations(self.__class__).items():
            print(params.sections())
            if not attr in params.sections():
                raise ValueError(f"Section '{attr}' not present in config.")

            value = set_parameter_attributes(dict(params[attr]), attr_type)
            setattr(self, attr, value)

    neat: NEATParameters
    genome: GenomeParameters
    speciation: SpeciationParameters
    evaluation: EvaluationParameters
    reproduction: ReproductionParameters


def set_parameter_attributes(input: dict, params_type: Type):
    params = params_type()

    for attr, attr_type in get_annotations(params_type).items():
        input_value = input.get(attr)
        if input_value is None:
            raise ValueError(f"Parameter '{attr}' is not defined.")

        setattr(params, attr, attr_type(input_value))

    return params_type
