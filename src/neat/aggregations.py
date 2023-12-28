from enum import Enum, auto
from operator import mul
from typing import Callable

import numpy as np
from numba import njit
from numba.core.typeinfer import reduce


class AggregationFuncs(Enum):
    MAX = auto()
    MIN = auto()
    MAXABS = auto()
    MEAN = auto()
    SUM = auto()
    PRODUCT = auto()
    MEDIAN = auto()


def get_aggregation_func(func: AggregationFuncs) -> Callable[[list[float]], float]:
    aggregation_func_mapper = {
        AggregationFuncs.MAX: np.max,
        AggregationFuncs.MIN: np.min,
        AggregationFuncs.MAXABS: maxabs,
        AggregationFuncs.MEAN: np.mean,
        AggregationFuncs.SUM: np.sum,
        AggregationFuncs.PRODUCT: product,
        AggregationFuncs.MEDIAN: np.median,
    }
    return aggregation_func_mapper[func]


@njit
def product(values: list[int]):
    return reduce(mul, values, 1.0)


@njit
def maxabs(values: list[int]):
    return max(values, key=abs)
