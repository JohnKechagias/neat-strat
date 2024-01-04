from dataclasses import dataclass
from random import random
from typing import Iterable, Literal

from numba import njit


@njit
def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


@njit
def randon_sign() -> Literal[-1, 1]:
    return 1 if random() > 0.5 else -1


@dataclass
class ValueRange:
    lo: float
    hi: float


GENOME_ID = -1


def get_genome_id() -> int:
    global GENOME_ID
    GENOME_ID += 1
    return GENOME_ID


def mean(values: Iterable):
    values = list(values)
    return sum(map(float, values)) / len(values)
