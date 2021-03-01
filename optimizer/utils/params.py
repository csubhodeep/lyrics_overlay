from dataclasses import dataclass
from typing import Tuple


@dataclass
class Costs:
    WRONG_COORDINATE_COST: int = 40000
    OVERLAPPING_COST: int = 20000
    TEXT_NOT_FITTING_COST: int = 10000
    MIN_DISTANCE_COST: int = 4000


@dataclass
class FontLimits:
    FONT_SIZE_LIMIT: Tuple[int, int] = (1, 10)
    FORM_LIMIT: Tuple[int, int] = (1, 1)


@dataclass
class LossFunctionParameters:
    MIN_DISTANCE_THRESHOLD: int = 20


@dataclass
class OptimizerParameters:
    POPULATION_SIZE: int = 100
