from dataclasses import dataclass
from typing import Tuple


@dataclass
class FontLimits:
    FONT_SIZE_LIMIT: Tuple[int, int] = (1, 30)
    FORM_LIMIT: Tuple[int, int] = (1, 4)


@dataclass
class LossFunctionParameters:
    MIN_DISTANCE_THRESHOLD: int = 20
    MIN_DISTANCE_COST: int = 4000
    OVERLAPPING_COST: int = 20000
    SMALL_BOX_COST: int = 20050
    WRONG_COORDINATE_COST: int = 40000
    MAXIMUM_LOSS_THRESHOLD: int = 4000


@dataclass
class OptimizerParameters:
    POPULATION_SIZE: int = 100
