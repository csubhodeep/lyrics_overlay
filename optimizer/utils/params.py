from dataclasses import dataclass
from typing import Tuple


@dataclass
class FontLimits:
    FONT_SIZE_LIMIT: Tuple[int, int] = (1, 30)
    FORM_LIMIT: Tuple[int, int] = (1, 4)


@dataclass
class LossFunctionParameters:
    OVERLAP_WEIGHTAGE: float = 0.5
    MIN_DISTANCE_WEIGHTAGE: float = 0.48
    UNIFORM_DISTANCE_WEIGHTAGE: float = 0.4
    BOX_AREA_WEIGHTAGE: float = 0.01
    MAXIMUM_LOSS_THRESHOLD: float = 3
    DISTANCE_BIAS: int = 3000


@dataclass
class OptimizerParameters:
    POPULATION_SIZE: int = 100
