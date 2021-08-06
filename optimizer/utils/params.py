from dataclasses import dataclass
from typing import Tuple


@dataclass
class FontLimits:
    FONT_SIZE_LIMIT: Tuple[int, int] = (1, 30)
    FORM_LIMIT: Tuple[int, int] = (1, 4)


@dataclass
class LossFunctionParameters:
    OVERLAP_WEIGHTAGE: float = 0.5
    MIN_DISTANCE_WEIGHTAGE: float = 0.4
    UNIFORM_DISTANCE_WEIGHTAGE: float = 0.05
    BOX_AREA_WEIGHTAGE: float = 0.05
    MAXIMUM_LOSS_THRESHOLD: float = 200


@dataclass
class OptimizerParameters:
    POPULATION_SIZE: int = 15
