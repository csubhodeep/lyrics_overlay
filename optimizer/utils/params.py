from dataclasses import dataclass
from typing import Tuple


@dataclass
class FontLimits:
    FONT_SIZE_LIMIT: Tuple[int, int] = (1, 30)
    FORM_LIMIT: Tuple[int, int] = (1, 4)


@dataclass
class LossFunctionParameters:
    OVERLAP_WEIGHTAGE: float = 0.2
    MIN_DISTANCE_WEIGHTAGE: float = 0.02
    UNIFORM_DISTANCE_WEIGHTAGE: float = 0.08
    BOX_AREA_WEIGHTAGE: float = 0.7
    MAXIMUM_LOSS_THRESHOLD: float = 200


@dataclass
class OptimizerParameters:
    POPULATION_SIZE: int = 15
