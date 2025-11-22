from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingModel:
    id: int
    name: str
    var_code: str
    cost_per_query: float
    response_time: float
    precision: float
    recall: float


@dataclass
class LPConstraints:
    p_min: float
    r_min: float
    t_max: float


@dataclass
class LPSolution:
    x: List[float]
    objective: float
