from typing import List
from models import EmbeddingModel, LPConstraints


def get_default_models() -> List[EmbeddingModel]:
    return [
        EmbeddingModel(
            id=1,
            name="fastText",
            var_code="x1",
            cost_per_query=0.20,
            response_time=0.15,
            precision=0.72,
            recall=0.68,
        ),
        EmbeddingModel(
            id=2,
            name="SBERT",
            var_code="x2",
            cost_per_query=0.40,
            response_time=0.40,
            precision=0.82,
            recall=0.80,
        ),
        EmbeddingModel(
            id=3,
            name="Хмарна модель",
            var_code="x3",
            cost_per_query=0.80,
            response_time=0.70,
            precision=0.90,
            recall=0.88,
        ),
    ]


def get_default_constraints() -> LPConstraints:
    return LPConstraints(
        p_min=0.80,
        r_min=0.78,
        t_max=0.50,
    )
