from typing import Union
import functools
from .strategies import (
    RandomSearch,
    GridSearch,
    SMBOSearch,
    NevergradSearch,
    CoordinateSearch,
    HalvingSearch,
    HyperbandSearch,
    PBTSearch,
)


def hyperopt(
    strategy_type: str,
    num_search_iters: int,
    real: Union[dict, None] = None,
    integer: Union[dict, None] = None,
    categorical: Union[dict, None] = None,
    search_config: Union[dict, None] = None,
    maximize_objective: bool = False,
    fixed_params: Union[dict, None] = None,
):
    """
    Simple search decorator for all strategies. Example usage:
    @hyperopt(strategy_type="grid",
              num_search_iters=25,
              real={"x": {"begin": 0., "end": 0.5, "bins": 5},
                    "y": {"begin": 0, "end": 0.5, "bins": 5}})
    def distance_from_circle(config):
        distance = abs((config["x"] ** 2 + config["y"] ** 2))
        return distance

    strategy = distance_from_circle()
    strategy.log
    """
    assert strategy_type in [
        "Random",
        "Grid",
        "SMBO",
        "Nevergrad",
        "Coordinate",
        "Halving",
        "Hyperband",
        "PBT",
    ]

    if strategy_type == "Random":
        strategy = RandomSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "Grid":
        strategy = GridSearch(real, integer, categorical, fixed_params)
    elif strategy_type == "SMBO":
        strategy = SMBOSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "Nevergrad":
        strategy = NevergradSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "Coordinate":
        strategy = CoordinateSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "Halving":
        strategy = HalvingSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "Hyperband":
        strategy = HyperbandSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "PBT":
        strategy = PBTSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for iter_id in range(num_search_iters):
                config = strategy.ask()
                result = function(config, *args, **kwargs)
                strategy.tell(config, [result])
            return strategy

        return wrapper

    return decorator
