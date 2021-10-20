from typing import Union
import functools
from .strategies import (
    RandomSearch,
    GridSearch,
    SMBOSearch,
    NevergradSearch,
    CoordinateSearch,
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

    if strategy_type == "random":
        strategy = RandomSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "grid":
        strategy = GridSearch(real, integer, categorical, fixed_params)
    elif strategy_type == "smbo":
        strategy = SMBOSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "nevergrad":
        strategy = NevergradSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )
    elif strategy_type == "coordinate":
        strategy = CoordinateSearch(
            real, integer, categorical, search_config, maximize_objective, fixed_params
        )

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for iter_id in range(num_search_iters):
                config = strategy.ask()
                result = function(config)
                strategy.tell(config, [result])
            return strategy

        return wrapper

    return decorator
