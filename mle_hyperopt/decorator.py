from typing import Optional, Callable, Any
import functools
from .strategies import Strategies


def hyperopt(
    strategy_type: str,
    num_search_iters: int,
    real: Optional[dict] = None,
    integer: Optional[dict] = None,
    categorical: Optional[dict] = None,
    search_config: Optional[dict] = None,
    maximize_objective: bool = False,
    fixed_params: Optional[dict] = None,
) -> Callable[[Any], None]:
    """Simple search decorator for all strategies. Example usage:
    @hyperopt(strategy_type="grid",
              num_search_iters=25,
              real={"x": {"begin": 0., "end": 0.5, "bins": 5},
                    "y": {"begin": 0, "end": 0.5, "bins": 5}})
    def distance_from_circle(config):
        distance = abs((config["x"] ** 2 + config["y"] ** 2))
        return distance

    strategy = distance_from_circle()
    strategy.log

    Args:
        strategy_type (str): Name of search strategy.
        num_search_iters (int): Number of iterations to run.
        real (Optional[dict], optional):
                Dictionary of real-valued search variables & their resolution.
                E.g. {"lrate": {"begin": 0.1, "end": 0.5, "bins": 5}}
                Defaults to None.
            integer (Optional[dict], optional):
                Dictionary of integer-valued search variables & their resolution.
                E.g. {"batch_size": {"begin": 1, "end": 5, "bins": 5}}
                Defaults to None.
            categorical (Optional[dict], optional):
                Dictionary of categorical-valued search variables.
                E.g. {"arch": ["mlp", "cnn"]}
                Defaults to None.
            search_config (dict, optional): Grid search hyperparameters.
                Defaults to None.
            maximize_objective (bool, optional): Whether to maximize objective.
                Defaults to False.
            fixed_params (Optional[dict], optional):
                Fixed parameters that will be added to all configurations.
                Defaults to None.

    Returns:
        Callable[Any]: _description_
    """
    assert strategy_type in Strategies.keys()

    strategy = Strategies[strategy_type](
        real,
        integer,
        categorical,
        search_config,
        maximize_objective,
        fixed_params,
    )

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for _ in range(num_search_iters):
                config = strategy.ask()
                result = function(config, *args, **kwargs)
                strategy.tell(config, [result])
            return strategy

        return wrapper

    return decorator
