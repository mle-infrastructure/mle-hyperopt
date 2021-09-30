from collections.abc import Mapping, Iterable
from functools import partial, reduce
import operator
from itertools import product
from typing import Union
from .base import HyperOpt
from ..hyperspace import construct_hyperparam_range


class GridHyperoptimisation(HyperOpt):
    def __init__(
        self,
        hyper_log: HyperoptLogger,
        resource_to_run: str,
        job_arguments: dict,
        config_fname: str,
        job_fname: str,
        experiment_dir: str,
        search_params: dict,
        search_type: str = "grid",
        search_schedule: str = "sync",
        message_id: Union[str, None] = None,
    ):
        BaseHyperOptimisation.__init__(
            self,
            hyper_log,
            resource_to_run,
            job_arguments,
            config_fname,
            job_fname,
            experiment_dir,
            search_params,
            search_type,
            search_schedule,
            message_id,
        )
        # Generate all possible combinations of param configs in list & loop
        # over the list when doing the grid search
        self.param_range = construct_hyperparam_range(
            self.search_params, self.search_type
        )
        self.param_grid = self.generate_search_grid()
        self.num_param_configs = len(self.param_grid)
        self.eval_counter = len(hyper_log)

    def get_hyperparam_proposal(self, num_iter_per_batch: int):
        """Get proposals to eval next (in batches) - Grid Search"""
        param_batch = []
        # Sample a new configuration for each eval in the batch
        while (
            len(param_batch) < num_iter_per_batch
            and self.eval_counter < self.num_param_configs
        ):
            # Get parameter batch from the grid
            proposal_params = self.param_grid[self.eval_counter]
            if proposal_params not in (
                self.hyper_log.all_evaluated_params + param_batch
            ):
                # Add parameter proposal to the batch list
                param_batch.append(proposal_params)
                self.eval_counter += 1
            else:
                # Otherwise continue sampling proposals
                continue
        return param_batch

    def generate_search_grid(self):
        """Construct the parameter grid & return as a list to index"""
        return list(ParameterGrid(self.param_range))


class ParameterGrid:
    """Param Grid Class taken from sklearn: https://tinyurl.com/yj53efc9"""

    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(
                "Parameter grid is not a dict or " "a list ({!r})".format(param_grid)
            )

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError("Parameter grid is not a " "dict ({!r})".format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError(
                        "Parameter grid value is not iterable "
                        "(key={!r}, value={!r})".format(key, grid[key])
                    )

        self.param_grid = param_grid

    def __iter__(self):
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(
            product(len(v) for v in p.values()) if p else 1 for p in self.param_grid
        )
