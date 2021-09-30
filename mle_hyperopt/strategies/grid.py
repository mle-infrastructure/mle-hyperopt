from collections.abc import Mapping, Iterable
from functools import partial, reduce
import operator
from itertools import product
from .base import HyperOpt
from ..hyperspace import construct_hyperparam_range


class GridSearch(HyperOpt):
    def __init__(
        self,
        search_params: dict,
    ):
        HyperOpt.__init__(self, search_params)
        # Generate all possible combinations of param configs in list & loop
        # over the list when doing the grid search
        self.param_range = construct_hyperparam_range(
            self.search_params, "grid"
        )
        self.param_grid = list(ParameterGrid(self.param_range))
        self.num_param_configs = len(self.param_grid)
        self.eval_counter = 0

    def ask(self, batch_size: int):
        """Get proposals to eval next (in batches) - Grid Search"""
        param_batch = []
        # Sample a new configuration for each eval in the batch
        while (
            len(param_batch) < batch_size
            and self.eval_counter < self.num_param_configs
        ):
            # Get parameter batch from the grid
            proposal_params = self.param_grid[self.eval_counter]
            param_batch.append(proposal_params)
            self.eval_counter += 1
            # if proposal_params not in (
            #     self.hyper_log.all_evaluated_params + param_batch
            # ):
            #     # Add parameter proposal to the batch list
            #     param_batch.append(proposal_params)
            #     self.eval_counter += 1
            # else:
            #     # Otherwise continue sampling proposals
            #     continue
        return param_batch

    def tell(self, batch_proposals, perf_measures):
        """Update search log data - Grid Search"""


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
