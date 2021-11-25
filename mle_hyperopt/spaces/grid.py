import numpy as np
from collections.abc import Mapping, Iterable
from functools import partial, reduce
import operator
from itertools import product
from typing import Union
from ..space import HyperSpace


class GridSpace(HyperSpace):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
    ):
        """For grid hyperopt generate numpy lists with desired resolution"""
        HyperSpace.__init__(self, real, integer, categorical)

    def check(self):
        """Check that all inputs are provided correctly."""
        if self.real is not None:
            real_keys = ["begin", "end", "bins"]
            for key in real_keys:
                for k, v in self.real.items():
                    assert key in v
                    assert float(v["begin"]) <= float(v["end"])

        if self.integer is not None:
            integer_keys = ["begin", "end", "bins"]
            for key in integer_keys:
                for k, v in self.integer.items():
                    assert key in v
                    assert int(v["begin"]) <= int(v["end"])
                    assert type(v[key]) == int

        if self.categorical is not None:
            for k, v in self.categorical.items():
                if type(v) is not list:
                    self.categorical[k] = [v]

    def sample(self, grid_counter):
        """'Sample' from the hyperparameter space. - Return next config."""
        return self.param_grid[grid_counter]

    def construct(self):
        self.param_range = {}
        if self.categorical is not None:
            for k, v in self.categorical.items():
                self.param_range[k] = v

        if self.real is not None:
            for k, v in self.real.items():
                self.param_range[k] = np.linspace(
                    float(v["begin"]), float(v["end"]), int(v["bins"])
                ).tolist()

        if self.integer is not None:
            for k, v in self.integer.items():
                self.param_range[k] = (
                    np.linspace(int(v["begin"]), int(v["end"]), int(v["bins"]))
                    .astype(int)
                    .tolist()
                )
        self.param_grid = list(ParameterGrid(self.param_range))

    def __len__(self) -> int:
        """Return number of runs stored in meta_log."""
        return len(self.param_grid)

    def describe(self):
        """Get space statistics/parameters printed out."""
        all_vars = []
        if self.categorical is not None:
            for k, v in self.categorical.items():
                data_dict = {"name": k, "type": "categorical", "extra": str(v)}
                all_vars.append(data_dict)

        if self.real is not None:
            for k, v in self.real.items():
                data_dict = {
                    "name": k,
                    "type": "real",
                    "extra": f'Begin: {v["begin"]}, End: {v["end"]}, Bins: {v["bins"]}',
                }
                all_vars.append(data_dict)

        if self.integer is not None:
            for k, v in self.integer.items():
                data_dict = {
                    "name": k,
                    "type": "integer",
                    "extra": f'Begin: {v["begin"]}, End: {v["end"]}, Bins: {v["bins"]}',
                }
                all_vars.append(data_dict)
        return all_vars

    def contains(self, candidate):
        """Check whether a candidate is in the search space."""
        # Define a separate function for discrete grid case!
        for k, v in candidate.items():
            if not (v in self.param_range[k]):
                return False
        return True


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
