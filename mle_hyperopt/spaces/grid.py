import numpy as np
from collections.abc import Mapping, Iterable
from functools import partial, reduce
import operator
from itertools import product
from typing import Optional, List
from ..space import HyperSpace


class GridSpace(HyperSpace):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
    ):
        """Grid search hyperparameter space with desired discrete resolution.

        Args:
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
        """
        HyperSpace.__init__(self, real, integer, categorical)

    def check(self) -> None:
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

    def sample(self, grid_counter: int) -> dict:
        """Sample' from the hyperparameter space. - Return next config in list.

        Args:
            grid_counter (int): Counter for where to index flat grid.

        Returns:
            dict: Parameter configuration in the grid.
        """
        return self.param_grid[grid_counter]

    def construct(self) -> None:
        """Construct the parameter grid for the different search variables."""
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
        """Return total number of configurations in parameter grid."""
        return len(self.param_grid)

    def describe(self) -> List[dict]:
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
                    "extra": (
                        f'Begin: {v["begin"]}, End: {v["end"]}, Bins:'
                        f' {v["bins"]}'
                    ),
                }
                all_vars.append(data_dict)

        if self.integer is not None:
            for k, v in self.integer.items():
                data_dict = {
                    "name": k,
                    "type": "integer",
                    "extra": (
                        f'Begin: {v["begin"]}, End: {v["end"]}, Bins:'
                        f' {v["bins"]}'
                    ),
                }
                all_vars.append(data_dict)
        return all_vars

    def contains(self, candidate: dict) -> bool:
        """Check whether a parameter candidate is in the search space.

        Args:
            candidate (dict): Candidate parameter dictionary.

        Returns:
            bool: Boolean indicating whether candidate is in the grid.
        """
        # Define a separate function for discrete grid case!
        for k, v in candidate.items():
            if not (v in self.param_range[k]):
                return False
        return True


class ParameterGrid:
    def __init__(self, param_grid: dict):
        """Constructs parameter grid from dictionary of possible parameter
        values. Taken from sklearn: https://tinyurl.com/yj53efc9.

        Args:
            param_grid (dict): Dict containing lists of discrete possible
                values of independent variables to search over.
        """
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(
                "Parameter grid is not a dict or a list ({!r})".format(
                    param_grid
                )
            )

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(
                    "Parameter grid is not a dict ({!r})".format(grid)
                )
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError(
                        "Parameter grid value is not iterable "
                        "(key={!r}, value={!r})".format(key, grid[key])
                    )

        self.param_grid = param_grid

    def __iter__(self):
        """Iterator looping over different configurations in grid."""
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
        """Number of points/configurations in the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        return sum(
            product(len(v) for v in p.values()) if p else 1
            for p in self.param_grid
        )
