from typing import Union
from ..search import HyperOpt
from ..spaces import GridSpace
import numpy as np


class CoordinateSearch(HyperOpt):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        search_config: dict = {},
        maximize_objective: bool = False,
        fixed_params: Union[dict, None] = None,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        HyperOpt.__init__(
            self,
            real,
            integer,
            categorical,
            search_config,
            maximize_objective,
            fixed_params,
            reload_path,
            reload_list,
            seed_id,
            verbose,
        )
        self.evals_per_coord = [0]
        for k in self.search_config["order"]:
            if self.real is not None:
                if k in self.real.keys():
                    self.evals_per_coord.append(self.real[k]["bins"])

            if self.integer is not None:
                if k in self.integer.keys():
                    range_int = (
                        np.linspace(
                            int(self.integer[k]["begin"]),
                            int(self.integer[k]["end"]),
                            int(self.integer[k]["bins"]),
                        )
                        .astype(int)
                        .tolist()
                    )
                    self.evals_per_coord.append(len(range_int))

            if self.categorical is not None:
                for k in self.categorical.keys():
                    self.evals_per_coord.append(len(self.categorical[k]))
        self.range_per_coord = np.cumsum(self.evals_per_coord)

        # Sequentially set-up different grid spaces - initialize 1st one
        self.grid_var_counter = 0
        self.var_counter = 0
        self.construct_active_space()
        self.search_name = "Coordinate-Wise Search"

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Coordinate Search"""
        # Set grid counter to eval_counter in order ensure while
        # That results for grid configuration are collected before continuation
        self.grid_var_counter = (
            self.eval_counter - self.range_per_coord[self.var_counter]
        )

        param_batch = []
        # Sample a new configuration for each eval in the batch
        while len(param_batch) < batch_size and self.grid_var_counter < len(self.space):
            # Get parameter batch from the grid
            proposal_params = self.space.param_grid[self.grid_var_counter]
            if proposal_params not in (self.all_evaluated_params + param_batch):
                # Add parameter proposal to the batch list
                param_batch.append(proposal_params)
                self.grid_var_counter += 1
            else:
                # Otherwise continue sampling proposals
                self.grid_var_counter += 1
                continue
        return param_batch

    def tell_search(self, batch_proposals: list, perf_measures: list):
        """Update search log data - Coordinate Search"""
        # Update/reset variable and grid counter based on eval_counter
        # And evals per search space (makes it easier to reload)
        self.grid_var_counter = (
            self.eval_counter - self.range_per_coord[self.var_counter]
        )
        if self.grid_var_counter == len(self.space):
            self.var_counter += 1
            if self.var_counter < len(self.search_config["order"]):
                self.construct_active_space()
                self.grid_var_counter = 0

    def construct_active_space(self):
        """Construct the active search space."""
        # Update the parameter defaults with the best performers
        if self.eval_counter > 0:
            idx, config, eval = self.get_best()
            for k, v in config.items():
                self.search_config["defaults"][k] = v

        # Increase active variable counter and reset grid counter
        self.active_var = self.search_config["order"][self.var_counter]

        # Create new grid search space - if fixed: Create categorical
        # Note: Only one variable is 'active' at every time
        real_sub, integer_sub, categorical_sub = {}, {}, {}
        if self.real is not None:
            for k in self.real.keys():
                if k == self.active_var:
                    real_sub[k] = self.real[k]
                else:
                    categorical_sub[k] = [self.search_config["defaults"][k]]

        if self.integer is not None:
            for k in self.integer.keys():
                if k == self.active_var:
                    integer_sub[k] = self.integer[k]
                else:
                    categorical_sub[k] = [self.search_config["defaults"][k]]

        if self.categorical is not None:
            for k in self.categorical.keys():
                if k == self.active_var:
                    categorical_sub[k] = self.categorical[k]
                else:
                    categorical_sub[k] = [self.search_config["defaults"][k]]

        # Construct new grid space with fixed coordinates!
        self.space = GridSpace(real_sub, integer_sub, categorical_sub)
