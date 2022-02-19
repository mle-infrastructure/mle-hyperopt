from typing import Optional, List
from ..strategy import Strategy
from ..spaces import GridSpace
import numpy as np
from rich.console import Console


class CoordinateSearch(Strategy):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
        search_config: dict = {},
        maximize_objective: bool = False,
        fixed_params: Optional[dict] = None,
        reload_path: Optional[str] = None,
        reload_list: Optional[list] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        """Coordinate-wise Search Strategy.

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
            search_config (dict, optional): Coordinate-wies search hyperparams.
                E.g. {"order": ["lrate", "batch_size", "arch"],
                      "defaults": {"lrate": 0.1,
                                   "batch_size": 3,
                                   "arch": "mlp"}}
                Defaults to None.
            maximize_objective (bool, optional): Whether to maximize objective.
                Defaults to False.
            fixed_params (Optional[dict], optional):
                Fixed parameters that will be added to all configurations.
                Defaults to None.
            reload_path (Optional[str], optional):
                Path to load previous search log from. Defaults to None.
            reload_list (Optional[list], optional):
                List of previous results to reload. Defaults to None.
            seed_id (int, optional):
                Random seed for reproducibility. Defaults to 42.
            verbose (bool, optional):
                Option to print intermediate results. Defaults to False.
        """
        self.search_name = "Coordinate"
        Strategy.__init__(
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
        var_counter = 0
        for k in self.search_config["order"]:
            if self.real is not None:
                if k in self.real.keys():
                    self.evals_per_coord.append(
                        self.real[k]["bins"] + var_counter
                    )
                    var_counter += 1

            if self.integer is not None:
                if k in self.integer.keys():
                    self.evals_per_coord.append(
                        self.integer[k]["bins"] + var_counter
                    )
                    var_counter += 1

            if self.categorical is not None:
                for k in self.categorical.keys():
                    self.evals_per_coord.append(
                        len(self.categorical[k]) + var_counter
                    )
                    var_counter += 1
        self.range_per_coord = np.cumsum(self.evals_per_coord)

        # Sequentially set-up different grid spaces - initialize 1st one
        self.grid_var_counter = 0
        self.var_counter = 0
        self.construct_active_space()

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def ask_search(self, batch_size: int) -> List[dict]:
        """Get proposals to eval next (in batches) - Grid Search.

        Args:
            batch_size (int): Number of desired configurations

        Returns:
            List[dict]: List of configuration dictionaries
        """
        # Set grid counter to eval_counter in order ensure while
        # That results for grid configuration are collected before continuation
        grid_var_counter = (
            self.eval_counter - self.range_per_coord[self.var_counter]
        )

        param_batch = []
        # Sample a new configuration for each eval in the batch
        while len(param_batch) < batch_size and grid_var_counter < len(
            self.space
        ):
            # Get parameter batch from the grid
            proposal_params = self.space.param_grid[grid_var_counter]
            if proposal_params not in (self.all_evaluated_params + param_batch):
                # Add parameter proposal to the batch list
                param_batch.append(proposal_params)
                grid_var_counter += 1
            else:
                # Otherwise continue sampling proposals
                grid_var_counter += 1
        return param_batch

    def update_search(self) -> None:
        """Update search log data - Coordinate Search"""
        # Update/reset variable and grid counter based on eval_counter
        # And evals per search space (makes it easier to reload)
        self.grid_var_counter = (
            self.eval_counter - self.range_per_coord[self.var_counter]
        )
        if self.grid_var_counter >= len(self.space) - self.var_counter:
            self.var_counter += 1
            if self.var_counter < len(self.search_config["order"]):
                self.construct_active_space()
                self.grid_var_counter = 0

    def construct_active_space(self) -> None:
        """Construct the active search space."""
        # Update the parameter defaults with the best performers
        if self.eval_counter > 0:
            idx, config, eval, _ = self.get_best()
            for k, v in config.items():
                if k == self.search_config["order"][self.var_counter - 1]:
                    self.search_config["defaults"][k] = v
                    if self.verbose:
                        Console().log(f"Fixed `{k}` hyperparameter to {v}.")

        # Increase active variable counter and reset grid counter
        self.active_var = self.search_config["order"][self.var_counter]
        if self.verbose:
            Console().log(
                f"New active variable/coordinate `{self.active_var}`."
            )

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
