from typing import Union
from ..search import HyperOpt
from ..spaces import GridSpace


class GridSearch(HyperOpt):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        fixed_params: Union[dict, None] = None,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
        seed_id: int = 42,
    ):
        HyperOpt.__init__(
            self,
            real,
            integer,
            categorical,
            fixed_params,
            reload_path,
            reload_list,
            seed_id,
        )
        # Generate all possible combinations of param configs in list & loop
        # over the list when doing the grid search
        self.space = GridSpace(real, integer, categorical)
        self.num_param_configs = len(self.space)
        self.grid_counter = self.eval_counter

        # TODO: Add start-up message printing the search space

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Grid Search"""
        param_batch = []
        # Sample a new configuration for each eval in the batch
        while (
            len(param_batch) < batch_size and self.grid_counter < self.num_param_configs
        ):
            # Get parameter batch from the grid
            proposal_params = self.space.sample(self.grid_counter)
            if proposal_params not in (self.all_evaluated_params + param_batch):
                # Add parameter proposal to the batch list
                param_batch.append(proposal_params)
                self.grid_counter += 1
            else:
                # Otherwise continue sampling proposals
                continue
        return param_batch

    def tell_search(self, batch_proposals: list, perf_measures: list):
        """Update search log data - Grid Search"""
        # Make sure that the grid_counter equals the eval eval_counter
        # This is only relevant if we load in new log data mid-search
        self.grid_counter = self.eval_counter

    def plot_grid(self):
        """Plot 2D heatmap of evaluations."""
        # TODO Add example with simple square optimization
        raise NotImplementedError
