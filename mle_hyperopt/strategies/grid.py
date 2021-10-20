from typing import Union, List
from ..search import HyperOpt
from ..spaces import GridSpace
from ..utils import visualize_2D_grid


class GridSearch(HyperOpt):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        search_config: Union[dict, None] = None,
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
        # Generate all possible combinations of param configs in list & loop
        # over the list when doing the grid search
        self.space = GridSpace(real, integer, categorical)
        self.num_param_configs = len(self.space)
        self.grid_counter = self.eval_counter
        self.search_name = "Gird Search"

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Grid Search"""
        # Set grid counter to eval_counter in order ensure while
        # That results for grid configuration are collected before continuation
        self.grid_counter = self.eval_counter
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
        # Make sure that the grid_counter equals the eval_counter
        # This is only relevant if we load in new log data mid-search
        self.grid_counter = self.eval_counter

    def plot_grid(
        self,
        fixed_params: Union[None, dict] = None,
        params_to_plot: list = [],
        target_to_plot: str = "objective",
        plot_title: str = "Temp Title",
        plot_subtitle: Union[None, str] = None,
        xy_labels: Union[None, List[str]] = ["x-label", "y-label"],
        variable_name: Union[None, str] = "Var Label",
        every_nth_tick: int = 1,
    ):
        """Plot 2D heatmap of evaluations."""
        df = self.to_df()
        fig, ax = visualize_2D_grid(
            df,
            fixed_params,
            params_to_plot,
            target_to_plot,
            plot_title,
            plot_subtitle,
            xy_labels,
            variable_name,
            every_nth_tick,
        )
        return fig, ax
