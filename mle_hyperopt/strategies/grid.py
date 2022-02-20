from typing import Optional, List
from ..strategy import Strategy
from ..spaces import GridSpace
from ..utils import visualize_2D_grid, print_grid_hello


class GridSearch(Strategy):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
        search_config: Optional[dict] = None,
        maximize_objective: bool = False,
        fixed_params: Optional[dict] = None,
        reload_path: Optional[str] = None,
        reload_list: Optional[list] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        """Grid Search Strategy.

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
            search_config (dict, optional): Grid search hyperparameters.
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
        self.search_name = "Grid"
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
        # Generate all possible combinations of param configs in list & loop
        # over the list when doing the grid search
        self.space = GridSpace(real, integer, categorical)
        self.num_param_configs = len(self.space)
        self.grid_counter = self.eval_counter
        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()
            self.print_hello_strategy()

    def ask_search(self, batch_size: int) -> List[dict]:
        """Get proposals to eval next (in batches) - Grid Search.

        Args:
            batch_size (int): Number of desired configurations

        Returns:
            List[dict]: List of configuration dictionaries
        """
        # Set grid counter to eval_counter in order ensure while
        # That results for grid configuration are collected before continuation
        grid_counter = self.eval_counter
        param_batch = []
        # Sample a new configuration for each eval in the batch
        while (
            len(param_batch) < batch_size
            and grid_counter < self.num_param_configs
        ):
            # Get parameter batch from the grid
            proposal_params = self.space.sample(grid_counter)
            if proposal_params not in (self.all_evaluated_params + param_batch):
                # Add parameter proposal to the batch list
                param_batch.append(proposal_params)
                grid_counter += 1
            else:
                # Otherwise continue sampling proposals
                continue
        return param_batch

    def update_search(self) -> None:
        """Update search log data - Grid Search"""
        # Make sure that the grid_counter equals the eval_counter
        # This is only relevant if we load in new log data mid-search
        self.grid_counter = self.eval_counter

    def plot_grid(
        self,
        fixed_params: Optional[dict] = None,
        params_to_plot: List[str] = [],
        target_to_plot: str = "objective",
        plot_title: str = "Temp Title",
        plot_subtitle: Optional[str] = None,
        xy_labels: Optional[List[str]] = ["x-label", "y-label"],
        variable_name: Optional[str] = "Var Label",
        every_nth_tick: int = 1,
        fname: Optional[str] = None,
    ):
        """Plot 2D heatmap of evaluations.

        Args:
            fixed_params (Optional[dict], optional):
                Dict of parameter keys and values to fix for plot. Defaults to None.
            params_to_plot (List[str], optional):
                Parameter names to plot. Defaults to [].
            target_to_plot (str, optional):
                Name of variable to plot. Defaults to "objective".
            plot_title (str, optional):
                Title of figure plot. Defaults to "Temp Title".
            plot_subtitle (Optional[str], optional):
                Subtitle of figure plot. Defaults to None.
            xy_labels (Optional[List[str]], optional):
                Label names. Defaults to ["x-label", "y-label"].
            variable_name (Optional[str], optional):
                Name of variable in heatmap bar. Defaults to "Var Label".
            every_nth_tick (int, optional):
                Controls spacing between ticks. Defaults to 1.
            fname (Optional[str], optional):
                File name to save plot to. Defaults to None.

        Returns:
            [type]: Figure and axis matplotlib objects
        """
        fig, ax = visualize_2D_grid(
            self.df,
            fixed_params,
            params_to_plot,
            target_to_plot,
            plot_title,
            plot_subtitle,
            xy_labels,
            variable_name,
            every_nth_tick,
        )

        # Save the figure if a filename was provided
        if fname is not None:
            fig.savefig(fname, dpi=300)
        else:
            return fig, ax

    def print_hello_strategy(self) -> None:
        """Hello message specific to grid search."""
        print_grid_hello(self.num_param_configs, self.space.num_dims)
