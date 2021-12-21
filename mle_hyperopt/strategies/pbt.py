from typing import Union
from ..strategy import Strategy


class PBTSearch(Strategy):
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
        self.search_name = "Population-Based Training"

        assert "noise_scale" in search_config.keys()
        assert "truncation_selection" in search_config.keys()

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Sync PBT"""
        if self.eval_counter == 0:
            pass
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
        """Update search log data - Sync PBT"""
        # Make sure that the grid_counter equals the eval_counter
        # This is only relevant if we load in new log data mid-search
