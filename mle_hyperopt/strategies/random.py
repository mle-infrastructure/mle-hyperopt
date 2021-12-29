from typing import Union
from ..strategy import Strategy
from ..spaces import RandomSpace


class RandomSearch(Strategy):
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
        self.search_name = "Random"
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
        self.space = RandomSpace(real, integer, categorical)

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        param_batch = []
        # Sample a new configuration for each eval in the batch
        while len(param_batch) < batch_size:
            proposal_params = self.space.sample()
            if proposal_params not in (self.all_evaluated_params + param_batch):
                # Add parameter proposal to the batch list
                param_batch.append(proposal_params)
            else:
                # Otherwise continue sampling proposals
                continue
        return param_batch

    def setup_search(self):
        """Initialize search settings at startup."""
        # Set up search space refinement - random, SMBO, nevergrad
        if self.search_config is not None:
            if "refine_top_k" in self.search_config.keys():
                self.refine_counter = 0
                assert self.search_config["refine_top_k"] > 1
                self.refine_after = self.search_config["refine_after"]
                # Make sure that refine iteration is list
                if type(self.refine_after) == int:
                    self.refine_after = [self.refine_after]
                self.refine_top_k = self.search_config["refine_top_k"]
                self.last_refined = 0
            else:
                self.refine_after = None
        else:
            self.refine_after = None

    def update_search(self):
        """Refine search space boundaries after set of search iterations."""
        if self.refine_after is not None:
            # Check whether there are still refinements open
            # And whether we have already passed last refinement point
            if len(self.refine_after) > self.refine_counter:
                exact = self.eval_counter == self.refine_after[self.refine_counter]
                skip = (
                    self.eval_counter > self.refine_after[self.refine_counter]
                    and self.last_refined != self.refine_after[self.refine_counter]
                )
                if exact or skip:
                    self.refine(self.refine_top_k)
                    self.last_refined = self.refine_after[self.refine_counter]
                    self.refine_counter += 1

    def refine_space(self, real, integer, categorical):
        """Update the random search space."""
        self.space.update(real, integer, categorical)
