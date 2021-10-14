from typing import Union
import numpy as np
from ..search import HyperOpt
from ..spaces import RandomSpace


class RandomSearch(HyperOpt):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        search_config: Union[dict, None] = None,
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
            fixed_params,
            reload_path,
            reload_list,
            seed_id,
            verbose,
        )
        self.space = RandomSpace(real, integer, categorical)
        self.search_config = search_config

        if self.search_config is not None:
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

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello("Random Search")

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

    def tell_search(self, batch_proposals: list, perf_measures: list):
        """Perform post-iteration clean-up by updating surrogate model."""
        # Refine search space boundaries after set of search iterations
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

    def refine(self, top_k: int):
        """Refine the space boundaries based on top-k performers."""
        top_idx, top_k_configs, top_k_evals = self.get_best(top_k)
        # Loop over real, integer and categorical variable keys
        # Get boundaries and re-define the search space
        if self.categorical is not None:
            categorical_refined = {}
            for var in self.categorical.keys():
                top_k_var = [config[var] for config in top_k_configs]
                categorical_refined[var] = list(set(top_k_var))
        else:
            categorical_refined = None

        if self.real is not None:
            real_refined = {}
            for var in self.real.keys():
                top_k_var = [config[var] for config in top_k_configs]
                real_refined[var] = {
                    "begin": np.min(top_k_var),
                    "end": np.max(top_k_var),
                    "prior": self.real[var]["prior"],
                }
        else:
            real_refined = None

        if self.integer is not None:
            integer_refined = {}
            for var in self.integer.keys():
                top_k_var = [config[var] for config in top_k_configs]
                integer_refined[var] = {
                    "begin": int(np.min(top_k_var)),
                    "end": int(np.max(top_k_var)),
                    "prior": self.integer[var]["prior"],
                }
        else:
            integer_refined = None

        self.space.update(real_refined, integer_refined, categorical_refined)
        if self.verbose:
            self.print_hello(
                f"After {self.eval_counter} Evals - Refined Random Search"
            )
