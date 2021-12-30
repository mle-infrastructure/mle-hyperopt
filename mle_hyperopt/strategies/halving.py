import math
import numpy as np
from typing import Union, List
from ..strategy import Strategy
from ..spaces import RandomSpace
from ..utils import print_halving_hello, print_halving_update


class HalvingSearch(Strategy):
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
        self.search_name = "Halving"
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
        for k in ["min_budget", "num_arms", "halving_coeff"]:
            assert k in self.search_config.keys()

        # Pre-compute number of configs per batch across SH loop
        def logeta(x):
            return math.log(x) / math.log(self.search_config["halving_coeff"])

        self.num_sh_batches = math.floor(logeta(self.search_config["num_arms"]) + 1)
        self.evals_per_batch = [self.search_config["num_arms"]]
        for i in range(self.num_sh_batches - 1):
            self.evals_per_batch.append(
                math.floor(
                    self.evals_per_batch[-1] / self.search_config["halving_coeff"]
                )
            )

        # Pre-compute number of step iterations per batch in SH loop
        self.iters_per_batch = []
        for i in range(self.num_sh_batches):
            iter_batch = (
                self.search_config["min_budget"]
                * self.search_config["halving_coeff"] ** i
            )
            # Cap off maximum update budget to spend on a single iteration
            if "max_budget" in self.search_config.keys():
                iter_batch = min(iter_batch, self.search_config["max_budget"])
            self.iters_per_batch.append(iter_batch)
        self.total_num_iters = np.sum(
            np.array(self.evals_per_batch) * np.array(self.iters_per_batch)
        )
        self.sh_counter = 0
        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()
            self.print_hello_strategy()

    def ask_search(self, batch_size: Union[int, None] = None):
        """Get proposals to eval next (in batches) - Random Sampling."""
        param_batch = []
        num_iters = self.iters_per_batch[self.sh_counter]
        if self.sh_counter > 0:
            num_prev_iters = self.iters_per_batch[self.sh_counter - 1]
        else:
            num_prev_iters = 0

        # Sample a new configuration for each eval in the batch
        if self.eval_counter == 0:
            while len(param_batch) < self.evals_per_batch[0]:
                proposal_params = self.space.sample()
                if proposal_params not in param_batch:
                    # Add parameter proposal to the batch list
                    proposal = {
                        "params": proposal_params,
                        "extra": {
                            "sh_num_total_iters": num_iters,
                            "sh_num_add_iters": num_iters - num_prev_iters,
                            "sh_counter": self.sh_counter,
                        },
                    }
                    param_batch.append(proposal)
                else:
                    # Otherwise continue sampling proposals
                    continue
        elif self.completed:
            raise ValueError("You already completed all Successive Halving iterations.")
        else:
            num_iters = self.iters_per_batch[self.sh_counter]
            num_prev_iters = self.iters_per_batch[self.sh_counter - 1]
            for i, c in enumerate(self.haved_configs):
                extra_dict = {
                    "sh_num_total_iters": num_iters,
                    "sh_num_add_iters": num_iters - num_prev_iters,
                    "sh_counter": self.sh_counter,
                }
                if self.haved_ckpt is not None:
                    extra_dict["sh_ckpt"] = self.haved_ckpt[i]
                proposal = {
                    "params": self.haved_configs[i],
                    "extra": extra_dict,
                }
                param_batch.append(proposal)
        return param_batch

    def tell_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[List[str], None] = None,
    ):
        """Perform post-iteration clean-up - no surrogate model."""
        self.sh_counter += 1
        if not self.completed:
            num_configs = self.evals_per_batch[self.sh_counter]

            # Sort last performances & set haved configs & corresponding ckpts
            sorted_idx = np.argsort(perf_measures)
            if not self.maximize_objective:
                best_idx = sorted_idx[:num_configs]
            else:
                best_idx = sorted_idx[::-1][:num_configs]
            self.haved_ids = best_idx
            self.haved_configs = [batch_proposals[idx] for idx in best_idx]
            if ckpt_paths is not None:
                self.haved_ckpt = [ckpt_paths[idx] for idx in best_idx]
            else:
                self.haved_ckpt = None

    def update_search(self):
        if self.verbose:
            self.print_update_strategy()

    def log_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[None, List[str], str] = None,
    ):
        """Log info specific to search strategy."""
        strat_data = []
        for i in range(len(batch_proposals)):
            c_data = {}
            if i in self.haved_ids:
                c_data["sh_continued"] = True
            else:
                c_data["sh_continued"] = False
            strat_data.append(c_data)
        return strat_data

    @property
    def completed(self):
        """Return boolean if all SH rounds were completed."""
        return self.sh_counter >= self.num_sh_batches

    def print_hello_strategy(self):
        """Hello message specific to successive halving search."""
        print_halving_hello(
            self.num_sh_batches,
            self.evals_per_batch,
            self.iters_per_batch,
            self.search_config["halving_coeff"],
            self.total_num_iters,
        )

    def print_update_strategy(self):
        """Update message specific to successive halving search."""
        print_halving_update(
            self.sh_counter,
            self.num_sh_batches,
            self.evals_per_batch,
            self.iters_per_batch,
            self.total_num_iters,
        )
