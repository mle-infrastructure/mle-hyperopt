import math
import numpy as np
from typing import Optional, List, Union
from ..strategy import Strategy
from ..spaces import RandomSpace
from ..utils import print_halving_hello, print_halving_update


class HalvingSearch(Strategy):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
        search_config: dict = {
            "min_budget": 1,
            "num_arms": 20,
            "halving_coeff": 2,
        },
        maximize_objective: bool = False,
        fixed_params: Optional[dict] = None,
        reload_path: Optional[str] = None,
        reload_list: Optional[list] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        """Successive Halving Iterative Search Strategy.
        Reference: https://proceedings.mlr.press/v28/karnin13.html

        Args:
            real (Optional[dict], optional):
                Dictionary of real-valued search variables & their priors.
                E.g. {"lrate": {"begin": 0.1, "end": 0.5, "prior": "log-uniform"}}
                Defaults to None.
            integer (Optional[dict], optional):
                Dictionary of integer-valued search variables & their priors.
                E.g. {"batch_size": {"begin": 1, "end": 5, "bins": "uniform"}}
                Defaults to None.
            categorical (Optional[dict], optional):
                Dictionary of categorical-valued search variables.
                E.g. {"arch": ["mlp", "cnn"]}
                Defaults to None.
            search_config (dict, optional): Halving search hyperparameters.
                Defaults to {"min_budget": 1,
                             "num_arms": 20,
                             "halving_coeff": 2}.
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

        self.num_sh_batches = math.floor(
            logeta(self.search_config["num_arms"]) + 1
        )
        self.evals_per_batch = [self.search_config["num_arms"]]
        for i in range(self.num_sh_batches - 1):
            self.evals_per_batch.append(
                math.floor(
                    self.evals_per_batch[-1]
                    / self.search_config["halving_coeff"]
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

    def ask_search(self, batch_size: Optional[int] = None) -> List[dict]:
        """Get proposals to eval next (in batches) - Halving Search.

        Args:
            batch_size (Optional[int]): Number of desired configurations
            - Not applicable here since number of configurations is prescribed

        Returns:
            List[dict]: List of configuration dictionaries
        """
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
            raise ValueError(
                "You already completed all Successive Halving iterations."
            )
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
        ckpt_paths: Optional[List[str]] = None,
    ) -> None:
        """Perform post-iteration clean-up by updating surrogate model.

        Args:
            batch_proposals (list): List of evaluated configurations
            perf_measures (list): List of corresponding performances
            ckpt_paths (Optional[List[str]], optional):
                List of corresponding model ckpts to store. Defaults to None.
        """
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

    def update_search(self) -> None:
        """Print state of halving search."""
        if self.verbose:
            self.print_update_strategy()

    def log_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Optional[Union[List[str], str]] = None,
    ) -> None:
        """Log info specific to search strategy.

        Args:
            batch_proposals (list): List of evaluated configurations
            perf_measures (list): List of corresponding performances
            ckpt_paths (Optional[List[str]], optional):
                List of corresponding model ckpts to store. Defaults to None.

        Returns:
            [list]: Hyperband data to log.
        """
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
    def completed(self) -> bool:
        """Return boolean if all SH rounds were completed."""
        return self.sh_counter >= self.num_sh_batches

    def print_hello_strategy(self) -> None:
        """Hello message specific to successive halving search."""
        print_halving_hello(
            self.num_sh_batches,
            self.evals_per_batch,
            self.iters_per_batch,
            self.search_config["halving_coeff"],
            self.total_num_iters,
        )

    def print_update_strategy(self) -> None:
        """Update message specific to successive halving search."""
        print_halving_update(
            self.sh_counter,
            self.num_sh_batches,
            self.evals_per_batch,
            self.iters_per_batch,
            self.total_num_iters,
        )
