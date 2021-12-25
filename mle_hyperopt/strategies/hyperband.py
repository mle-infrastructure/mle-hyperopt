import math
from typing import Union, List
from ..strategy import Strategy
from ..spaces import RandomSpace
from .halving import SuccessiveHalvingSearch


class HyperbandSearch(Strategy):
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
        self.search_name = "Hyperband"
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
        for k in ["max_resource", "eta"]:
            assert k in self.search_config.keys()

        # Pre-compute number of configs & iters per config across HB batches
        def logeta(x):
            return math.log(x) / math.log(self.search_config["eta"])

        self.s_max = math.floor(logeta(self.search_config["max_resource"]))
        self.B = (self.s_max + 1) * self.search_config["max_resource"]
        self.sh_num_arms = [
            math.ceil(
                self.B
                / self.search_config["max_resource"]
                * ((self.search_config["eta"] ** s) / (s + 1))
            )
            for s in reversed(range(self.s_max + 1))
        ]
        self.sh_budgets = [
            int(self.search_config["max_resource"] * self.search_config["eta"] ** (-s))
            for s in reversed(range(self.s_max + 1))
        ]
        self.hb_counter = 0

        """
        # TODO: Add method `update_search` for refinement/coord-update/sh-switch
        # TODO: Make reloading work for new strategies
        # TODO: Add changelog/contributing to all mle-infrastructure packages
        # TODO: Separate hyperparams and stratparams [[{}, {}]]
        # In loop go over individual SH loops
        self.sub_strategy = SuccessiveHalvingSearch(
            real=self.real,
            integer=self.integer,
            categorical=self.categorical,
            search_config={
                "budget": self.sh_budgets[self.hb_counter],
                "num_arms": self.sh_num_arms[self.hb_counter],
                "halving_coeff": self.search_config["eta"],
            },
            seed_id=self.hb_counter + self.seed_id,
        )
        """

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

    def tell_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[List[str], None] = None,
    ):
        """Perform post-iteration clean-up - no surrogate model."""

    def update_search(self):
        """Check whether to switch to next successive halving strategy."""

    def log_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[None, List[str], str] = None,
    ):
        """Log info specific to search strategy."""
        strat_data = []
        num_iters = self.iters_per_batch[self.sh_counter - 1]
        num_prev_iters = self.iters_per_batch[self.sh_counter - 2]
        for i in range(len(batch_proposals)):
            c_data = {}
            if i in self.haved_ids:
                c_data["hb_continued"] = True
            else:
                c_data["hb_continued"] = False
            c_data["hb_counter"] = self.sh_counter - 1
            c_data["hb_total_iters"] = num_iters
            c_data["hb_add_iters"] = num_iters - num_prev_iters
            strat_data.append(c_data)
        return strat_data
