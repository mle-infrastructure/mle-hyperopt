from typing import Union, List
from ...strategy import Strategy
from ...spaces import RandomSpace
from .exploit import Exploit
from .explore import Explore


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
        self.search_name = "Population-Based Training"
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
        for k in ["explore", "exploit", "steps_until_ready", "num_workers"]:
            assert k in self.search_config.keys()
        self.explore = Explore(self.search_config["explore"], self.space)
        self.exploit = Exploit(self.search_config["exploit"])
        self.pbt_step_counter = 0

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Sync PBT"""
        param_batch = []
        if self.eval_counter == 0:
            # Sample a new configuration for each eval in the batch
            while len(param_batch) < self.search_config["num_workers"]:
                proposal_params = self.space.sample()
                if proposal_params not in param_batch:
                    # Add parameter proposal to the batch list
                    proposal = {
                        "params": proposal_params,
                        "extra": {
                            "pbt_num_total_iters": self.pbt_step_counter
                            * self.search_config["steps_until_ready"],
                            "pbt_num_add_iters": self.search_config[
                                "steps_until_ready"
                            ],
                            "pbt_worker_id": len(param_batch),
                            "pbt_counter": self.pbt_step_counter,
                        },
                    }
                    param_batch.append(proposal)
                else:
                    # Otherwise continue sampling proposals
                    continue
        else:
            for worker_id in range(self.search_config["num_workers"]):
                if self.copy_info[worker_id]:
                    # Explore parameters if previously not exploited
                    param_dict = self.explore(self.hyperparams[worker_id])
                    explore_step = True
                else:
                    # Continue previously used hyperparams
                    param_dict = self.hyperparams[worker_id]
                    explore_step = False

                # Add pointer to checkpoint to reload
                extra_dict = {
                    "pbt_num_total_iters": self.pbt_step_counter
                    * self.search_config["steps_until_ready"],
                    "pbt_num_add_iters": self.search_config["steps_until_ready"],
                    "pbt_worker_id": len(param_batch),
                    "pbt_counter": self.pbt_step_counter,
                    "pbt_explore": explore_step,
                }
                if self.pbt_ckpt is not None:
                    extra_dict["pbt_ckpt"] = self.ckpt[worker_id]
                proposal = {
                    "params": param_dict,
                    "extra": extra_dict,
                }
                param_batch.append(param_dict)
        return param_batch

    def tell_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[List[str], None] = None,
    ):
        """Update search log data - Sync PBT"""
        self.pbt_step_counter += 1
        self.copy_info, self.hyperparams, self.ckpt = self.exploit(
            batch_proposals, self.log
        )

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
            # TODO: Add copy info
            strat_data.append(c_data)
        return strat_data
