from typing import Union
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
        self.search_name = "Population-Based Training"
        self.space = RandomSpace(real, integer, categorical)
        self.explore = Explore(search_config["explore_config"], self.space)
        self.exploit = Exploit(search_config["exploit_config"])
        self.pbt_counter = 0

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Sync PBT"""
        param_batch = []
        if self.pbt_counter == 0:
            # Sample a new configuration for each eval in the batch
            while len(param_batch) < batch_size:
                proposal_params = self.space.sample()
                if proposal_params not in (self.all_evaluated_params + param_batch):
                    # Add parameter proposal to the batch list
                    param_batch.append(proposal_params)
                else:
                    # Otherwise continue sampling proposals
                    continue
        else:
            for worker_id in range(self.batch_size):
                if self.copy_info[worker_id]:
                    # Explore parameters if previously not exploited
                    param_dict = self.explore(self.hyperparams[worker_id])
                    param_dict["explore"] = True
                else:
                    # Continue previously used hyperparams
                    param_dict = self.hyperparams[worker_id]
                    param_dict["exploit"] = False
                # Add pointer to checkpoint to reload
                param_dict["ckpt"] = self.ckpt[worker_id]
                param_batch.append(param_dict)
        return param_batch

    def tell_search(self, batch_proposals: list, perf_measures: list):
        """Update search log data - Sync PBT"""
        self.pbt_counter += 1
        self.copy_info, self.hyperparams, self.ckpt = self.exploit(
            batch_proposals, self.log
        )
