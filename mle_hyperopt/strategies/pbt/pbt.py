from typing import Optional, List
from ...strategy import Strategy
from ...spaces import RandomSpace
from ...utils import print_pbt_hello, print_pbt_update
from .exploit import Exploit
from .explore import Explore


class PBTSearch(Strategy):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
        search_config: dict = {
            "exploit": {"strategy": "truncation", "selection_percent": 0.2},
            "explore": {
                "strategy": "perturbation",
                "perturb_coeffs": [0.8, 1.2],
            },
            "steps_until_ready": 4,
            "num_workers": 5,
        },
        maximize_objective: bool = False,
        fixed_params: Optional[dict] = None,
        reload_path: Optional[str] = None,
        reload_list: Optional[list] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        """Population-Based Training Search Strategy.
        Reference: https://arxiv.org/abs/1711.09846

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
            search_config (dict, optional): PBT hyperparameters.
                Defaults to {"exploit": {"strategy": "truncation",
                                         "selection_percent": 0.2},
                             "explore": {"strategy": "perturbation",
                                         "perturb_coeffs": [0.8, 1.2]},
                             "steps_until_ready": 4,
                             "num_workers": 5}.
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
        self.search_name = "PBT"
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
        self.num_workers = self.search_config["num_workers"]
        self.steps_until_ready = self.search_config["steps_until_ready"]
        self.explore = Explore(self.search_config["explore"], self.space)
        self.exploit = Exploit(
            self.search_config["exploit"], self.maximize_objective
        )
        self.pbt_step_counter = 0

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()
            self.print_hello_strategy()

    def ask_search(self, batch_size: Optional[int] = None) -> List[dict]:
        """Get proposals to eval next (in batches) - SMBO Search.

        Args:
            batch_size (Optional[int]): Number of desired configurations
            - Not applicable here since number of configurations is prescribed

        Returns:
            List[dict]: List of configuration dictionaries
        """
        param_batch = []
        if self.eval_counter == 0:
            # Sample a new configuration for each eval in the batch
            while len(param_batch) < self.num_workers:
                proposal_params = self.space.sample()
                if proposal_params not in param_batch:
                    # Add parameter proposal to the batch list
                    proposal = {
                        "params": proposal_params,
                        "extra": {
                            "pbt_num_total_iters": (self.pbt_step_counter + 1)
                            * self.search_config["steps_until_ready"],
                            "pbt_num_add_iters": self.search_config[
                                "steps_until_ready"
                            ],
                            "pbt_worker_id": len(param_batch),
                            "pbt_step_counter": self.pbt_step_counter,
                        },
                    }
                    param_batch.append(proposal)
                else:
                    # Otherwise continue sampling proposals
                    continue
        else:
            for worker_id in range(self.search_config["num_workers"]):
                if self.copy_info[worker_id]["explore"]:
                    # Explore parameters if previously not exploited
                    param_dict = self.explore(self.hyperparams[worker_id])
                else:
                    # Continue previously used hyperparams
                    param_dict = self.hyperparams[worker_id]

                # Add pointer to checkpoint to reload
                extra_dict = {
                    "pbt_num_total_iters": (self.pbt_step_counter + 1)
                    * self.search_config["steps_until_ready"],
                    "pbt_num_add_iters": self.search_config[
                        "steps_until_ready"
                    ],
                    "pbt_worker_id": worker_id,
                    "pbt_step_counter": self.pbt_step_counter,
                    "pbt_explore": self.copy_info[worker_id]["explore"],
                    "pbt_copy_id": self.copy_info[worker_id]["copy_id"],
                    "pbt_old_params": self.copy_info[worker_id]["old_params"],
                    "pbt_copy_params": self.copy_info[worker_id]["copy_params"],
                    "pbt_old_performance": self.copy_info[worker_id][
                        "old_performance"
                    ],
                    "pbt_copy_performance": self.copy_info[worker_id][
                        "copy_performance"
                    ],
                    "pbt_ckpt": self.ckpt[worker_id],
                }
                proposal = {
                    "params": param_dict,
                    "extra": extra_dict,
                }
                param_batch.append(proposal)
        return param_batch

    def tell_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: List[str],
    ) -> None:
        """Update search log data - Sync PBT

        Args:
            batch_proposals (list): List of evaluated configurations
            perf_measures (list): List of corresponding performances
            ckpt_paths (List[str]):
                List of corresponding model ckpts to store.
        """
        self.copy_info, self.hyperparams, self.ckpt = self.exploit(
            batch_proposals, perf_measures, ckpt_paths
        )

    def update_search(self) -> None:
        """Update PBT search counter."""
        self.pbt_step_counter += 1
        if self.verbose:
            self.print_update_strategy()

    def print_hello_strategy(self) -> None:
        """Hello message specific to PBT search."""
        print_pbt_hello(
            self.num_workers,
            self.steps_until_ready,
            self.search_config["explore"]["strategy"],
            self.search_config["exploit"]["strategy"],
        )

    def print_update_strategy(self) -> None:
        """Update message specific to PBT search."""
        print_pbt_update(
            self.pbt_step_counter,
            self.pbt_step_counter * self.steps_until_ready,
            self.copy_info,
        )
