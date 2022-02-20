import math
import numpy as np
from typing import Optional, List, Tuple, Union
from ..strategy import Strategy
from .halving import HalvingSearch
from ..spaces import RandomSpace
from ..utils import print_hyperband_hello, print_hyperband_update


def get_batch_resources(sh_num_arms: list, eta: int) -> Tuple[int, List[int]]:
    """Compute the number of batches & iterations per batch

    Args:
        sh_num_arms (list): List of number of arms per iteration
        eta (int): Halving parameter

    Returns:
        Tuple[int, List[int]]: number of batches & iterations per batch
    """
    num_total_batches = 0
    all_evals_per_batch = []

    def logeta(x):
        return math.log(x) / math.log(eta)

    for i in range(len(sh_num_arms)):
        num_sh_batches = math.floor(logeta(sh_num_arms[i]) + 1)
        evals_per_batch = [sh_num_arms[i]]
        for i in range(num_sh_batches - 1):
            evals_per_batch.append(math.floor(evals_per_batch[-1] / eta))
        num_total_batches += num_sh_batches
        all_evals_per_batch.extend(evals_per_batch)
    return num_total_batches, all_evals_per_batch


class HyperbandSearch(Strategy):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
        search_config: dict = {"max_resource": 27, "eta": 3},
        maximize_objective: bool = False,
        fixed_params: Optional[dict] = None,
        reload_path: Optional[str] = None,
        reload_list: Optional[list] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        """Hyperband Iterative Search Strategy.
        Reference: https://homes.cs.washington.edu/~jamieson/hyperband.html

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
            search_config (dict, optional): Hyperband search hyperparameters.
                Defaults to {"max_resource": 27, "eta": 3}.
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
            int(
                math.ceil(
                    int(self.B / self.search_config["max_resource"] / (s + 1))
                    * self.search_config["eta"] ** s
                )
            )
            for s in reversed(range(self.s_max + 1))
        ]
        self.sh_budgets = [
            int(
                self.search_config["max_resource"]
                * self.search_config["eta"] ** (-s)
            )
            for s in reversed(range(self.s_max + 1))
        ]
        self.hb_counter, self.hb_batch_counter = 0, 0
        self.num_hb_loops = len(self.sh_budgets)

        # Calculate resources for all SH loops -> used in mle-toolbox
        self.num_hb_batches, self.evals_per_batch = get_batch_resources(
            self.sh_num_arms, self.search_config["eta"]
        )

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()
            self.print_hello_strategy()

        # Define first SH Loop to evaluate
        self.sub_strategy = HalvingSearch(
            real=self.real,
            integer=self.integer,
            categorical=self.categorical,
            search_config={
                "min_budget": self.sh_budgets[self.hb_counter],
                "num_arms": self.sh_num_arms[self.hb_counter],
                "halving_coeff": self.search_config["eta"],
            },
            seed_id=self.hb_counter + self.seed_id,
        )

    def ask_search(self, batch_size: Optional[int] = None) -> List[dict]:
        """Get proposals to eval next (in batches) - Hyperband Search.

        Args:
            batch_size (Optional[int]): Number of desired configurations
            - Not applicable here since number of configurations is prescribed

        Returns:
            List[dict]: List of configuration dictionaries
        """
        param_batch = self.sub_strategy.ask()
        if type(param_batch) != list:
            param_batch = [param_batch]

        # Add Hyperband iter counter to extra dictionary
        for c in param_batch:
            c["extra"]["hb_counter"] = self.hb_counter
        return param_batch

    def tell_search(
        self,
        batch_proposals: List[dict],
        perf_measures: List[Union[float, np.ndarray, int]],
        ckpt_paths: Optional[List[str]] = None,
    ) -> None:
        """Perform post-iteration clean-up by updating surrogate model.

        Args:
            batch_proposals (List[dict]): List of evaluated configurations.
            perf_measures (List[float, np.ndarray]):
                List of corresponding performances.
            ckpt_paths (Optional[List[str]], optional):
                List of corresponding model ckpts to store. Defaults to None.
        """
        self.sub_strategy.tell(batch_proposals, perf_measures, ckpt_paths)
        self.hb_batch_counter += 1

    def update_search(self) -> None:
        """Check whether to switch to next successive halving strategy."""
        if self.sub_strategy.completed:
            self.hb_counter += 1
            if self.hb_counter < self.num_hb_loops:
                self.sub_strategy = HalvingSearch(
                    real=self.real,
                    integer=self.integer,
                    categorical=self.categorical,
                    search_config={
                        "min_budget": self.sh_budgets[self.hb_counter],
                        "num_arms": self.sh_num_arms[self.hb_counter],
                        "halving_coeff": self.search_config["eta"],
                    },
                    seed_id=self.hb_counter + self.seed_id,
                )
        if self.verbose:
            self.print_update_strategy()

    def log_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Optional[Union[List[str], str]] = None,
    ) -> list:
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
            c_data["hb_counter"] = self.hb_counter
            if i in self.sub_strategy.haved_ids:
                c_data["sh_continued"] = True
            else:
                c_data["sh_continued"] = False
            strat_data.append(c_data)
        return strat_data

    def print_hello_strategy(self) -> None:
        """Hello message specific to hyperband search."""
        print_hyperband_hello(
            self.num_hb_loops,
            self.sh_num_arms,
            self.sh_budgets,
            self.num_hb_batches,
            self.evals_per_batch,
        )

    def print_update_strategy(self) -> None:
        """Update message specific to hyperband search."""
        print_hyperband_update(
            self.hb_counter,
            self.num_hb_loops,
            self.sh_num_arms,
            self.sh_budgets,
            self.num_hb_batches,
            self.hb_batch_counter,
            self.evals_per_batch,
        )
