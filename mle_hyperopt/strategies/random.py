from typing import Optional, List
from ..strategy import Strategy
from ..spaces import RandomSpace


class RandomSearch(Strategy):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
        search_config: Optional[dict] = None,
        maximize_objective: bool = False,
        fixed_params: Optional[dict] = None,
        reload_path: Optional[str] = None,
        reload_list: Optional[list] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        """Random Search Strategy.

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
            search_config (dict, optional): Random search hyperparameters.
                Can specify timepoints of search space refinements.
                E.g. {"refine_after": [5, 10], "refine_top_k": 2}
                Defaults to None.
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

    def ask_search(self, batch_size: int) -> List[dict]:
        """Get proposals to eval next (in batches) - Random Search.

        Args:
            batch_size (int): Number of desired configurations

        Returns:
            List[dict]: List of configuration dictionaries
        """
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

    def setup_search(self) -> None:
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

    def update_search(self) -> None:
        """Refine search space boundaries after set of search iterations."""
        if self.refine_after is not None:
            # Check whether there are still refinements open
            # And whether we have already passed last refinement point
            if len(self.refine_after) > self.refine_counter:
                exact = (
                    self.eval_counter == self.refine_after[self.refine_counter]
                )
                skip = (
                    self.eval_counter > self.refine_after[self.refine_counter]
                    and self.last_refined
                    != self.refine_after[self.refine_counter]
                )
                if exact or skip:
                    self.refine(self.refine_top_k)
                    self.last_refined = self.refine_after[self.refine_counter]
                    self.refine_counter += 1

    def refine_space(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
    ) -> None:
        """Update the random search space based on refined dictionaries.

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
        """
        self.space.update(real, integer, categorical)
