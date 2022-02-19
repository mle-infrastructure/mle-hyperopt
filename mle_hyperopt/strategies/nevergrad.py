from ..strategy import Strategy
from ..spaces import NevergradSpace
from typing import Optional, List, Tuple
import nevergrad as ng


class NevergradSearch(Strategy):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
        search_config: dict = {
            "optimizer": "NGOpt",
            "budget_size": 100,
            "num_workers": 10,
        },
        maximize_objective: bool = False,
        fixed_params: Optional[dict] = None,
        reload_path: Optional[str] = None,
        reload_list: Optional[list] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        """Nevergrad Search Strategy.
        Wraps around https://facebookresearch.github.io/nevergrad/
        Reference: https://engineering.fb.com/2018/12/20/ai-research/nevergrad/

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
            search_config (dict, optional): Nevergrad search hyperparameters.
                Defaults to {"optimizer": "NGOpt",
                             "budget_size": 100,
                             "num_workers": 10}.
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
        self.search_name = "Nevergrad"
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
        self.space = NevergradSpace(real, integer, categorical)
        self.init_optimizer()

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    @property
    def optimizers(self) -> list:
        """Returns list of available nevergrad optimizers."""
        return sorted(ng.optimizers.registry.keys())

    def init_optimizer(self) -> None:
        """Initialize the surrogate model/hyperparam config proposer."""
        assert self.search_config["optimizer"] in list(
            dict(ng.optimizers.registry).keys()
        )
        self.hyper_optimizer = ng.optimizers.registry[
            self.search_config["optimizer"]
        ](
            parametrization=self.space.dimensions,
            budget=self.search_config["budget_size"],
            num_workers=self.search_config["num_workers"],
        )

    def ask_search(self, batch_size: int) -> List[dict]:
        """Get proposals to eval next (in batches) - Nevergrad Search.

        Args:
            batch_size (int): Number of desired configurations

        Returns:
            List[dict]: List of configuration dictionaries
        """
        # Generate list of dictionaries with different hyperparams to evaluate
        last_batch_params = [
            self.hyper_optimizer.ask() for i in range(batch_size)
        ]
        param_batch = [params.value[1] for params in last_batch_params]
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
        for i, prop in enumerate(batch_proposals):
            # Need to update hyperoptimizer with ng Instrumentation candidate
            prop_conf = dict(prop)
            if self.fixed_params is not None:
                for k in self.fixed_params.keys():
                    if k in prop_conf.keys():
                        del prop_conf[k]
            # Only update space if candidate is in bounds
            if self.space.contains(prop_conf):
                x = self.hyper_optimizer.parametrization.spawn_child(
                    new_value=((), prop_conf)
                )
                # Get performance for each objective - Negate values for max
                if not self.maximize_objective:
                    self.hyper_optimizer.tell(x, perf_measures[i])
                else:
                    self.hyper_optimizer.tell(
                        x, [-1 * p for p in perf_measures[i]]
                    )

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

    def refine_space(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
    ) -> None:
        """Update the Nevergrad search space based on refined dictionaries.

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
        # Reinitialize the optimizer and provide data from previous updates
        self.init_optimizer()
        for iter in self.log:
            self.tell_search([iter["params"]], [iter["objective"]])

    def get_pareto_front(self) -> Tuple[None, list, list, None]:
        """Get the pareto-front of the optimizer."""
        pareto_configs, pareto_evals = [], []
        for param in sorted(
            self.hyper_optimizer.pareto_front(), key=lambda p: p.losses[0]
        ):
            pareto_configs.append(param.value[1])
            if self.maximize_objective:
                eff_loss = [-1 * l for l in param.losses]
            else:
                eff_loss = param.losses
            pareto_evals.append(eff_loss)
        return None, pareto_configs, pareto_evals, None
