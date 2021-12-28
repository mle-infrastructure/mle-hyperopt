from ..strategy import Strategy
from ..spaces import NevergradSpace
from typing import Union, List
import nevergrad as ng


class NevergradSearch(Strategy):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        search_config: dict = {
            "optimizer": "NGOpt",
            "budget_size": 100,
            "num_workers": 10,
        },
        maximize_objective: bool = False,
        fixed_params: Union[dict, None] = None,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
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
    def optimizers(self):
        """Returns list of available nevergrad optimizers."""
        return sorted(ng.optimizers.registry.keys())

    def init_optimizer(self):
        """Initialize the surrogate model/hyperparam config proposer."""
        assert self.search_config["optimizer"] in list(
            dict(ng.optimizers.registry).keys()
        )
        self.hyper_optimizer = ng.optimizers.registry[self.search_config["optimizer"]](
            parametrization=self.space.dimensions,
            budget=self.search_config["budget_size"],
            num_workers=self.search_config["num_workers"],
        )

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        # Generate list of dictionaries with different hyperparams to evaluate
        last_batch_params = [self.hyper_optimizer.ask() for i in range(batch_size)]
        param_batch = [params.value[1] for params in last_batch_params]
        return param_batch

    def tell_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[List[str], None] = None,
    ):
        """Perform post-iteration clean-up by updating surrogate model."""
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
                    self.hyper_optimizer.tell(x, [-1 * p for p in perf_measures[i]])

    def update_search(self):
        """Refine search space boundaries after set of search iterations."""
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

    def setup_search(self):
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

    def refine_space(self, real, integer, categorical):
        """Update the nevergrad search space."""
        self.space.update(real, integer, categorical)
        # Reinitialize the optimizer and provide data from previous updates
        self.init_optimizer()
        for iter in self.log:
            self.tell_search([iter["params"]], [iter["objective"]])

    def get_pareto_front(self):
        """Get the pareto-front of the optimizer."""
        pareto_configs, pareto_evals, pareto_ckpt = [], [], None
        for param in sorted(
            self.hyper_optimizer.pareto_front(), key=lambda p: p.losses[0]
        ):
            pareto_configs.append(param.value[1])
            pareto_evals.append(param.losses)
        return pareto_configs, pareto_evals, pareto_ckpt
