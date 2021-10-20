from ..search import HyperOpt
from ..spaces import NevergradSpace
from typing import Union
import nevergrad as ng


class NevergradSearch(HyperOpt):
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
        HyperOpt.__init__(
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
        self.search_name = "Nevergrad Wrapper Search"

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello()

    def init_optimizer(self):
        """Initialize the surrogate model/hyperparam config proposer."""
        assert self.search_config["optimizer"] in list(dict(ng.optimizers.registry).keys())
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

    def tell_search(self, batch_proposals, perf_measures):
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

    def refine_space(self, real, integer, categorical):
        """Update the nevergrad search space."""
        self.space.update(real, integer, categorical)
        # Reinitialize the optimizer and provide data from previous updates
        self.init_optimizer()
        for iter in self.log:
            self.tell_search([iter["params"]], [iter["objective"]])

    def get_pareto_front(self):
        """Get the pareto-front of the optimizer."""
        pareto_configs, pareto_evals = [], []
        for param in sorted(
            self.hyper_optimizer.pareto_front(), key=lambda p: p.losses[0]
        ):
            pareto_configs.append(param.value[1])
            pareto_evals.append(param.losses)
        return pareto_configs, pareto_evals
