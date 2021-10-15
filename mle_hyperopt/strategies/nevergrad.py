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
            fixed_params,
            reload_path,
            reload_list,
            seed_id,
            verbose,
        )
        self.space = NevergradSpace(real, integer, categorical)
        # Initialize the surrogate model/hyperparam config proposer
        if self.search_config["optimizer"] == "CMA":
            self.hyper_optimizer = ng.optimizers.CMA(
                parametrization=self.space.dimensions,
                budget=self.search_config["budget_size"],
                num_workers=self.search_config["num_workers"],
            )
        elif self.search_config["optimizer"] == "NGOpt":
            self.hyper_optimizer = ng.optimizers.NGOpt(
                parametrization=self.space.dimensions,
                budget=self.search_config["budget_size"],
                num_workers=self.search_config["num_workers"],
            )
        else:
            raise ValueError("Please provide valid nevergrad optimizer type.")

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello("Nevergrad Wrapper Search")

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        # Generate list of dictionaries with different hyperparams to evaluate
        self.last_batch_params = [self.hyper_optimizer.ask() for i in range(batch_size)]
        param_batch = [params.value[1] for params in self.last_batch_params]
        return param_batch

    def tell_search(self, batch_proposals, perf_measures):
        """Perform post-iteration clean-up by updating surrogate model."""
        for i, prop in enumerate(batch_proposals):
            # Need to update hyperoptimizer with ng Instrumentation candidate
            prop_conf = dict(prop)
            if self.fixed_params is not None:
                for k in self.fixed_params.keys():
                    del prop_conf[k]
            for last_prop in self.last_batch_params:
                if last_prop.value[1] == prop_conf:
                    x = last_prop
                    break
            # Get performance for each objective - tell individually to optim
            self.hyper_optimizer.tell(x, perf_measures[i])

    def get_pareto_front(self):
        """Get the pareto-front of the optimizer."""
        pareto_configs, pareto_evals = [], []
        for param in sorted(
            self.hyper_optimizer.pareto_front(), key=lambda p: p.losses[0]
        ):
            pareto_configs.append(param.value[1])
            pareto_evals.append(param.losses)
        return pareto_configs, pareto_evals
