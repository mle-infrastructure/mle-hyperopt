from ..base import HyperOpt
from ..hyperspace import nevergrad_space


class NevergradSearch(HyperOpt):
    def __init__(
        self,
        search_params: dict,
        nevergrad_config: dict = {
            "base_estimator": "GP",
            "acq_function": "gp_hedge",
            "n_initial_points": 5,
        },
    ):
        try:
            import nevergrad as ng
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to"
                "install `nevergrad` to use "
                "the `mle_toolbox.hyperopt.nevergrad` module."
            )

        HyperOpt.__init__(self, search_params)
        self.param_range = nevergrad_space(self.search_params, "nevergrad")

        # Initialize the surrogate model/hyperparam config proposer
        self.nevergrad_config = nevergrad_config
        if self.nevergrad_config["optimizer"] == "CMA":
            self.hyper_optimizer = ng.optimizers.CMA(
                parametrization=self.param_range,
                budget=self.nevergrad_config["budget_size"],
                num_workers=self.nevergrad_config["num_workers"],
            )
        elif self.nevergrad_config["optimizer"] == "NGOpt":
            self.hyper_optimizer = ng.optimizers.NGOpt(
                parametrization=self.param_range,
                budget=self.nevergrad_config["budget_size"],
                num_workers=self.nevergrad_config["num_workers"],
            )
        else:
            raise ValueError("Please provide valid nevergrad optimizer type.")

    def ask(self, batch_size: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        # Generate list of dictionaries with different hyperparams to evaluate
        self.last_batch_params = [self.hyper_optimizer.ask() for i in range(batch_size)]
        param_batch = [params.value[1] for params in self.last_batch_params]
        return param_batch

    def tell(self, batch_proposals, perf_measures):
        """Perform post-iteration clean-up by updating surrogate model."""
        # First key of all metrics is used to update the surrogate!
        to_model = list(perf_measures.keys())
        eval_ids = list(perf_measures[to_model[0]].keys())

        for i, prop in enumerate(batch_proposals):
            # Need to update hyperoptimizer with ng Instrumentation candidate
            for last_prop in self.last_batch_params:
                if last_prop.value[1] == prop:
                    x = last_prop
                    break
            # Get performance for each objective - tell individually to optim
            perf = []
            for k in to_model:
                perf.append(
                    -1 * perf_measures[k][eval_ids[i]]
                    if self.hyper_log.max_objective
                    else perf_measures[k][eval_ids[i]]
                )
            self.hyper_optimizer.tell(x, perf)
