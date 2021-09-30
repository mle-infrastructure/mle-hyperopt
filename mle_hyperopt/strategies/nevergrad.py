from typing import Union
from .base import HyperOpt
from .hyperspace import construct_hyperparam_range


class NevergradHyperoptimisation(HyperOpt):
    def __init__(
        self,
        hyper_log: HyperoptLogger,
        resource_to_run: str,
        job_arguments: dict,
        config_fname: str,
        job_fname: str,
        experiment_dir: str,
        search_params: dict,
        search_type: str = "nevergrad",
        search_schedule: str = "sync",
        nevergrad_config: dict = {
            "base_estimator": "GP",
            "acq_function": "gp_hedge",
            "n_initial_points": 5,
        },
        message_id: Union[str, None] = None,
    ):
        try:
            import nevergrad as ng
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to"
                "install `nevergrad` to use "
                "the `mle_toolbox.hyperopt.nevergrad` module."
            )

        # Check that SMBO uses synchronous scheduling
        assert search_schedule == "sync", "Batch nevergrad schedules jobs synchronously"
        BaseHyperOptimisation.__init__(
            self,
            hyper_log,
            resource_to_run,
            job_arguments,
            config_fname,
            job_fname,
            experiment_dir,
            search_params,
            search_type,
            search_schedule,
            message_id,
        )
        self.param_range = construct_hyperparam_range(
            self.search_params, self.search_type
        )

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

    def get_hyperparam_proposal(self, num_iter_per_batch: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        # Generate list of dictionaries with different hyperparams to evaluate
        self.last_batch_params = [
            self.hyper_optimizer.ask() for i in range(num_iter_per_batch)
        ]
        param_batch = [params.value[1] for params in self.last_batch_params]
        return param_batch

    def clean_up_after_batch_iteration(self, batch_proposals, perf_measures):
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
