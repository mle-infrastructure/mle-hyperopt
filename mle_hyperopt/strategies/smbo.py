import numpy as np
from typing import Union
from .base import HyperOpt
from .hyperspace import construct_hyperparam_range


class SMBOHyperoptimisation(HyperOpt):
    def __init__(
        self,
        hyper_log: HyperoptLogger,
        resource_to_run: str,
        job_arguments: dict,
        config_fname: str,
        job_fname: str,
        experiment_dir: str,
        search_params: dict,
        search_type: str = "smbo",
        search_schedule: str = "sync",
        smbo_config: dict = {
            "base_estimator": "GP",
            "acq_function": "gp_hedge",
            "n_initial_points": 5,
        },
        message_id: Union[str, None] = None,
    ):
        try:
            from skopt import Optimizer
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to"
                "install `scikit-optimize` to use "
                "the `mle_toolbox.hyperopt.smbo` module."
            )

        # Check that SMBO uses synchronous scheduling
        assert search_schedule == "sync", "Batch SMBO schedules jobs synchronously"
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
        self.smbo_config = smbo_config
        self.hyper_optimizer = Optimizer(
            dimensions=list(self.param_range.values()),
            random_state=42,
            base_estimator=smbo_config["base_estimator"],
            acq_func=smbo_config["acq_function"],
            n_initial_points=smbo_config["n_initial_points"],
        )

    def get_hyperparam_proposal(self, num_iter_per_batch: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        param_batch = []
        proposals = self.hyper_optimizer.ask(n_points=num_iter_per_batch)
        # Generate list of dictionaries with different hyperparams to evaluate
        for prop in proposals:
            proposal_params = {}
            for i, p_name in enumerate(self.param_range.keys()):
                if type(prop[i]) == np.int64:
                    proposal_params[p_name] = int(prop[i])
                else:
                    proposal_params[p_name] = prop[i]
            param_batch.append(proposal_params)
        return param_batch

    def clean_up_after_batch_iteration(self, batch_proposals, perf_measures):
        """Perform post-iteration clean-up by updating surrogate model."""
        x, y = [], []
        # First key of all metrics is used to update the surrogate!
        to_model = list(perf_measures.keys())[0]
        eval_ids = list(perf_measures[to_model].keys())
        # TODO: Do multi-objective SMBO?!
        for i, prop in enumerate(batch_proposals):
            x.append(list(prop.values()))
            # skopt assumes we want to minimize surrogate model
            # Make performance negative if we maximize
            effective_perf = (
                -1 * perf_measures[to_model][eval_ids[i]]
                if self.hyper_log.max_objective
                else perf_measures[to_model][eval_ids[i]]
            )
            y.append(effective_perf)
        self.hyper_optimizer.tell(x, y)
