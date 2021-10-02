from typing import Union
import numpy as np
from ..base import HyperOpt
from ..hyperspace import smbo_space


class SMBOSearch(HyperOpt):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        smbo_config: dict = {
            "base_estimator": "GP",
            "acq_function": "gp_hedge",
            "n_initial_points": 5,
        },
        fixed_params: Union[dict, None] = None,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
        seed_id: int = 42,
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
        HyperOpt.__init__(
            self, real, integer, categorical, fixed_params, reload_path, reload_list
        )
        self.param_range = smbo_space(real, integer, categorical)

        # Initialize the surrogate model/hyperparam config proposer
        self.smbo_config = smbo_config
        self.hyper_optimizer = Optimizer(
            dimensions=list(self.param_range.values()),
            random_state=self.seed_id,
            base_estimator=smbo_config["base_estimator"],
            acq_func=smbo_config["acq_function"],
            n_initial_points=smbo_config["n_initial_points"],
        )

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        param_batch = []
        proposals = self.hyper_optimizer.ask(n_points=batch_size)
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

    def tell_search(self, batch_proposals, perf_measures):
        """Perform post-iteration clean-up by updating surrogate model."""
        x = []
        for i, prop in enumerate(batch_proposals):
            prop_conf = dict(prop)
            if self.fixed_params is not None:
                for k in self.fixed_params.keys():
                    del prop_conf[k]
            x.append(list(prop_conf.values()))
        self.hyper_optimizer.tell(x, perf_measures)
