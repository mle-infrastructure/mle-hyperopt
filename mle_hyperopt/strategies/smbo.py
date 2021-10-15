from typing import Union
import numpy as np
from ..search import HyperOpt
from ..spaces import SMBOSpace
from skopt import Optimizer


class SMBOSearch(HyperOpt):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        search_config: dict = {
            "base_estimator": "GP",
            "acq_function": "gp_hedge",
            "n_initial_points": 5,
        },
        fixed_params: Union[dict, None] = None,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
        seed_id: int = 42,
        verbose: bool = False,
    ):
        # Check that SMBO uses synchronous scheduling
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
        self.space = SMBOSpace(real, integer, categorical)

        # Initialize the surrogate model/hyperparam config proposer
        self.hyper_optimizer = Optimizer(
            dimensions=self.space.dimensions,
            random_state=self.seed_id,
            base_estimator=self.search_config["base_estimator"],
            acq_func=self.search_config["acq_function"],
            n_initial_points=self.search_config["n_initial_points"],
        )

        # Add start-up message printing the search space
        if self.verbose:
            self.print_hello("SMBO Search")

    def ask_search(self, batch_size: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        param_batch = []
        proposals = self.hyper_optimizer.ask(n_points=batch_size)
        # Generate list of dictionaries with different hyperparams to evaluate
        for prop in proposals:
            proposal_params = {}
            for i, p_name in enumerate(self.space.param_range.keys()):
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
