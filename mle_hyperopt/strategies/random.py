import numpy as np
from .base import HyperOpt
from ..hyperspace import construct_hyperparam_range


class RandomSearch(HyperOpt):
    def __init__(
        self,
        search_params: dict,
    ):
        HyperOpt.__init__(self, search_params)
        self.param_range = construct_hyperparam_range(
            self.search_params, "random"
        )
        self.eval_counter = 0

    def ask(self, batch_size: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        param_batch = []
        # Sample a new configuration for each eval in the batch
        while len(param_batch) < batch_size:
            proposal_params = {}
            # Sample the parameters individually at random from the ranges
            for p_name, p_range in self.param_range.items():
                if p_range["value_type"] in ["integer", "categorical"]:
                    eval_param = np.random.choice(p_range["values"])
                    if type(eval_param) == np.int64:
                        eval_param = int(eval_param)
                elif p_range["value_type"] == "real":
                    eval_param = np.random.uniform(*p_range["values"])
                proposal_params[p_name] = eval_param

            if proposal_params not in (
                self.hyper_log.all_evaluated_params + param_batch
            ):
                # Add parameter proposal to the batch list
                param_batch.append(proposal_params)
                self.eval_counter += 1
            else:
                # Otherwise continue sampling proposals
                continue
        return param_batch

    def tell(self, batch_proposals, perf_measures):
        """Perform post-iteration clean-up by updating surrogate model."""
