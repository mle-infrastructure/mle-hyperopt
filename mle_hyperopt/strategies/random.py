import numpy as np
from typing import Union
from .base import HyperOpt
from .hyperspace import construct_hyperparam_range


class RandomHyperoptimisation(HyperOpt):
    def __init__(
        self,
        hyper_log: HyperoptLogger,
        resource_to_run: str,
        job_arguments: dict,
        config_fname: str,
        job_fname: str,
        experiment_dir: str,
        search_params: dict,
        search_type: str = "random",
        search_schedule: str = "sync",
        message_id: Union[str, None] = None,
    ):
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
        self.eval_counter = len(hyper_log)

    def get_hyperparam_proposal(self, num_evals_per_batch: int):
        """Get proposals to eval next (in batches) - Random Sampling."""
        param_batch = []
        # Sample a new configuration for each eval in the batch
        while len(param_batch) < num_evals_per_batch:
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
