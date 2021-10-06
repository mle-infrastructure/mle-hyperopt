import numpy as np
from ..hyperspace import HyperSpace


class RandomSpace(HyperSpace):
    def __init__(self, real, integer, categorical):
        """For random hyperopt generate list/1d-line range to sample from."""
        HyperSpace.__init__(self, real, integer, categorical)

    def construct(self):
        """ Setup/construct the space. """
        param_range = {}
        if self.categorical is not None:
            for k, v in self.categorical.items():
                param_range[k] = {"value_type": "categorical", "values": v}

        if self.real is not None:
            for k, v in self.real.items():
                param_range[k] = {
                    "value_type": "real",
                    "values": [float(v["begin"]), float(v["end"])],
                }

        if self.integer is not None:
            for k, v in self.integer.items():
                param_range[k] = {
                    "value_type": "integer",
                    "values": np.arange(int(v["begin"]), int(v["end"]) + 1, 1).tolist(),
                }
        self.param_range = param_range

    def sample(self):
        """ 'Sample' from the hyperparameter space. """
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
        return proposal_params
