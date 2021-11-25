from typing import Union
import numpy as np
from ..space import HyperSpace


class RandomSpace(HyperSpace):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
    ):
        """For random hyperopt generate list/1d-line range to sample from."""
        HyperSpace.__init__(self, real, integer, categorical)

    def check(self):
        """Check that all inputs are provided correctly."""
        if self.real is not None:
            real_keys = ["begin", "end", "prior"]
            for key in real_keys:
                for k, v in self.real.items():
                    assert key in v
                    assert float(v["begin"]) <= float(v["end"])
                if key == "prior":
                    assert v[key] in ["uniform", "log-uniform"]

        if self.integer is not None:
            integer_keys = ["begin", "end", "prior"]
            for key in integer_keys:
                for k, v in self.integer.items():
                    assert key in v
                    assert int(v["begin"]) <= int(v["end"])
                    if key != "prior":
                        assert type(v[key]) == int
                    else:
                        assert v[key] in ["uniform", "log-uniform"]

        if self.categorical is not None:
            for k, v in self.categorical.items():
                if type(v) is not list:
                    self.categorical[k] = [v]

    def construct(self):
        """Setup/construct the search space."""
        param_range = {}
        if self.categorical is not None:
            for k, v in self.categorical.items():
                param_range[k] = {"value_type": "categorical", "values": v}

        if self.real is not None:
            for k, v in self.real.items():
                param_range[k] = {
                    "value_type": "real",
                    "values": [float(v["begin"]), float(v["end"])],
                    "prior": v["prior"],
                }

        if self.integer is not None:
            for k, v in self.integer.items():
                param_range[k] = {
                    "value_type": "integer",
                    "values": np.arange(int(v["begin"]), int(v["end"]) + 1, 1).tolist(),
                    "prior": v["prior"],
                }
        self.param_range = param_range

    def sample(self):
        """'Sample' from the hyperparameter space."""
        proposal_params = {}
        # Sample the parameters individually at random from the ranges
        for p_name, p_range in self.param_range.items():
            if p_range["value_type"] == "categorical":
                eval_param = np.random.choice(p_range["values"])
            elif p_range["value_type"] == "integer":
                if p_range["prior"] == "uniform":
                    eval_param = int(np.random.choice(p_range["values"]))
                elif p_range["prior"] == "log-uniform":
                    x = np.random.uniform(
                        np.log(p_range["values"][0]), np.log(p_range["values"][-1])
                    )
                    eval_param = int(np.exp(x))
            elif p_range["value_type"] == "real":
                if p_range["prior"] == "uniform":
                    eval_param = np.random.uniform(*p_range["values"])
                elif p_range["prior"] == "log-uniform":
                    x = np.random.uniform(
                        np.log(p_range["values"][0]), np.log(p_range["values"][1])
                    )
                    eval_param = np.exp(x)
            proposal_params[p_name] = eval_param
        return proposal_params
