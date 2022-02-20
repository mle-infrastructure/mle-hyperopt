from typing import Optional
import numpy as np
from ..space import HyperSpace


class RandomSpace(HyperSpace):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
    ):
        """Random search hyperparameter space with desired priors.

        Args:
            real (Optional[dict], optional):
                Dictionary of real-valued search variables & their priors.
                E.g. {"lrate": {"begin": 0.1, "end": 0.5, "prior": "log-uniform"}}
                Defaults to None.
            integer (Optional[dict], optional):
                Dictionary of integer-valued search variables & their priors.
                E.g. {"batch_size": {"begin": 1, "end": 5, "bins": "uniform"}}
                Defaults to None.
            categorical (Optional[dict], optional):
                Dictionary of categorical-valued search variables.
                E.g. {"arch": ["mlp", "cnn"]}
                Defaults to None.
        """
        HyperSpace.__init__(self, real, integer, categorical)

    def check(self) -> None:
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

    def construct(self) -> None:
        """Setup/construct the random search space."""
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
                    "values": np.arange(
                        int(v["begin"]), int(v["end"]) + 1, 1
                    ).tolist(),
                    "prior": v["prior"],
                }
        self.param_range = param_range

    def sample(self) -> dict:
        """Sample from the random hyperparameter space.

        Returns:
            dict: Randomly sampled parameter configuration dictionary.
        """
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
                        np.log(p_range["values"][0]),
                        np.log(p_range["values"][-1]),
                    )
                    eval_param = int(np.exp(x))
            elif p_range["value_type"] == "real":
                if p_range["prior"] == "uniform":
                    eval_param = np.random.uniform(*p_range["values"])
                elif p_range["prior"] == "log-uniform":
                    x = np.random.uniform(
                        np.log(p_range["values"][0]),
                        np.log(p_range["values"][1]),
                    )
                    eval_param = np.exp(x)
            proposal_params[p_name] = eval_param
        return proposal_params
