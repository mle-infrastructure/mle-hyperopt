from typing import Union
from ..space import HyperSpace
from skopt.space import Real, Integer, Categorical


class SMBOSpace(HyperSpace):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
    ):
        """For SMBO-based hyperopt generate spaces with skopt classes"""
        HyperSpace.__init__(self, real, integer, categorical)
        self.dimensions = list(self.param_range.values())

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
                param_range[k] = Categorical(v, name=k)

        if self.real is not None:
            for k, v in self.real.items():
                param_range[k] = Real(
                    float(v["begin"]), float(v["end"]), prior=v["prior"], name=k
                )

        if self.integer is not None:
            for k, v in self.integer.items():
                param_range[k] = Integer(
                    int(v["begin"]), int(v["end"]) + 1, prior=v["prior"], name=k
                )
        self.param_range = param_range
