from typing import Union
from ..space import HyperSpace
import nevergrad as ng


class NevergradSpace(HyperSpace):
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
    ):
        """For SMBO-based hyperopt generate spaces with skopt classes"""
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
        param_dict = {}
        if self.categorical is not None:
            for k, v in self.categorical.items():
                param_dict[k] = ng.p.Choice(v)

        if self.real is not None:
            for k, v in self.real.items():
                if v["prior"] == "uniform":
                    param_dict[k] = ng.p.Scalar(
                        lower=float(v["begin"]), upper=float(v["end"])
                    )
                elif v["prior"] == "log-uniform":
                    param_dict[k] = ng.p.Log(
                        lower=float(v["begin"]), upper=float(v["end"])
                    )

        if self.integer is not None:
            for k, v in self.integer.items():
                if v["prior"] == "uniform":
                    param_dict[k] = ng.p.Scalar(
                        lower=float(v["begin"]), upper=float(v["end"]) + 1
                    ).set_integer_casting()
                elif v["prior"] == "log-uniform":
                    param_dict[k] = ng.p.Log(
                        lower=float(v["begin"]), upper=float(v["end"]) + 1
                    ).set_integer_casting()
        self.dimensions = ng.p.Instrumentation(**param_dict)
