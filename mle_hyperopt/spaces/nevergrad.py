from typing import Optional
from ..space import HyperSpace
import nevergrad as ng


class NevergradSpace(HyperSpace):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
    ):
        """Search hyperparameter space for FAIR's nevergrad.
        Reference: https://facebookresearch.github.io/nevergrad/

        Args:
            real (Optional[dict], optional):
                Dictionary of real-valued search variables & their prior.
                E.g. {"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}}
                Defaults to None.
            integer (Optional[dict], optional):
                Dictionary of integer-valued search variables & their prior.
                E.g. {"batch_size": {"begin": 1, "end": 5, "prior": "log-uniform"}}
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
        """Setup/construct the search space for nevergrad."""
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
