from ..hyperspace import HyperSpace
import nevergrad as ng


class NevergradSpace(HyperSpace):
    def __init__(self, real, integer, categorical):
        """For SMBO-based hyperopt generate spaces with skopt classes"""
        HyperSpace.__init__(self, real, integer, categorical)

    def construct(self):
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
                    param_dict[k] = ng.p.Log(lower=float(v["begin"]), upper=float(v["end"]))

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
