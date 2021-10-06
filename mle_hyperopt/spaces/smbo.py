from ..hyperspace import HyperSpace
from skopt.space import Real, Integer, Categorical


class SMBOSpace(HyperSpace):
    def __init__(self, real, integer, categorical):
        """For SMBO-based hyperopt generate spaces with skopt classes"""
        HyperSpace.__init__(self, real, integer, categorical)
        self.dimensions = list(self.param_range.values())

    def construct(self):
        param_range = {}
        # Can specify prior distribution over hyperp. distrib
        # log-uniform samples more from the lower tail of the hyperparam range
        #   real: ["uniform", "log-uniform"]
        #   integer: ["uniform", "log-uniform"]
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
