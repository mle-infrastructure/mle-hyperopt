import numpy as np
from ...spaces import RandomSpace


class Explore(object):
    def __init__(self, explore_config: dict, space: RandomSpace):
        """Exploration Strategies for PBT (Jaderberg et al. 17)."""
        self.explore_config = explore_config
        self.space = space
        assert self.explore_config["strategy"] in [
            "perturbation",
            "resampling",
            "additive-noise",
        ]

    def perturb(self, hyperparams: dict) -> dict:
        """Multiply hyperparam independently by random factor of 0.8/1.2/etc."""
        new_hyperparams = {}
        for param_name, param_value in hyperparams.items():
            new_hyperparams[param_name] = (
                np.random.choice(self.explore_config["perturb_coeffs"]) * param_value
            )
        return new_hyperparams

    def resample(self) -> dict:
        """Resample hyperparam from original prior distribution."""
        hyperparams = self.space.sample()
        return hyperparams

    def noisify(self, hyperparams: dict) -> dict:
        """Add independent Gaussian noise to all float hyperparams."""
        # Uses scale of 0.2 as in example notebook
        # https://github.com/bkj/pbt/blob/master/pbt.ipynb
        new_hyperparams = {}
        for param_name, param_value in hyperparams.items():
            # Sample gaussian noise and add it to the parameter
            eps = np.random.normal() * self.explore_config["noise_scale"]
            new_hyperparams[param_name] = param_value + eps
        return new_hyperparams

    def __call__(self, hyperparams: dict) -> dict:
        """Perform an exploration step."""
        if self.strategy == "perturbation":
            hyperparams = self.perturb(hyperparams)
        elif self.strategy == "resampling":
            hyperparams = self.resample()
        elif self.strategy == "additive-noise":
            hyperparams = self.noisify(hyperparams)
        return hyperparams
