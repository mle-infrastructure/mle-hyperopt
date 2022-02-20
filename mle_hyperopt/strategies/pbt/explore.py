import numpy as np
from ...spaces import RandomSpace


class Explore(object):
    def __init__(self, explore_config: dict, space: RandomSpace):
        """Exploration Strategies for PBT (Jaderberg et al. 17).

        Args:
            explore_config (dict): Configuration for exploration step.
                E.g. {"strategy": "perturbation", "perturb_coeffs": [0.8, 1.2]}
            space (RandomSpace): Random hyperparameter space to sample from.
        """
        self.explore_config = explore_config
        self.space = space
        assert self.explore_config["strategy"] in [
            "perturbation",
            "resampling",
            "additive-noise",
        ]

    def perturb(self, hyperparams: dict) -> dict:
        """Multiply hyperparam independently by random sampled factor.

        Args:
            hyperparams (dict): Dictionary of hyperparameters of a worker.

        Returns:
            dict: Perturbed/resampled hyperparameter dictionary.
        """
        new_hyperparams = {}
        for param_name, param_value in hyperparams.items():
            # If real-valued -> multiply by sampled coefficient
            if param_name in self.space.real_names:
                new_hyperparams[param_name] = (
                    np.random.choice(self.explore_config["perturb_coeffs"])
                    * param_value
                )
            # If integer-valued -> multiply by sampled coefficient & round
            elif param_name in self.space.integer_names:
                new_hyperparams[param_name] = round(
                    np.random.choice(self.explore_config["perturb_coeffs"])
                    * param_value
                )
            # If categorical-valued -> resample from space
            else:
                sample_config = self.space.sample()
                new_hyperparams[param_name] = sample_config[param_name]
        return new_hyperparams

    def resample(self) -> dict:
        """Resample hyperparam from original prior distribution.

        Returns:
            dict: Resampled hyperparameter dictionary.
        """
        return self.space.sample()

    def noisify(self, hyperparams: dict) -> dict:
        """Add independent Gaussian noise to all float hyperparams.

        Returns:
            dict: Noisified hyperparameter dictionary.
        """
        new_hyperparams = {}
        for param_name, param_value in hyperparams.items():
            if param_name in self.space.real_names:
                # If real-valued -> add sampled noise
                eps = np.random.normal() * self.explore_config["noise_scale"]
                new_hyperparams[param_name] = param_value + eps
            elif param_name in self.space.integer_names:
                # If integer-valued -> add sampled noise & round
                eps = np.random.normal() * self.explore_config["noise_scale"]
                new_hyperparams[param_name] = round(param_value + eps)
            else:
                # If categorical-valued -> resample from space
                sample_config = self.space.sample()
                new_hyperparams[param_name] = sample_config[param_name]
        return new_hyperparams

    def __call__(self, hyperparams: dict) -> dict:
        """Perform an exploration step for a single worker hyperparameter dict.

        Args:
            hyperparams (dict): Hyperparameters of worker.

        Returns:
            dict: Hyperparameters after exploration step.
        """
        if self.explore_config["strategy"] == "perturbation":
            hyperparams = self.perturb(hyperparams)
        elif self.explore_config["strategy"] == "resampling":
            hyperparams = self.resample()
        elif self.explore_config["strategy"] == "additive-noise":
            hyperparams = self.noisify(hyperparams)
        return hyperparams
