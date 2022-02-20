from typing import Optional, List


class HyperSpace(object):
    def __init__(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
    ):
        """General Class Wrapper for HyperSpace configuration setup.

        Args:
            real (Optional[dict], optional):
                Dictionary of real-valued search variables & their resolution.
                E.g. {"lrate": {"begin": 0.1, "end": 0.5, "bins": 5}}
                Defaults to None.
            integer (Optional[dict], optional):
                Dictionary of integer-valued search variables & their resolution.
                E.g. {"batch_size": {"begin": 1, "end": 5, "bins": 5}}
                Defaults to None.
            categorical (Optional[dict], optional):
                Dictionary of categorical-valued search variables.
                E.g. {"arch": ["mlp", "cnn"]}
                Defaults to None.
        """
        self.update(real, integer, categorical)
        if self.real is not None:
            self.real_names = list(self.real.keys())
        else:
            self.real_names = []
        if self.integer is not None:
            self.integer_names = list(self.integer.keys())
        else:
            self.integer_names = []
        if self.categorical is not None:
            self.categorical_names = list(self.categorical.keys())
        else:
            self.categorical_names = []
        self.variable_names = (
            self.real_names + self.integer_names + self.categorical_names
        )

    def check(self):
        """Check that all inputs are provided correctly."""
        raise NotImplementedError

    def sample(self):
        """'Sample' from the hyperparameter space."""
        raise NotImplementedError

    def construct(self):
        """Setup/construct the search space."""
        raise NotImplementedError

    def update(
        self,
        real: Optional[dict] = None,
        integer: Optional[dict] = None,
        categorical: Optional[dict] = None,
    ) -> None:
        """Update the search variables and update the space.

        Args:
            real (Optional[dict], optional):
                Dictionary of real-valued search variables & their resolution.
                E.g. {"lrate": {"begin": 0.1, "end": 0.5, "bins": 5}}
                Defaults to None.
            integer (Optional[dict], optional):
                Dictionary of integer-valued search variables & their resolution.
                E.g. {"batch_size": {"begin": 1, "end": 5, "bins": 5}}
                Defaults to None.
            categorical (Optional[dict], optional):
                Dictionary of categorical-valued search variables.
                E.g. {"arch": ["mlp", "cnn"]}
                Defaults to None.
        """
        self.real = real
        self.integer = integer
        self.categorical = categorical
        # Check for correctness of search space
        self.check()
        # Run the space setup
        self.construct()

    @property
    def num_dims(self) -> int:
        """Get number of variables to search over."""
        num_total_dims = 0
        if self.real is not None:
            num_total_dims += len(self.real)
        if self.integer is not None:
            num_total_dims += len(self.integer)
        if self.categorical is not None:
            num_total_dims += len(self.categorical)
        return num_total_dims

    @property
    def bounds(self) -> dict:
        """Return bounds of real and integer valued variables."""
        bounds_dict = {}
        if self.real is not None:
            for k, v in self.real.items():
                bounds_dict[k] = ["real", v["begin"], v["end"]]
        if self.integer is not None:
            for k, v in self.integer.items():
                bounds_dict[k] = ["integer", v["begin"], v["end"]]
        if self.categorical is not None:
            for k, v in self.categorical.items():
                bounds_dict[k] = ["categorical"] + v
        return bounds_dict

    def describe(self) -> List[dict]:
        """Get space statistics/parameters printed out."""
        all_vars = []
        if self.categorical is not None:
            for k, v in self.categorical.items():
                data_dict = {"name": k, "type": "categorical", "extra": str(v)}
                all_vars.append(data_dict)

        if self.real is not None:
            for k, v in self.real.items():
                data_dict = {
                    "name": k,
                    "type": "real",
                    "extra": (
                        f'Begin: {v["begin"]}, End: {v["end"]}, Prior:'
                        f' {v["prior"]}'
                    ),
                }
                all_vars.append(data_dict)

        if self.integer is not None:
            for k, v in self.integer.items():
                data_dict = {
                    "name": k,
                    "type": "integer",
                    "extra": (
                        f'Begin: {v["begin"]}, End: {v["end"]}, Prior:'
                        f' {v["prior"]}'
                    ),
                }
                all_vars.append(data_dict)
        return all_vars

    def contains(self, candidate: dict) -> bool:
        """Check whether a candidate is in the search space."""
        # Define a separate function for discrete grid case!
        for k, v in candidate.items():
            bound_k = self.bounds[k]
            if bound_k[0] == "real":
                if not (bound_k[1] <= v <= bound_k[2]):
                    return False
            elif bound_k[0] == "integer":
                if not (type(v) == int) or not (bound_k[1] <= v <= bound_k[2]):
                    return False
            elif bound_k[0] == "categorical":
                if not v in bound_k[1:]:
                    return False
        return True
