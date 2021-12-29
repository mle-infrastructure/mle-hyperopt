class HyperSpace(object):
    def __init__(self, real, integer, categorical):
        """General Class Wrapper for HyperSpace configuration setup."""
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

    def check(self):
        """Check that all inputs are provided correctly."""
        raise NotImplementedError

    def sample(self):
        """'Sample' from the hyperparameter space."""
        raise NotImplementedError

    def construct(self):
        """Setup/construct the search space."""
        raise NotImplementedError

    def update(self, real, integer, categorical):
        """Update the search variables and update the space."""
        self.real = real
        self.integer = integer
        self.categorical = categorical
        # Check for correctness of search space
        self.check()
        # Run the space setup
        self.construct()

    @property
    def num_dims(self):
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
    def bounds(self):
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

    def describe(self):
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
                    "extra": f'Begin: {v["begin"]}, End: {v["end"]}, Prior: {v["prior"]}',
                }
                all_vars.append(data_dict)

        if self.integer is not None:
            for k, v in self.integer.items():
                data_dict = {
                    "name": k,
                    "type": "integer",
                    "extra": f'Begin: {v["begin"]}, End: {v["end"]}, Prior: {v["prior"]}',
                }
                all_vars.append(data_dict)
        return all_vars

    def contains(self, candidate):
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
