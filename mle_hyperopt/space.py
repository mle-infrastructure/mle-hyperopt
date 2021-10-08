class HyperSpace(object):
    def __init__(self, real, integer, categorical):
        """General Class Wrapper for HyperSpace configuration setup."""
        self.real = real
        self.integer = integer
        self.categorical = categorical
        # Check for correctness of search space
        self.check()
        # Run the space setup
        self.construct()

    def check(self):
        """Check that all inputs are provided correctly."""
        raise NotImplementedError

    def sample(self):
        """'Sample' from the hyperparameter space."""
        raise NotImplementedError

    def construct(self):
        """Setup/construct the search space."""
        raise NotImplementedError

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
