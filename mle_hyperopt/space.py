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
        raise NotImplementedError
