

class HyperSpace(object):
    def __init__(self, real, integer, categorical):
        """General Class Wrapper for HyperSpace configuration setup."""
        self.real = real
        self.integer = integer
        self.categorical = categorical
        # Run the space setup
        self.construct()

    def sample(self):
        """ 'Sample' from the hyperparameter space."""
        raise NotImplementedError

    def construct(self):
        """ Setup/construct the space."""
        raise NotImplementedError

    def describe(self):
        """ Get space statistics/parameters printed out. """
        raise NotImplementedError
