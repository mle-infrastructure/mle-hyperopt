

class HyperOpt(object):
    """ Base Class for Running Hyperparameter Optimisation Searches."""
    def __init__(
        self,
        search_params: dict,
    ):
        # Key Input: Specify which params to optimize & in which ranges (dict)
        self.search_params = search_params  # param space specs
        self.current_iter = 0

    def ask(self, num_iter_batch: int):
        """Get proposals to eval - implemented by specific hyperopt algo"""
        raise NotImplementedError

    def tell(self, batch_proposals, perf_measures):
        """Perform post-iteration clean-up. (E.g. update surrogate model)"""
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def refine(self):
        raise NotImplementedError
