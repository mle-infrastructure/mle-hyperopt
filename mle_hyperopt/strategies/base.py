from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils import load_pkl_object, save_pkl_object, write_configs_to_file


sns.set(
    context="poster",
    style="white",
    palette="Paired",
    font="sans-serif",
    font_scale=1.0,
    color_codes=True,
    rc=None,
)


class HyperOpt(object):
    """ Base Class for Running Hyperparameter Optimisation Searches."""
    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        fixed_params: Union[dict, None] = None,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
    ):
        # Key Input: Specify which params to optimize & in which ranges (dict)
        self.real = real
        self.integer = integer
        self.categorical = categorical
        self.fixed_params = fixed_params
        self.eval_counter = 0
        self.log = []
        self.all_evaluated_params = []
        self.load(reload_path, reload_list)

    def ask(self, batch_size: int, store: bool = False,
            config_fnames: Union[None, List[str]] = None):
        """Get proposals to eval - implemented by specific hyperopt algo"""
        param_batch = self.ask_search(batch_size)
        # If fixed params are not none add them to config dicts
        if self.fixed_params is not None:
            for i in range(len(param_batch)):
                param_batch[i] = {**param_batch[i], **self.fixed_params}

        # If string for storage is given: Save configs as .yaml
        if store:
            if config_fnames is None:
                config_fnames = [f"eval_{self.eval_counter + i}.yaml"
                                 for i in range(len(param_batch))]
            else:
                assert len(config_fnames) == len(param_batch)
            self.store_configs(param_batch, config_fnames)
        return param_batch

    def ask_search(self, batch_size: int):
        """Search method-specific proposal generation."""
        raise NotImplementedError

    def tell(self,
             batch_proposals: Union[List[dict], dict],
             perf_measures: Union[List[float], float]):
        """Perform post-iteration clean-up. (E.g. update surrogate model)"""
        self.tell_search(batch_proposals, perf_measures)

        for i in range(len(batch_proposals)):
            # Check whether proposals were already previously added
            # If so -- ignore (and print message?)
            if batch_proposals[i] in self.all_evaluated_params:
                print(f"{batch_proposals[i]} were previously evaluated.")
            else:
                self.log.append({"eval_id": self.eval_counter,
                                 "params": batch_proposals[i],
                                 "objective": perf_measures[i]})
                self.all_evaluated_params.append(batch_proposals[i])
                self.eval_counter += 1
                print(f"Loaded {batch_proposals[i]}, Obj: {perf_measures[i]}.")

    def tell_search(self, batch_proposals: list, perf_measures: list):
        """Search method-specific strategy update."""
        raise NotImplementedError

    def save(self, save_path: str = "search_log.pkl"):
        """Store the state of the optimizer (parameters, values) as .pkl."""
        save_pkl_object(self.log, save_path)
        print(f"Stored {self.eval_counter} search iterations.")

    def load(self,
             reload_path: Union[str, None] = None,
             reload_list: Union[list, None] = None):
        """Reload the state of the optimizer (parameters, values) as .pkl."""
        # Simply loop over param, value pairs and `tell` the strategy.
        prev_evals = int(self.eval_counter)
        if reload_path is not None:
            reloaded = load_pkl_object(reload_path)
            for iter in reloaded:
                self.tell([iter["params"]], [iter["objective"]])

        if reload_list is not None:
            for iter in reload_list:
                self.tell([iter["params"]], [iter["objective"]])

        if reload_path is not None or reload_list is not None:
            print(f"Reloaded {self.eval_counter - prev_evals}"
                  " previous search iterations.")

    def get_best(self, top_k: int = 1):
        """Return top-k best performing parameter configurations."""
        assert top_k <= self.eval_counter
        objective_evals = [it["objective"] for it in self.log]
        best_idx = np.argsort(objective_evals)[:top_k]
        best_configs = [self.log[idx] for idx in best_idx]
        return best_configs

    def print_ranking(self, top_k: int = 5):
        """Pretty print archive of best configurations."""
        # TODO: Add nice rich-style print statement!
        raise NotImplementedError

    def store_configs(self,
                      config_dicts: List[dict],
                      config_fnames: Union[str, List[str], None] = None):
        """Store configuration as .json files to file path."""
        write_configs_to_file(config_dicts, config_fnames)

    def plot_best(self):
        """Plot the evolution of best model performance over evaluations."""
        objective_evals = [it["objective"] for it in self.log]
        timeseries = np.minimum.accumulate(objective_evals)
        fig, ax = plt.subplots()
        ax.plot(timeseries)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Best Objective Value")
        ax.set_xlabel("# Config Evaluations")
        ax.set_ylabel("Objective")
        fig.tight_layout()
        return fig, ax
