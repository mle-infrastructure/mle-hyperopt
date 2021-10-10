from typing import Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils.helpers import load_json, save_json, write_configs_to_file
from .comms import welcome_message, update_message, ranking_message


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
    """Base Class for Running Hyperparameter Optimisation Searches."""

    def __init__(
        self,
        real: Union[dict, None] = None,
        integer: Union[dict, None] = None,
        categorical: Union[dict, None] = None,
        fixed_params: Union[dict, None] = None,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
        seed_id: int = 42,
        verbose: bool = True,
    ):
        # Key Input: Specify which params to optimize & in which ranges (dict)
        self.real = real
        self.integer = integer
        self.categorical = categorical
        self.fixed_params = fixed_params
        self.seed_id = seed_id
        self.verbose = verbose
        self.eval_counter = 0
        self.log = []
        self.all_evaluated_params = []
        self.load(reload_path, reload_list)

        # Set random seed for reproduction for all strategies
        np.random.seed(self.seed_id)

    def ask(
        self,
        batch_size: int = 1,
        store: bool = False,
        config_fnames: Union[None, List[str]] = None,
    ):
        """Get proposals to eval - implemented by specific hyperopt algo"""
        # Ask the specific strategy for a batch of configs to evaluate
        param_batch = self.ask_search(batch_size)

        # If fixed params are not none add them to config dicts
        if self.fixed_params is not None:
            for i in range(len(param_batch)):
                param_batch[i] = {**param_batch[i], **self.fixed_params}

        # If string for storage is given: Save configs as .yaml
        if store:
            if config_fnames is None:
                config_fnames = [
                    f"eval_{self.eval_counter + i}.yaml"
                    for i in range(len(param_batch))
                ]
            else:
                assert len(config_fnames) == len(param_batch)
            self.store_configs(param_batch, config_fnames)
            if batch_size == 1:
                return param_batch[0], config_fnames[0]
            else:
                return param_batch, config_fnames
        else:
            if batch_size == 1:
                return param_batch[0]
            else:
                return param_batch

    def ask_search(self, batch_size: int):
        """Search method-specific proposal generation."""
        raise NotImplementedError

    def tell(
        self,
        batch_proposals: Union[List[dict], dict],
        perf_measures: Union[List[Union[float, int]], float],
        reload: bool = False,
    ):
        """Perform post-iteration clean-up. (E.g. update surrogate model)"""
        # Ensure that update data is list to loop over
        if type(batch_proposals) == dict:
            batch_proposals = [batch_proposals]
        if type(perf_measures) in [float, int]:
            perf_measures = [perf_measures]

        for i in range(len(batch_proposals)):
            # Check whether proposals were already previously added
            # If so -- ignore (and print message?)
            proposal_clean = dict(batch_proposals[i])
            if self.fixed_params is not None:
                for k in self.fixed_params.keys():
                    del proposal_clean[k]

            if proposal_clean in self.all_evaluated_params:
                print(f"{batch_proposals[i]} was previously evaluated.")
            else:
                self.log.append(
                    {
                        "eval_id": self.eval_counter,
                        "params": proposal_clean,
                        "objective": perf_measures[i],
                    }
                )
                self.all_evaluated_params.append(proposal_clean)
                self.eval_counter += 1

        self.tell_search(batch_proposals, perf_measures)

        # Print update message
        if self.verbose and not reload:
            self.print_update(batch_proposals, perf_measures)

    def tell_search(self, batch_proposals: list, perf_measures: list):
        """Search method-specific strategy update."""
        raise NotImplementedError

    def save(self, save_path: str = "search_log.pkl"):
        """Store the state of the optimizer (parameters, values) as .pkl."""
        save_json(self.log, save_path)
        print(f"Stored {self.eval_counter} search iterations.")

    def load(
        self,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
    ):
        """Reload the state of the optimizer (parameters, values) as .pkl."""
        # Simply loop over param, value pairs and `tell` the strategy.
        prev_evals = int(self.eval_counter)
        if reload_path is not None:
            reloaded = load_json(reload_path)
            for iter in reloaded:
                self.tell([iter["params"]], [iter["objective"]], True)

        if reload_list is not None:
            for iter in reload_list:
                self.tell([iter["params"]], [iter["objective"]], True)

        if reload_path is not None or reload_list is not None:
            print(
                f"Reloaded {self.eval_counter - prev_evals}"
                " previous search iterations."
            )

    def get_best(self, top_k: int = 1):
        """Return top-k best performing parameter configurations."""
        assert top_k <= self.eval_counter
        # Mono-objective case - get best objective evals
        if type(self.log[0]["objective"]) in [float, int, np.int64]:
            objective_evals = [it["objective"] for it in self.log]
            best_idx = np.argsort(objective_evals)[:top_k]
            best_iters = [self.log[idx] for idx in best_idx]
            best_configs = [it["params"] for it in best_iters]
            best_evals = [it["objective"] for it in best_iters]
        # Multi-objective case - get pareto front
        else:
            pareto_configs, pareto_evals = self.get_pareto_front()
            best_configs, best_evals = pareto_configs[:top_k], pareto_evals[:top_k]
            best_idx = top_k * ["-"]
        if top_k == 1:
            return best_idx[0], best_configs[0], best_evals[0]
        else:
            return best_idx, best_configs, best_evals

    def print_ranking(self, top_k: int = 5):
        """Pretty print archive of best configurations."""
        best_idx, best_configs, best_evals = self.get_best(top_k)
        ranking_message(best_idx, best_configs, best_evals)

    def store_configs(
        self,
        config_dicts: List[dict],
        config_fnames: Union[str, List[str], None] = None,
    ):
        """Store configuration as .json files to file path."""
        write_configs_to_file(config_dicts, config_fnames)

    def plot_best(self):
        """Plot the evolution of best model performance over evaluations."""
        assert type(self.log[0]["objective"]) in [float, int]
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

    def to_df(self):
        """Return log as pandas dataframe."""
        flat_log = []
        for l in self.log:
            sub_log = {}
            sub_log["eval_id"] = l["eval_id"]
            # TODO: Add if-clause for multi-objective list case
            sub_log["objective"] = l["objective"]
            sub_log.update(l["params"])
            flat_log.append(sub_log)
        return pd.DataFrame(flat_log)

    def __len__(self) -> int:
        """Return number of evals stored in log."""
        return self.eval_counter

    def print_hello(self, search_type: str):
        """Print start-up message."""
        # Get search data in table format
        space_data = self.space.describe()
        welcome_message(space_data, search_type)

    def print_update(
        self, batch_proposals: List[dict], perf_measures: List[Union[float, int]]
    ):
        """Print strategy update."""
        best_eval_id, best_config, best_eval = self.get_best(top_k=1)
        best_batch_idx = np.argmin(perf_measures)
        best_batch_eval_id = self.eval_counter - len(perf_measures) + best_batch_idx
        best_batch_config, best_batch_eval = (
            batch_proposals[best_batch_idx],
            perf_measures[best_batch_idx],
        )
        # Print best data in log - and best data in last batch
        update_message(self.eval_counter, best_eval_id, best_config, best_eval,
                       best_batch_eval_id, best_batch_config, best_batch_eval)
