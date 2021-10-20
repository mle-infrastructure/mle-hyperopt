from typing import Union, List
import numpy as np
import pandas as pd
from mle_hyperopt.utils import load_json, save_json, write_configs_to_file
from mle_hyperopt.comms import welcome_message, update_message, ranking_message
import matplotlib.pyplot as plt
import seaborn as sns

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
        search_config: Union[dict, None] = None,
        maximize_objective: bool = False,
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
        self.search_config = search_config
        self.maximize_objective = maximize_objective
        self.fixed_params = fixed_params
        self.seed_id = seed_id
        self.verbose = verbose
        self.eval_counter = 0
        self.log = []
        self.all_evaluated_params = []

        # Set random seed for reproduction for all strategies
        np.random.seed(self.seed_id)

        # Set up search space refinement
        if self.search_config is not None:
            if "refine_top_k" in self.search_config.keys():
                self.refine_counter = 0
                assert self.search_config["refine_top_k"] > 1
                self.refine_after = self.search_config["refine_after"]
                # Make sure that refine iteration is list
                if type(self.refine_after) == int:
                    self.refine_after = [self.refine_after]
                self.refine_top_k = self.search_config["refine_top_k"]
                self.last_refined = 0
            else:
                self.refine_after = None
        else:
            self.refine_after = None

        # Reload previously stored search data
        self.load(reload_path, reload_list)

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

        # Refine search space boundaries after set of search iterations
        if self.refine_after is not None:
            # Check whether there are still refinements open
            # And whether we have already passed last refinement point
            if len(self.refine_after) > self.refine_counter:
                exact = self.eval_counter == self.refine_after[self.refine_counter]
                skip = (
                    self.eval_counter > self.refine_after[self.refine_counter]
                    and self.last_refined != self.refine_after[self.refine_counter]
                )
                if exact or skip:
                    self.refine(self.refine_top_k)
                    self.last_refined = self.refine_after[self.refine_counter]
                    self.refine_counter += 1

    def tell_search(self, batch_proposals: list, perf_measures: list):
        """Search method-specific strategy update."""
        raise NotImplementedError

    def save(self, save_path: str = "search_log.json", verbose: bool = False):
        """Store the state of the optimizer (parameters, values) as .pkl."""
        save_json(self.log, save_path)
        if verbose:
            print(f"Stored {self.eval_counter} search iterations --> {save_path}.")

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
            sorted_idx = np.argsort(objective_evals)
            if not self.maximize_objective:
                best_idx = sorted_idx[:top_k]
            else:
                best_idx = sorted_idx[::-1][:top_k]
            best_iters = [self.log[idx] for idx in best_idx]
            best_configs = [it["params"] for it in best_iters]
            best_evals = [it["objective"] for it in best_iters]

        # Multi-objective case - get pareto front
        else:
            pareto_configs, pareto_evals = self.get_pareto_front()
            if not self.maximize_objective:
                best_configs, best_evals = pareto_configs[:top_k], pareto_evals[:top_k]
            else:
                best_configs, best_evals = (
                    pareto_configs[::-1][:top_k],
                    pareto_evals[::-1][:top_k],
                )

            best_idx = top_k * ["-"]

        # Unravel retrieved lists if there is only single config
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

        if not self.maximize_objective:
            timeseries = np.minimum.accumulate(objective_evals)
        else:
            timeseries = np.maximum.accumulate(objective_evals)

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

    def print_hello(self, message: Union[str, None] = None):
        """Print start-up message."""
        # Get search data in table format
        space_data = self.space.describe()
        if message is not None:
            print_out = self.search_name + " - " + message
        else:
            print_out = self.search_name
        welcome_message(space_data, print_out)

    def print_update(
        self, batch_proposals: List[dict], perf_measures: List[Union[float, int]]
    ):
        """Print strategy update."""
        best_eval_id, best_config, best_eval = self.get_best(top_k=1)
        if not self.maximize_objective:
            best_batch_idx = np.argmin(perf_measures)
        else:
            best_batch_idx = np.argmax(perf_measures)

        best_batch_eval_id = self.eval_counter - len(perf_measures) + best_batch_idx
        best_batch_config, best_batch_eval = (
            batch_proposals[best_batch_idx],
            perf_measures[best_batch_idx],
        )
        # Print best data in log - and best data in last batch
        update_message(
            self.eval_counter,
            best_eval_id,
            best_config,
            best_eval,
            best_batch_eval_id,
            best_batch_config,
            best_batch_eval,
        )

    def refine_space(self, top_k: int):
        """Search method-specific search space update."""
        raise NotImplementedError

    def refine(self, top_k: int):
        """Refine the space boundaries based on top-k performers."""
        top_idx, top_k_configs, top_k_evals = self.get_best(top_k)
        # Loop over real, integer and categorical variable keys
        # Get boundaries and re-define the search space
        if self.categorical is not None:
            categorical_refined = {}
            for var in self.categorical.keys():
                top_k_var = [config[var] for config in top_k_configs]
                categorical_refined[var] = list(set(top_k_var))
        else:
            categorical_refined = None

        if self.real is not None:
            real_refined = {}
            for var in self.real.keys():
                top_k_var = [config[var] for config in top_k_configs]
                real_refined[var] = {
                    "begin": np.min(top_k_var),
                    "end": np.max(top_k_var),
                }
                # Copy prior/number of bins to loop over
                if "prior" in self.real[var].keys():
                    real_refined[var]["prior"] = self.real[var]["prior"]
                # elif "bins" in self.real[var].keys():
                #     # TODO: Will increase the grid resolution! Do we want this?
                #     real_refined[var]["bins"] = self.real[var]["bins"]
        else:
            real_refined = None

        if self.integer is not None:
            integer_refined = {}
            for var in self.integer.keys():
                top_k_var = [config[var] for config in top_k_configs]
                integer_refined[var] = {
                    "begin": int(np.min(top_k_var)),
                    "end": int(np.max(top_k_var)),
                }
            # Copy prior/number of bins to loop over
            if "prior" in self.integer[var].keys():
                integer_refined[var]["prior"] = self.integer[var]["prior"]
            # elif "bins" in self.integer[var].keys():
            #     # TODO: Will increase the grid resolution! Do we want this?
            #     integer_refined[var]["bins"] = self.integer[var]["bins"]
        else:
            integer_refined = None

        self.refine_space(real_refined, integer_refined, categorical_refined)
        if self.verbose:
            self.print_hello(f"{self.eval_counter} Evals - Top {top_k} - Refined")

    def get_pareto_front(self):
        """Get pareto front for multi-objective problems."""
        raise NotImplementedError
