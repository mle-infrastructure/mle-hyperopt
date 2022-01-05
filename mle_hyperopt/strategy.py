from typing import Union, List
import os
import numpy as np
import pandas as pd
import numbers
from .utils import (
    load_log,
    save_log,
    load_strategy,
    save_strategy,
    write_configs,
    welcome_message,
    update_message,
    ranking_message,
)
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console

sns.set(
    context="poster",
    style="white",
    palette="Paired",
    font="sans-serif",
    font_scale=1.0,
    color_codes=True,
    rc=None,
)


class Strategy(object):
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

        # Set up the search strategy settings
        self.setup_search()

        # Reload previously stored search data
        self.load(reload_path, reload_list)

    def ask(
        self,
        batch_size: Union[int, None] = 1,
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
            if len(param_batch) == 1:
                return param_batch[0], config_fnames[0]
            else:
                return param_batch, config_fnames
        else:
            if len(param_batch) == 1:
                return param_batch[0]
            else:
                return param_batch

    def tell(
        self,
        batch_proposals: Union[List[dict], dict],
        perf_measures: Union[List[Union[float, int]], float],
        ckpt_paths: Union[None, List[str], str] = None,
        save: bool = False,
        save_path: str = "search_log.yaml",
        reload: bool = False,
    ):
        """Perform post-iteration clean-up. (E.g. update surrogate model)"""
        # Ensure that update data is list to loop over
        if type(batch_proposals) == dict:
            batch_proposals = [batch_proposals]
        if isinstance(perf_measures, numbers.Number):
            perf_measures = [perf_measures]
        if type(ckpt_paths) == str:
            ckpt_paths = [ckpt_paths]

        # Check that checkpoints are provided if using iterative search methods
        if self.search_name in ["Halving", "Hyperband", "PBT"]:
            assert ckpt_paths is not None

        log_data, clean_prop, clean_perf, clean_ckpt = self.clean_data(
            batch_proposals, perf_measures, ckpt_paths
        )

        # Update search strategy - specific to each strategy
        self.tell_search(clean_prop, clean_perf, clean_ckpt)

        # Update the log with additional search result data - merge dicts
        strat_data = self.log_search(batch_proposals, perf_measures, ckpt_paths)
        for i in range(len(clean_prop)):
            if strat_data is not None:
                if "extra" in log_data[i].keys():
                    log_data[i]["extra"] = {**log_data[i]["extra"], **strat_data[i]}
                else:
                    log_data[i]["extra"] = strat_data[i]
            self.log.append(log_data[i])

        # Print update message
        if self.verbose and not reload:
            self.print_update(clean_prop, clean_perf, clean_ckpt)

        # Update the search strategy - space refinement/switches
        self.update_search()

        # Save the log if desired (default to search_log.yaml in root)
        if save:
            self.save(save_path)

    def clean_data(
        self,
        batch_proposals: Union[List[dict], dict],
        perf_measures: Union[List[Union[float, int]], float],
        ckpt_paths: Union[None, List[str], str] = None,
    ):
        """Remove duplicate evals (reload) & strategy non-relevant data."""
        log_data, clean_proposals, clean_performance, clean_ckpt = [], [], [], []
        for i in range(len(batch_proposals)):
            # Check whether proposals were already previously added
            # If so -- ignore (and print message?)
            proposal_clean = dict(batch_proposals[i])

            if "extra" in proposal_clean.keys():
                extra_data = proposal_clean["extra"]
                proposal_clean = proposal_clean["params"]
            else:
                extra_data = None
            if self.fixed_params is not None:
                for k in self.fixed_params.keys():
                    del proposal_clean[k]

            if (
                proposal_clean in self.all_evaluated_params
                and self.search_name not in ["Halving", "Hyperband", "PBT"]
            ):
                Console().log(f"{batch_proposals[i]} was previously evaluated.")
            else:
                data_to_append = {
                    "eval_id": self.eval_counter,
                    "params": proposal_clean,
                    "objective": perf_measures[i],
                }
                if extra_data is not None:
                    data_to_append["extra"] = extra_data
                # Add checkpoint path to data if it is provided!
                if ckpt_paths is not None:
                    data_to_append["ckpt"] = ckpt_paths[i]
                log_data.append(data_to_append)
                clean_proposals.append(proposal_clean)
                clean_performance.append(perf_measures[i])
                if type(ckpt_paths) == list:
                    clean_ckpt.append(ckpt_paths[i])
                else:
                    clean_ckpt = None
                self.all_evaluated_params.append(proposal_clean)
                self.eval_counter += 1
        return log_data, list(clean_proposals), clean_performance, clean_ckpt

    def ask_search(self, batch_size: int):
        """Search method-specific proposal generation."""
        raise NotImplementedError

    def tell_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[None, List[str], str] = None,
    ):
        """Search method-specific strategy update."""

    def setup_search(self):
        """Initialize search settings at startup."""

    def log_search(
        self,
        batch_proposals: list,
        perf_measures: list,
        ckpt_paths: Union[None, List[str], str] = None,
    ):
        """Log info specific to search strategy."""

    def update_search(self):
        """Update the strategy settings - e.g. refine/coord/halving switch."""

    def save(self, save_path: str = "search_log.yaml"):
        """Store the state of the optimizer (parameters, values) as .pkl."""
        fname, fext = os.path.splitext(save_path)
        if fext in [".yaml", ".json"]:
            save_log(self.log, save_path)
        elif fext == ".pkl":
            save_strategy(self, save_path)
        else:
            raise ValueError("Only YAML, JSON or PKL file paths supported.")
        if self.verbose:
            Console().log(
                f"Stored {self.eval_counter} search iterations ➞ {save_path}."
            )

    def load(
        self,
        reload_path: Union[str, None] = None,
        reload_list: Union[list, None] = None,
    ):
        """Reload the state of the optimizer (parameters, values) as .pkl."""
        # Simply loop over param, value pairs and `tell` the strategy.
        prev_evals = int(self.eval_counter)
        if reload_path is not None:
            fname, fext = os.path.splitext(reload_path)
            if fext in [".yaml", ".json"]:
                if self.search_name in ["PBT", "SuccessiveHalving", "Hyperband"]:
                    raise ValueError(
                        "Iterative search logs can only be loaded from .pkl."
                    )
                reloaded = load_log(reload_path)
                for iter in reloaded:
                    if "ckpt" in iter.keys():
                        self.tell(
                            [iter["params"]],
                            [iter["objective"]],
                            [iter["ckpt"]],
                            reload=True,
                        )
                    else:
                        self.tell([iter["params"]], [iter["objective"]], reload=True)
            elif fext == ".pkl":
                self.__dict__ = load_strategy(reload_path).__dict__

        if reload_list is not None:
            for iter in reload_list:
                if "ckpt" in iter.keys():
                    self.tell(
                        [iter["params"]],
                        [iter["objective"]],
                        [iter["ckpt"]],
                        reload=True,
                    )
                else:
                    self.tell([iter["params"]], [iter["objective"]], reload=True)

        if reload_path is not None or reload_list is not None:
            Console().log(
                f"Reloaded {self.eval_counter - prev_evals}"
                " previous search iterations."
            )

    def get_best(self, top_k: int = 1):
        """Return top-k best performing parameter configurations."""
        assert top_k <= self.eval_counter

        # Mono-objective case - get best objective evals
        if isinstance(self.log[0]["objective"], numbers.Number):
            objective_evals = [it["objective"] for it in self.log]
            sorted_idx = np.argsort(objective_evals)
            if not self.maximize_objective:
                best_idx = sorted_idx[:top_k]
            else:
                best_idx = sorted_idx[::-1][:top_k]
            best_iters = [self.log[idx] for idx in best_idx]
            best_configs = [it["params"] for it in best_iters]
            best_evals = [it["objective"] for it in best_iters]

            if "ckpt" in best_iters[0].keys():
                best_ckpt = [it["ckpt"] for it in best_iters]
            else:
                best_ckpt = None

        # Multi-objective case - get pareto front
        else:
            pareto_configs, pareto_evals, pareto_ckpt = self.get_pareto_front()
            if not self.maximize_objective:
                best_configs, best_evals = pareto_configs[:top_k], pareto_evals[:top_k]
                if pareto_ckpt is not None:
                    best_ckpt = pareto_ckpt[:top_k]
                else:
                    best_ckpt = None
            else:
                best_configs, best_evals = (
                    pareto_configs[::-1][:top_k],
                    pareto_evals[::-1][:top_k],
                )
                if pareto_ckpt is not None:
                    best_ckpt = pareto_ckpt[::-1][:top_k]
                else:
                    best_ckpt = None

            best_idx = top_k * ["-"]

        # Unravel retrieved lists if there is only single config
        if top_k == 1:
            if best_ckpt is not None:
                ckpt_to_return = best_ckpt[0]
            else:
                ckpt_to_return = None
            return best_idx[0], best_configs[0], best_evals[0], ckpt_to_return
        else:
            return best_idx, best_configs, best_evals, best_ckpt

    def print_ranking(self, top_k: int = 5):
        """Pretty print archive of best configurations."""
        best_idx, best_configs, best_evals, _ = self.get_best(top_k)
        ranking_message(best_idx, best_configs, best_evals)

    def improvement(self, score: float) -> bool:
        """Return boolean if score is better than best logged one."""
        best_idx, best_config, best_eval, _ = self.get_best()
        if self.maximize_objective:
            improved = score >= best_eval
        else:
            improved = score <= best_eval
        return improved

    def store_configs(
        self,
        config_dicts: List[dict],
        config_fnames: Union[str, List[str], None] = None,
    ):
        """Store configuration as .json files to file path."""
        write_configs(config_dicts, config_fnames)

    def plot_best(self, fname: Union[None, str] = None):
        """Plot the evolution of best model performance over evaluations."""
        assert isinstance(self.log[0]["objective"], numbers.Number)
        objective_evals = [it["objective"] for it in self.log]

        if not self.maximize_objective:
            timeseries = np.minimum.accumulate(objective_evals)
        else:
            timeseries = np.maximum.accumulate(objective_evals)

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(timeseries) + 1), timeseries)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Best Objective Value")
        ax.set_xlabel("# Config Evaluations")
        ax.set_ylabel("Objective")
        fig.tight_layout()

        # Save the figure if a filename was provided
        if fname is not None:
            fig.savefig(fname, dpi=300)
        else:
            return fig, ax

    @property
    def df(self):
        """Return log as pandas dataframe."""
        flat_log = []
        for l in self.log:
            sub_log = {}
            sub_log["eval_id"] = l["eval_id"]
            # TODO: Add if-clause for multi-objective list case
            # TODO: Add extra data stored in log
            sub_log["objective"] = l["objective"]
            sub_log.update(l["params"])
            if "extra" in l.keys():
                sub_log.update(l["extra"])
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
        self,
        batch_proposals: List[dict],
        perf_measures: List[Union[float, int]],
        ckpt_paths: Union[None, List[str], str] = None,
    ):
        """Print strategy update."""
        best_eval_id, best_config, best_eval, best_ckpt = self.get_best(top_k=1)
        if not self.maximize_objective:
            best_batch_idx = np.argmin(perf_measures)
        else:
            best_batch_idx = np.argmax(perf_measures)

        best_batch_eval_id = self.eval_counter - len(perf_measures) + best_batch_idx
        best_batch_config, best_batch_eval = (
            batch_proposals[best_batch_idx],
            perf_measures[best_batch_idx],
        )
        if ckpt_paths is not None:
            best_batch_ckpt = ckpt_paths[best_batch_idx]
        else:
            best_batch_ckpt = None
        # Print best data in log - and best data in last batch
        update_message(
            self.eval_counter,
            best_eval_id,
            best_config,
            best_eval,
            best_ckpt,
            best_batch_eval_id,
            best_batch_config,
            best_batch_eval,
            best_batch_ckpt,
        )

    def refine_space(self, top_k: int):
        """Search method-specific search space update."""
        raise NotImplementedError

    def refine(self, top_k: int):
        """Refine the space boundaries based on top-k performers."""
        assert self.search_name in ["Random", "SMBO", "Nevergrad"]
        top_idx, top_k_configs, top_k_evals, _ = self.get_best(top_k)
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
        else:
            integer_refined = None

        self.refine_space(real_refined, integer_refined, categorical_refined)
        if self.verbose:
            self.print_hello(f"{self.eval_counter} Evals - Top {top_k} - Refined")

    def get_pareto_front(self):
        """Get pareto front for multi-objective problems."""
        raise NotImplementedError
