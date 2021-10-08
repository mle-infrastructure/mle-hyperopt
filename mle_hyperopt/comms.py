from typing import Union
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


def welcome_message(space_data):
    for l in space_data:
        print(l)
    return


def update_message(best_eval, best_config, best_batch_eval, best_batch_config):
    print("Best overall", best_eval, best_config)
    print("Best in batch", best_batch_eval, best_batch_config)


def ranking_message(best_configs, best_evals):
    # Ensure that update data is list to loop over
    if type(best_configs) == dict:
        best_configs = [best_configs]
    if type(best_evals) in [float, int]:
        best_evals = [best_evals]
    for i in range(len(best_configs)):
        print(f"Rank {i+1}", best_evals[i], best_configs[i])
