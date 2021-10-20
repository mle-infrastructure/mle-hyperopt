from datetime import datetime
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box
from rich.align import Align

console_width = 80


def welcome_message(space_data, search_type: str):
    """Print startup configuration of search space."""
    console = Console(width=console_width)
    table = Table(show_footer=False)
    table.add_column(":sunflower: Variable", no_wrap=True)
    table.add_column("Type")
    table.add_column("Search Range :left_right_arrow:")
    table.title = "MLE-Hyperopt " + search_type + " Hyperspace :rocket:"
    for row in space_data:
        table.add_row(*list(row.values()))
    table.columns[2].justify = "left"
    table.columns[2].header_style = "bold red"
    table.columns[2].style = "red"
    table.row_styles = ["none"]
    table.box = box.SIMPLE
    console.print(Align.center(table))


def update_message(
    total_eval_id,
    best_eval_id,
    best_config,
    best_eval,
    best_batch_eval_id,
    best_batch_config,
    best_batch_eval,
):
    """Print current best performing configurations."""
    time_t = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    console = Console(width=console_width)
    table = Table(show_header=True)
    table.add_column(f":inbox_tray: Total: {total_eval_id}", style="dim")
    table.add_column("ID")
    table.add_column("Obj. :chart_with_downwards_trend:")
    table.add_column(f"Configuration :bookmark: - {time_t}")
    # Round all the values for prettier printing
    best_eval = round(best_eval, 3)
    best_batch_eval = round(best_batch_eval, 3)
    for k, v in best_config.items():
        if type(v) == float:
            best_config[k] = round(v, 3)
    for k, v in best_batch_config.items():
        if type(v) == float:
            best_batch_config[k] = round(v, 3)
    table.add_row(
        "Best Overall", str(best_eval_id), str(best_eval), str(best_config)[1:-1]
    )
    table.add_row(
        "Best in Batch",
        str(best_batch_eval_id),
        str(best_batch_eval),
        str(best_batch_config)[1:-1],
    )
    console.print(Align.center(table))


def ranking_message(best_eval_ids, best_configs, best_evals):
    """Print top-k performing configurations."""
    # Ensure that update data is list to loop over
    if type(best_eval_ids) in [int, np.int64]:
        best_eval_ids = [best_eval_ids]
    if type(best_configs) == dict:
        best_configs = [best_configs]
    if type(best_evals) in [float, int]:
        best_evals = [best_evals]

    console = Console()
    table = Table(show_header=True)
    table.add_column(f":1st_place_medal: Rank", style="dim")
    table.add_column("ID")
    table.add_column("Obj. :chart_with_downwards_trend:")
    table.add_column("Configuration :bookmark:")
    for i in range(len(best_configs)):
        # Round all the values for prettier printing
        if type(best_evals[i]) == np.ndarray:
            best_evals[i] = best_evals[i].tolist()
            best_eval = [round(best_evals[i][j], 3) for j in range(len(best_evals[i]))]
        else:
            best_eval = round(best_evals[i], 3)
        for k, v in best_configs[i].items():
            if type(v) == float:
                best_configs[i][k] = round(v, 3)
        table.add_row(
            f"{i+1}", str(best_eval_ids[i]), str(best_eval), str(best_configs[i])[1:-1]
        )
    console.print(Align.center(table))
