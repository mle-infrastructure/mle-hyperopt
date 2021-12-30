from datetime import datetime
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box
from rich.align import Align
from typing import List

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
    best_ckpt,
    best_batch_eval_id,
    best_batch_config,
    best_batch_eval,
    best_batch_ckpt,
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
    best_c = dict(best_config)
    if best_ckpt is not None:
        best_c["ckpt"] = best_ckpt
    table.add_row("Best Overall", str(best_eval_id), str(best_eval), str(best_c)[1:-1])
    best_b_c = dict(best_batch_config)
    if best_batch_ckpt is not None:
        best_b_c["ckpt"] = best_batch_ckpt
    table.add_row(
        "Best in Batch",
        str(best_batch_eval_id),
        str(best_batch_eval),
        str(best_b_c)[1:-1],
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

    console = Console(width=console_width)
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


def print_grid_hello(num_total_configs: int, num_dims_grid: int):
    """Hello message specific to grid search."""
    console = Console(width=console_width)
    console.log(
        f"Start running {num_total_configs} configuration for {num_dims_grid}D grid."
    )


def print_halving_hello(
    num_sh_batches: int,
    evals_per_batch: List[int],
    iters_per_batch: List[int],
    halving_coeff: int,
    num_total_iters: int,
):
    """Hello message specific to SH search."""
    console = Console(width=console_width)
    console.log(f"Start running {num_sh_batches} batches of Successive Halving.")
    console.log(f"➞ Configurations per batch: {evals_per_batch}")
    console.log(f"➞ Iterations per batch: {iters_per_batch}")
    console.log(f"➞ Halving coefficient: {halving_coeff}")
    console.log(f"➞ Total Number of Iterations: {num_total_iters}")
    console.log(
        f"➞ Batch No. 1/{num_sh_batches}: {evals_per_batch[0]} configs for {iters_per_batch[0]} iters."
    )
    return


def print_halving_update(
    sh_counter: int,
    num_sh_batches: int,
    evals_per_batch: List[int],
    iters_per_batch: List[int],
    num_total_iters: int,
):
    """Update message specific to SH search."""
    console = Console(width=console_width)
    done_iters = np.sum(
        np.array(evals_per_batch)[:sh_counter] * np.array(iters_per_batch)[:sh_counter]
    )
    console.log(
        f"Completed {sh_counter}/{num_sh_batches} batches of SH ➢ {done_iters}/{num_total_iters} iters."
    )
    if sh_counter < num_sh_batches:
        console.log(
            f"➞ Next - Batch No. {sh_counter+1}/{num_sh_batches}: {evals_per_batch[sh_counter]} configs for {iters_per_batch[sh_counter]} iters."
        )


def print_hyperband_hello(
    num_hb_loops: int,
    sh_num_arms: List[int],
    sh_budgets: List[int],
    num_hb_batches: int,
    evals_per_batch: List[int],
):
    """Hello message specific to Hyperband search."""
    console = Console(width=console_width)
    console.log(f"Start running {num_hb_batches} batches of Hyperband evaluations.")
    console.log(f"➞ Evals per batch: {evals_per_batch}")
    console.log(f"➞ Total SH loops: {num_hb_loops} | Arms per loop: {sh_num_arms}")
    console.log(f"➞ Min. budget per loop: {sh_budgets}")
    console.log(
        f"➞ Start Loop No. 1/{num_hb_loops}: {sh_num_arms[0]} arms & {sh_budgets[0]} min budget."
    )


def print_hyperband_update(
    hb_counter: int,
    num_hb_loops: int,
    sh_num_arms: List[int],
    sh_budgets: List[int],
    num_hb_batches: int,
    hb_batch_counter: int,
    evals_per_batch: List[int],
):
    """Update message specific to Hyperband search."""
    console = Console(width=console_width)
    console.log(
        f"Completed {hb_batch_counter}/{num_hb_batches} of Hyperband evaluation batches."
    )
    console.log(f"➞ Done with {hb_counter}/{num_hb_loops} loops of SH.")
    if hb_counter < num_hb_loops:
        console.log(
            f"➞ Active Loop No. {hb_counter + 1}/{num_hb_loops}: {sh_num_arms[hb_counter]} arms & {sh_budgets[hb_counter]} min budget."
        )
        console.log(f"➞ Next batch of SH: {evals_per_batch[hb_batch_counter]} evals.")


def print_pbt_hello(
    num_workers: int, steps_until_ready: int, explore_type: str, exploit_type: str
):
    """Hello message specific to PBT search."""
    console = Console(width=console_width)
    console.log(f"Start running PBT w. {num_workers} workers.")
    console.log(f"➞ Steps until ready: {steps_until_ready}")
    console.log(f"➞ Exploration strategy: {explore_type}")
    console.log(f"➞ Exploitation strategy: {exploit_type}")


def print_pbt_update(step_counter: int, num_total_steps: int, copy_info: dict):
    """Update message specific to PBT search."""
    console = Console(width=console_width)
    console.log(f"Completed {step_counter} batches of PBT.")
    console.log(f"➞ Number of total steps: {num_total_steps}")
    for w_id in range(len(copy_info)):
        if w_id != copy_info[w_id]["copy_id"]:
            console.log(
                f"➞ 👨‍🚒 W{w_id} (P: {round(copy_info[w_id]['old_performance'], 3)}) exploits W{copy_info[w_id]['copy_id']} (P: {round(copy_info[w_id]['copy_performance'], 3)})"
            )
            console.log(f"-- E/E Params: {copy_info[w_id]['copy_params']}")
        else:
            console.log(
                f"➞ 👨‍🚒 W{w_id} (P: {round(copy_info[w_id]['old_performance'], 3)}) continues own trajectory."
            )
            console.log(f"-- Old Params: {copy_info[w_id]['copy_params']}")
