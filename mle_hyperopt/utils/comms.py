from datetime import datetime
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box
from rich.align import Align
from typing import List, Optional, Union

console_width = 80


def welcome_message(
    space_data: List[dict],
    search_type: str,
    fixed_params: Optional[dict] = None,
) -> None:
    """Print startup configuration of search space.

    Args:
        space_data (List[dict]): List of search variable descriptions.
        search_type (str): Name of search strategy
        fixed_params (Optional[dict], optional):
            Fixed parameter names and values. Defaults to None.
    """
    console = Console(width=console_width)
    table = Table(show_footer=False)
    table.add_column(":sunflower: Variable", no_wrap=True)
    table.add_column("Type")
    table.add_column("Search Range :left_right_arrow:")
    table.title = "MLE-Hyperopt " + search_type + " Hyperspace :rocket:"
    for row in space_data:
        table.add_row(*list(row.values()))

    if fixed_params is not None:
        for k, v in fixed_params.items():
            table.add_row(k, "fixed", str(v))
    table.columns[2].justify = "left"
    table.columns[2].header_style = "bold red"
    table.columns[2].style = "red"
    table.row_styles = ["none"]
    table.box = box.SIMPLE
    console.print(Align.center(table))


def update_message(
    total_eval_id: int,
    best_eval_id: List[int],
    best_config: List[dict],
    best_eval: List[Union[float, np.ndarray]],
    best_ckpt: Optional[List[str]],
    best_batch_eval_id: List[int],
    best_batch_config: List[dict],
    best_batch_eval: List[Union[float, np.ndarray]],
    best_batch_ckpt: Optional[List[str]],
) -> None:
    """Print current best performing configurations.

    Args:
        total_eval_id (int): Number of total evaluations so far.
        best_eval_id (List[int]): ID of top-k performing evaluations.
        best_config (List[dict]): Top-k performing parameter configurations.
        best_eval (List[float, np.ndarray]): Top-k performance values.
        best_ckpt (Optional[List[str]]): Top-k checkpoint paths.
        best_batch_eval_id (List[int]): Top-k performing evaluations in batch.
        best_batch_config (List[dict]): Top-k performing configurations in batch.
        best_batch_eval (List[float, np.ndarray]):
            Top-k performance values in batch.
        best_batch_ckpt (Optional[List[str]]): Top-k checkpoint paths in batch.
    """
    time_t = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    console = Console(width=console_width)
    table = Table(show_header=True)
    table.add_column(f":inbox_tray: Total: {total_eval_id}", style="dim")
    table.add_column("ID")
    table.add_column("Obj. :chart_with_downwards_trend:")
    table.add_column(f"Configuration :bookmark: - {time_t}")
    print()
    for i in range(len(best_eval_id)):
        best_e = np.round_(best_eval[i], 3)
        for k, v in best_config[i].items():
            if type(v) == float:
                best_config[i][k] = np.round_(v, 3)
        best_c = dict(best_config[i])
        if best_ckpt is not None:
            best_c["ckpt"] = best_ckpt[i]
        table.add_row(
            "Best Overall", str(best_eval_id[i]), str(best_e), str(best_c)[1:-1]
        )

    # Add row(s) for best config(s) in batch
    for i in range(len(best_eval_id)):
        best_batch_e = np.round_(best_batch_eval[i], 3)
        for k, v in best_batch_config[i].items():
            if type(v) == float:
                best_batch_config[i][k] = np.round_(v, 3)
        best_b_c = dict(best_batch_config[i])
        if best_batch_ckpt is not None:
            best_b_c["ckpt"] = best_batch_ckpt[i]
        table.add_row(
            "Best in Batch",
            str(best_batch_eval_id[i]),
            str(best_batch_e),
            str(best_b_c)[1:-1],
        )
    console.print(Align.center(table))


def ranking_message(
    best_eval_ids: List[int],
    best_configs: List[dict],
    best_evals: List[Union[float, np.ndarray]],
) -> None:
    """Print top-k performing configurations.

    Args:
        best_eval_ids (List[int]): ID of top-k performing evaluations.
        best_configs (List[dict]): Top-k performing parameter configurations.
        best_evals (List[float, np.ndarray]): Top-k performance values.
    """
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
            best_eval = [
                round(best_evals[i][j], 3) for j in range(len(best_evals[i]))
            ]
        else:
            best_eval = round(best_evals[i], 3)
        for k, v in best_configs[i].items():
            if type(v) == float:
                best_configs[i][k] = round(v, 3)
        table.add_row(
            f"{i+1}",
            str(best_eval_ids[i]),
            str(best_eval),
            str(best_configs[i])[1:-1],
        )
    console.print(Align.center(table))


def print_grid_hello(num_total_configs: int, num_dims_grid: int) -> None:
    """Hello message specific to grid search.

    Args:
        num_total_configs (int): Number of total configurations in grid.
        num_dims_grid (int): Number of variables to search over.
    """
    console = Console(width=console_width)
    console.log(
        f"Start running {num_dims_grid}D grid with "
        f"{num_total_configs} total configurations."
    )


def print_halving_hello(
    num_sh_batches: int,
    evals_per_batch: List[int],
    iters_per_batch: List[int],
    halving_coeff: int,
    num_total_iters: int,
) -> None:
    """Hello message specific to SH search.

    Args:
        num_sh_batches (int): Total number of SH batches.
        evals_per_batch (List[int]): List of number of evaluations per batch.
        iters_per_batch (List[int]): List of number of iterations per batch.
        halving_coeff (int): Halving coefficient.
        num_total_iters (int): Number of total evaluations at the end of search.
    """
    console = Console(width=console_width)
    console.log(
        f"Start running {num_sh_batches} batches of Successive Halving."
    )
    console.log(f"➞ Configurations per batch: {evals_per_batch}")
    console.log(f"➞ Iterations per batch: {iters_per_batch}")
    console.log(f"➞ Halving coefficient: {halving_coeff}")
    console.log(f"➞ Total Number of Iterations: {num_total_iters}")
    console.log(
        f"➞ Batch No. 1/{num_sh_batches}: {evals_per_batch[0]} configs for"
        f" {iters_per_batch[0]} iters."
    )
    return


def print_halving_update(
    sh_counter: int,
    num_sh_batches: int,
    evals_per_batch: List[int],
    iters_per_batch: List[int],
    num_total_iters: int,
) -> None:
    """Update message specific to SH search.

    Args:
        sh_counter (int): Number of completed SH batches.
        num_sh_batches (int): Total number of SH batches.
        evals_per_batch (List[int]): List of number of evaluations per batch.
        iters_per_batch (List[int]): List of number of iterations per batch.
        num_total_iters (int): Number of total evaluations at the end of search.
    """
    console = Console(width=console_width)
    done_iters = np.sum(
        np.array(evals_per_batch)[:sh_counter]
        * np.array(iters_per_batch)[:sh_counter]
    )
    console.log(
        f"Completed {sh_counter}/{num_sh_batches} batches of SH ➢"
        f" {done_iters}/{num_total_iters} iters."
    )
    if sh_counter < num_sh_batches:
        console.log(
            f"➞ Next - Batch No. {sh_counter+1}/{num_sh_batches}:"
            f" {evals_per_batch[sh_counter]} configs for"
            f" {iters_per_batch[sh_counter]} iters."
        )


def print_hyperband_hello(
    num_hb_loops: int,
    sh_num_arms: List[int],
    sh_budgets: List[int],
    num_hb_batches: int,
    evals_per_batch: List[int],
) -> None:
    """Hello message specific to Hyperband search.

    Args:
        num_hb_loops (int): Number of total SH loops in hyperband.
        sh_num_arms (List[int]): List of active bandit arms in all SH loops.
        sh_budgets (List[int]): List of iteration budgets in all SH loops.
        num_hb_batches (int): Number of total job batches in hyperband search.
        evals_per_batch (List[int]): List of number of jobs in all batches.
    """
    console = Console(width=console_width)
    console.log(
        f"Start running {num_hb_batches} batches of Hyperband evaluations."
    )
    console.log(f"➞ Evals per batch: {evals_per_batch}")
    console.log(
        f"➞ Total SH loops: {num_hb_loops} | Arms per loop: {sh_num_arms}"
    )
    console.log(f"➞ Min. budget per loop: {sh_budgets}")
    console.log(
        f"➞ Start Loop No. 1/{num_hb_loops}: {sh_num_arms[0]} arms &"
        f" {sh_budgets[0]} min budget."
    )


def print_hyperband_update(
    hb_counter: int,
    num_hb_loops: int,
    sh_num_arms: List[int],
    sh_budgets: List[int],
    num_hb_batches: int,
    hb_batch_counter: int,
    evals_per_batch: List[int],
) -> None:
    """Update message specific to Hyperband search.

    Args:
        hb_counter (int): Number of completed SH loops in hyperband.
        num_hb_loops (int): Number of total SH loops in hyperband.
        sh_num_arms (List[int]): List of active bandit arms in all SH loops.
        sh_budgets (List[int]): List of iteration budgets in all SH loops.
        num_hb_batches (int): Number of total job batches in hyperband search.
        hb_batch_counter (int): Number of completed job batches.
        evals_per_batch (List[int]): List of number of jobs in all batches.
    """
    console = Console(width=console_width)
    console.log(
        f"Completed {hb_batch_counter}/{num_hb_batches} of Hyperband evaluation"
        " batches."
    )
    console.log(f"➞ Done with {hb_counter}/{num_hb_loops} loops of SH.")
    if hb_counter < num_hb_loops:
        console.log(
            f"➞ Active Loop No. {hb_counter + 1}/{num_hb_loops}:"
            f" {sh_num_arms[hb_counter]} arms & {sh_budgets[hb_counter]} min"
            " budget."
        )
        console.log(
            f"➞ Next batch of SH: {evals_per_batch[hb_batch_counter]} evals."
        )


def print_pbt_hello(
    num_workers: int,
    steps_until_ready: int,
    explore_type: str,
    exploit_type: str,
) -> None:
    """Hello message specific to PBT search.

    Args:
        num_workers (int): Number of synchronous PBT workers.
        steps_until_ready (int): Number of (SGD) steps between PBT iterations.
        explore_type (str): Exploration strategy name.
        exploit_type (str): Exploitation strategy name.
    """
    console = Console(width=console_width)
    console.log(f"Start running PBT w. {num_workers} workers.")
    console.log(f"➞ Steps until ready: {steps_until_ready}")
    console.log(f"➞ Exploration strategy: {explore_type}")
    console.log(f"➞ Exploitation strategy: {exploit_type}")


def print_pbt_update(
    step_counter: int, num_total_steps: int, copy_info: dict
) -> None:
    """Update message specific to PBT search.

    Args:
        step_counter (int): Number of completed PBT batches.
        num_total_steps (int): Number of total steps (e.g. SGD intervals).
        copy_info (dict): Info about which worker exploited/explored.
    """
    console = Console(width=console_width)
    console.log(f"Completed {step_counter} batches of PBT.")
    console.log(f"➞ Number of total steps: {num_total_steps}")
    for w_id in range(len(copy_info)):
        if w_id != copy_info[w_id]["copy_id"]:
            console.log(
                f"➞ 👨‍🚒 W{w_id} (P:"
                f" {round(copy_info[w_id]['old_performance'], 3)}) exploits"
                f" W{copy_info[w_id]['copy_id']} (P:"
                f" {round(copy_info[w_id]['copy_performance'], 3)})"
            )
            console.log(f"-- E/E Params: {copy_info[w_id]['copy_params']}")
        else:
            console.log(
                f"➞ 👨‍🚒 W{w_id} (P:"
                f" {round(copy_info[w_id]['old_performance'], 3)}) continues"
                " own trajectory."
            )
            console.log(f"-- Old Params: {copy_info[w_id]['copy_params']}")
