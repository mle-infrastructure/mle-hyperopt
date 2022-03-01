import argparse
import os
import sys
import importlib
from dotmap import DotMap
from .strategies import Strategies
from .utils import load_yaml


def get_search_args() -> None:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exec_fname",
        metavar="C",
        type=str,
        default="main.py",
        help="Filename to import `main(config)` function from.",
    )
    parser.add_argument(
        "-base",
        "--base_config",
        type=str,
        default=None,
        help="Filename to load base configuration from.",
    )
    parser.add_argument(
        "-search",
        "--search_config",
        type=str,
        default="search.yaml",
        help="Filename to load search configuration from.",
    )
    parser.add_argument(
        "-iters",
        "--num_iters",
        type=int,
        default=None,
        help="Number of desired search iterations.",
    )
    parser.add_argument(
        "-log",
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save search_log.yaml in.",
    )
    parser.add_argument(
        "-reload",
        "--reload_log",
        default=False,
        action="store_true",
        help="Attempt to reload log file.",
    )
    args = parser.parse_args()
    return args


def search() -> None:
    """Command line tool for running a sequential search given a python script
    `<script>.py` containing a function `main(config)`, a default configuration
    file `<base>.yaml` & a search configuration `<search>.yaml`. The `main`
    function should return a single scalar performance score.
    You can then start the search via:

        mle-search <script>.py
            --base_config <base>.yaml
            --search_config <search>.yaml
            --num_iters <num_iters>
            --log_dir <log_dir>

    Or short:

        mle-search <script>.py -base <base>.yaml -search <search>.yaml
            -iters <num_iters >-log <log_dir>

    This will spawn single runs for different configurations and walk through a
    set of search iterations.
    """
    args = get_search_args()

    # Setup log storage path & effective search iterations
    save_path = (
        os.path.join(args.log_dir, "search_log.yaml")
        if args.log_dir is not None
        else "search_log.yaml"
    )

    # Load base configuration and search configuration
    search_config = load_yaml(args.search_config, False)
    base_config = load_yaml(args.base_config, False)

    # Allow `mle-search` to work with experiment configuration
    # Note: Will still require a main(config) call to exec run
    if "param_search_args" in search_config.keys():
        conf_temp = dict(search_config["param_search_args"]["search_config"])
        conf_temp["maximize_objective"] = search_config["param_search_args"][
            "search_logging"
        ]["max_objective"]
        if "search_config" not in conf_temp.keys():
            conf_temp["search_config"] = {}
        search_config = conf_temp

    if "train_config" in base_config.keys():
        base_config = base_config["train_config"]

    num_search_iters = (
        args.num_iters
        if args.num_iters is not None
        else search_config["num_iters"]
    )

    if args.reload_log:
        reload_path = save_path
    else:
        reload_path = None
    strategy = Strategies[search_config["search_type"]](
        **search_config["search_params"],
        search_config=search_config["search_config"],
        maximize_objective=search_config["maximize_objective"],
        fixed_params=base_config,
        reload_path=reload_path,
        verbose=True,
    )

    # Append path for correct imports & load the main function module
    sys.path.append(os.getcwd())
    spec = importlib.util.spec_from_file_location(
        "main", os.path.join(os.getcwd(), args.exec_fname)
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    # Run the search loop and store results to path
    for s_iter in range(strategy.eval_counter, num_search_iters, 1):
        config = strategy.ask()
        # Add search id for logging inside main call
        config["search_eval_id"] = (
            search_config["search_type"].lower() + f"_{s_iter}"
        )
        result = foo.main(DotMap(config))
        del config["search_eval_id"]
        strategy.tell(config, result, save=True, save_path=save_path)
