import argparse
import os
import importlib
from mle_logging import load_config
from .strategies import Strategies


def get_search_args() -> None:
    """Get command line arguments."""
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
        default="base.yaml",
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
    args = parser.parse_args()
    return args


def search():
    """Command line tool for running a sequential search given a python script
    `<script>.py` containing a function `main(config)`, a default configuration
    file `<base>.yaml` & a search configuration `<search>.yaml`. The `main`
    function should return a single scalar performance score.
    You can then start the search via:

        mle-search <script>.py --base_config <base>.yaml --search_config <search>.yaml

    Or short:

        mle-search <script>.py -base <base>.yaml -search <search>.yaml

    This will spawn single runs for different configurations and walk through a
    set of search iterations.
    """
    args = get_search_args()

    # Load base configuration and search configuration
    search_config = load_config(args.search_config, True)
    base_config = load_config(args.base_config, True)

    # Setup search instance
    real = (
        search_config.search_config.real
        if "real" in search_config.search_config.keys()
        else None
    )
    integer = (
        search_config.search_config.integer
        if "integer" in search_config.search_config.keys()
        else None
    )
    categorical = (
        search_config.search_config.categorical
        if "categorical" in search_config.search_config.keys()
        else None
    )

    strategy = Strategies[search_config.search_type](
        real,
        integer,
        categorical,
        search_config.search_config,
        search_config.maximize_objective,
        fixed_params=base_config.toDict(),
        verbose=search_config.verbose,
    )

    # Load the main function module
    spec = importlib.util.spec_from_file_location(
        "main", os.path.join(os.getcwd(), args.exec_fname)
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    num_search_iters = (
        args.num_iters
        if args.num_iters is not None
        else search_config.num_iters
    )
    for _ in range(num_search_iters):
        config = strategy.ask()
        result = foo.main(config)
        strategy.tell(config, result, save=True)
