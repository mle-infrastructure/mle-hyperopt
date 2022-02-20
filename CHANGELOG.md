## [v0.0.6] - [02/19/2022]
### Added

- Adds a command line interface for running a sequential search given a python script `<script>.py` containing a function `main(config)`, a default configuration file `<base>.yaml` & a search configuration `<search>.yaml`. The `main` function should return a single scalar performance score. You can then start the search via:

  ```
  mle-search <script>.py --base_config <base>.yaml --search_config <search>.yaml --num_iters <search_iters>
  ```
  Or short via:
  ```
  mle-search <script>.py -base <base>.yaml -search <search>.yaml -iters <search_iters>
  ```
- Adds doc-strings to all functionalities.
### Changed

- Make it possible to optimize parameters in nested dictionaries. Added helpers `flatten_config` and `unflatten_config`. For shaping `'sub1/sub2/vname' <-> {sub1: {sub2: {vname: v}}}`
- Make start-up message also print fixed parameter settings.
- Cleaned up decorator with the help of `Strategies` wrapper.

## [v0.0.5] - [01/05/2022]

### Added

- Adds possibility to store and reload entire strategies as pkl file (as asked for in issue #2).
- Adds `improvement` method indicating if score is better than best stored one
- Adds save option for best plot
- Adds `args, kwargs` into decorator
- Adds synchronous Successive Halving (`SuccessiveHalvingSearch` - issue #3)
- Adds synchronous HyperBand (`HyperbandSearch` - issue #3)
- Adds synchronous PBT (`PBTSearch` - issue #4)
- Adds option to save log in `tell` method
- Adds small torch mlp example for SH/Hyperband/PBT w. logging/scheduler
- Adds print welcome/update message for strategy specific info

### Changed
- Major internal restructuring:
  - `clean_data`: Get rid of extra data provided in configuration file
  - `tell_search`: Update model of search strategy (e.g. SMBO/Nevergrad)
  - `log_search`: Add search specific log data to evaluation log
  - `update_search`: Refine search space/change active strategy etc.
- Also allow to store checkpoint of trained models in `tell` method.
- Fix logging message when log is stored
- Make json serializer more robust for numpy data types
- Robust type checking with `isinstance(self.log[0]["objective"], (float, int, np.integer, np.float))`
- Update NB to include `mle-scheduler` example
- Make PBT explore robust for integer/categorical valued hyperparams
- Calculate total batches & their sizes for hyperband

## [v0.0.4] - [12/10/2021]

### Fixed
- Bug Fix Data Types & internal refactor 🔺 for internal talk.

## [v0.0.3] - [10/24/2021]

### Added
- Adds rich logging to all console print statements.
- Updates documentation and adds text to `getting_started.ipynb`.

### Changed
- Generalizes `NevergradSearch` to wrap around all search strategies.

### Fixed
- Fixes `CoordinateSearch` active grid search dimension updating. We have to account for the fact that previous coordinates are not evaluated again after switching the active variable.

## [v0.0.2] - [10/20/2021]

### Added
- Adds search space refinement for nevergrad and smbo search strategies via `refine_after` and `refine_top_k`:

```python
strategy = SMBOSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={
            "base_estimator": "GP",
            "acq_function": "gp_hedge",
            "n_initial_points": 5,
            "refine_after": 5,
            "refine_top_k": 2,
        },
        seed_id=42,
        verbose=True
    )
```
- Adds additional strategy boolean option `maximize_objective` to maximize instead of performing default black-box minimization.

### Changed
- Enhances documentation and test coverage.

### Fixed
- Fixes import bug when using PyPi installation.


## [v0.0.1] - [10/16/2021]

### Added
- Base API implementation:

```python
from mle_hyperopt import RandomSearch

# Instantiate random search class
strategy = RandomSearch(real={"lrate": {"begin": 0.1,
                                        "end": 0.5,
                                        "prior": "log-uniform"}},
                        integer={"batch_size": {"begin": 32,
                                                "end": 128,
                                                "prior": "uniform"}},
                        categorical={"arch": ["mlp", "cnn"]})

# Simple ask - eval - tell API
configs = strategy.ask(5)
values = [train_network(**c) for c in configs]
strategy.tell(configs, values)
```

- Different search strategies: Grid, random, SMBO, base nevergrad
