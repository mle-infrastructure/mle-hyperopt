## [v0.0.5] - [Unreleased]

### Added
- Adds `SuccessiveHalvingSearch`, `HyperbandSearch`, `PBTSearch` and examples/NB documentation (see issues #3, #4).
- Adds possibility to store and reload entire strategies as pkl file (as asked for in issue #2).

### Changed
- Major internal restructuring:
  - `clean_data`: Get rid of extra data provided in configuration file
  - `tell_search`: Update model of search strategy (e.g. SMBO/Nevergrad)
  - `log_search`: Add search specific log data to evaluation log
  - `update_search`: Refine search space/change active strategy etc.
- Also allow to store checkpoint of trained models in `tell` method.

### Fixed

## [v0.0.4] - 12/10/2021

### Fixed
- Bug Fix Data Types & internal refactor 🔺 for internal talk.

## [v0.0.3] - 10/24/2021

### Added
- Adds rich logging to all console print statements.
- Updates documentation and adds text to `getting_started.ipynb`.

### Changed
- Generalizes `NevergradSearch` to wrap around all search strategies.

### Fixed
- Fixes `CoordinateSearch` active grid search dimension updating. We have to account for the fact that previous coordinates are not evaluated again after switching the active variable.

## [v0.0.2] - 10/20/2021

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


## [v0.0.1] - 10/16/2021

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
