# Lightweight Hyperparameter Optimization 🚂
[![Pyversions](https://img.shields.io/pypi/pyversions/mle-hyperopt.svg?style=flat-square)](https://pypi.python.org/pypi/mle-hyperopt)
[![PyPI version](https://badge.fury.io/py/mle-hyperopt.svg)](https://badge.fury.io/py/mle-hyperopt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/mle-infrastructure/mle-hyperopt/branch/main/graph/badge.svg)](https://codecov.io/gh/mle-infrastructure/mle-hyperopt)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mle-infrastructure/mle-hyperopt/blob/main/examples/getting_started.ipynb)
<a href="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/logo_transparent.png?raw=true"><img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/logo_transparent.png?raw=true" width="200" align="right" /></a>

The `mle-hyperopt` package provides a simple and intuitive API for hyperparameter optimization of your Machine Learning Experiment (MLE) pipeline. It supports real, integer & categorical search variables and single- or multi-objective optimization.

Core features include the following:

- **API Simplicity**: `strategy.ask()`, `strategy.tell()` interface & space definition.
- **Strategy Diversity**: Grid, random, coordinate search, SMBO & wrapping FAIR's [`nevergrad`](https://facebookresearch.github.io/nevergrad/), Successive Halving, Hyperband, Population-Based Training.
- **Search Space Refinement** based on the top performing configs via `strategy.refine(top_k=10)`.
- **Export of configurations** to execute via e.g. `python train.py --config_fname config.yaml`.
- **Storage & reload search logs** via `strategy.save(<log_fname>)`,  `strategy.load(<log_fname>)`.

For a quickstart check out the [notebook blog](https://github.com/mle-infrastructure/mle-hyperopt/blob/main/examples/getting_started.ipynb) 📖.

## The API 🎮

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

### Implemented Search Types 	🔭

| | Search Type           | Description | `search_config` |
|----|----------------------- | ----------- | --------------- |
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/grid.png?raw=true" alt="drawing" width="65"/>|  `GridSearch`          |  Search over list of discrete values  | - |
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/random.png?raw=true" alt="drawing" width="65"/>|  `RandomSearch`        |  Random search over variable ranges         | `refine_after`, `refine_top_k` |
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/coordinate.png?raw=true" alt="drawing" width="65"/>|  `CoordinateSearch`    |  Coordinate-wise optimization with fixed defaults | `order`, `defaults`
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/smbo.png?raw=true" alt="drawing" width="65"/>|  `SMBOSearch`          |  Sequential model-based optimization [(Hutter et al., 2011)](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/11-LION5-SMAC.pdf)   | `base_estimator`, `acq_function`, `n_initial_points`
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/nevergrad.png?raw=true" alt="drawing" width="65"/>|  `NevergradSearch`     |  Multi-objective [nevergrad](https://facebookresearch.github.io/nevergrad/) wrapper | `optimizer`, `budget_size`, `num_workers`
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/halving.png?raw=true" alt="drawing" width="65"/>|  `HalvingSearch`     | Successive Halving [(Karmin et al., 2013)](https://proceedings.mlr.press/v28/karnin13.html) | `min_budget`, `num_arms`, `halving_coeff`
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/hyperband.png?raw=true" alt="drawing" width="65"/>|  `HyperbandSearch`     | Hyperband [(Li et al., 2018)](https://arxiv.org/pdf/1603.06560.pdf) | `max_resource`, `eta`
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/pbt.png?raw=true" alt="drawing" width="65"/>|  `PBTSearch`     | Population-Based Training [(Jaderberg et al., 2017)](https://arxiv.org/pdf/1711.09846.pdf) | `explore`, `exploit`

### Variable Types & Hyperparameter Spaces 🌍

| | Variable            | Type | Space Specification |
| --- |----------------------- | ----------- | --------------- |
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/real.png?raw=true" alt="drawing" width="65"/> |  **`real`**          |  Real-valued  | `Dict`: `begin`, `end`, `prior`/`bins` (grid) |
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/integer.png?raw=true" alt="drawing" width="65"/>  |  **`integer`**        |  Integer-valued         | `Dict`: `begin`, `end`, `prior`/`bins` (grid) |
|<img src="https://github.com/mle-infrastructure/mle-hyperopt/blob/main/docs/categorical.png?raw=true" alt="drawing" width="65"/> |  **`categorical`**  |  Categorical        | `List`: Values to search over


## Installation ⏳

A PyPI installation is available via:

```
pip install mle-hyperopt
```

Alternatively, you can clone this repository and afterwards 'manually' install it:

```
git clone https://github.com/mle-infrastructure/mle-hyperopt.git
cd mle-hyperopt
pip install -e .
```

## Search Method Highlights 🔎

### Grid Search 🟥

```python
strategy = GridSearch(
    real={"lrate": {"begin": 0.1,
                    "end": 0.5,
                    "bins": 5}},
    integer={"batch_size": {"begin": 1,
                            "end": 5,
                            "bins": 1}},
    categorical={"arch": ["mlp", "cnn"]},
    fixed_params={"momentum": 0.9})  # Add fixed param setting to each config

configs = strategy.ask()
```

### Hyperband 🎸

```python
strategy = HyperbandSearch(
    real={"lrate": {"begin": 0.1,
                    "end": 0.5,
                    "prior": "uniform"}},
    integer={"batch_size": {"begin": 1,
                            "end": 5,
                            "prior": "log-uniform"}},
    categorical={"arch": ["mlp", "cnn"]},
    search_config={"max_resource": 81,
                   "eta": 3},
    seed_id=42,  # Fix randomness for reproducibility
    verbose=True)

configs = strategy.ask()
```

### Population-Based Training 🦎

```python
strategy = PBTSearch(
    real={"lrate": {"begin": 0.1,
                    "end": 0.5,
                    "prior": "uniform"}}
    search_config={
        "exploit": {"strategy": "truncation", "selection_percent": 0.2},
        "explore": {"strategy": "perturbation", "perturb_coeffs": [0.8, 1.2]},
        "steps_until_ready": 4,
        "num_workers": 10,
    },
    maximize_objective=True  # Max score instead of min
)

configs = strategy.ask()
```

## Further Options 🚴

### Saving & Reloading Logs 🏪

```python
# Storing & reloading of results from .json/.yaml/.pkl
strategy.save("search_log.json")
strategy = RandomSearch(..., reload_path="search_log.json")

# Or manually add info after class instantiation
strategy = RandomSearch(...)
strategy.load("search_log.json")
```

### Search Decorator 🧶

```python
from mle_hyperopt import hyperopt

@hyperopt(strategy_type="Grid",
          num_search_iters=25,
          real={"x": {"begin": 0., "end": 0.5, "bins": 5},
                "y": {"begin": 0, "end": 0.5, "bins": 5}})
def circle(config):
    distance = abs((config["x"] ** 2 + config["y"] ** 2))
    return distance

strategy = circle()
```

### Storing Configuration Files 📑


```python
# Store 2 proposed configurations - eval_0.yaml, eval_1.yaml
strategy.ask(2, store=True)
# Store with explicit configuration filenames - conf_0.yaml, conf_1.yaml
strategy.ask(2, store=True, config_fnames=["conf_0.yaml", "conf_1.yaml"])
```

### Storing Checkpoint Paths 🛥️


```python
# Ask for 5 configurations to evaluate and get their scores
configs = strategy.ask(5)
values = ...
# Get list of checkpoint paths corresponding to config runs
ckpts = [f"ckpt_{i}.pt" for i in range(len(configs))]
# `tell` parameter configs, eval scores & ckpt paths
# Required for Halving, Hyperband and PBT
strategy.tell(configs, scores, ckpts)
```

### Retrieving Top Performers & Visualizing Results 📉

```python
# Get the top k best performing configurations
id, configs, values = strategy.get_best(top_k=4)

# Plot timeseries of best performing score over search iterations
strategy.plot_best()

# Print out ranking of best performers
strategy.print_ranking(top_k=3)
```

### Refining the Search Space of Your Strategy 🪓

```python
# Refine the search space after 5 & 10 iterations based on top 2 configurations
strategy = RandomSearch(real={"lrate": {"begin": 0.1,
                                        "end": 0.5,
                                        "prior": "log-uniform"}},
                        integer={"batch_size": {"begin": 1,
                                                "end": 5,
                                                "prior": "uniform"}},
                        categorical={"arch": ["mlp", "cnn"]},
                        search_config={"refine_after": [5, 10],
                                       "refine_top_k": 2})

# Or do so manually using `refine` method
strategy.tell(...)
strategy.refine(top_k=2)
```

Note that the search space refinement is only implemented for random, SMBO and `nevergrad`-based search strategies.


### Citing the MLE-Infrastructure ✏️

If you use `mle-hyperopt` in your research, please cite it as follows:

```
@software{mle_infrastructure2021github,
  author = {Robert Tjarko Lange},
  title = {{MLE-Infrastructure}: A Set of Lightweight Tools for Distributed Machine Learning Experimentation},
  url = {http://github.com/mle-infrastructure},
  year = {2021},
}
```

## Development 👷

You can run the test suite via `python -m pytest -vv tests/`. If you find a bug or are missing your favourite feature, feel free to create an issue and/or start [contributing](CONTRIBUTING.md) :hugs:.
