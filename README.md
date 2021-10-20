# Lightweight Hyperparameter Optimization 🚀
[![Pyversions](https://img.shields.io/pypi/pyversions/mle-hyperopt.svg?style=flat-square)](https://pypi.python.org/pypi/mle-hyperopt)
[![PyPI version](https://badge.fury.io/py/mle-hyperopt.svg)](https://badge.fury.io/py/mle-hyperopt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobertTLange/mle-hyperopt/blob/main/examples/getting_started.ipynb)
<a href="docs/logo_transparent.png_2"><img src="docs/logo_transparent.png" width="200" align="right" /></a>

The `mle-hyperopt` package provides a simple and intuitive API for hyperparameter optimization of your Machine Learning Experiment (MLE) pipeline. It supports real, integer & categorical search variables and single- or multi-objective optimization.

Core features include the following:

- **API Simplicity**: `strategy.ask()`, `strategy.tell()` interface & space definition.
- **Strategy Diversity**: Grid, random, coordinate search, SMBO & wrapping around FAIR's nevergrad.
- **Search Space Refinement** based on the top performing configs via `strategy.refine(top_k=10)`.
- **Export of configurations** to execute via e.g. `python train.py --config_fname config.yaml`.
- **Storage & reload search logs** via `strategy.save(<log_fname>)`,  `strategy.load(<log_fname>)`.

For a quickstart check out the [notebook blog](https://github.com/RobertTLange/mle-hyperopt/blob/main/examples/getting_started.ipynb) 📖.

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
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/grid.png?raw=true" alt="drawing" width="65"/>|  `GridSearch`          |  Search over list of discrete values  | - |
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/random.png?raw=true" alt="drawing" width="65"/>|  `RandomSearch`        |  Random search over variable ranges         | `refine_after`, `refine_top_k` |
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/coordinate.png?raw=true" alt="drawing" width="65"/>|  `CoordinateSearch`    |  Coordinate-wise optimization with fixed defaults | `order`, `defaults`
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/smbo.png?raw=true" alt="drawing" width="65"/>|  `SMBOSearch`          |  Sequential model-based optimization        | `base_estimator`, `acq_function`, `n_initial_points`
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/nevergrad.png?raw=true" alt="drawing" width="65"/>|  `NevergradSearch`     |  Multi-objective [nevergrad](https://facebookresearch.github.io/nevergrad/) wrapper | `optimizer`, `budget_size`, `num_workers`

### Variable Types & Hyperparameter Spaces 🌍

| | Variable            | Type | Space Specification |
| --- |----------------------- | ----------- | --------------- |
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/real.png?raw=true" alt="drawing" width="65"/> |  **`real`**          |  Real-valued  | `Dict`: `begin`, `end`, `prior`/`bins` (grid) |
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/integer.png?raw=true" alt="drawing" width="65"/>  |  **`integer`**        |  Integer-valued         | `Dict`: `begin`, `end`, `prior`/`bins` (grid) |
|<img src="https://github.com/RobertTLange/mle-hyperopt/blob/main/docs/categorical.png?raw=true" alt="drawing" width="65"/> |  **`categorical`**  |  Categorical        | `List`: Values to search over


## Installation ⏳

A PyPI installation is available via:

```
pip install mle-hyperopt
```

Alternatively, you can clone this repository and afterwards 'manually' install it:

```
git clone https://github.com/RobertTLange/mle-hyperopt.git
cd mle-hyperopt
pip install -e .
```

## Further Options 🚴

### Saving & Reloading Logs 🏪

```python
# Storing & reloading of results from .pkl
strategy.save("search_log.json")
strategy = RandomSearch(..., reload_path="search_log.json")

# Or manually add info after class instantiation
strategy = RandomSearch(...)
strategy.load("search_log.json")
```

### Search Decorator 🧶

```python
from mle_hyperopt import hyperopt

@hyperopt(strategy_type="grid",
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
                                        "prior": "uniform"}},
                        integer={"batch_size": {"begin": 1,
                                                "end": 5,
                                                "prior": "log-uniform"}},
                        categorical={"arch": ["mlp", "cnn"]},
                        search_config={"refine_after": [5, 10],
                                       "refine_top_k": 2})

# Or do so manually using `refine` method
strategy.tell(...)
strategy.refine(top_k=2)
```

Note the search space refinement is only implemented for random, SMBO and nevergrad-based search strategies.

## Development & Milestones for Next Release

You can run the test suite via `python -m pytest -vv tests/`. If you find a bug or are missing your favourite feature, feel free to contact me [@RobertTLange](https://twitter.com/RobertTLange) or create an issue :hugs:. Here are some features I want to implement for the next release:

- [ ] Add text to notebook for what is implemented
- [ ] Update Readme text
- [ ] Update mle-toolbox webpage intro
- [ ] Release and make sure installation works
- [ ] Draft tweet for release
- [ ] Synergies with mle-logging
