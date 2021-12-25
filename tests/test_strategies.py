from mle_hyperopt import (
    RandomSearch,
    GridSearch,
    SMBOSearch,
    NevergradSearch,
    CoordinateSearch,
)
import os
import numpy as np


def fake_train(lrate, batch_size, arch):
    """Optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = (lrate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if arch == "conv" else 10)
    return f1


def test_core():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)

    # Check internal logging of tell results
    assert len(strategy) == 5
    assert strategy.log[-1]["objective"] == values[-1]
    assert strategy.log[-1]["params"] == configs[-1]

    # Check that configs are correctly stored
    configs = strategy.ask(2, store=True)
    assert os.path.exists("eval_5.yaml")
    assert os.path.exists("eval_6.yaml")
    os.remove("eval_5.yaml")
    os.remove("eval_6.yaml")

    configs = strategy.ask(2, store=True, config_fnames=["conf_0.yaml", "conf_1.yaml"])
    assert os.path.exists("conf_0.yaml")
    assert os.path.exists("conf_1.yaml")
    os.remove("conf_0.yaml")
    os.remove("conf_1.yaml")

    # Check that get_best retrieves best performers
    best_id, best_config, best_value, _ = strategy.get_best(2)
    assert (best_id == [4, 3]).all()
    assert (np.array(best_value) == [10.005996236277564, 11.00112114105261]).all()
    assert best_config[0] == {
        "arch": "cnn",
        "lrate": 0.12256463161084011,
        "batch_size": 4,
    }
    assert best_config[1] == {
        "arch": "cnn",
        "lrate": 0.23348344445560876,
        "batch_size": 3,
    }


def test_maximize_objective():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        maximize_objective=True,
    )
    configs = [{"lrate": 0.2}, {"lrate": 0.3}, {"lrate": 0.4}]
    values = [0.25, 0.35, 0.45]
    strategy.tell(configs, values)
    id, conf, val, _ = strategy.get_best()
    assert id == 2
    assert conf == {"lrate": 0.4}
    assert val == 0.45


def test_fixed_params():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        fixed_params={"momentum": 0.9},
    )
    configs = strategy.ask(5)
    for c in configs:
        assert c["momentum"] == 0.9


def test_random():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    return


def test_grid():
    strategy = GridSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "bins": 5}},
        integer={"batch_size": {"begin": 1, "end": 5, "bins": 1}},
        categorical={"arch": ["mlp", "cnn"]},
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    return


def test_smbo():
    strategy = SMBOSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={
            "base_estimator": "GP",
            "acq_function": "gp_hedge",
            "n_initial_points": 5,
        },
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    return


def test_nevergrad():
    strategy = NevergradSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={"optimizer": "NGOpt", "budget_size": 100, "num_workers": 5},
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    return


def test_coordinate():
    strategy = CoordinateSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "bins": 5}},
        integer={"batch_size": {"begin": 1, "end": 5, "bins": 2}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={
            "order": ["lrate", "batch_size", "arch"],
            "defaults": {"lrate": 0.1, "batch_size": 3, "arch": "mlp"},
        },
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    return
