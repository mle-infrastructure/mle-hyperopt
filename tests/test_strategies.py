from mle_hyperopt import (
    RandomSearch,
    GridSearch,
    SMBOSearch,
    NevergradSearch,
    CoordinateSearch,
)


def fake_train(lrate, batch_size, arch):
    """Optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = (lrate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if arch == "conv" else 10)
    return f1


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
        integer={"batch_size": {"begin": 1, "end": 5, "spacing": 1}},
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
        integer={"batch_size": {"begin": 1, "end": 5, "spacing": 1}},
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
