from mle_hyperopt import RandomSearch, NevergradSearch


def multi_fake_train(lrate, batch_size, arch):
    # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
    f1 = (lrate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if arch == "conv" else 10)
    # optimal for learning_rate=0.3, batch_size=2, architecture="mlp"
    f2 = (lrate - 0.3) ** 2 + (batch_size - 2) ** 2 + (0 if arch == "mlp" else 5)
    return f1, f2


def test_multi_random():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        verbose=1,
    )
    configs = strategy.ask(5)
    values = [multi_fake_train(**c) for c in configs]
    strategy.tell(configs, values)


def test_multi_nevergrad():
    strategy = NevergradSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={"optimizer": "NGOpt", "budget_size": 100, "num_workers": 5},
        verbose=1,
    )
    configs = strategy.ask(5)
    values = [multi_fake_train(**c) for c in configs]
    strategy.tell(configs, values)
