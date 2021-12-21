from mle_hyperopt import (
    RandomSearch,
    PBTSearch,
)


def fake_train(lrate, batch_size, arch):
    """Optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = (lrate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if arch == "conv" else 10)
    return f1


def test_store_ckpt():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        maximize_objective=True,
    )
    configs = [{"lrate": 0.2}, {"lrate": 0.3}, {"lrate": 0.4}]
    values = [0.25, 0.35, 0.45]
    ckpts = ["ckpt1.pt", "ckpt2.pt", "ckpt3.pt"]
    strategy.tell(configs, values, ckpts)
    id, conf, val, ck = strategy.get_best(3, return_ckpt=True)
    assert ck[0] == "ckpt3.pt"
    assert ck[1] == "ckpt2.pt"
    assert ck[2] == "ckpt1.pt"


def test_pbt():
    strategy = PBTSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={"noise_scale": 0.1, "truncation_selection": 0.2},
    )
    # configs = strategy.ask(5)
    # values = [fake_train(**c) for c in configs]
    # strategy.tell(configs, values)
    return
