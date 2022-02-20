from mle_hyperopt import (
    RandomSearch,
    SMBOSearch,
    NevergradSearch,
)


def fake_train(lrate, batch_size, arch):
    """Optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = (
        (lrate - 0.2) ** 2
        + (batch_size - 4) ** 2
        + (0 if arch == "conv" else 10)
    )
    return f1


def test_refinement_random():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "log-uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={"refine_after": 5, "refine_top_k": 2},
        seed_id=42,
    )

    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    assert strategy.last_refined == 5
    assert strategy.space.bounds["lrate"][1] == 0.12256463161084011
    assert strategy.space.bounds["lrate"][2] == 0.34044600469728353
    assert strategy.space.bounds["batch_size"][1] == 3
    assert strategy.space.bounds["batch_size"][2] == 3
    assert strategy.space.bounds["arch"][1] in ["mlp", "cnn"]
    assert strategy.space.bounds["arch"][2] in ["mlp", "cnn"]


def test_refinement_smbo():
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
    )

    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    assert strategy.last_refined == 5
    assert strategy.space.bounds["lrate"][1] == 0.3680591793075739
    assert strategy.space.bounds["lrate"][2] == 0.39580169367616824
    assert strategy.space.bounds["batch_size"][1] == 3
    assert strategy.space.bounds["batch_size"][2] == 3
    assert strategy.space.bounds["arch"][1] in ["mlp", "cnn"]


def test_refinement_nevergrad():
    strategy = NevergradSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={
            "optimizer": "NGOpt",
            "budget_size": 100,
            "num_workers": 5,
            "refine_after": 5,
            "refine_top_k": 2,
        },
        seed_id=42,
    )

    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    assert strategy.last_refined == 5
    # assert strategy.space.bounds["lrate"][1] == 0.30915641385130954
    # assert strategy.space.bounds["lrate"][2] == 0.34261384229290115
    # assert strategy.space.bounds["batch_size"][1] == 4
    # assert strategy.space.bounds["batch_size"][2] == 4
    # assert strategy.space.bounds["arch"][1] in ["mlp", "cnn"]
    # assert strategy.space.bounds["arch"][2] in ["mlp", "cnn"]
