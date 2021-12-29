from mle_hyperopt import (
    RandomSearch,
    HalvingSearch,
    HyperbandSearch,
)
import numpy as np


def get_iteration_score(
    epoch: int, seed_id: int, lrate: float, batch_size: int, arch: str, **kwargs
) -> (float, float):
    """Surrogate Objective w. optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = (lrate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if arch == "conv" else 10)
    train_loss = f1 + seed_id * 0.5
    test_loss = f1 + seed_id * 0.5 + np.random.uniform(0, 0.3)
    return train_loss / epoch, test_loss / epoch


def test_store_ckpt():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        maximize_objective=True,
    )
    configs = [{"lrate": 0.2}, {"lrate": 0.3}, {"lrate": 0.4}]
    values = [0.25, 0.35, 0.45]
    ckpts = ["ckpt1.pt", "ckpt2.pt", "ckpt3.pt"]
    strategy.tell(configs, values, ckpts)
    id, conf, val, ck = strategy.get_best(3)
    assert ck[0] == "ckpt3.pt"
    assert ck[1] == "ckpt2.pt"
    assert ck[2] == "ckpt1.pt"


def test_successive_halving():
    strategy = HalvingSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "log-uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={"min_budget": 1, "num_arms": 20, "halving_coeff": 2},
        seed_id=42,
    )
    assert strategy.num_sh_batches == 5
    assert strategy.evals_per_batch == [20, 10, 5, 2, 1]
    assert strategy.iters_per_batch == [1, 2, 4, 8, 16]

    configs = strategy.ask()
    assert len(configs) == 20
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_0_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)

    configs = strategy.ask()
    assert len(configs) == 10
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_1_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)


def test_hyperband():
    strategy = HyperbandSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "log-uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={"max_resource": 81, "eta": 3},
        seed_id=42,
    )

    assert strategy.sh_num_arms == [81, 27, 9, 6, 5]
    assert strategy.sh_budgets == [1, 3, 9, 27, 81]

    configs = strategy.ask()
    assert len(configs) == 81
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_0_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)

    configs = strategy.ask()
    assert len(configs) == 27
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_1_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)

    configs = strategy.ask()
    assert len(configs) == 9
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_2_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)

    configs = strategy.ask()
    assert len(configs) == 3
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_3_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)

    configs = strategy.ask()
    assert type(configs) == dict
    configs = [configs]
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_4_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)

    configs = strategy.ask()
    assert len(configs) == 27
    scores = [
        get_iteration_score(c["extra"]["sh_num_total_iters"], 0, **c["params"])[1]
        for c in configs
    ]
    ckpts = [f"ckpt_5_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)
