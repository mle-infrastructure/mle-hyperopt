from mle_hyperopt import (
    RandomSearch,
    PBTSearch,
    SuccessiveHalvingSearch,
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


class QuadraticProblem(object):
    def __init__(self, theta=None, lrate: float = 0.1):
        self.theta = theta
        if self.theta is None:
            self.theta = np.array([[0.9, 0.9]])
        self.lrate = lrate

    def step(self, h):
        """Perform GradAscent step on quadratic surrogate objective (max!)."""
        surrogate_grad = -2.0 * h * self.theta
        self.theta += self.lrate * surrogate_grad

    def evaluate(self):
        """Ground truth objective (e.g. val loss) - Jaderberg et al. 2016."""
        return 1.2 - np.sum(self.theta ** 2)

    def surrogate_objective(self, h):
        """Surrogate objective (with hyperparams h) - Jaderberg et al. 2016."""
        return 1.2 - np.sum(h * self.theta ** 2)

    def __call__(self, hyperparams):
        h = np.array([hyperparams["h0"], hyperparams["h1"]])
        self.step(h)
        exact = self.evaluate()
        surrogate = self.surrogate_objective(h)
        return exact, surrogate


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
        real={
            "h0": {"begin": 0.0, "end": 1.0, "prior": "uniform"},
            "h1": {"begin": 0.0, "end": 1.0, "prior": "uniform"},
        },
        search_config={
            "exploit_config": {"strategy": "truncation", "truncation_selection": 0.2},
            "explore_config": {"strategy": "additive-noise", "noise_scale": 0.1},
        },
    )
    configs = strategy.ask(2)
    values = []
    theta = [[np.array([[0.9, 0.9]]), np.array([[0.9, 0.9]])]]
    new_theta = []
    for i in range(2):
        problem = QuadraticProblem(theta[-1][i])
        exact, surrogate = problem(configs[i])
        values.append(exact)
        new_theta.append(problem.theta)
    theta.append(new_theta)

    ckpts = ["ckpt1.pt", "ckpt2.pt"]
    # strategy.tell(configs, values, ckpts)
    return


def test_successive_halving():
    strategy = SuccessiveHalvingSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "log-uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        search_config={"budget": 100, "num_arms": 20},
        seed_id=42,
        verbose=True,
    )
    assert strategy.num_batches == 5
    assert strategy.evals_per_batch == [20, 10, 5, 2, 1]
    assert strategy.iters_per_batch == [1, 2, 4, 10, 20]

    configs = strategy.ask()
    assert len(configs) == 20
    scores = [get_iteration_score(c["num_total_sh_iters"], 0, **c)[1] for c in configs]
    ckpts = [f"ckpt_0_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)

    configs = strategy.ask()
    assert len(configs) == 10
    scores = [get_iteration_score(c["num_total_sh_iters"], 0, **c)[1] for c in configs]
    ckpts = [f"ckpt_1_{i}.pt" for i in range(len(configs))]
    strategy.tell(configs, scores, ckpts)


# def test_hyperband():
#     return
