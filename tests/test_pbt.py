from mle_hyperopt import PBTSearch
import numpy as np


class QuadraticProblem(object):
    def __init__(self, theta=None, lrate: float = 0.1):
        self.lrate = lrate

    def step(self, h, theta):
        """Perform GradAscent step on quadratic surrogate objective (max!)."""
        surrogate_grad = -2.0 * h * theta
        return theta + self.lrate * surrogate_grad

    def evaluate(self, theta):
        """Ground truth objective (e.g. val loss) - Jaderberg et al. 2016."""
        return 1.2 - np.sum(theta ** 2)

    def surrogate_objective(self, h, theta):
        """Surrogate objective (with hyperparams h) - Jaderberg et al. 2016."""
        return 1.2 - np.sum(h * theta ** 2)

    def __call__(self, theta, hyperparams):
        h = np.array([hyperparams["h0"], hyperparams["h1"]])
        theta = self.step(h, theta)
        exact = self.evaluate(theta)
        # surrogate = self.surrogate_objective(h, theta)
        return theta.tolist(), exact


def test_pbt():
    strategy = PBTSearch(
        real={
            "h0": {"begin": 0.0, "end": 1.0, "prior": "uniform"},
            "h1": {"begin": 0.0, "end": 1.0, "prior": "uniform"},
        },
        search_config={
            "exploit": {"strategy": "truncation", "selection_percent": 0.2},
            "explore": {"strategy": "additive-noise", "noise_scale": 0.35},
            "steps_until_ready": 4,
            "num_workers": 2,
        },
    )
    configs = strategy.ask()

    problem = QuadraticProblem()
    ckpts = [[0.9, 0.9], [0.9, 0.9]]
    value = [
        problem(ckpts[0], {"h0": 0.9, "h1": 0.9})[1],
        problem(ckpts[1], {"h0": 0.9, "h1": 0.9})[1],
    ]

    theta_log = [ckpts]
    value_log = [value]
    steps_until_ready = 4

    for s in range(steps_until_ready):
        theta_l, value_l = [], []
        for i in range(len(configs)):
            theta, exact = problem(theta_log[-1][i], configs[i]["params"])
            theta_l.append(theta)
            value_l.append(exact)
        value_log.append(value_l)
        theta_log.append(theta_l)

    ckpts = theta_log[-1]
    values = value_log[-1]
    strategy.tell(configs, values, ckpts)

    configs = strategy.ask()
    for k in [
        "pbt_num_total_iters",
        "pbt_num_add_iters",
        "pbt_worker_id",
        "pbt_step_counter",
        "pbt_explore",
        "pbt_copy_id",
        "pbt_old_params",
        "pbt_copy_params",
        "pbt_old_performance",
        "pbt_copy_performance",
        "pbt_ckpt",
    ]:
        assert k in configs[0]["extra"].keys()
