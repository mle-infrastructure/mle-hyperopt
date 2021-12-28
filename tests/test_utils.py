from mle_hyperopt import RandomSearch
from mle_hyperopt import hyperopt
import os


def fake_train(lrate, batch_size, arch):
    """Optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = (lrate - 0.2) ** 2 + (batch_size - 4) ** 2 + (0 if arch == "conv" else 10)
    return f1


def test_comms():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        verbose=True
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    strategy.print_ranking()


def test_best_plot():
    strategy = RandomSearch(
        real={"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}},
        integer={"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}},
        categorical={"arch": ["mlp", "cnn"]},
        verbose=True
    )
    configs = strategy.ask(5)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)

    strategy.plot_best("best_plot.png")
    assert os.path.exists("best_plot.png")
    os.remove("best_plot.png")


def test_plot_grid():
    @hyperopt(strategy_type="Grid",
              num_search_iters=400,
              real={"x": {"begin": -0.5, "end": 0.5, "bins": 20},
                    "y": {"begin": -0.5, "end": 0.5, "bins": 20}})
    def circle_objective(config):
        distance = abs((config["x"] ** 2 + config["y"] ** 2))
        return distance

    strategy = circle_objective()
    strategy.plot_grid(params_to_plot=["x", "y"],
                       target_to_plot="objective",
                       plot_title="Quadratics for Life",
                       plot_subtitle="How beautiful can they be?",
                       xy_labels= ["x", "y"],
                       variable_name="Objective Name",
                       every_nth_tick=3,
                       fname="grid_plot.png")
    assert os.path.exists("grid_plot.png")
    os.remove("grid_plot.png")
