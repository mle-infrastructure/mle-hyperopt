from mle_hyperopt import hyperopt


def test_decorator():
    @hyperopt(
        strategy_type="Grid",
        num_search_iters=25,
        real={
            "x": {"begin": 0.0, "end": 0.5, "bins": 5},
            "y": {"begin": 0, "end": 0.5, "bins": 5},
        },
    )
    def circle(config):
        distance = abs((config["x"] ** 2 + config["y"] ** 2))
        return distance

    strategy = circle()
    assert len(strategy) == 25
