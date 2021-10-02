import os
from mle_hyperopt import GridSearch


def fake_train(lrate, batch_size, arch):
    """ Optimum: lrate=0.2, batch_size=4, arch='conv'."""
    f1 = ((lrate - 0.2) ** 2 + (batch_size - 4) ** 2
          + (0 if arch == "conv" else 10))
    return f1


def test_save():
    strategy = GridSearch(real={"lrate": {"begin": 0.1,
                                      "end": 0.5,
                                      "bins": 5}},
                      integer={"batch_size": {"begin": 1,
                                              "end": 5,
                                              "spacing": 1}},
                      categorical={"arch": ["mlp", "cnn"]})
    configs = strategy.ask(batch_size=2)
    values = [fake_train(**c) for c in configs]
    strategy.tell(configs, values)
    strategy.save("search_log.pkl")
    assert os.path.exists("search_log.pkl")
    os.remove("search_log.pkl")
