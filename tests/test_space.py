from mle_hyperopt.spaces import (
    RandomSpace,
    GridSpace,
    SMBOSpace,
    NevergradSpace,
)


def test_random_space():
    real = {"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}}
    integer = {"batch_size": {"begin": 1, "end": 5, "prior": "log-uniform"}}
    categorical = {"arch": ["mlp", "cnn"]}
    space = RandomSpace(real, integer, categorical)
    return


def test_grid_space():
    real = {"lrate": {"begin": 0.1, "end": 0.5, "bins": 5}}
    integer = {"batch_size": {"begin": 1, "end": 5, "spacing": 1}}
    categorical = {"arch": ["mlp", "cnn"]}
    space = GridSpace(real, integer, categorical)
    return


def test_smbo_space():
    real = {"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}}
    integer = {"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}}
    space = SMBOSpace(real, integer)
    return


def test_nevergrad_space():
    real = {"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}}
    integer = {"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}}
    space = NevergradSpace(real, integer)
    return
