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
    candidate_pos = space.sample()
    assert space.contains(candidate_pos)
    candidate_neg = {"lrate": 0.8, "integer": 7, "arch": "bert"}
    assert not space.contains(candidate_neg)


def test_grid_space():
    real = {"lrate": {"begin": 0.1, "end": 0.5, "bins": 5}}
    integer = {"batch_size": {"begin": 1, "end": 5, "bins": 1}}
    categorical = {"arch": ["mlp", "cnn"]}
    space = GridSpace(real, integer, categorical)
    candidate_pos = space.sample(grid_counter=0)
    assert space.contains(candidate_pos)
    candidate_neg = {"lrate": 0.8, "batch_size": 7, "arch": "bert"}
    assert not space.contains(candidate_neg)


def test_smbo_space():
    real = {"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}}
    integer = {"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}}
    categorical = {"arch": ["mlp", "cnn"]}
    space = SMBOSpace(real, integer, categorical)
    candidate_pos = {"lrate": 0.3, "batch_size": 5, "arch": "mlp"}
    assert space.contains(candidate_pos)
    candidate_neg = {"lrate": 0.8, "batch_size": 7, "arch": "bert"}
    assert not space.contains(candidate_neg)


def test_nevergrad_space():
    real = {"lrate": {"begin": 0.1, "end": 0.5, "prior": "uniform"}}
    integer = {"batch_size": {"begin": 1, "end": 5, "prior": "uniform"}}
    categorical = {"arch": ["mlp", "cnn"]}
    space = NevergradSpace(real, integer, categorical)
    candidate_pos = {"lrate": 0.3, "batch_size": 5, "arch": "mlp"}
    assert space.contains(candidate_pos)
    candidate_neg = {"lrate": 0.8, "batch_size": 7, "arch": "bert"}
    assert not space.contains(candidate_neg)
