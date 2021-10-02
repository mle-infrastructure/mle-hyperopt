import numpy as np


def grid_space(real, integer, categorical) -> dict:
    """ For grid hyperopt generate numpy lists with desired resolution"""
    param_range = {}
    if categorical is not None:
        for k, v in categorical.items():
            param_range[k] = v
    if real is not None:
        for k, v in real.items():
            param_range[k] = np.linspace(
                float(v["begin"]), float(v["end"]), int(v["bins"])
            ).tolist()
    if integer is not None:
        for k, v in integer.items():
            param_range[k] = np.arange(
                int(v["begin"]), int(v["end"]), int(v["spacing"])
            ).tolist()
    return param_range


def random_space(real, integer, categorical) -> dict:
    """For random hyperopt generate list/1d-line range to sample from."""
    param_range = {}
    if categorical is not None:
        for k, v in categorical.items():
            param_range[k] = {"value_type": "categorical", "values": v}
    if real is not None:
        for k, v in real.items():
            param_range[k] = {
                "value_type": "real",
                "values": [float(v["begin"]), float(v["end"])],
            }
    if integer is not None:
        for k, v in integer.items():
            param_range[k] = {
                "value_type": "integer",
                "values": np.arange(int(v["begin"]), int(v["end"]), 1).tolist(),
            }
    return param_range


def smbo_space(real, integer, categorical) -> dict:
    """For SMBO-based hyperopt generate spaces with skopt classes"""
    param_range = {}
    # Can specify prior distribution over hyperp. distrib
    # log-uniform samples more from the lower tail of the hyperparam range
    #   real: ["uniform", "log-uniform"]
    #   integer: ["uniform", "log-uniform"]
    try:
        from skopt.space import Real, Integer, Categorical
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to"
            "install `scikit-optimize` to use "
            "the `mle_toolbox.hyperopt` module."
        )

    if categorical is not None:
        for k, v in categorical.items():
            param_range[k] = Categorical(v, name=k)
    if real is not None:
        for k, v in real.items():
            param_range[k] = Real(
                float(v["begin"]), float(v["end"]), prior=v["prior"], name=k
            )
    if integer is not None:
        for k, v in integer.items():
            param_range[k] = Integer(
                int(v["begin"]), int(v["end"]), prior=v["prior"], name=k
            )
    return param_range


def nevergrad_space(real, integer, categorical) -> dict:
    """ For Nevergrad-based hyperopt generate spaces with parametrization"""
    param_range = {}
    try:
        import nevergrad as ng
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to"
            "install `nevergrad` to use "
            "the `mle_toolbox.hyperopt.nevergrad` module."
        )
    param_dict = {}
    if categorical is not None:
        for k, v in categorical.items():
            param_dict[k] = ng.p.Choice(v)
    if real is not None:
        for k, v in real.items():
            if v["prior"] == "uniform":
                param_dict[k] = ng.p.Scalar(
                    lower=float(v["begin"]), upper=float(v["end"])
                )
            elif v["prior"] == "log-uniform":
                param_dict[k] = ng.p.Log(
                    lower=float(v["begin"]), upper=float(v["end"])
                )
    if integer is not None:
        for k, v in integer.items():
            if v["prior"] == "uniform":
                param_dict[k] = ng.p.Scalar(
                    lower=float(v["begin"]), upper=float(v["end"])
                ).set_integer_casting()
            elif v["prior"] == "log-uniform":
                param_dict[k] = ng.p.Log(
                    lower=float(v["begin"]), upper=float(v["end"])
                ).set_integer_casting()
    param_range = ng.p.Instrumentation(**param_dict)
    return param_range
