import numpy as np


def construct_hyperparam_range(params_to_search: dict, search_type: str) -> dict:
    """Helper to generate list of hyperparam ranges from YAML dictionary."""
    param_range = {}
    # For grid hyperopt generate numpy lists with desired resolution
    if search_type == "grid":
        if "categorical" in params_to_search.keys():
            for k, v in params_to_search["categorical"].items():
                param_range[k] = v
        if "real" in params_to_search.keys():
            for k, v in params_to_search["real"].items():
                param_range[k] = np.linspace(
                    float(v["begin"]), float(v["end"]), int(v["bins"])
                ).tolist()
        if "integer" in params_to_search.keys():
            for k, v in params_to_search["integer"].items():
                param_range[k] = np.arange(
                    int(v["begin"]), int(v["end"]), int(v["spacing"])
                ).tolist()

    # For random hyperopt generate list/1d-line range to sample from
    elif search_type == "random":
        if "categorical" in params_to_search.keys():
            for k, v in params_to_search["categorical"].items():
                param_range[k] = {"value_type": "categorical", "values": v}
        if "real" in params_to_search.keys():
            for k, v in params_to_search["real"].items():
                param_range[k] = {
                    "value_type": "real",
                    "values": [float(v["begin"]), float(v["end"])],
                }
        if "integer" in params_to_search.keys():
            for k, v in params_to_search["integer"].items():
                param_range[k] = {
                    "value_type": "integer",
                    "values": np.arange(int(v["begin"]), int(v["end"]), 1).tolist(),
                }

    # For SMBO-based hyperopt generate spaces with skopt classes
    elif search_type == "smbo":
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

        if "categorical" in params_to_search.keys():
            for k, v in params_to_search["categorical"].items():
                param_range[k] = Categorical(v, name=k)
        if "real" in params_to_search.keys():
            for k, v in params_to_search["real"].items():
                param_range[k] = Real(
                    float(v["begin"]), float(v["end"]), prior=v["prior"], name=k
                )
        if "integer" in params_to_search.keys():
            for k, v in params_to_search["integer"].items():
                param_range[k] = Integer(
                    int(v["begin"]), int(v["end"]), prior=v["prior"], name=k
                )

    # For Nevergrad-based hyperopt generate spaces with parametrization
    elif search_type == "nevergrad":
        try:
            import nevergrad as ng
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                f"{err}. You need to"
                "install `nevergrad` to use "
                "the `mle_toolbox.hyperopt.nevergrad` module."
            )
        param_dict = {}
        if "categorical" in params_to_search.keys():
            for k, v in params_to_search["categorical"].items():
                param_dict[k] = ng.p.Choice(v)
        if "real" in params_to_search.keys():
            for k, v in params_to_search["real"].items():
                if v["prior"] == "uniform":
                    param_dict[k] = ng.p.Scalar(
                        lower=float(v["begin"]), upper=float(v["end"])
                    )
                elif v["prior"] == "log-uniform":
                    param_dict[k] = ng.p.Log(
                        lower=float(v["begin"]), upper=float(v["end"])
                    )
        if "integer" in params_to_search.keys():
            for k, v in params_to_search["integer"].items():
                if v["prior"] == "uniform":
                    param_dict[k] = ng.p.Scalar(
                        lower=float(v["begin"]), upper=float(v["end"])
                    ).set_integer_casting()
                elif v["prior"] == "log-uniform":
                    param_dict[k] = ng.p.Log(
                        lower=float(v["begin"]), upper=float(v["end"])
                    ).set_integer_casting()
        param_range = ng.p.Instrumentation(**param_dict)
    else:
        raise ValueError("Please provide a valid hyperparam search type.")
    return param_range
