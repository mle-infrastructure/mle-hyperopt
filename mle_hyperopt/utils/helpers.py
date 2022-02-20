import pickle
import pickle5
from typing import List
import os
import json
import yaml
import ast
import numpy as np
import collections


def convert(obj):
    """Conversion helper instead of JSON encoder for handling booleans."""
    if isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [convert(item) for item in obj]
    if isinstance(obj, dict):
        return {convert(key): convert(value) for key, value in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return convert(obj.tolist())
    if isinstance(obj, np.bool_):
        return int(obj)
    return obj


def save_json(obj, filename: str) -> None:
    """Save object as json file."""
    with open(filename, "w") as fout:
        json.dump(convert(obj), fout, indent=1)


def save_yaml(obj, filename: str) -> None:
    """Save object as yaml file."""
    # Case 1: Save list of logged data
    if type(obj) == list:
        data = {}
        for i in range(len(obj)):
            e_id = obj[i]["eval_id"]
            data[f"{e_id}"] = obj[i]
        data_dump = json.dumps(convert(data), indent=1)
        with open(filename, "w") as f:
            yaml.safe_dump(json.loads(data_dump), f, default_flow_style=False)
    # Case 2: Save configuration to file
    else:
        data = json.dumps(convert(obj), indent=1)
        data_dump = ast.literal_eval(data)
        with open(filename, "w") as f:
            yaml.safe_dump(data_dump, f, default_flow_style=False)


def save_log(obj: List[dict], filename: str) -> None:
    """Save log results as json or yaml files.

    Args:
        obj (List[dict]): Logger from strategy.
        filename (str): File path to store log to.

    Raises:
        ValueError: Make sure that filename has correct extension.
    """
    fname, fext = os.path.splitext(filename)
    if fext == ".json":
        # Write config dictionary to json file
        save_json(obj, filename)
    elif fext == ".yaml":
        save_yaml(obj, filename)
    else:
        raise ValueError("Log can only be stored as YAML & JSON files.")


def load_strategy(filename: str):
    """Helper to reload pickle objects.

    Args:
        filename (str): Filename to reload strategy from.

    Returns:
        _type_: Instantiated strategy with previous results.
    """
    with open(filename, "rb") as input:
        obj = pickle5.load(input)
    return obj


def save_strategy(obj, filename: str) -> None:
    """Helper to store pickle objects."""
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_log(filename: str) -> List[dict]:
    """Load JSON/YAML config depending on file ending.

    Args:
        filename (str): Filename to load log results from.

    Raises:
        ValueError: Make sure that filename has correct extension.

    Returns:
        List[dict]: Return list of dictionaries with search results.
    """
    _, fext = os.path.splitext(filename)
    if fext == ".yaml":
        log_results = load_yaml(filename)
    elif fext == ".json":
        log_results = load_json(filename)
    else:
        raise ValueError("Only YAML & JSON configuration can be loaded.")
    return log_results


def load_yaml(filename: str) -> List[dict]:
    """Load in YAML file.

    Args:
        filename (str): YAML filename to load from.

    Returns:
        List[dict]: List of evaluation results.
    """
    with open(filename) as file:
        yaml_temp = yaml.load(file, Loader=yaml.FullLoader)
    # From dict of evals to list of evals
    yaml_log = []
    for k in yaml_temp.keys():
        yaml_log.append(yaml_temp[k])
    return yaml_log


def load_json(filename: str):
    """ "Load in JSON file.

    Args:
        filename (str): JSON filename to load from.
    """
    return json.load(open(filename))


def write_configs(params_batch: List[dict], config_fnames: List[str]) -> None:
    """Write batch-list of configs to json/yaml files.

    Args:
        params_batch (List[dict]): List of parameter configurations.
        config_fnames (List[str]): List of filenames to write configurations to.

    Raises:
        ValueError: Make sure that filenames have correct extension.
    """
    for s_id in range(len(params_batch)):
        _, config_fext = os.path.splitext(config_fnames[s_id])
        if config_fext == ".json":
            # Write config dictionary to json file
            save_json(params_batch[s_id], config_fnames[s_id])
        elif config_fext == ".yaml":
            save_yaml(params_batch[s_id], config_fnames[s_id])
        else:
            raise ValueError("Only YAML & JSON configuration can be stored.")


def unflatten_config(dictionary, sep="/") -> dict:
    """Transform flat composed parameter keys into corresponding nested dict.
    Example: {'sub1/sub2/vname': v} -> {sub1: {sub2: {vname: v}}}

    Args:
        dictionary (_type_): Dictionary with flat keys.
        sep (str, optional): Separator to unflatten by. Defaults to "/".

    Returns:
        dict: Unflattened dictionary.
    """
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(sep)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def flatten_config(dictionary, parent_key="", sep="/") -> dict:
    """Transform nested dict keys into flat composed parameter keys.
    Example: {sub1: {sub2: {vname: v}}} -> {'sub1/sub2/vname': v}

    Args:
        dictionary (_type_): Dictionary with nested structure.
        parent_key (str, optional): Parent key. Defaults to "".
        sep (str, optional): Separator used to merge. Defaults to "/".

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_config_dicts(dict1: dict, dict2: dict):
    """Merge two potentially nested dictionaries.
    Important: dict2 overwrites dict1 in case of shared entries.

    Args:
        dict1 (dict): Fixed parameter dictionary.
        dict2 (dict): New hyperparameters to evaluate.

    Yields:
        _type_: Generator - wrap with dict outside of function.
    """
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(merge_config_dicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
                # Alternatively, replace this with exception raiser to alert you of value conflicts
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])
