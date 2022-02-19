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


def save_json(obj, filename: str):
    with open(filename, "w") as fout:
        json.dump(convert(obj), fout, indent=1)


def save_yaml(obj, filename: str):
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


def save_log(obj, filename: str):
    fname, fext = os.path.splitext(filename)
    if fext == ".json":
        # Write config dictionary to json file
        save_json(obj, filename)
    elif fext == ".yaml":
        save_yaml(obj, filename)
    else:
        raise ValueError("Log can only be stored as YAML & JSON files.")


def load_strategy(filename: str):
    """Helper to reload pickle objects."""
    with open(filename, "rb") as input:
        obj = pickle5.load(input)
    return obj


def save_strategy(obj, filename: str):
    """Helper to store pickle objects."""
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_log(filename: str):
    """Load JSON/YAML config depending on file ending."""
    fname, fext = os.path.splitext(filename)
    if fext == ".yaml":
        config = load_yaml(filename)
    elif fext == ".json":
        config = load_json(filename)
    else:
        raise ValueError("Only YAML & JSON configuration can be loaded.")
    return config


def load_yaml(filename: str):
    """Load in YAML file."""
    with open(filename) as file:
        yaml_temp = yaml.load(file, Loader=yaml.FullLoader)
    # From dict of evals to list of evals
    yaml_log = []
    for k in yaml_temp.keys():
        yaml_log.append(yaml_temp[k])
    return yaml_log


def load_json(filename: str):
    """Load in json file."""
    return json.load(open(filename))


def write_configs(params_batch: List[dict], config_fnames: List[str]):
    """Take batch-list of configs & write to jsons. Return fnames."""
    for s_id in range(len(params_batch)):
        filename, config_fext = os.path.splitext(config_fnames[s_id])
        if config_fext == ".json":
            # Write config dictionary to json file
            save_json(params_batch[s_id], config_fnames[s_id])
        elif config_fext == ".yaml":
            save_yaml(params_batch[s_id], config_fnames[s_id])
        else:
            raise ValueError("Only YAML & JSON configuration can be stored.")


def unflatten_config(dictionary, sep="/"):
    """Transform flat composed parameter keys into corresponding nested dict.
    Example: 'sub1/sub2/vname' -> {sub1: {sub2: {vname: v}}}"""
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


def flatten_config(dictionary, parent_key="", sep="/"):
    """Transform nested dict keys into flat composed parameter keys.
    Example: {sub1: {sub2: {vname: v}}} -> 'sub1/sub2/vname'"""
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
