import pickle
import pickle5
from typing import List
import os
import json
import yaml
import ast
import numpy as np


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(CustomJSONizer, self).default(obj)


def save_json(obj, filename: str):
    with open(filename, "w") as fout:
        json.dump(obj, fout, indent=1, cls=CustomJSONizer)


def save_yaml(obj, filename: str):
    data_dump = json.dumps(obj, indent=1, cls=CustomJSONizer)
    data_dump_dict = ast.literal_eval(data_dump)
    with open(filename, "w") as f:
        yaml.dump(data_dump_dict, f, default_flow_style=False)


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
        yaml_config = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_config


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
