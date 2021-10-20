import pickle
from typing import Any, List
import os
import json
import yaml
import numpy as np


def load_pkl_object(filename: str) -> Any:
    """Helper to reload pickle objects."""
    with open(filename, "rb") as input:
        obj = pickle.load(input)
    return obj


def save_pkl_object(obj, filename: str):
    """Helper to store pickle objects."""
    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        return (
            super().encode(bool(obj))
            if isinstance(obj, np.bool_)
            else super().default(obj)
        )


def save_json(obj, filename: str):
    with open(filename, "w") as fout:
        json.dump(obj, fout, indent=1, cls=CustomJSONizer)


def load_json(filename):
    # Opening JSON file
    f = open(
        filename,
    )
    data = json.load(f)
    return data


def write_configs_to_file(params_batch: List[dict], config_fnames: List[str]):
    """Take batch-list of configs & write to jsons. Return fnames."""
    for s_id in range(len(params_batch)):
        filename, config_fext = os.path.splitext(config_fnames[s_id])
        if config_fext == ".json":
            # Write config dictionary to json file
            with open(config_fnames[s_id], "w") as f:
                json.dump(params_batch[s_id], f)
        elif config_fext == ".yaml":
            with open(config_fnames[s_id], "w") as f:
                yaml.dump(params_batch[s_id], f, default_flow_style=False)
