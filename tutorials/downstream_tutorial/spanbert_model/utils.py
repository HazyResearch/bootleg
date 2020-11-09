import json
import os

import numpy as np


def write_to_file(file_path, value):
    """
    Write value to file.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    if not isinstance(value, str):
        value = str(value)
    fout = open(file_path, "w")
    fout.write(value + "\n")
    fout.close()


def write_to_json_file(file_path, dict):
    """
    Write dict to json file.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    for k in dict.keys():
        if isinstance(dict[k], (np.float32, np.float64)):
            dict[k] = dict[k].item()
    json_obj = json.dumps(dict)
    fout = open(file_path, "w")
    fout.write(json_obj)
    fout.close()


def load_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data
