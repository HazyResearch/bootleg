'''
Useful functions
'''
import copy
from importlib import import_module
from itertools import islice, chain

import marisa_trie
import ujson
import json # we need this for dumping nans
import logging
import os
import pickle
import sys
import torch
from tqdm import tqdm


def recursive_transform(x, test_func, transform):
    """Applies a transformation recursively to each member of a dictionary

    Args:
        x: a (possibly nested) dictionary
        test_func: a function that returns whether this element should be transformed
        transform: a function that transforms a value
    """
    for k, v in x.items():
        if test_func(v):
            x[k] = transform(v)
        if isinstance(v, dict):
            recursive_transform(v, test_func, transform)
    return x


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def exists_dir(d):
    return os.path.exists(d)

def dump_json_file(filename, contents):
    with open(filename, 'w') as f:
        try:
            ujson.dump(contents, f)
        except OverflowError:
            json.dump(contents, f)

def load_json_file(filename):
    with open(filename, 'r') as f:
        contents = ujson.load(f)
    return contents

def dump_pickle_file(filename, contents):
    with open(filename, 'wb') as f:
        pickle.dump(contents, f)

def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        contents = pickle.load(f)
    return contents

def create_single_item_trie(in_dict, out_file=""):
    keys = []
    values = []
    for k in tqdm(in_dict, total=len(in_dict), desc="Reading values for marisa trie"):
        assert type(in_dict[k]) is int
        keys.append(k)
        # Tries require list of item for the record trie
        values.append(tuple([in_dict[k]]))
    fmt = "<l"
    trie = marisa_trie.RecordTrie(fmt, zip(keys, values))
    if out_file != "":
        trie.save(out_file)
    return trie

def load_single_item_trie(file):
    assert exists_dir(file)
    return marisa_trie.RecordTrie('<l').mmap(file)

def flatten(arr):
    return [item for sublist in arr for item in sublist]

def chunks(iterable, n):
   """chunks(ABCDE,2) => AB CD E"""
   iterable = iter(iterable)
   while True:
       try:
           yield chain([next(iterable)], islice(iterable, n-1))
       except StopIteration:
           return None

def chunk_file(in_file, out_dir, num_lines, prefix="out_"):
    ensure_dir(out_dir)
    out_files = {}
    total_lines = 0
    ending = os.path.splitext(in_file)[1]
    with open(in_file) as bigfile:
        i = 0
        while True:
            try:
                lines = next(chunks(bigfile, num_lines))
            except StopIteration:
                break
            except RuntimeError:
                break
            file_split = os.path.join(out_dir, f"{prefix}{i}{ending}")
            total_file_lines = 0
            i += 1
            with open(file_split, 'w') as f:
                while True:
                    try:
                        line = next(lines)
                    except StopIteration:
                        break
                    total_lines += 1
                    total_file_lines += 1
                    f.write(line)
            out_files[file_split] = total_file_lines
    return total_lines, out_files

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

# iterates over a list of dicts with values that are tensors or numbers
# and concatenates values with corresponding keys together
def merge_dicts(list_of_dicts):
    merged_dict = {}
    for key in list_of_dicts[0].keys():
        merged_dict[key] = torch.tensor([d[key] for d in list_of_dicts])
    return merged_dict

# In case weird stuff is in config, we sanitize it
def sanitize_config(args):
    args = copy.deepcopy(args)
    # Replace individual functions
    is_func = lambda x: callable(x)
    replace_with_name = lambda f: str(f)
    args = recursive_transform(args, is_func, replace_with_name)
    # Replace lists of functions
    is_func_list = lambda x: isinstance(x, list) and all(is_func(f) for f in x)
    replace_with_names = lambda x: [replace_with_name(f) for f in x]
    args = recursive_transform(args, is_func_list, replace_with_names)
    return args

# Takes the prefix path and import that plus all but the rightmost modules in base string
# Eg: prefix_string = bootleg.embeddings.word_embeddings
#     base_string = bert.BERTWordEmbedding
# This will import bootleg.embeddings.word_embeddings.bert and give a class of BERTWordEmbedding
def import_class(prefix_string, base_string):
    if "." in base_string:
        path, load_class = base_string.rsplit(".", 1)
        mod = import_module(f"{prefix_string}.{path}")
    else:
        load_class = base_string
        mod = import_module(f"{prefix_string}")
    return mod, load_class

def remove_dots(str):
    return str.replace(".", "_")