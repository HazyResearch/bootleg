"""Bootleg utils."""
import collections
import json
import logging
import math
import os
import pathlib
import shutil
import time
import unicodedata
from itertools import chain, islice

import marisa_trie
import ujson
import yaml

from bootleg import log_rank_0_info
from bootleg.utils.classes.dotted_dict import DottedDict

logger = logging.getLogger(__name__)


def ensure_dir(d):
    """
    Check if a directory exists. If not, it makes it.

    Args:
        d: path
    """
    pathlib.Path(d).mkdir(exist_ok=True, parents=True)


def exists_dir(d):
    """
    Check if directory exists.

    Args:
        d: path
    """
    return pathlib.Path(d).exists()


def dump_json_file(filename, contents):
    """
    Dump dictionary to json file.

    Args:
        filename: file to write to
        contents: dictionary to save
    """
    filename = pathlib.Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "w") as f:
        try:
            ujson.dump(contents, f)
        except OverflowError:
            json.dump(contents, f)


def dump_yaml_file(filename, contents):
    """
    Dump dictionary to yaml file.

    Args:
        filename: file to write to
        contents: dictionary to save
    """
    filename = pathlib.Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "w") as f:
        yaml.dump(contents, f)


def load_json_file(filename):
    """
    Load dictionary from json file.

    Args:
        filename: file to read from

    Returns: Dict
    """
    with open(filename, "r") as f:
        contents = ujson.load(f)
    return contents


def load_yaml_file(filename):
    """
    Load dictionary from yaml file.

    Args:
        filename: file to read from

    Returns: Dict
    """
    with open(filename) as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
    return contents


def recurse_redict(d):
    """
    Cast all DottedDict values in a dictionary to be dictionaries.

    Useful for YAML dumping.

    Args:
        d: Dict

    Returns: Dict with no DottedDicts
    """
    d = dict(d)
    for k, v in d.items():
        if isinstance(v, (DottedDict, dict)):
            d[k] = recurse_redict(dict(d[k]))
    return d


def write_to_file(filename, value):
    """
    Write generic value to a file.

    If value is not string, will cast to str().

    Args:
        filename: file to write to
        value: context to write

    Returns: Dict
    """
    ensure_dir(os.path.dirname(filename))
    if not isinstance(value, str):
        value = str(value)
    fout = open(filename, "w")
    fout.write(value + "\n")
    fout.close()


def write_jsonl(filepath, values):
    """
    Write List[Dict] data to jsonlines file.

    Args:
        filepath: file to write to
        values: list of dictionary data to write

    """
    with open(filepath, "w") as out_f:
        for val in values:
            out_f.write(ujson.dumps(val) + "\n")
    return


def chunks(iterable, n):
    """
    Chunk data.

    chunks(ABCDE,2) => AB CD E.

    Args:
        iterable: iterable input
        n: number of chunks

    Returns: next chunk
    """
    iterable = iter(iterable)
    while True:
        try:
            yield chain([next(iterable)], islice(iterable, n - 1))
        except StopIteration:
            return None


def chunk_file(in_file, out_dir, num_lines, prefix="out_"):
    """
    Chunk a file into num_lines chunks.

    Args:
        in_file: input file
        out_dir: output directory
        num_lines: number of lines in each chunk
        prefix: prefix for output files in out_dir

    Returns: total number of lines read, dictionary of output file path -> number of lines in that file (for tqdms)
    """
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
            with open(file_split, "w") as f:
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


def create_single_item_trie(in_dict, out_file=""):
    """
    Create marisa trie.

    Creates a marisa trie from the input dictionary. We assume the
    dictionary has string keys and integer values.

    Args:
        in_dict: Dict[str] -> Int
        out_file: marisa file to save (useful for reading as memmap) (optional)

    Returns: marisa trie of in_dict
    """
    keys = []
    values = []
    for k in in_dict:
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
    """
    Load a marisa trie with integer values from memmap file.

    Args:
        file: marisa input file

    Returns: marisa trie
    """
    assert exists_dir(file)
    return marisa_trie.RecordTrie("<l").mmap(file)


def get_lnrm(s, strip, lower):
    """
    Convert to lnrm form.

    Convert a string to its lnrm form We form the lower-cased normalized
    version l(s) of a string s by canonicalizing its UTF-8 characters,
    eliminating diacritics, lower-casing the UTF-8 and throwing out all ASCII-
    range characters that are not alpha-numeric.

    from http://nlp.stanford.edu/pubs/subctackbp.pdf Section 2.3

    Args:
        s: input string
        strip: boolean for stripping alias or not
        lower: boolean for lowercasing alias or not

    Returns: the lnrm form of the string
    """
    if not strip and not lower:
        return s
    lnrm = str(s)
    if lower:
        lnrm = lnrm.lower()
    if strip:
        lnrm = unicodedata.normalize("NFD", lnrm)
        lnrm = "".join(
            [
                x
                for x in lnrm
                if (not unicodedata.combining(x) and x.isalnum() or x == " ")
            ]
        ).strip()
    # will remove if there are any duplicate white spaces e.g. "the  alias    is here"
    lnrm = " ".join(lnrm.split())
    return lnrm


def strip_nan(input_list):
    """
    Replace float('nan') with nulls.

    Used for ujson loading/dumping.

    Args:
        input_list: list of items to remove the Nans from

    Returns: list or nested list where Nan is not None
    """
    final_list = []
    for item in input_list:
        if isinstance(item, collections.abc.Iterable):
            final_list.append(strip_nan(item))
        else:
            final_list.append(item if not math.isnan(item) else None)
    return final_list


def try_rmtree(rm_dir):
    """
    Try to remove a directory tree.

    In the case a resource is open, rmtree will fail. This retries to rmtree
    after 1 second waits for 5 times.

    Args:
        rm_dir: directory to remove
    """
    num_retries = 0
    max_retries = 5
    while num_retries < max_retries:
        try:
            shutil.rmtree(rm_dir)
            break
        except OSError:
            time.sleep(1)
            num_retries += 1
            if num_retries >= max_retries:
                log_rank_0_info(
                    logger,
                    f"{rm_dir} was not able to be deleted. This is okay but will have to manually be removed.",
                )
