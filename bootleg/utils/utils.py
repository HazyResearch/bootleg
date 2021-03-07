import json
import os
import time
import unicodedata
from importlib import import_module
from itertools import chain, islice

import marisa_trie
import ujson
import yaml
from tqdm import tqdm


def ensure_dir(d):
    """Checks if a directory exists. If not, it makes it.

    Args:
        d: path

    Returns:
    """
    if len(d) > 0:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def exists_dir(d):
    """Checks if directory exists.

    Args:
        d: path

    Returns:
    """
    return os.path.exists(d)


def dump_json_file(filename, contents):
    """Dumps dictionary to json file.

    Args:
        filename: file to write to
        contents: dictionary to dump

    Returns:
    """
    ensure_dir(os.path.dirname(filename))
    with open(filename, "w") as f:
        try:
            ujson.dump(contents, f)
        except OverflowError:
            json.dump(contents, f)


def dump_yaml_file(filename, contents):
    """Dumps dictionary to yaml file.

    Args:
        filename: file to write to
        contents: dictionary to dump

    Returns:
    """
    ensure_dir(os.path.dirname(filename))
    with open(filename, "w") as f:
        yaml.dump(contents, f)


def load_json_file(filename):
    """Loads dictionary from json file.

    Args:
        filename: file to read from

    Returns: Dict
    """
    with open(filename, "r") as f:
        contents = ujson.load(f)
    return contents


def load_yaml_file(filename):
    """Loads dictionary from yaml file.

    Args:
        filename: file to read from

    Returns: Dict
    """
    with open(filename) as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
    return contents


def assert_keys_in_dict(allowable_keys, d):
    """
    Checks that all keys in d are in allowable keys
    Args:
        allowable_keys: Set or List of allowable keys
        d: Dict

    Returns: Boolean if satisfied, None if correct/key that is not in allowable keys
    """
    for k in d:
        if k not in allowable_keys:
            return False, k
    return True, None


def write_to_file(filename, value):
    """Write generic value to a file. If value is not string, will cast to
    str()

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


def load_jsonl(filepath):
    """Loads jsonlines data from jsonl file.

    Args:
        filepath: file to read from

    Returns: List[Dict] of data
    """
    st = time.time()
    lines = []
    num_lines = sum([1 for _ in open(filepath)])
    with open(filepath, "r") as in_f:
        for line in tqdm(in_f, total=num_lines):
            lines.append(ujson.loads(line))
    print("time", time.time() - st)
    return lines


def write_jsonl(filepath, values):
    """Writes List[Dict] data to jsonlines file.

    Args:
        filepath: file to write to
        values: list of dictionary data to write

    Returns:
    """
    with open(filepath, "w") as out_f:
        for val in values:
            out_f.write(ujson.dumps(val) + "\n")
    return


def chunks(iterable, n):
    """Chunks data.

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
    """Chunks a file into num_lines chunks.

    Args:
        in_file: input file
        out_dir: output directory
        num_lines: number of lines in each chunk
        prefix: prefix for output files in out_dir

    Returns: total number of lines read, dictionary of output file path -> number of lines in that file (useful for tqdms)
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
    """Creates a marisa trie from the input dictionary. We assume the
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
    """Load a marisa trie with integer values from memmap file.

    Args:
        file: marisa input file

    Returns: marisa trie
    """
    assert exists_dir(file)
    return marisa_trie.RecordTrie("<l").mmap(file)


def import_class(prefix_string, base_string):
    """
    Takes the prefix path and import that plus all but the rightmost modules in base string.
    This can be used in conjunciton with
        `getattr(mod, load_class)(...)`
    to initialize the base_string class
    Ex: import_class("bootleg.embeddings", "LearnedEntityEmb") will return bootleg.embeddings module and "LearnedEntityEmb"

    Args:
        prefix_string: prefix path
        base_string: base string

    Returns: imported module, class string

    """
    if "." in base_string:
        path, load_class = base_string.rsplit(".", 1)
        mod = import_module(f"{prefix_string}.{path}")
    else:
        load_class = base_string
        mod = import_module(f"{prefix_string}")
    return mod, load_class


def get_lnrm(s, strip, lower):
    """Convert a string to its lnrm form We form the lower-cased normalized
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
