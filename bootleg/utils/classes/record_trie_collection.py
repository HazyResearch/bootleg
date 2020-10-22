from datetime import datetime
import os
from functools import partial
import numpy as np
import marisa_trie
from typing import Dict, Any, List, Tuple, Callable
import bootleg.utils.utils as utils

def get_qid_cand_with_score(max_value:int, value: List[Tuple[str, int]], vocabulary: Dict[str, int]):
    assert type(value) is list
    assert len(value) > 0
    assert all(type(v[0]) is str for v in value)
    assert all((type(v[1]) is float) or (type(v[1]) is int) for v in value)
    new_value = utils.flatten([[vocabulary[p[0]], p[1]] for p in value])
    assert -1 not in new_value
    new_value.extend([-1]*(2*max_value - len(new_value)))
    return tuple(new_value)

def inverse_qid_cand_with_score(value: List[int], itos: np.ndarray):
    assert len(value)%2 == 0
    new_value = []
    for i in range(0, len(value), 2):
        # -1 values are only added at the end as padding
        if value[i] == -1:
            break
        new_value.append([itos[value[i]], value[i+1]])
    return new_value

def get_single_str_val(max_value:int, value: List[str], vocabulary: Dict[str, int]):
    assert type(value) is list or type(value) is set or type(value) is np.ndarray
    assert len(value) > 0
    assert all(type(v) is str for v in value)
    new_value = [vocabulary[p] for p in value]
    assert -1 not in new_value
    new_value.extend([-1]*(max_value - len(new_value)))
    return tuple(new_value)

def inverse_single_str_val(return_type: Callable[[Any], Any], value: List[int], itos: np.ndarray):
    new_value = [itos[p] for p in value if p != -1]
    return return_type(new_value)

def get_type_ids(max_value:int, value: List[int], vocabulary: Dict[str, int]):
    assert type(value) is list or type(value) is np.ndarray
    assert len(value) > 0
    assert all(type(v) is int or type(v) is np.int64 for v in value)
    new_value = list(value)
    assert -1 not in new_value
    new_value.extend([-1]*(max_value - len(new_value)))
    return tuple(new_value)

def inverse_type_ids(value: List[int], itos: np.ndarray):
    new_value = [p for p in value if p != -1]
    return new_value

class RecordTrieCollection:
    def __init__(self, load_dir: str=None, input_dicts: Dict[str, Dict[str, Any]]=None, vocabulary: Dict[str, int]=None,
                 fmt_types: Dict[str, Any]=None, max_values: Dict[str, Any]=None) -> None:
        self._get_fmt_strings = {
            "qid_cand": lambda x: f"<{'l'*x}",  # K long integers
            "type_ids": lambda x: f"<{'l'*x}",  # K long integers
            "kg_relations": lambda x: f"<{'l'*x}",  # K long integers
            "qid_cand_with_score": lambda x: f"<{'lf'*x}" # integer for QID and float for score
        }
        self._get_fmt_funcs_map = {
            "qid_cand": get_single_str_val,
            "type_ids": get_type_ids,
            "kg_relations": get_single_str_val,
            "qid_cand_with_score": get_qid_cand_with_score
        }
        self._get_fmt_funcs_inv = {
            "qid_cand": partial(inverse_single_str_val, return_type=lambda x: list(x)),
            "type_ids": inverse_type_ids,
            "kg_relations": partial(inverse_single_str_val, return_type=lambda x: set(x)),
            "qid_cand_with_score": inverse_qid_cand_with_score
        }

        if load_dir is not None:
            self.load(load_dir)
            self._loaded_from_dir = load_dir
        else:
            for tri_name in fmt_types:
                assert fmt_types[tri_name] in self._get_fmt_strings
            assert fmt_types.keys() == max_values.keys()
            assert fmt_types.keys() == input_dicts.keys()

            self._fmt_types = fmt_types
            self._max_values = max_values
            self._stoi: Dict[str, int] = vocabulary
            self._itos: np.ndarray = self.get_itos()
            self._record_tris = {}
            for tri_name in self._fmt_types:
                print("FOUND TRI NAME", tri_name)
                self._record_tris[tri_name] = self.build_trie(input_dicts[tri_name], self._fmt_types[tri_name], self._max_values[tri_name])
            self._loaded_from_dir = None


    def dump(self, save_dir):
        #memmapped files bahve badly if you try to overwrite them in memory, which is what we'd be doing if load_dir == save_dir
        if self._loaded_from_dir is None or self._loaded_from_dir != save_dir:
            utils.ensure_dir(save_dir)
            utils.dump_json_file(filename=os.path.join(save_dir, "fmt_types.json"), contents=self._fmt_types)
            utils.dump_json_file(filename=os.path.join(save_dir, "max_values.json"), contents=self._max_values)
            utils.dump_json_file(filename=os.path.join(save_dir, "vocabulary.json"), contents=self._stoi)
            np.save(file=os.path.join(save_dir, "itos.npy"), arr=self._itos, allow_pickle=True)
            for tri_name in self._fmt_types:
                self._record_tris[tri_name].save(os.path.join(save_dir, f'record_trie_{tri_name}.marisa'))

    def load(self, load_dir):
        self._fmt_types = utils.load_json_file(filename=os.path.join(load_dir, "fmt_types.json"))
        self._max_values = utils.load_json_file(filename=os.path.join(load_dir, "max_values.json"))
        self._stoi = utils.load_json_file(filename=os.path.join(load_dir, "vocabulary.json"))
        self._itos = np.load(file=os.path.join(load_dir, "itos.npy"), allow_pickle=True)
        assert self._fmt_types.keys() == self._max_values.keys()
        for tri_name in self._fmt_types:
            assert f'record_trie_{tri_name}.marisa' in os.listdir(load_dir), f"Missing record_trie_{tri_name}.marisa in {load_dir}"
        self._record_tris = {}
        for tri_name in self._fmt_types:
            self._record_tris[tri_name] = marisa_trie.RecordTrie(
                self._get_fmt_strings[self._fmt_types[tri_name]](self._max_values[tri_name])).mmap(os.path.join(load_dir,
                                                                                                                f'record_trie_{tri_name}.marisa'))

    def get_itos(self) -> np.ndarray:
        vocabulary_inv = {v:k for k,v in self._stoi.items()}
        max_v = max(vocabulary_inv.keys())
        id2vocab = np.array([vocabulary_inv[i] if i in vocabulary_inv else None for i in range(max_v+1)])
        return id2vocab

    def build_trie(self, input_dict: Dict[str, Any], fmt_type: str, max_value: int):
        all_values = []
        all_keys = sorted(list(input_dict.keys()))
        for key in all_keys:
            value = input_dict[key]
            new_value = self._get_fmt_funcs_map[fmt_type](max_value=max_value, value=value, vocabulary=self._stoi)
            all_values.append(new_value)
        trie = marisa_trie.RecordTrie(self._get_fmt_strings[fmt_type](max_value), zip(all_keys, all_values))
        return trie

    def get_value(self, tri_name, key, getter=lambda x: x):
        record_trie = self._record_tris[tri_name]
        assert key in record_trie
        value = record_trie[key]
        # Record trie allows keys to have multiple values and returns a list of values for each key.
        # As we make the value for each key a list already (to control order/not have to sort again), we need to assert there is only a single value
        assert len(value) == 1
        value = value[0]
        return_value = self._get_fmt_funcs_inv[self._fmt_types[tri_name]](value=value, itos=self._itos)
        res = (type(return_value))(map(getter, return_value))
        assert len(res) <= self._max_values[tri_name]
        return res

    def get_keys(self, tri_name):
        return self._record_tris[tri_name].keys()

    def is_key_in_trie(self, tri_name, key):
        return key in self._record_tris[tri_name]

    def is_tri_in_collection(self, tri_name):
        return tri_name in self._record_tris