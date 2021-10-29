"""Alias to candidate trie."""
import json
import os
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import marisa_trie
import ujson


def flatten(arr):
    """Flatten array."""
    return [item for sublist in arr for item in sublist]


def load_json_file(filename):
    """Load json file."""
    with open(filename, "r") as f:
        contents = ujson.load(f)
    return contents


def dump_json_file(filename, contents):
    """Dump json file."""
    with open(filename, "w") as f:
        try:
            ujson.dump(contents, f)
        except OverflowError:
            json.dump(contents, f)


def get_qid_cand_with_score(
    max_value: int, value: List[Tuple[str, int]], vocabulary: marisa_trie
):
    """Get the entity candidate with prob score numerical values."""
    assert type(value) is list
    if len(value) > 0:
        assert all(type(v[0]) is str for v in value)
        assert all((type(v[1]) is float) or (type(v[1]) is int) for v in value)
    new_value = flatten([[vocabulary[p[0]], p[1]] for p in value])
    assert -1 not in new_value
    new_value.extend([-1] * (2 * max_value - len(new_value)))
    return tuple(new_value)


def inverse_qid_cand_with_score(value: List[int], itos: Callable[[int], str]):
    """Return entity candidate and prob score from numerical values."""
    assert len(value) % 2 == 0
    new_value = []
    for i in range(0, len(value), 2):
        # -1 values are only added at the end as padding
        if value[i] == -1:
            break
        new_value.append([itos(value[i]), value[i + 1]])
    return new_value


class VocabularyPairedListTrie:
    """VocabularyPairedListTrie.

    This creates a record tri from a string to a list of string candidates. These candidates are either a single
    list of string items. Or a list of pairs [string item, float score].
    """

    def __init__(
        self,
        load_dir: str = None,
        input_dict: Dict[str, Any] = None,
        vocabulary: Union[Dict[str, Any], Set[str]] = None,
        max_value: int = None,
    ) -> None:
        """Paired vocab initializer."""
        self._get_fmt_string = lambda x: f"<{'lf'*x}"

        if load_dir is not None:
            self.load(load_dir)
            self._loaded_from_dir = load_dir
        else:
            if max_value is None:
                raise ValueError("max_value cannot be None when creating trie")
            self._max_value = max_value
            if isinstance(vocabulary, dict):
                vocabulary = set(vocabulary.keys())
            self._stoi: marisa_trie = marisa_trie.Trie(vocabulary)
            self._itos: Callable[[int], str] = lambda x: self._stoi.restore_key(x)
            self._record_trie = self.build_trie(input_dict, self._max_value)
            self._loaded_from_dir = None

    def dump(self, save_dir):
        """Dump."""
        # memmapped files bahve badly if you try to overwrite them in memory,
        # which is what we'd be doing if load_dir == save_dir
        if self._loaded_from_dir is None or self._loaded_from_dir != save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            dump_json_file(
                filename=os.path.join(save_dir, "max_value.json"),
                contents=self._max_value,
            )
            self._stoi.save(os.path.join(save_dir, "vocabulary_trie.marisa"))
            self._record_trie.save(os.path.join(save_dir, "record_trie.marisa"))

    def load(self, load_dir):
        """Load."""
        self._max_value = load_json_file(
            filename=os.path.join(load_dir, "max_value.json")
        )
        self._stoi = marisa_trie.Trie().mmap(
            os.path.join(load_dir, "vocabulary_trie.marisa")
        )
        self._itos = lambda x: self._stoi.restore_key(x)
        self._record_trie = marisa_trie.RecordTrie(
            self._get_fmt_string(self._max_value)
        ).mmap(os.path.join(load_dir, "record_trie.marisa"))

    def to_dict(self, keep_score=True):
        """Convert to dictionary."""
        res_dict = {}
        for key in self.keys():
            res_dict[key] = self.get_value(key, keep_score)
        return res_dict

    def build_trie(self, input_dict: Dict[str, Any], max_value: int):
        """Build trie."""
        all_values = []
        all_keys = sorted(list(input_dict.keys()))
        for key in all_keys:
            # Extract the QID candidate
            cand_list = input_dict[key]
            # If the scores are not in the candidate list, set them as default 0.0
            if len(cand_list) > 0 and not isinstance(cand_list[0], list):
                cand_list = [[c, 0.0] for c in cand_list]
            new_value = get_qid_cand_with_score(
                max_value=max_value, value=cand_list, vocabulary=self._stoi
            )
            all_values.append(new_value)
        trie = marisa_trie.RecordTrie(
            self._get_fmt_string(max_value), zip(all_keys, all_values)
        )
        return trie

    def get_value(self, key, keep_score=True):
        """Get value for key."""
        record_trie = self._record_trie
        assert key in record_trie
        value = record_trie[key]
        # Record trie allows keys to have multiple values and returns a list of values for each key.
        # As we make the value for each key a list already (to control order/not have to sort again),
        # we need to assert there is only a single value
        assert len(value) == 1
        value = value[0]
        return_value = inverse_qid_cand_with_score(value=value, itos=self._itos)
        if not keep_score:
            return_value = [x[0] for x in return_value]
        assert len(return_value) <= self._max_value
        return return_value

    def keys(self):
        """Get keys."""
        return self._record_trie.keys()

    def vocab_keys(self):
        """Get vocab keys."""
        return self._stoi.keys()

    def is_key_in_trie(self, key):
        """Return if key in trie."""
        return key in self._record_trie
