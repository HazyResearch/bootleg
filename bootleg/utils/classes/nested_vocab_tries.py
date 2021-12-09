"""Nested vocab tries."""
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import marisa_trie
import numpy as np
import ujson
from numba import njit
from tqdm import tqdm

from bootleg.utils.utils import dump_json_file, load_json_file

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)

def flatten(arr):
    """Flatten array."""
    return [item for sublist in arr for item in sublist]


@njit
def index(array, item):
    """Retrun index of imte in array."""
    for idx, val in np.ndenumerate(array):
        if val == item:
            # ndenumerate retuns tuple of index; we have 1D array and only care about first value
            return idx[0]
    return None


def get_cand_with_score(
    max_value: int, value: List[Tuple[str, int]], vocabulary: marisa_trie
):
    """Get the keys with score numerical values as list of ints."""
    assert type(value) is list
    if len(value) > 0:
        assert all(type(v[0]) is str for v in value)
        assert all((type(v[1]) is float) or (type(v[1]) is int) for v in value)
    new_value = flatten([[vocabulary[p[0]], p[1]] for p in value])[: (2 * max_value)]
    assert -1 not in new_value
    overflowed = len(new_value) == (2 * max_value)
    new_value.extend([-1] * (2 * max_value - len(new_value)))
    return tuple(new_value), overflowed


def get_key_value_pair(
    max_value: int,
    value: List[Tuple[str, int]],
    key_vocabulary: marisa_trie,
    value_vocabulary: marisa_trie,
):
    """Get the key value pairs as list of ints."""
    new_value = flatten(
        [[key_vocabulary[p[0]], value_vocabulary[p[1]]] for p in value]
    )[: (2 * max_value)]
    assert -1 not in new_value
    overflowed = len(new_value) == (2 * max_value)
    new_value.extend([-1] * (2 * max_value - len(new_value)))
    return tuple(new_value), overflowed


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


def inverse_key_value_pair(
    value: List[int], key_itos: Callable[[int], str], value_itos: Callable[[int], str]
):
    """Return list of key value pairs from numerical values."""
    assert len(value) % 2 == 0
    new_value = []
    for i in range(0, len(value), 2):
        # -1 values are only added at the end as padding
        if value[i] == -1:
            break
        new_value.append([key_itos(value[i]), value_itos(value[i + 1])])
    return new_value


class VocabularyTrie:
    """String (vocabulary) to int trie.

    This is basically a marisa trie except that we maintain the original indexes given in the input dict.
    This helps keep indexes the same even if underlying trie is different.
    """

    def __init__(
        self,
        load_dir: str = None,
        input_dict: Dict[str, int] = None,
    ) -> None:
        """Vocab trie initializer."""
        # One long integer
        self._get_fmt_string = lambda x: "<'l'"
        if load_dir is not None:
            self.load(load_dir)
            self._loaded_from_dir = load_dir
        else:
            self._stoi: marisa_trie = marisa_trie.Trie(input_dict.keys())
            # Array from internal trie id to external id from dict
            self._itoexti: np.array = (np.ones(len(input_dict)) * -1).astype(int)
            # Keep track external ids to prevent duplicates
            extis: set = set()
            self._max_id = next(iter(input_dict.values()))
            for k, exti in input_dict.items():
                i = self._stoi[k]
                self._itoexti[i] = exti
                self._max_id = max(self._max_id, exti)
                if exti in extis:
                    raise ValueError(f"All ids must be unique. {exti} is a duplicate.")
            self._loaded_from_dir = None

    def dump(self, save_dir):
        """Dump."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ujson.dump({"max_id": self._max_id}, open(save_dir / "config.json", "w", encoding="utf-8"))
        self._stoi.save(str(save_dir / "vocabulary_trie.marisa"))
        np.save(str(save_dir / "itoexti.npy"), self._itoexti)

    def load(self, load_dir):
        """Load."""
        load_dir = Path(load_dir)
        self._max_id = ujson.load(open(load_dir / "config.json", encoding="utf-8"))["max_id"]
        self._stoi = marisa_trie.Trie().mmap(str(load_dir / "vocabulary_trie.marisa"))
        self._itoexti = np.load(str(load_dir / "itoexti.npy")).astype(int)

    def to_dict(self):
        """Convert to dictionary."""
        res_dict = {}
        for key in self.keys():
            res_dict[key] = self.get_value(key)
        return res_dict

    def get_value(self, key):
        """Get value for key."""
        i_value = self._stoi[key]
        ext_value = int(self._itoexti[i_value])
        return ext_value

    def get_key(self, value):
        """Get key for value."""
        i_value = index(self._itoexti, value)
        if i_value is None:
            raise KeyError(f"{value} not in Trie")
        return self._stoi.restore_key(i_value)

    def keys(self):
        """Get keys."""
        return self._stoi.keys()

    def is_key_in_trie(self, key):
        """Return if key in trie."""
        return key in self._stoi

    def is_value_in_trie(self, value):
        """Return if value in trie."""
        try:
            self.get_key(value)
            return True
        except KeyError:
            return False

    def get_max_id(self):
        """Get max id."""
        return self._max_id

    def __getitem__(self, item):
        """Get item."""
        return self.get_value(item)

    def __len__(self):
        """Get length."""
        return len(self.keys())

    def __contains__(self, key):
        """Contain key or not."""
        return self.is_key_in_trie(key)


class TwoLayerVocabularyScoreTrie:
    """TwoLayerVocabularyScoreTrie.

    This creates a record trie from a string to a list of string candidates. These candidates are either a single
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
        # memmapped files behave badly if you try to overwrite them in memory,
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
        total_overflow = 0
        for key in tqdm(all_keys, desc="Creating trie"):
            # Extract the QID candidate
            cand_list = input_dict[key]
            # If the scores are not in the candidate list, set them as default 0.0
            if len(cand_list) > 0 and not isinstance(cand_list[0], list):
                cand_list = [[c, 0.0] for c in cand_list]
            new_value, overflow = get_cand_with_score(
                max_value=max_value, value=cand_list, vocabulary=self._stoi
            )
            total_overflow += overflow
            all_values.append(new_value)
        trie = marisa_trie.RecordTrie(
            self._get_fmt_string(max_value), zip(all_keys, all_values)
        )
        logger.debug(
            f"There were {total_overflow/len(all_keys)}% of items that lost information because max_connections"
            f" was too small."
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


class ThreeLayerVocabularyTrie:
    """ThreeLayerVocabularyTrie.

    This creates a dict from query -> key -> list of values but
    saves as trie with query -> flatten lower level dict.

    Note that max_value is the maximum number of values for each possible key.
    """

    def __init__(
        self,
        load_dir: str = None,
        input_dict: Dict[str, Any] = None,
        key_vocabulary: Union[Dict[str, Any], Set[str]] = None,
        value_vocabulary: Union[Dict[str, Any], Set[str]] = None,
        max_value: int = None,
    ) -> None:
        """Doct vocab initializer."""
        self._get_fmt_string = lambda x: f"<{'ll'*x}"

        if load_dir is not None:
            self.load(load_dir)
            self._loaded_from_dir = load_dir
        else:
            if max_value is None:
                raise ValueError("max_value cannot be None when creating trie")
            if isinstance(key_vocabulary, dict):
                key_vocabulary = set(key_vocabulary.keys())
            if isinstance(value_vocabulary, dict):
                value_vocabulary = set(value_vocabulary.keys())
            self._max_value = (
                max_value * 2
            )  # Add a buffer to try to keep all connections - it's imperfect
            self._key_stoi: marisa_trie = marisa_trie.Trie(key_vocabulary)
            self._key_itos: Callable[[int], str] = lambda x: self._key_stoi.restore_key(
                x
            )
            self._value_stoi: marisa_trie = marisa_trie.Trie(value_vocabulary)
            self._value_itos: Callable[
                [int], str
            ] = lambda x: self._value_stoi.restore_key(x)
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
            self._key_stoi.save(os.path.join(save_dir, "key_vocabulary_trie.marisa"))
            self._value_stoi.save(
                os.path.join(save_dir, "value_vocabulary_trie.marisa")
            )
            self._record_trie.save(os.path.join(save_dir, "record_trie.marisa"))

    def load(self, load_dir):
        """Load."""
        self._max_value = load_json_file(
            filename=os.path.join(load_dir, "max_value.json")
        )
        self._key_stoi = marisa_trie.Trie().mmap(
            os.path.join(load_dir, "key_vocabulary_trie.marisa")
        )
        self._key_itos = lambda x: self._key_stoi.restore_key(x)
        self._value_stoi = marisa_trie.Trie().mmap(
            os.path.join(load_dir, "value_vocabulary_trie.marisa")
        )
        self._value_itos = lambda x: self._value_stoi.restore_key(x)
        self._record_trie = marisa_trie.RecordTrie(
            self._get_fmt_string(self._max_value)
        ).mmap(os.path.join(load_dir, "record_trie.marisa"))

    def to_dict(self, keep_score=True):
        """Convert to dictionary."""
        res_dict = {}
        for key in self.keys():
            res_dict[key] = self.get_value(key)
        return res_dict

    def build_trie(self, input_dict: Dict[str, Any], max_value: int):
        """Build trie."""
        all_values = []
        all_keys = sorted(list(input_dict.keys()))
        total_overflow = 0
        from tqdm import tqdm

        for key in tqdm(all_keys, desc="Prepping trie data"):
            # Extract the QID candidate
            cand_list = [
                [key2, val]
                for key2, values in input_dict[key].items()
                for val in values
            ]
            # If the scores are not in the candidate list, set them as default 0.0
            new_value, overflow = get_key_value_pair(
                max_value=max_value,
                value=cand_list,
                key_vocabulary=self._key_stoi,
                value_vocabulary=self._value_stoi,
            )
            total_overflow += overflow
            all_values.append(new_value)
        print(
            f"Creating trie with {len(all_keys)} values. This can take a few minutes."
        )
        trie = marisa_trie.RecordTrie(
            self._get_fmt_string(max_value), zip(all_keys, all_values)
        )
        logger.debug(
            f"There were {total_overflow/len(all_keys)}% of items that lost information because max_connections"
            f" was too small."
        )
        return trie

    def get_value(self, key):
        """Get value for query as dict of key -> values."""
        assert key in self._record_trie
        flattened_value = self._record_trie[key]
        # Record trie allows keys to have multiple values and returns a list of values for each key.
        # As we make the value for each key a list already (to control order/not have to sort again),
        # we need to assert there is only a single value
        assert len(flattened_value) == 1
        flattened_value = flattened_value[0]
        flattened_return_value = inverse_key_value_pair(
            value=flattened_value, key_itos=self._key_itos, value_itos=self._value_itos
        )
        assert len(flattened_return_value) <= self._max_value
        return_dict = {}
        for k, grped_v in itertools.groupby(flattened_return_value, key=lambda x: x[0]):
            return_dict[k] = list(map(lambda x: x[1], grped_v))
        return return_dict

    def keys(self):
        """Get keys."""
        return self._record_trie.keys()

    def key_vocab_keys(self):
        """Get key vocab keys."""
        return self._key_stoi.keys()

    def value_vocab_keys(self):
        """Get value vocab keys."""
        return self._value_stoi.keys()

    def is_key_in_trie(self, key):
        """Return if key in trie."""
        return key in self._record_trie
