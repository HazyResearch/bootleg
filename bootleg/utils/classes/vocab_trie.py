"""String to int (offset by +1) trie."""
from pathlib import Path
from typing import Dict

import marisa_trie
import numpy as np
import ujson
from numba import njit


@njit
def index(array, item):
    """Retrun index of imte in array."""
    for idx, val in np.ndenumerate(array):
        if val == item:
            # ndenumerate retuns tuple of index; we have 1D array and only care about first value
            return idx[0]
    return None


class VocabularyTrie:
    """String (vocabulary) to int trie.

    This is basically a marisa trie except that we maintain the original indexes given in the input dict.
    This helps keep indexes the same even if underlying tri is different.
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
        ujson.dump({"max_id": self._max_id}, open(save_dir / "config.json", "w"))
        self._stoi.save(str(save_dir / "vocabulary_trie.marisa"))
        np.save(str(save_dir / "itoexti.npy"), self._itoexti)

    def load(self, load_dir):
        """Load."""
        load_dir = Path(load_dir)
        self._max_id = ujson.load(open(load_dir / "config.json"))["max_id"]
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
