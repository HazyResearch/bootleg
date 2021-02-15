"""Entity symbols."""
import os
from datetime import datetime
from typing import Dict

import marisa_trie

import bootleg.utils.utils as utils
from bootleg.symbols.constants import UNK_AL


class EntitySymbols:
    """Entity symbols class.

    Args:
        load_dir: loading directory (defaults to None)
        max_candidates: maximum number of candidates
        max_alias_len: maximum length of a single alias w.r.t len(string.split())
        alias2qids: Dict mapping of alias to list of QID candidates
        qid2title: Dict mapping of qid to title
        alias_cand_map_file: alias to candidates file name to use when dumping

    Returns:
    """

    def __init__(
        self,
        load_dir=None,
        max_candidates=None,
        max_alias_len=None,
        alias2qids=None,
        qid2title=None,
        alias_cand_map_file="alias2qids.json",
    ):
        # We support different candidate mappings for the same set of entities
        self.alias_cand_map_file = alias_cand_map_file
        if load_dir is not None:
            self.load(load_dir)
        else:
            self.max_candidates = max_candidates
            # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
            self.max_alias_len = max_alias_len
            self._alias2qids: Dict[str, list] = alias2qids
            self._qid2title: Dict[str, str] = qid2title
            # Assert that max_candidates is true
            # Sometimes the alias2qids are manually modified so we need to recheck this condition
            for al in self._alias2qids:
                assert (
                    len(self._alias2qids[al]) <= self.max_candidates
                ), f"You have a alias {al} that has more than {self.max_candidates} candidates. This can't happen."
            # Add 1 for the noncand class
            # We only make these inside the else because of json ordering being nondeterministic
            # If we load stuff up in self.load() and regenerate these, the eid values may be nondeterministic
            self._qid2eid: Dict[str, int] = {v: i + 1 for i, v in enumerate(qid2title)}
        self._eid2qid: Dict[int, str] = {eid: qid for qid, eid in self._qid2eid.items()}
        assert -1 not in self._eid2qid, f"-1 can't be an eid"
        assert (
            0 not in self._eid2qid
        ), f"0 can't be an eid. It's reserved for null candidate"
        # generate trie of aliases for quick entity generation in sentences (trie generates alias ids, too)
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        # this assumes that eid of 0 is NO_CAND and eid of -1 is NULL entity
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2

    def dump(self, save_dir, stats=None, args=None):
        """Dumps the entity symbols.

        Args:
            save_dir: directory string to save
            stats: statistics to dump
            args: other args to dump

        Returns:
        """
        if stats is None:
            stats = {}
        self._sort_alias_cands()
        utils.ensure_dir(save_dir)
        utils.dump_json_file(
            filename=os.path.join(save_dir, "config.json"),
            contents={
                "max_candidates": self.max_candidates,
                "max_alias_len": self.max_alias_len,
                "datetime": str(datetime.now()),
            },
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, self.alias_cand_map_file),
            contents=self._alias2qids,
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, "qid2title.json"), contents=self._qid2title
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, "qid2eid.json"), contents=self._qid2eid
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, "filter_stats.json"), contents=stats
        )
        if args is not None:
            utils.dump_json_file(
                filename=os.path.join(save_dir, "args.json"), contents=vars(args)
            )

    def load(self, load_dir):
        """Loads entity symbols from load_dir.

        Args:
            load_dir: directory to load from

        Returns:
        """
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        self.max_candidates = config["max_candidates"]
        self.max_alias_len = config["max_alias_len"]
        self._alias2qids: Dict[str, list] = utils.load_json_file(
            filename=os.path.join(load_dir, self.alias_cand_map_file)
        )
        self._qid2title: Dict[str, str] = utils.load_json_file(
            filename=os.path.join(load_dir, "qid2title.json")
        )
        self._qid2eid: Dict[str, int] = utils.load_json_file(
            filename=os.path.join(load_dir, "qid2eid.json")
        )
        self._sort_alias_cands()

    def _sort_alias_cands(self):
        """
        Sorts the candidate lists for each alias from largest to smallest score (each candidate is a pair [QID, sort_value])
        Returns:

        """
        for alias in self._alias2qids:
            # Add second key for determinism in case of same counts
            self._alias2qids[alias] = sorted(
                self._alias2qids[alias], key=lambda x: (x[1], x[0]), reverse=True
            )

    def get_qid2eid(self):
        """
        Gets the qid2eid mapping
        Returns: Dict qid2eid mapping

        """
        return self._qid2eid

    def get_alias2qids(self):
        """
        Gets the alias2qids mapping (key is alias, value is list of candidate tuple of length two of [QID, sort_value])
        Returns: Dict alias2qids mapping

        """
        return self._alias2qids

    def get_qid2title(self):
        """
        Gets the qid2title mapping
        Returns: Dict qid2title mapping

        """
        return self._qid2title

    def get_all_qids(self):
        """
        Gets all QIDs
        Returns: Dict_keys of all QIDs

        """
        return self._qid2eid.keys()

    def get_all_aliases(self):
        """
        Gets all aliases
        Returns: Dict_keys of all aliases

        """
        return self._alias2qids.keys()

    def get_all_titles(self):
        """
        Gets all QID titles
        Returns: Dict_values of all titles

        """
        return self._qid2title.values()

    def get_qid(self, id):
        """Gets the QID associated with EID.

        Args:
            id: EID

        Returns: QID string
        """
        assert id in self._eid2qid
        return self._eid2qid[id]

    def alias_exists(self, alias):
        """Does alias exist.

        Args:
            alias: alias string

        Returns: boolean
        """
        return alias in self._alias_trie

    def qid_exists(self, qid):
        """Does QID exist.

        Args:
            alias: QID string

        Returns: boolean
        """
        return qid in self._qid2eid

    def eid_exists(self, eid):
        """Does EID exist.

        Args:
            alias: EID int

        Returns: boolean
        """
        return eid in self._eid2qid[eid]

    def get_eid(self, id):
        """Gets the QID for the EID.

        Args:
            id: EID int

        Returns: QID string
        """
        assert id in self._qid2eid
        return self._qid2eid[id]

    def get_qid_cands(self, alias, max_cand_pad=False):
        """Get the QID candidates for an alias.

        Args:
            alias: alias
            max_cand_pad: whether to pad with '-1' or not if fewer than max_candidates candidates

        Returns: List of QID strings
        """
        assert alias in self._alias2qids, f"{alias} not in alias2qid mapping"
        res = [qid_pair[0] for qid_pair in self._alias2qids[alias]]
        if max_cand_pad:
            res = res + ["-1"] * (self.max_candidates - len(res))
        return res

    def get_qid_count_cands(self, alias, max_cand_pad=False):
        """Get the [QID, sort_value] candidates for an alias.

        Args:
            alias: alias
            max_cand_pad: whether to pad with ['-1',-1] or not if fewer than max_candidates candidates

        Returns: List of [QID, sort_value]
        """
        assert alias in self._alias2qids
        res = self._alias2qids[alias]
        if max_cand_pad:
            res = res + ["-1", -1] * (self.max_candidates - len(res))
        return res

    def get_eid_cands(self, alias, max_cand_pad=False):
        """Get the EID candidates for an alias.

        Args:
            alias: alias
            max_cand_pad: whether to pad with -1 or not if fewer than max_candidates candidates

        Returns: List of EID ints
        """
        assert alias in self._alias2qids
        res = [self._qid2eid[qid_pair[0]] for qid_pair in self._alias2qids[alias]]
        if max_cand_pad:
            res = res + [-1] * (self.max_candidates - len(res))
        return res

    def get_title(self, id):
        """Gets title for QID.

        Args:
            id: QID string

        Returns: title string
        """
        assert id in self._qid2title
        return self._qid2title[id]

    def get_alias_idx(self, alias):
        """Gets the numeric index of an alias.

        Args:
            alias: alias

        Returns: integer representation of alias
        """
        return self._alias_trie[alias]

    def get_alias_from_idx(self, alias_idx):
        """Gets the alias from the numeric index.

        Args:
            alias: alias numeric index

        Returns: alias string
        """
        try:
            res = self._alias_trie.restore_key(alias_idx)
        except KeyError:
            res = UNK_AL
        return res
