"""Entity symbols"""
from datetime import datetime
import os
from typing import Dict
import marisa_trie

import bootleg.utils.utils as utils
from bootleg.symbols.constants import UNK_AL

class EntitySymbols:
    """
    Entity symbols class. QID is associated with Wikidata.

    Attributes:
          max_candidates: maximum number of candidates per mention (alias)
          max_alias_len: maximum number of words for any alias
          _alias2qids: dictionary of alias to list of [QID, sort_value] pairs; sort_value can be any numeric quantity for sort the candidate lists
          _qid2title: QID to title
          _qid2eid: QID to embedding row id
          _eid2qid: embedding row id to QID
          _alias_trie: alias Trie
    """
    def __init__(self, load_dir=None, max_candidates=None, max_alias_len=None,
        alias2qids=None, qid2title=None, alias_cand_map_file="alias2qids.json"):
        # We support different candidate mappings for the same set of entities
        self.alias_cand_map_file = alias_cand_map_file
        if load_dir is not None:
            self.load(load_dir)
        else:
            self.max_candidates = max_candidates
            # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
            self.max_alias_len = max_alias_len
            self._alias2qids: Dict[str, list] = alias2qids
            # Assert that max_candidates is true
            for al in self._alias2qids:
                assert len(self._alias2qids[al]) <= self.max_candidates, f"You have a alias {al} that has more than {self.max_candidates} candidates. This can't happen."
            self._qid2title: Dict[str, str] = qid2title
            # Add 1 for the noncand class
            # We only make these inside the else because of json ordering being nondeterministic
            # If we load stuff up in self.load() and regenerate these, the eid values may be nondeterministic
            self._qid2eid: Dict[str, int] = {v: i+1 for i, v in enumerate(qid2title)}
        self._eid2qid: Dict[int, str] = {eid:qid for qid, eid in self._qid2eid.items()}
        assert -1 not in self._eid2qid, f"-1 can't be an eid"
        assert 0 not in self._eid2qid, f"0 can't be an eid. It's reserved for null candidate"
        # generate trie of aliases for quick entity generation in sentences (trie generates alias ids, too)
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        # this assumes that eid of 0 is NO_CAND and eid of -1 is NULL entity
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2

    def dump(self, save_dir, stats=None, args=None):
        if stats is None:
            stats = {}
        self._sort_alias_cands()
        utils.ensure_dir(save_dir)
        utils.dump_json_file(filename=os.path.join(save_dir, "config.json"), contents={"max_candidates":self.max_candidates,
                                                                                       "max_alias_len":self.max_alias_len,
                                                                                       "datetime": str(datetime.now())})
        utils.dump_json_file(filename=os.path.join(save_dir, self.alias_cand_map_file), contents=self._alias2qids)
        utils.dump_json_file(filename=os.path.join(save_dir, "qid2title.json"), contents=self._qid2title)
        utils.dump_json_file(filename=os.path.join(save_dir, "qid2eid.json"), contents=self._qid2eid)
        utils.dump_json_file(filename=os.path.join(save_dir, "filter_stats.json"), contents=stats)
        if args is not None:
            utils.dump_json_file(filename=os.path.join(save_dir, "args.json"), contents=vars(args))

    def load(self, load_dir):
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        self.max_candidates = config["max_candidates"]
        self.max_alias_len = config["max_alias_len"]
        self._alias2qids: Dict[str, list] = utils.load_json_file(filename=os.path.join(load_dir, self.alias_cand_map_file))
        self._qid2title: Dict[str, str] = utils.load_json_file(filename=os.path.join(load_dir, "qid2title.json"))
        self._qid2eid: Dict[str, int] = utils.load_json_file(filename=os.path.join(load_dir, "qid2eid.json"))
        self._sort_alias_cands()

    def _sort_alias_cands(self):
        for alias in self._alias2qids:
            # Add second key for determinism in case of same counts
            self._alias2qids[alias] = sorted(self._alias2qids[alias], key = lambda x: (x[1], x[0]), reverse=True)

    def get_qid2eid(self):
        return self._qid2eid

    def get_alias2qids(self):
        return self._alias2qids

    def get_qid2title(self):
        return self._qid2title

    def get_all_qids(self):
        return self._qid2eid.keys()

    def get_all_aliases(self):
        return self._alias2qids.keys()

    def get_all_titles(self):
        return self._qid2title.values()

    def get_qid(self, id):
        assert id in self._eid2qid
        return self._eid2qid[id]

    def qid_exists(self, qid):
        """Does QID exist"""
        return (qid in self._qid2eid)

    def eid_exists(self, eid):
        """Does embedding row id exist"""
        return (eid in self._eid2qid[eid])

    def get_eid(self, id):
        """Get the embedding row id associated with the QID"""
        assert id in self._qid2eid
        return self._qid2eid[id]

    def get_qid_cands(self, alias, max_cand_pad=False):
        """Get the QID candidates for an alias"""
        assert alias in self._alias2qids, f"{alias} not in alias2qid mapping"
        res = [qid_pair[0] for qid_pair in self._alias2qids[alias]]
        if max_cand_pad:
            res = res + ["-1"]*(self.max_candidates-len(res))
        return res

    def get_qid_count_cands(self, alias, max_cand_pad=False):
        """Get the [QID, sort_value] pairs for an alias"""
        assert alias in self._alias2qids
        res = self._alias2qids[alias]
        if max_cand_pad:
            res = res + ["-1,-1"]*(self.max_candidates-len(res))
        return res

    def get_eid_cands(self, alias, max_cand_pad=False):
        """Get the embedding row ids of the candidates for an alias"""
        assert alias in self._alias2qids
        res = [self._qid2eid[qid_pair[0]] for qid_pair in self._alias2qids[alias]]
        if max_cand_pad:
            res = res + [-1]*(self.max_candidates-len(res))
        return res

    def get_title(self, id):
        assert id in self._qid2title
        return self._qid2title[id]

    def get_alias_idx(self, alias):
        """Get the numeric index of an alias"""
        return self._alias_trie[alias]

    def get_alias_from_idx(self, alias_idx):
        """Get the alias from the numeric index"""
        try:
            res = self._alias_trie.restore_key(alias_idx)
        except KeyError:
            res = UNK_AL
        return res
