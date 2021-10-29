"""Entity symbols."""
import logging
import os
import time
from datetime import datetime
from typing import Callable, Dict, Optional, Union

from tqdm import tqdm

import bootleg.utils.utils as utils
from bootleg.symbols.constants import edit_op
from bootleg.utils.classes.vocab_trie import VocabularyTrie
from bootleg.utils.classes.vocabularypairedlist_trie import VocabularyPairedListTrie

logger = logging.getLogger(__name__)


class EntitySymbols:
    """Entity Symbols class for managing entity metadata."""

    def __init__(
        self,
        alias2qids: Union[Dict[str, list], VocabularyPairedListTrie],
        qid2title: Dict[str, str],
        qid2desc: Union[Dict[str, str]] = None,
        qid2eid: Optional[VocabularyTrie] = None,
        alias2id: Optional[VocabularyTrie] = None,
        max_candidates: int = 30,
        alias_cand_map_fld: str = "alias2qids",
        alias_idx_fld: str = "alias2id",
        edit_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Entity symbols initializer."""
        # We support different candidate mappings for the same set of entities
        self.alias_cand_map_fld = alias_cand_map_fld
        self.alias_idx_fld = alias_idx_fld
        self.max_candidates = max_candidates
        self.edit_mode = edit_mode
        self.verbose = verbose

        if qid2eid is None:
            # +1 as 0 is reserved for not-in-cand list entity
            qid2eid = {q: i + 1 for i, q in enumerate(qid2title.keys())}
        if alias2id is None:
            alias2id = {a: i for i, a in enumerate(alias2qids.keys())}

        # If edit mode is ON, we must load everything as a dictionary
        if self.edit_mode:
            self._load_edit_mode(
                alias2qids,
                qid2title,
                qid2desc,
                qid2eid,
                alias2id,
            )
        else:
            self._load_non_edit_mode(
                alias2qids,
                qid2title,
                qid2desc,
                qid2eid,
                alias2id,
            )

        # This assumes that eid of 0 is NO_CAND and eid of -1 is NULL entity; neither are in dict
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2

    def _load_edit_mode(
        self,
        alias2qids: Union[Dict[str, list], VocabularyPairedListTrie],
        qid2title: Dict[str, str],
        qid2desc: Union[Dict[str, str]],
        qid2eid: Union[Dict[str, int], VocabularyTrie],
        alias2id: Union[Dict[str, int], VocabularyTrie],
    ):
        """Load in edit mode.

        Loading in edit mode requires all inputs be cast to dictionaries. Tries do not allow value changes.
        """
        # Convert to dict for editing
        if isinstance(alias2qids, VocabularyPairedListTrie):
            alias2qids = alias2qids.to_dict()

        self._alias2qids: Union[Dict[str, list], VocabularyPairedListTrie] = alias2qids
        self._qid2title: Dict[str, str] = qid2title
        self._qid2desc: Dict[str, str] = qid2desc

        # Sort by score and filter to max candidates
        self._sort_alias_cands(
            self._alias2qids, truncate=True, max_cands=self.max_candidates
        )

        # Cast to dicts in edit mode
        if isinstance(qid2eid, VocabularyTrie):
            self._qid2eid: Union[Dict[str, int], VocabularyTrie] = qid2eid.to_dict()
        else:
            self._qid2eid: Union[Dict[str, int], VocabularyTrie] = qid2eid

        if isinstance(alias2id, VocabularyTrie):
            self._alias2id: Union[Dict[str, int], VocabularyTrie] = alias2id.to_dict()
        else:
            self._alias2id: Union[Dict[str, int], VocabularyTrie] = alias2id

        # Generate reverse indexes for fast editing
        self._id2alias: Union[Dict[int, str], Callable[[int], str]] = {
            id: al for al, id in self._alias2id.items()
        }
        self._eid2qid: Union[Dict[int, str], Callable[[int], str]] = {
            eid: qid for qid, eid in self._qid2eid.items()
        }

        self._qid2aliases: Union[Dict[str, set], None] = {}
        for al in tqdm(
            self._alias2qids,
            total=len(self._alias2qids),
            desc="Building edit mode objs",
            disable=not self.verbose,
        ):
            for qid_pair in self._alias2qids[al]:
                if qid_pair[0] not in self._qid2aliases:
                    self._qid2aliases[qid_pair[0]] = set()
                self._qid2aliases[qid_pair[0]].add(al)

        assert len(self._qid2eid) == len(self._eid2qid), (
            "The qid2eid mapping is not invertable. "
            "This means there is a duplicate id value."
        )
        assert -1 not in self._eid2qid, "-1 can't be an eid"
        assert (
            0 not in self._eid2qid
        ), "0 can't be an eid. It's reserved for null candidate"

        # For when we need to add new entities
        self.max_eid = max(self._eid2qid.keys())
        self.max_alid = max(self._id2alias.keys())

    def _load_non_edit_mode(
        self,
        alias2qids: Union[Dict[str, list], VocabularyPairedListTrie],
        qid2title: Dict[str, str],
        qid2desc: Union[Dict[str, str]],
        qid2eid: Optional[VocabularyTrie],
        alias2id: Optional[VocabularyTrie],
    ):
        """Load items in read-only Trie mode."""
        # Convert to record trie
        st = time.time()
        if isinstance(alias2qids, dict):
            self._sort_alias_cands(
                alias2qids, truncate=True, max_cands=self.max_candidates
            )
            alias2qids = VocabularyPairedListTrie(
                input_dict=alias2qids,
                vocabulary=qid2title,
                max_value=self.max_candidates,
            )
        print(f"Time for creating alias trie {time.time() - st}")

        self._alias2qids: Union[Dict[str, list], VocabularyPairedListTrie] = alias2qids
        self._qid2title: Dict[str, str] = qid2title
        self._qid2desc: Dict[str, str] = qid2desc

        st = time.time()
        # Convert to Tries for non edit mode
        if isinstance(qid2eid, dict):
            self._qid2eid: Union[Dict[str, int], VocabularyTrie] = VocabularyTrie(
                input_dict=qid2eid
            )
        else:
            self._qid2eid: Union[Dict[str, int], VocabularyTrie] = qid2eid

        if isinstance(alias2id, dict):
            self._alias2id: Union[Dict[str, int], VocabularyTrie] = VocabularyTrie(
                input_dict=alias2id
            )
        else:
            self._alias2id: Union[Dict[str, int], VocabularyTrie] = alias2id
        print(f"Time for creating vocab trie {time.time() - st}")

        st = time.time()
        # Make reverse functions for each of use
        self._id2alias: Union[
            Dict[int, str], Callable[[int], str]
        ] = lambda x: self._alias2id.get_key(x)
        self._eid2qid: Union[
            Dict[int, str], Callable[[int], str]
        ] = lambda x: self._qid2eid.get_key(x)
        print(f"Time for iterating indexes {time.time() - st}")

        self._qid2aliases: Union[Dict[str, set], None] = None

        assert not self._qid2eid.is_value_in_trie(
            0
        ), "0 can't be an eid. It's reserved for null candidate"

        # For when we need to add new entities
        st = time.time()
        self.max_eid = self._qid2eid.get_max_id()
        self.max_alid = self._alias2id.get_max_id()
        print(f"Time for max {time.time() - st}")

    def save(self, save_dir):
        """Dump the entity symbols.

        Args:
            save_dir: directory string to save
        """
        utils.ensure_dir(save_dir)
        utils.dump_json_file(
            filename=os.path.join(save_dir, "config.json"),
            contents={
                "max_candidates": self.max_candidates,
                "datetime": str(datetime.now()),
            },
        )
        # If in edit mode, must convert back to tris for saving
        if isinstance(self._alias2qids, dict):
            alias2qids = VocabularyPairedListTrie(
                input_dict=self._alias2qids,
                vocabulary=self._qid2title,
                max_value=self.max_candidates,
            )
            alias2qids.dump(os.path.join(save_dir, self.alias_cand_map_fld))
        else:
            self._alias2qids.dump(os.path.join(save_dir, self.alias_cand_map_fld))

        if isinstance(self._alias2id, dict):
            alias2id = VocabularyTrie(input_dict=self._alias2id)
            alias2id.dump(os.path.join(save_dir, self.alias_idx_fld))
        else:
            self._alias2id.dump(os.path.join(save_dir, self.alias_idx_fld))

        if isinstance(self._qid2eid, dict):
            qid2eid = VocabularyTrie(input_dict=self._qid2eid)
            qid2eid.dump(os.path.join(save_dir, "qid2eid"))
        else:
            self._qid2eid.dump(os.path.join(save_dir, "qid2eid"))

        utils.dump_json_file(
            filename=os.path.join(save_dir, "qid2title.json"), contents=self._qid2title
        )
        if self._qid2desc is not None:
            utils.dump_json_file(
                filename=os.path.join(save_dir, "qid2desc.json"),
                contents=self._qid2desc,
            )

    @classmethod
    def load_from_cache(
        cls,
        load_dir,
        alias_cand_map_fld="alias2qids",
        alias_idx_fld="alias2id",
        edit_mode=False,
        verbose=False,
    ):
        """Load entity symbols from load_dir.

        Args:
            load_dir: directory to load from
            alias_cand_map_fld: alias2qid file
            alias_idx_fld: alias2id file
            edit_mode: edit mode flag
            verbose: verbose flag
        """
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        max_candidates = config["max_candidates"]
        # For backwards compatibility, check if folder exists - if not, load from json
        # Future versions will assume folders exist
        st = time.time()
        alias_load_dir = os.path.join(load_dir, alias_cand_map_fld)
        if not os.path.exists(alias_load_dir):
            alias2qids: Dict[str, list] = utils.load_json_file(
                filename=os.path.join(load_dir, "alias2qids.json")
            )
        else:
            alias2qids: VocabularyPairedListTrie = VocabularyPairedListTrie(
                load_dir=alias_load_dir
            )
        print(f"Time to load alias {time.time() - st}")
        st = time.time()
        alias_id_load_dir = os.path.join(load_dir, alias_idx_fld)
        alias2id = None
        if os.path.exists(alias_id_load_dir):
            alias2id: VocabularyTrie = VocabularyTrie(load_dir=alias_id_load_dir)
        print(f"Time to load aliasid {time.time() - st}")
        st = time.time()
        eid_load_dir = os.path.join(load_dir, "qid2eid")
        qid2eid = None
        if os.path.exists(eid_load_dir):
            qid2eid: VocabularyTrie = VocabularyTrie(load_dir=eid_load_dir)
        print(f"Time to load qideid {time.time() - st}")
        st = time.time()
        qid2title: Dict[str, str] = utils.load_json_file(
            filename=os.path.join(load_dir, "qid2title.json")
        )
        print(f"Time to load alias {time.time() - st}")
        st = time.time()
        qid2desc = None
        if os.path.exists(os.path.join(load_dir, "qid2desc.json")):
            qid2desc: Dict[str, str] = utils.load_json_file(
                filename=os.path.join(load_dir, "qid2desc.json")
            )
        print(f"Time to load desc {time.time() - st}")
        return cls(
            alias2qids,
            qid2title,
            qid2desc,
            qid2eid,
            alias2id,
            max_candidates,
            alias_cand_map_fld,
            alias_idx_fld,
            edit_mode,
            verbose,
        )

    def _sort_alias_cands(
        self, alias2qids: Dict[str, list], truncate: bool = False, max_cands: int = 30
    ):
        """Sort the candidates for each alias from largest to smallest score, truncating if desired."""
        for alias in alias2qids:
            # Add second key for determinism in case of same counts
            alias2qids[alias] = sorted(
                alias2qids[alias], key=lambda x: (x[1], x[0]), reverse=True
            )
            if truncate:
                alias2qids[alias] = alias2qids[alias][:max_cands]
        return alias2qids

    def get_qid2eid_dict(self):
        """
        Get the qid2eid mapping.

        Returns: Dict qid2eid mapping
        """
        if isinstance(self._qid2eid, dict):
            return self._qid2eid
        else:
            return self._qid2eid.to_dict()

    def get_alias2qids_dict(self):
        """
        Get the alias2qids mapping.

        Key is alias, value is list of candidate tuple of length two of [QID, sort_value].

        Returns: Dict alias2qids mapping
        """
        if isinstance(self._alias2qids, dict):
            return self._alias2qids
        else:
            return self._alias2qids.to_dict()

    def get_qid2title_dict(self):
        """
        Get the qid2title mapping.

        Returns: Dict qid2title mapping
        """
        return self._qid2title

    def get_allalias_vocabtrie(self):
        """
        Get a trie of all aliases.

        Returns: Vocab trie of all aliases.
        """
        if isinstance(self._alias2id, VocabularyTrie):
            return self._alias2id
        else:
            return VocabularyTrie(input_dict=self._alias2id)

    def get_all_qids(self):
        """
        Get all QIDs.

        Returns: Dict_keys of all QIDs
        """
        return self._qid2eid.keys()

    def get_all_aliases(self):
        """
        Get all aliases.

        Returns: Dict_keys of all aliases
        """
        return self._alias2qids.keys()

    def get_all_titles(self):
        """
        Get all QID titles.

        Returns: Dict_values of all titles
        """
        return self._qid2title.values()

    def get_qid(self, id):
        """Get the QID associated with EID.

        Args:
            id: EID

        Returns: QID string
        """
        if isinstance(self._eid2qid, dict):
            return self._eid2qid[id]
        else:
            return self._eid2qid(id)

    def alias_exists(self, alias):
        """Check alias existance.

        Args:
            alias: alias string

        Returns: boolean
        """
        if isinstance(self._alias2qids, dict):
            return alias in self._alias2id
        else:
            return self._alias2qids.is_key_in_trie(alias)

    def qid_exists(self, qid):
        """Check QID existance.

        Args:
            alias: QID string

        Returns: boolean
        """
        if isinstance(self._qid2eid, dict):
            return qid in self._qid2eid
        else:
            return self._qid2eid.is_key_in_trie(qid)

    def get_eid(self, id):
        """Get the QID for the EID.

        Args:
            id: EID int

        Returns: QID string
        """
        return self._qid2eid[id]

    def _get_qid_pairs(self, alias):
        """Get the qid pairs for an alias.

        Args:
            alias: alias

        Returns: List of QID pairs
        """
        if isinstance(self._alias2qids, dict):
            qid_pairs = self._alias2qids[alias]
        else:
            qid_pairs = self._alias2qids.get_value(alias)
        return qid_pairs

    def get_qid_cands(self, alias, max_cand_pad=False):
        """Get the QID candidates for an alias.

        Args:
            alias: alias
            max_cand_pad: whether to pad with '-1' or not if fewer than max_candidates candidates

        Returns: List of QID strings
        """
        qid_pairs = self._get_qid_pairs(alias)
        res = [qid_pair[0] for qid_pair in qid_pairs]
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
        qid_pairs = self._get_qid_pairs(alias)
        res = qid_pairs
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
        qid_pairs = self._get_qid_pairs(alias)
        res = [self._qid2eid[qid_pair[0]] for qid_pair in qid_pairs]
        if max_cand_pad:
            res = res + [-1] * (self.max_candidates - len(res))
        return res

    def get_title(self, id):
        """Get title for QID.

        Args:
            id: QID string

        Returns: title string
        """
        return self._qid2title[id]

    def get_desc(self, id):
        """Get description for QID.

        Args:
            id: QID string

        Returns: title string
        """
        if self._qid2desc is None:
            return ""
        return self._qid2desc.get(id, "")

    def get_alias_idx(self, alias):
        """Get the numeric index of an alias.

        Args:
            alias: alias

        Returns: integer representation of alias
        """
        return self._alias2id[alias]

    def get_alias_from_idx(self, alias_idx):
        """Get the alias from the numeric index.

        Args:
            alias_idx: alias numeric index

        Returns: alias string
        """
        if isinstance(self._id2alias, dict):
            alias = self._id2alias[alias_idx]
        else:
            alias = self._id2alias(alias_idx)
        return alias

    # ============================================================
    # EDIT MODE OPERATIONS
    # ============================================================

    @edit_op
    def set_title(self, qid: str, title: str):
        """Set the title for a QID.

        Args:
            qid: QID
            title: title
        """
        assert qid in self._qid2eid
        self._qid2title[qid] = title

    @edit_op
    def set_desc(self, qid: str, desc: str):
        """Set the description for a QID.

        Args:
            qid: QID
            desc: description
        """
        assert qid in self._qid2eid
        self._qid2desc[qid] = desc

    @edit_op
    def set_score(self, qid: str, mention: str, score: float):
        """Change the mention QID score and resorts candidates.

        Highest score is first.

        Args:
            qid: QID
            mention: mention
            score: score
        """
        if mention not in self._alias2qids:
            raise ValueError(f"The mention {mention} is not in our mapping")
        qids_only = list(map(lambda x: x[0], self._alias2qids[mention]))
        if qid not in set(qids_only):
            raise ValueError(
                f"The qid {qid} is not already associated with that mention."
            )
        qid_idx = qids_only.index(qid)
        assert self._alias2qids[mention][qid_idx][0] == qid
        self._alias2qids[mention][qid_idx][1] = score
        self._alias2qids[mention] = sorted(
            self._alias2qids[mention], key=lambda x: x[1], reverse=True
        )
        return

    @edit_op
    def add_mention(self, qid: str, mention: str, score: float):
        """Add mention to QID with the associated score.

        The mention already exists, error thrown to call ``set_score`` instead.
        If there are already max candidates to that mention, the last candidate of the
        mention is removed in place of QID.

        Args:
            qid: QID
            mention: mention
            score: score
        """
        # Cast to lower and stripped for aliases
        mention = utils.get_lnrm(mention, strip=True, lower=True)

        # If mention is in mapping, make sure the qid is not
        if mention in self._alias2qids:
            if qid in set(map(lambda x: x[0], self._alias2qids[mention])):
                logger.warning(
                    f"The QID {qid} is already associated with {mention}. Use set_score if you want to change "
                    f"the score of an existing mention-qid pair"
                )
                return
        # If mention is not in mapping, add it
        if mention not in self._alias2qids:
            self._alias2qids[mention] = []
            new_al_id = self.max_alid + 1
            self.max_alid += 1
            assert (
                new_al_id not in self._id2alias
            ), f"{new_al_id} already in self_id2alias"
            self._alias2id[mention] = new_al_id
            self._id2alias[new_al_id] = mention
            # msg = f"You have added a new mention to the dataset. You MUST reprep you data for this to take effect.
            # Set data_config.overwrite_preprocessed_data to be True. This warning will now be supressed."
            # logger.warning(msg)
            # warnings.filterwarnings("ignore", message=msg)

        assert (
            mention not in self._qid2aliases[qid]
        ), f"{mention} was a mention for {qid} despite the alias mapping saying otherwise"
        # If adding will go beyond max candidates, remove the last candidate. Even if the score is higher,
        # the user still wants this mention added.
        if len(self._alias2qids[mention]) >= self.max_candidates:
            qid_to_remove = self._alias2qids[mention][-1][0]
            self.remove_mention(qid_to_remove, mention)
            assert (
                len(self._alias2qids[mention]) < self.max_candidates
            ), f"Invalid state: {mention} still has more than {self.max_candidates} candidates after removal"
        # Add pair
        self._alias2qids[mention].append([qid, score])
        self._alias2qids[mention] = sorted(
            self._alias2qids[mention], key=lambda x: x[1], reverse=True
        )
        self._qid2aliases[qid].add(mention)

    @edit_op
    def remove_mention(self, qid, mention):
        """Remove the mention from those associated with the QID.

        Args:
            qid: QID
            mention: mention to remove
        """
        # Make sure the mention and qid pair is already in the mapping
        if mention not in self._alias2qids:
            return
        qids_only = list(map(lambda x: x[0], self._alias2qids[mention]))
        if qid not in set(qids_only):
            return

        # Remove the QID
        idx_to_remove = qids_only.index(qid)
        self._alias2qids[mention].pop(idx_to_remove)

        # If the mention has NO candidates, remove it as a possible mention
        if len(self._alias2qids[mention]) == 0:
            del self._alias2qids[mention]
            al_id = self._alias2id[mention]
            del self._alias2id[mention]
            del self._id2alias[al_id]
            assert (
                mention not in self._alias2qids and mention not in self._alias2id
            ), f"Removal of no candidates mention {mention} failed"
            # msg = f"You have removed all candidates for an existing mention, which will now be removed.
            # You MUST reprep you data for this to take effect. Set data_config.overwrite_preprocessed_data to be
            # True. This warning will now be supressed."
            # logger.warning(msg)
            # warnings.filterwarnings("ignore", message=msg)

        # Remove mention from inverse mapping (will be not None in edit mode)
        assert (
            mention in self._qid2aliases[qid]
        ), f"{mention} was not a mention for {qid} despite the reverse being true"
        self._qid2aliases[qid].remove(mention)
        return

    @edit_op
    def add_entity(self, qid, mentions, title, desc=""):
        """Add entity QID to our mappings with its mentions and title.

        Args:
            qid: QID
            mentions: List of tuples [mention, score]
            title: title
            desc: description
        """
        assert (
            qid not in self._qid2eid
        ), "Something went wrong with the qid check that this entity doesn't exist"
        # Update eid
        new_eid = self.max_eid + 1
        assert new_eid not in self._eid2qid
        self._qid2eid[qid] = new_eid
        self._eid2qid[new_eid] = qid
        # Update title
        self._qid2title[qid] = title
        # Update description
        self._qid2desc[qid] = desc
        # Make empty list to add in add_mention
        self._qid2aliases[qid] = set()
        # Update mentions
        for mention_pair in mentions:
            self.add_mention(qid, mention_pair[0], mention_pair[1])
        # Update metrics at the end in case of failure
        self.max_eid += 1
        self.num_entities += 1
        self.num_entities_with_pad_and_nocand += 1

    @edit_op
    def reidentify_entity(self, old_qid, new_qid):
        """Rename ``old_qid`` to ``new_qid``.

        Args:
            old_qid: old QID
            new_qid: new QID
        """
        assert (
            old_qid in self._qid2eid and new_qid not in self._qid2eid
        ), f"Internal Error: checks on existing versus new qid for {old_qid} and {new_qid} failed"
        # Save state
        eid = self._qid2eid[old_qid]
        mentions = self.get_mentions(old_qid)
        # Update qid2eid
        self._qid2eid[new_qid] = self._qid2eid[old_qid]
        del self._qid2eid[old_qid]
        # Reassign eid
        self._eid2qid[eid] = new_qid
        # Update qid2title
        self._qid2title[new_qid] = self._qid2title[old_qid]
        del self._qid2title[old_qid]
        # Update qid2desc
        self._qid2desc[new_qid] = self.get_desc(old_qid)
        del self._qid2desc[old_qid]
        # Update qid2aliases
        self._qid2aliases[new_qid] = self._qid2aliases[old_qid]
        del self._qid2aliases[old_qid]
        # Update alias2qids
        for mention in mentions:
            for i in range(len(self._alias2qids[mention])):
                if self._alias2qids[mention][i][0] == old_qid:
                    self._alias2qids[mention][i][0] = new_qid
                    break

    @edit_op
    def prune_to_entities(self, entities_to_keep):
        """Remove all entities except those in ``entities_to_keep``.

        Args:
            entities_to_keep: Set of entities to keep
        """
        # Update qid based dictionaries
        self._qid2title = {
            k: v for k, v in self._qid2title.items() if k in entities_to_keep
        }
        if self._qid2desc is not None:
            self._qid2desc = {
                k: v for k, v in self._qid2desc.items() if k in entities_to_keep
            }
        self._qid2aliases = {
            k: v for k, v in self._qid2aliases.items() if k in entities_to_keep
        }
        # Reindex the entities to compress the embedding matrix (when model is update)
        self._qid2eid = {k: i + 1 for i, k in enumerate(sorted(entities_to_keep))}
        self._eid2qid = {eid: qid for qid, eid in self._qid2eid.items()}
        # Extract mentions to keep
        mentions_to_keep = set().union(*self._qid2aliases.values())
        # Reindex aliases
        self._alias2id = {v: i for i, v in enumerate(sorted(mentions_to_keep))}
        self._id2alias = {id: al for al, id in self._alias2id.items()}
        # Rebuild self._alias2qids
        new_alias2qids = {}
        for al in mentions_to_keep:
            new_alias2qids[al] = [
                pair for pair in self._alias2qids[al] if pair[0] in entities_to_keep
            ][: self.max_candidates]
            assert len(new_alias2qids[al]) > 0
        self._alias2qids = new_alias2qids

        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2
        assert self.num_entities == len(entities_to_keep)
        # For when we need to add new entities
        self.max_eid = max(self._eid2qid.keys())
        self.max_alid = max(self._id2alias.keys())

    @edit_op
    def get_mentions(self, qid):
        """Get the mentions for the QID.

        Args:
            qid: QID

        Returns: List of mentions
        """
        # qid2aliases is only created in edit mode to allow for removal of mentions associated with a qid
        return self._qid2aliases[qid]

    @edit_op
    def get_mentions_with_scores(self, qid):
        """Get the mentions and the associated score for the QID.

        Args:
            qid: QID

        Returns: List of tuples [mention, score]
        """
        mentions = self._qid2aliases[qid]
        res = []
        for men in mentions:
            for qid_pair in self._alias2qids[men]:
                if qid_pair[0] == qid:
                    res.append([men, qid_pair[1]])
                    break
        return list(sorted(res, key=lambda x: x[1], reverse=True))
