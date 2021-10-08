"""Entity symbols."""
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import marisa_trie
from tqdm import tqdm

import bootleg.utils.utils as utils
from bootleg.symbols.constants import edit_op

logger = logging.getLogger(__name__)


class EntitySymbols:
    """Entity Symbols class for managing entity metadata."""

    def __init__(
        self,
        alias2qids: Dict[str, list],
        qid2title: Dict[str, str],
        qid2desc: Dict[str, str] = None,
        qid2eid: Optional[Dict[str, int]] = None,
        alias2id: Optional[Dict[str, int]] = None,
        max_candidates: int = 30,
        alias_cand_map_file: str = "alias2qids.json",
        alias_idx_file: str = "alias2id.json",
        edit_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Entity symbols initializer."""
        # We support different candidate mappings for the same set of entities
        self.alias_cand_map_file = alias_cand_map_file
        self.alias_idx_file = alias_idx_file
        self.max_candidates = max_candidates
        self.edit_mode = edit_mode
        self.verbose = verbose
        # Used if we need to do any string searching for aliases. This keep track of
        # the largest n-gram needed.
        self._alias2qids: Dict[str, list] = alias2qids
        self._qid2title: Dict[str, str] = qid2title
        self._qid2desc: Dict[str, str] = qid2desc
        # Add empty description for entities
        if self._qid2desc is not None:
            for qid in self._qid2title:
                if qid not in self._qid2desc:
                    self._qid2desc[qid] = ""
        # Sort by score and filter to max candidates
        self._sort_alias_cands()
        for al in list(self._alias2qids.keys()):
            self._alias2qids[al] = self._alias2qids[al][: self.max_candidates]

        if qid2eid is None:
            # Add 1 for the noncand class
            # If we load stuff up in self.load() and regenerate these,
            # the eid values may be nondeterministic
            self._qid2eid: Dict[str, int] = {v: i + 1 for i, v in enumerate(qid2title)}
        else:
            self._qid2eid = qid2eid

        # Keep an index for each alias
        if alias2id is None:
            self._alias2id: Dict[str, int] = {
                v: i for i, v in enumerate(sorted(self._alias2qids.keys()))
            }
        else:
            self._alias2id = alias2id

        self._id2alias: Dict[int, str] = {id: al for al, id in self._alias2id.items()}
        self._eid2qid: Dict[int, str] = {eid: qid for qid, eid in self._qid2eid.items()}
        assert len(self._qid2eid) == len(self._eid2qid), (
            "The qid2eid mapping is not invertable. "
            "This means there is a duplicate id value."
        )
        assert -1 not in self._eid2qid, "-1 can't be an eid"
        assert (
            0 not in self._eid2qid
        ), "0 can't be an eid. It's reserved for null candidate"
        # this assumes that eid of 0 is NO_CAND and eid of -1 is NULL entity
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2
        # For when we need to add new entities
        self.max_eid = max(self._eid2qid.keys())
        self.max_alid = max(self._id2alias.keys())
        self._qid2aliases = None
        self._alias_trie = None
        if self.edit_mode:
            self._qid2aliases = {}
            for al in tqdm(
                self._alias2qids,
                total=len(self._alias2qids),
                desc="Building edit mode objs",
                disable=not verbose,
            ):
                for qid_pair in self._alias2qids[al]:
                    if qid_pair[0] not in self._qid2aliases:
                        self._qid2aliases[qid_pair[0]] = set()
                    self._qid2aliases[qid_pair[0]].add(al)
        else:
            # generate trie of aliases for quick entity generation in sentences
            self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())

    def save(self, save_dir):
        """Dump the entity symbols.

        Args:
            save_dir: directory string to save
        """
        self._sort_alias_cands()
        utils.ensure_dir(save_dir)
        utils.dump_json_file(
            filename=os.path.join(save_dir, "config.json"),
            contents={
                "max_candidates": self.max_candidates,
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
        if self._qid2desc is not None:
            utils.dump_json_file(
                filename=os.path.join(save_dir, "qid2desc.json"),
                contents=self._qid2desc,
            )
        utils.dump_json_file(
            filename=os.path.join(save_dir, "qid2eid.json"), contents=self._qid2eid
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, self.alias_idx_file),
            contents=self._alias2id,
        )

    @classmethod
    def load_from_cache(
        cls,
        load_dir,
        alias_cand_map_file="alias2qids.json",
        alias_idx_file="alias2id.json",
        edit_mode=False,
        verbose=False,
    ):
        """Load entity symbols from load_dir.

        Args:
            load_dir: directory to load from
            alias_cand_map_file: alias2qid file
            alias_idx_file: alias2id file
            edit_mode: edit mode flag
            verbose: verbose flag
        """
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        max_candidates = config["max_candidates"]
        alias2qids: Dict[str, list] = utils.load_json_file(
            filename=os.path.join(load_dir, alias_cand_map_file)
        )
        qid2title: Dict[str, str] = utils.load_json_file(
            filename=os.path.join(load_dir, "qid2title.json")
        )
        qid2desc = None
        if os.path.exists(os.path.join(load_dir, "qid2desc.json")):
            qid2desc: Dict[str, str] = utils.load_json_file(
                filename=os.path.join(load_dir, "qid2desc.json")
            )
        qid2eid: Dict[str, int] = utils.load_json_file(
            filename=os.path.join(load_dir, "qid2eid.json")
        )
        alias2id: Dict[str, int] = utils.load_json_file(
            filename=os.path.join(load_dir, alias_idx_file)
        )
        return cls(
            alias2qids,
            qid2title,
            qid2desc,
            qid2eid,
            alias2id,
            max_candidates,
            alias_cand_map_file,
            alias_idx_file,
            edit_mode,
            verbose,
        )

    def _sort_alias_cands(self):
        """Sort the candidates for each alias from largest to smallest score."""
        for alias in self._alias2qids:
            # Add second key for determinism in case of same counts
            self._alias2qids[alias] = sorted(
                self._alias2qids[alias], key=lambda x: (x[1], x[0]), reverse=True
            )

    def get_qid2eid(self):
        """
        Get the qid2eid mapping.

        Returns: Dict qid2eid mapping
        """
        return self._qid2eid

    def get_alias2qids(self):
        """
        Get the alias2qids mapping.

        Key is alias, value is list of candidate tuple of length two of [QID, sort_value].

        Returns: Dict alias2qids mapping
        """
        return self._alias2qids

    def get_qid2title(self):
        """
        Get the qid2title mapping.

        Returns: Dict qid2title mapping
        """
        return self._qid2title

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
        assert id in self._eid2qid
        return self._eid2qid[id]

    def alias_exists(self, alias):
        """Check alias existance.

        Args:
            alias: alias string

        Returns: boolean
        """
        if not self.edit_mode:
            return alias in self._alias_trie
        else:
            return alias in self._alias2id

    def qid_exists(self, qid):
        """Check QID existance.

        Args:
            alias: QID string

        Returns: boolean
        """
        return qid in self._qid2eid

    def eid_exists(self, eid):
        """Check EID existance.

        Args:
            alias: EID int

        Returns: boolean
        """
        return eid in self._eid2qid[eid]

    def get_eid(self, id):
        """Get the QID for the EID.

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
        """Get title for QID.

        Args:
            id: QID string

        Returns: title string
        """
        assert id in self._qid2title
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
        assert alias_idx in self._id2alias
        return self._id2alias[alias_idx]

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
        self._qid2desc[new_qid] = self._qid2desc[old_qid]
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
