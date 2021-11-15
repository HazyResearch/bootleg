"""KG symbols class."""
import copy
import os
import re
from typing import Dict, List, Optional, Set, Union

from tqdm import tqdm

from bootleg.symbols.constants import edit_op
from bootleg.utils import utils
from bootleg.utils.classes.dictvocabulary_tries import ThreeLayerVocabularyTrie


def _convert_to_trie(qid2relations, max_connections):
    all_relations = set()
    all_qids = set()
    qid2relations_filt = {}
    for q, rel_dict in qid2relations.items():
        qid2relations_filt[q] = {}
        for rel, tail_qs in rel_dict.items():
            all_qids.update(set(tail_qs))
            all_relations.add(rel)
            qid2relations_filt[q][rel] = tail_qs[:max_connections]
    qid2relations_trie = ThreeLayerVocabularyTrie(
        input_dict=qid2relations_filt,
        key_vocabulary=all_relations,
        value_vocabulary=all_qids,
        max_value=max_connections,
    )
    return qid2relations_trie


class KGSymbols:
    """KG Symbols class for managing KG metadata."""

    def __init__(
        self,
        qid2relations: Union[Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie],
        max_connections: Optional[int] = 50,
        edit_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """KG initializer.

        max_connections acts as the max single number of connections for a given relation.
        max_connections * 2 is the max number of connections across all relations for a
        given entity (see ThreeLayerVocabularyTrie).
        """
        self.max_connections = max_connections
        self.edit_mode = edit_mode
        self.verbose = verbose

        if self.edit_mode:
            self._load_edit_mode(
                qid2relations,
            )
        else:
            self._load_non_edit_mode(
                qid2relations,
            )

    def _load_edit_mode(
        self,
        qid2relations: Union[Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie],
    ):
        """Load relations in edit mode."""
        if isinstance(qid2relations, ThreeLayerVocabularyTrie):
            self._qid2relations: Union[
                Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie
            ] = qid2relations.to_dict()
        else:
            self._qid2relations: Union[
                Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie
            ] = {
                head_qid: {
                    rel: tail_qids[: self.max_connections]
                    for rel, tail_qids in rel_dict.items()
                }
                for head_qid, rel_dict in qid2relations.items()
            }
        self._obj2head: Union[Dict[str, set], None] = {}
        self._all_relations: Union[Set[str], None] = set()
        for qid in tqdm(
            self._qid2relations,
            total=len(self._qid2relations),
            desc="Checking relations and building edit mode objs",
            disable=not self.verbose,
        ):
            for rel in self._qid2relations[qid]:
                self._all_relations.add(rel)
                for qid2 in self._qid2relations[qid][rel]:
                    if qid2 not in self._obj2head:
                        self._obj2head[qid2] = set()
                    self._obj2head[qid2].add(qid)

    def _load_non_edit_mode(
        self,
        qid2relations: Union[Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie],
    ):
        """Load relations in not edit mode."""
        if isinstance(qid2relations, dict):
            self._qid2relations: Union[
                Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie
            ] = _convert_to_trie(qid2relations, self.max_connections)
        else:
            self._qid2relations: Union[
                Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie
            ] = qid2relations
        self._all_relations: Union[Set[str], None] = None
        self._obj2head: Union[Dict[str, set], None] = None

    def save(self, save_dir, prefix=""):
        """Dump the kg symbols.

        Args:
            save_dir: directory string to save
            prefix: prefix to add to beginning to file
        """
        utils.ensure_dir(str(save_dir))
        utils.dump_json_file(
            filename=os.path.join(save_dir, "config.json"),
            contents={
                "max_connections": self.max_connections,
            },
        )
        if isinstance(self._qid2relations, dict):
            qid2relations = _convert_to_trie(self._qid2relations, self.max_connections)
            qid2relations.dump(os.path.join(save_dir, f"{prefix}qid2relations"))
        else:
            self._qid2relations.dump(os.path.join(save_dir, f"{prefix}qid2relations"))

    @classmethod
    def load_from_cache(cls, load_dir, prefix="", edit_mode=False, verbose=False):
        """Load type symbols from load_dir.

        Args:
            load_dir: directory to load from
            prefix: prefix to add to beginning to file
            edit_mode: edit mode
            verbose: verbose flag

        Returns: TypeSymbols
        """
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        max_connections = config["max_connections"]
        # For backwards compatibility, check if trie directory exists, otherwise load from json
        rel_load_dir = os.path.join(load_dir, f"{prefix}qid2relations")
        if not os.path.exists(rel_load_dir):
            qid2relations: Union[
                Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie
            ] = utils.load_json_file(
                filename=os.path.join(load_dir, f"{prefix}qid2relations.json")
            )
            # Make sure relation is _not_ PID. The user should have the qid2relation dict that is pre-translated
            first_qid = next(iter(qid2relations.keys()))
            first_rel = next(iter(qid2relations[first_qid].keys()))
            if re.match("^P[0-9]+$", first_rel):
                raise ValueError(
                    "Your qid2relations dict has a relation as a PID identifier. Please replace "
                    "with human readable strings for training. "
                    "See https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all"
                )
        else:
            qid2relations: Union[
                Dict[str, Dict[str, List[str]]], ThreeLayerVocabularyTrie
            ] = ThreeLayerVocabularyTrie(
                load_dir=rel_load_dir, max_value=max_connections
            )
        return cls(qid2relations, max_connections, edit_mode, verbose)

    def get_qid2relations_dict(self):
        """Return a dictionary form of the relation to qid mappings object.

        Returns: Dict of relation to head qid to list of tail qids
        """
        if isinstance(self._qid2relations, dict):
            return copy.deepcopy(self._qid2relations)
        else:
            return self._qid2relations.to_dict()

    def get_all_relations(self):
        """Get all relations in our KG mapping.

        Returns: Set
        """
        if isinstance(self._qid2relations, dict):
            return self._all_relations
        else:
            return set(self._qid2relations.key_vocab_keys())

    def get_relation_between(self, qid1, qid2):
        """Check if two QIDs are connected in KG and returns their relation.

        Args:
            qid1: QID one
            qid2: QID two

        Returns: string relation or None
        """
        rel_dict = {}
        if isinstance(self._qid2relations, dict):
            rel_dict = self._qid2relations.get(qid1, {})
        else:
            if self._qid2relations.is_key_in_trie(qid1):
                rel_dict = self._qid2relations.get_value(qid1)

        for rel, tail_qids in rel_dict.items():
            if qid2 in set(tail_qids):
                return rel
        return None

    def get_relations_tails_for_qid(self, qid):
        """Get dict of relation to tail qids for given qid.

        Args:
            qid: QID

        Returns: Dict relation to list of tail qids for that relation
        """
        if isinstance(self._qid2relations, dict):
            return self._qid2relations.get(qid, {})
        else:
            rel_dict = {}
            if self._qid2relations.is_key_in_trie(qid):
                rel_dict = self._qid2relations.get_value(qid)
        return rel_dict

    # ============================================================
    # EDIT MODE OPERATIONS
    # ============================================================
    @edit_op
    def add_relation(self, qid, relation, qid2):
        """Add a relationship triple to our mapping.

        If the QID already has max connection through ``relation``,
        the last ``other_qid`` is removed and replaced by ``qid2``.

        Args:
            qid: head entity QID
            relation: relation
            qid2: tail entity QID:
        """
        if relation not in self._all_relations:
            self._all_relations.add(relation)
        if relation not in self._qid2relations[qid]:
            self._qid2relations[qid][relation] = []
        # Check if qid2 already in that relation
        if qid2 in self._qid2relations[qid][relation]:
            return
        if len(self._qid2relations[qid][relation]) >= self.max_connections:
            qid_to_remove = self._qid2relations[qid][relation][-1]
            self.remove_relation(qid, relation, qid_to_remove)
            assert len(self._qid2relations[qid][relation]) < self.max_connections, (
                f"Something went wrong and we still have more that {self.max_connections} "
                f"relations when removing {qid}, {relation}, {qid2}"
            )
        self._qid2relations[qid][relation].append(qid2)
        if qid2 not in self._obj2head:
            self._obj2head[qid2] = set()
        self._obj2head[qid2].add(qid)
        return

    @edit_op
    def remove_relation(self, qid, relation, qid2):
        """Remove a relation triple from our mapping.

        Args:
            qid: head entity QID
            relation: relation
            qid2: tail entity QID
        """
        if relation not in self._qid2relations[qid]:
            return
        if qid2 not in self._qid2relations[qid][relation]:
            return
        self._qid2relations[qid][relation].remove(qid2)
        self._obj2head[qid2].remove(qid)
        # If no connections, remove relation
        if len(self._qid2relations[qid][relation]) <= 0:
            del self._qid2relations[qid][relation]
        if len(self._obj2head[qid2]) <= 0:
            del self._obj2head[qid2]
        return

    @edit_op
    def add_entity(self, qid, relation_dict):
        """Add a new entity to our relation mapping.

        Args:
            qid: QID
            relation_dict: dictionary of relation -> list of connected other_qids by relation
        """
        if qid in self._qid2relations:
            raise ValueError(f"{qid} is already in kg symbols")
        for relation in relation_dict:
            if relation not in self._all_relations:
                self._all_relations.add(relation)
        self._qid2relations[qid] = relation_dict.copy()
        for rel in self._qid2relations[qid]:
            self._qid2relations[qid][rel] = self._qid2relations[qid][rel][
                : self.max_connections
            ]
        # Use self._qid2relations[qid] rather than relation_dict as the former is limited by max connections
        for rel in self._qid2relations[qid]:
            for obj_qid in self._qid2relations[qid][rel]:
                if obj_qid not in self._obj2head:
                    self._obj2head[obj_qid] = set()
                self._obj2head[obj_qid].add(qid)
        return

    @edit_op
    def reidentify_entity(self, old_qid, new_qid):
        """Rename ``old_qid`` to ``new_qid``.

        Args:
            old_qid: old QID
            new_qid: new QID
        """
        if old_qid not in self._qid2relations or new_qid in self._qid2relations:
            raise ValueError(
                f"Either old qid {old_qid} is not in kg symbols or new qid {new_qid} is already in kg symbols"
            )
        # Update all object qids (aka subjects-object pairs where the object is the old qid)
        for subj_qid in self._obj2head.get(old_qid, {}):
            for rel in self._qid2relations[subj_qid]:
                if old_qid in self._qid2relations[subj_qid][rel]:
                    for j in range(len(self._qid2relations[subj_qid][rel])):
                        if self._qid2relations[subj_qid][rel][j] == old_qid:
                            self._qid2relations[subj_qid][rel][j] = new_qid
        # Update all subject qids - take the set union in case a subject has the same object with different relations
        for obj_qid in set().union(
            *[
                set(self._qid2relations[old_qid][rel])
                for rel in self._qid2relations[old_qid]
            ]
        ):
            # May get cyclic relationship ann the obj qid qill already have been transformed
            if obj_qid == new_qid:
                obj_qid = old_qid
            assert (
                old_qid in self._obj2head[obj_qid]
            ), f"{old_qid} {obj_qid} {self._obj2head[obj_qid]}"
            self._obj2head[obj_qid].remove(old_qid)
            self._obj2head[obj_qid].add(new_qid)
        # Update qid2relations and the object2head mappings
        self._qid2relations[new_qid] = self._qid2relations[old_qid]
        del self._qid2relations[old_qid]
        if old_qid in self._obj2head:
            self._obj2head[new_qid] = self._obj2head[old_qid]
            del self._obj2head[old_qid]

    @edit_op
    def prune_to_entities(self, entities_to_keep):
        """Remove all entities except those in ``entities_to_keep``.

        Args:
            entities_to_keep: Set of entities to keep
        """
        # Update qid2relations
        self._qid2relations = {
            k: v for k, v in self._qid2relations.items() if k in entities_to_keep
        }
        new_obj2head = {}
        # Update all object qids
        for qid in self._qid2relations:
            for rel in list(self._qid2relations[qid].keys()):
                filtered_object_ents = [
                    j for j in self._qid2relations[qid][rel] if j in entities_to_keep
                ][: self.max_connections]
                # Keep relation only if more than one object
                if len(filtered_object_ents) > 0:
                    self._qid2relations[qid][rel] = filtered_object_ents
                    for obj_qid in filtered_object_ents:
                        if obj_qid not in new_obj2head:
                            new_obj2head[obj_qid] = set()
                        new_obj2head[obj_qid].add(qid)
                else:
                    del self._qid2relations[qid][rel]
        self._obj2head = new_obj2head
