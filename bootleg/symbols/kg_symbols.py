"""KG symbols class."""

import os
from datetime import datetime
from typing import Dict, List, Optional

import ujson as json
from tqdm import tqdm

from bootleg.symbols.constants import edit_op
from bootleg.utils import utils


class KGSymbols:
    def __init__(
        self,
        qid2relations: Dict[str, Dict[str, List[str]]],
        max_connections: Optional[int] = 150,
        edit_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        self.max_connections = max_connections
        self.edit_mode = edit_mode
        self.verbose = verbose
        self._qid2relations: Dict[str, Dict[str, List[str]]] = qid2relations
        self._all_relations = set()
        # EDIT MODE ONLY: Storing inverse edges to quickly reidentify entities
        self._obj2head = None
        if self.edit_mode:
            self._obj2head = dict()
        for qid in tqdm(
            self._qid2relations,
            total=len(self._qid2relations),
            desc="Checking relations and building edit mode objs",
            disable=not verbose,
        ):
            for rel in self._qid2relations[qid]:
                self._all_relations.add(rel)
                self._qid2relations[qid][rel] = self._qid2relations[qid][rel][
                    : self.max_connections
                ]
                if self.edit_mode:
                    for qid2 in self._qid2relations[qid][rel]:
                        if qid2 not in self._obj2head:
                            self._obj2head[qid2] = set()
                        self._obj2head[qid2].add(qid)

    def save(self, save_dir, prefix=""):
        """Dumps the kg symbols.

        Args:
            save_dir: directory string to save
            prefix: prefix to add to beginning to file

        Returns:
        """
        utils.ensure_dir(str(save_dir))
        utils.dump_json_file(
            filename=os.path.join(save_dir, "config.json"),
            contents={
                "max_connections": self.max_connections,
                "datetime": str(datetime.now()),
            },
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, f"{prefix}qid2relations.json"),
            contents=self._qid2relations,
        )
        # For backwards compatability with model, dump the adjacency matrix, too
        with open(os.path.join(save_dir, f"{prefix}qid2qid_adj.txt"), "w") as out_f:
            for qid in self._qid2relations:
                for rel in self._qid2relations[qid]:
                    for qid2 in self._qid2relations[qid][rel]:
                        out_f.write(f"{qid}\t{qid2}\n")

    @classmethod
    def load_from_cache(cls, load_dir, prefix="", edit_mode=False, verbose=False):
        """Loads type symbols from load_dir.

        Args:
            load_dir: directory to load from
            prefix: prefix to add to beginning to file
            edit_mode: edit mode

        Returns: TypeSymbols
        """
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        max_connections = config["max_connections"]
        qid2relations: Dict[str, Dict[str, List[str]]] = utils.load_json_file(
            filename=os.path.join(load_dir, f"{prefix}qid2relations.json")
        )
        return cls(qid2relations, max_connections, edit_mode, verbose)

    def get_connections_by_relation(self, qid, relation):
        return self._qid2relations[qid].get(relation, {})

    def get_all_connections(self, qid):
        return self._qid2relations[qid]

    def get_all_relations(self):
        return self._all_relations

    def is_connected(self, qid1, qid2):
        """Checks if two QIDs are connected in KG.

        Args:
            qid1: QID one
            qid2: QID two

        Returns: boolean
        """
        for rel in self._qid2relations.get(qid1, {}):
            if qid2 in self._qid2relations[qid1][rel]:
                return True
        return False

    # ============================================================
    # EDIT MODE OPERATIONS
    # ============================================================
    @edit_op
    def add_relation(self, qid, relation, qid2):
        if relation not in self._all_relations:
            raise ValueError(
                f"Tried adding {relation} to qid {qid}. We do not support new relations."
            )

        if relation not in self._qid2relations[qid]:
            self._qid2relations[qid][relation] = []
        # Check if qid2 already in that relation
        if qid2 in self._qid2relations[qid][relation]:
            return
        if len(self._qid2relations[qid][relation]) >= self.max_connections:
            qid_to_remove = self._qid2relations[qid][relation][-1]
            self.remove_relation(qid, relation, qid_to_remove)
            assert (
                len(self._qid2relations[qid][relation]) < self.max_connections
            ), f"Something went wrong and we still have more that {self.max_connections} relations when removing {qid}, {relation}, {qid2}"
        self._qid2relations[qid][relation].append(qid2)
        if qid2 not in self._obj2head:
            self._obj2head[qid2] = set()
        self._obj2head[qid2].add(qid)
        return

    @edit_op
    def remove_relation(self, qid, relation, qid2):
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
        for relation in relation_dict:
            if relation not in self._all_relations:
                raise ValueError(
                    f"Tried adding {relation} to new qid {qid}. We do not support new relations."
                )
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
        assert (
            old_qid in self._qid2relations and new_qid not in self._qid2relations
        ), f"Internal Error: checks on existing versus new qid for {old_qid} and {new_qid} failed where {old_qid in self._qid2relations} and {new_qid not in self._qid2relations}"
        # Update all object qids (aka subjects-object pairs where the object is the old qid)
        for subj_qid in self._obj2head.get(old_qid, {}):
            for rel in self._qid2relations[subj_qid]:
                if old_qid in self._qid2relations[subj_qid][rel]:
                    for j in range(len(self._qid2relations[subj_qid][rel])):
                        if self._qid2relations[subj_qid][rel][j] == old_qid:
                            self._qid2relations[subj_qid][rel][j] = new_qid
        # Update all subject qids - take the set union in case a subject has the same object with different relationships
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
