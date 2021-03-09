import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ujson
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from bootleg.symbols.constants import check_qid_exists, edit_op
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.kg_symbols import KGSymbols
from bootleg.symbols.type_symbols import TypeSymbols

logger = logging.getLogger(__name__)

ENTITY_SUBFOLDER = "entity_mappings"
TYPE_SUBFOLDER = "type_mappings"
KG_SUBFOLDER = "kg_mappings"
REQUIRED_KEYS = ["entity_id", "mentions"]
OTHER_KEYS = ["title", "types", "relations"]


class EntityObj(BaseModel):
    """Base entity object class to check types."""

    entity_id: str
    mentions: List[Tuple[str, float]]
    title: str
    types: Optional[Dict[str, List[str]]]
    relations: Optional[List[Dict[str, str]]]


class EntityProfile:
    def __init__(
        self,
        entity_symbols,
        type_systems=None,
        kg_symbols=None,
        edit_mode=False,
        verbose=False,
    ):
        self.edit_mode = edit_mode
        self.verbose = verbose
        self._entity_symbols = entity_symbols
        self._type_systems = type_systems
        self._kg_symbols = kg_symbols

    def save(self, save_dir):
        save_dir = Path(save_dir)
        self._entity_symbols.save(save_dir / ENTITY_SUBFOLDER)
        for type_sys in self._type_systems:
            self._type_systems[type_sys].save(save_dir / TYPE_SUBFOLDER / type_sys)
        self._kg_symbols.save(save_dir / KG_SUBFOLDER)

    @classmethod
    def load_from_cache(cls, load_dir, edit_mode=False, verbose=False):
        load_dir = Path(load_dir)
        if verbose:
            print("Loading Entity Symbols")
        entity_symbols = EntitySymbols.load_from_cache(
            load_dir / ENTITY_SUBFOLDER,
            edit_mode=edit_mode,
            verbose=verbose,
        )
        type_systems = {}
        type_subfolder = load_dir / TYPE_SUBFOLDER
        for fold in type_subfolder.iterdir():
            if fold.is_dir():
                if verbose:
                    print(f"Loading Type Symbols from {fold}")
                type_systems[fold.name] = TypeSymbols.load_from_cache(
                    type_subfolder / fold.name,
                    edit_mode=edit_mode,
                    verbose=verbose,
                )
        if verbose:
            print(f"Loading KG Symbols")
        kg_symbols = KGSymbols.load_from_cache(
            load_dir / KG_SUBFOLDER,
            edit_mode=edit_mode,
            verbose=verbose,
        )
        return cls(entity_symbols, type_systems, kg_symbols, edit_mode, verbose)

    @classmethod
    def load_from_jsonl(
        cls,
        profile_file,
        max_candidates=30,
        max_types=10,
        max_kg_connections=100,
        edit_mode=False,
    ):
        qid2title, alias2qids, type_systems, qid2relations = cls._read_profile_file(
            profile_file
        )
        entity_symbols = EntitySymbols(
            alias2qids=alias2qids,
            qid2title=qid2title,
            max_candidates=max_candidates,
            edit_mode=edit_mode,
        )

        all_type_symbols = {
            ty_name: TypeSymbols(
                qid2typenames=type_map, max_types=max_types, edit_mode=edit_mode
            )
            for ty_name, type_map in type_systems.items()
        }
        kg_symbols = KGSymbols(
            qid2relations, max_connections=max_kg_connections, edit_mode=edit_mode
        )
        return cls(entity_symbols, all_type_symbols, kg_symbols, edit_mode)

    @classmethod
    def _read_profile_file(cls, profile_file):
        qid2title: Dict[str, str] = {}
        alias2qids: Dict[str, list] = {}
        type_systems: Dict[str, Dict[str, List[str]]] = {}
        qid2relations: Dict[str, Dict[str, List[str]]] = {}

        num_lines = sum(1 for _ in open(profile_file))
        with open(profile_file, "r") as in_f:
            for line in tqdm(in_f, total=num_lines, desc="Reading profile"):
                line = ujson.loads(line)

                # Check keys and schema
                assert all(
                    k in line.keys() for k in REQUIRED_KEYS
                ), f"A key from {REQUIRED_KEYS} was not in {line}"
                try:
                    # Asserts the types are correct
                    ent = EntityObj(
                        entity_id=line["entity_id"],
                        mentions=line["mentions"],
                        title=line.get("title", line["entity_id"]),
                        types=line.get("types", {}),
                        relations=line.get("relations", []),
                    )
                except ValidationError as e:
                    print(e.json())
                    raise e
                if ent.entity_id in qid2title:
                    raise ValueError(f"{ent.entity_id} is already in our dump")
                qid2title[ent.entity_id] = ent.title
                # For each [mention, score] value, create a value of mention -> [qid, score] in the alias2qid dict
                for men_pair in ent.mentions:
                    if men_pair[0] not in alias2qids:
                        alias2qids[men_pair[0]] = []
                    alias2qids[men_pair[0]].append([ent.entity_id, men_pair[1]])
                # Add type systems of type_sys -> QID -> list of type names
                for type_sys in ent.types:
                    if type_sys not in type_systems:
                        type_systems[type_sys] = {}
                    type_systems[type_sys][ent.entity_id] = ent.types[type_sys]
                # Add kg relations QID -> relation -> list of object QIDs
                for rel_pair in ent.relations:
                    if "relation" not in rel_pair or "object" not in rel_pair:
                        raise ValueError(
                            f"For each value in relations, it must be a JSON with keys relation and object"
                        )
                    if ent.entity_id not in qid2relations:
                        qid2relations[ent.entity_id] = {}
                    if rel_pair["relation"] not in qid2relations[ent.entity_id]:
                        qid2relations[ent.entity_id][rel_pair["relation"]] = []
                    qid2relations[ent.entity_id][rel_pair["relation"]].append(
                        rel_pair["object"]
                    )

        # Sort mentions based on score, highest first
        for al in list(alias2qids.keys()):
            alias2qids[al] = sorted(alias2qids[al], key=lambda x: x[1], reverse=True)
        # Add all qids to the type systems and KG connections with empty values
        # This isn't strictly required but can make the sets more clean as they'll have consistent keys
        for qid in qid2title:
            for type_sys in type_systems:
                if qid not in type_systems[type_sys]:
                    type_systems[type_sys][qid] = []
            if qid not in qid2relations:
                qid2relations[qid] = {}
        return qid2title, alias2qids, type_systems, qid2relations

    # ============================================================
    # GETTERS
    # ============================================================
    def qid_exists(self, qid):
        return self._entity_symbols.qid_exists(qid)

    def mention_exists(self, mention):
        return self._entity_symbols.alias_exists(mention)

    def get_all_qids(self):
        return self._entity_symbols.get_all_qids()

    def get_all_mentions(self):
        return self._entity_symbols.get_all_aliases()

    def get_all_typesystems(self):
        return list(self._type_systems.keys())

    def get_all_types(self, type_system):
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        return self._type_systems[type_system].get_all_types()

    @check_qid_exists
    def get_title(self, qid):
        return self._entity_symbols.get_title(qid)

    @check_qid_exists
    def get_eid(self, qid):
        return self._entity_symbols.get_eid(qid)

    @check_qid_exists
    def get_qid_cands(self, qid):
        return self._entity_symbols.get_qid_cands(qid)

    @check_qid_exists
    def get_qid_count_cands(self, qid):
        return self._entity_symbols.get_qid_count_cands(qid)

    def get_num_entities_with_pad_and_nocand(self):
        return self._entity_symbols.num_entities_with_pad_and_nocand

    @check_qid_exists
    def get_types(self, qid, type_system):
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        return self._type_systems[type_system].get_types(qid)

    @check_qid_exists
    def get_connections_by_relation(self, qid, relation):
        return self._kg_symbols.get_connections_by_relation(qid, relation)

    @check_qid_exists
    def get_all_connections(self, qid):
        return self._kg_symbols.get_all_connections(qid)

    @check_qid_exists
    def is_connected(self, qid, qid2):
        self._kg_symbols.is_connected(qid, qid2)

    # ============================================================
    # EDIT MODE OPERATIONS
    # ============================================================
    # GETTERS
    # get_mentions is in edit mode due to needing the qid->mention dict
    @edit_op
    @check_qid_exists
    def get_mentions(self, qid):
        return self._entity_symbols.get_mentions(qid)

    @edit_op
    @check_qid_exists
    def get_mentions_with_scores(self, qid):
        return self._entity_symbols.get_mentions_with_scores(qid)

    @edit_op
    def get_entities_of_type(self, typename, type_system):
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        return self._type_systems[type_system].get_entities_of_type(typename)

    # UPDATES
    @edit_op
    def add_entity(self, entity_obj):
        if (
            type(entity_obj) is not dict
            or "entity_id" not in entity_obj
            or "mentions" not in entity_obj
        ):
            raise ValueError(
                f"The input to update_entity needs to be a dictionary with an entity_id key and mentions key as you are replacing the entity information in bulk."
            )
        try:
            ent = EntityObj(
                entity_id=entity_obj["entity_id"],
                mentions=entity_obj["mentions"],
                title=entity_obj.get("title", entity_obj["entity_id"]),
                types=entity_obj.get("types", {}),
                relations=entity_obj.get("relations", []),
            )
        except ValidationError as e:
            print(e.json())
            raise e
        # We assume this is a new entity
        if self._entity_symbols.qid_exists(ent.entity_id):
            raise ValueError(
                f"The entity {ent.entity_id} already exists. Please call update_entity instead."
            )
        # Add type systems of type_sys -> QID -> list of type names
        for type_sys in ent.types:
            if type_sys not in self._type_systems:
                raise ValueError(
                    f"Error {entity_obj}. When adding a new entity, you must use the same type system. We don't support new type systems."
                )
        # Add kg relations QID -> relation -> list of object QIDs
        parsed_rels = {}
        for rel_pair in ent.relations:
            if "relation" not in rel_pair or "object" not in rel_pair:
                raise ValueError(
                    f"For each value in relations, it must be a JSON with keys relation and object"
                )
            if rel_pair["relation"] not in self._kg_symbols.get_all_relations():
                raise ValueError(
                    f"Error {entity_obj}. When adding a new entity, you must use the same set of relations. We don't support new relations."
                )
            if rel_pair["relation"] not in parsed_rels:
                parsed_rels[rel_pair["relation"]] = []
            parsed_rels[rel_pair["relation"]].append(rel_pair["object"])
        self._entity_symbols.add_entity(ent.entity_id, ent.mentions, ent.title)
        for type_sys in self._type_systems:
            self._type_systems[type_sys].add_entity(
                ent.entity_id, ent.types.get(type_sys, [])
            )
        self._kg_symbols.add_entity(ent.entity_id, parsed_rels)
        # Warn user once about needing to update the model
        # msg = f"When adding an entity, you MUST call XXX refit the model to this profile. You MUST reprep your data. Set data_config.overwrite_preprocessed_data to be True. These messages will now be surpressed."
        # logger.warning(msg)
        # warnings.filterwarnings("ignore", message=msg)

    @edit_op
    @check_qid_exists
    def reidentify_entity(self, qid, new_qid):
        # We assume this is a new entity
        if self._entity_symbols.qid_exists(new_qid):
            raise ValueError(
                f"The entity {new_qid} already exists. Please call update_entity instead."
            )
        self._entity_symbols.reidentify_entity(qid, new_qid)
        for type_sys in self._type_systems:
            self._type_systems[type_sys].reidentify_entity(qid, new_qid)
        self._kg_symbols.reidentify_entity(qid, new_qid)

    @edit_op
    def update_entity(self, entity_obj):
        if (
            type(entity_obj) is not dict
            or "entity_id" not in entity_obj
            or "mentions" not in entity_obj
        ):
            raise ValueError(
                f"The input to update_entity needs to be a dictionary with an entity_id key and mentions key as you are replacing the entity information in bulk."
            )
        if not self._entity_symbols.qid_exists(entity_obj["entity_id"]):
            raise ValueError(f"The entity {entity_obj['entity_id']} is not in our dump")
        try:
            ent = EntityObj(
                entity_id=entity_obj["entity_id"],
                mentions=entity_obj["mentions"],
                title=entity_obj.get("title", entity_obj["entity_id"]),
                types=entity_obj.get("types", {}),
                relations=entity_obj.get("relations", []),
            )
        except ValidationError as e:
            print(e.json())
            raise e
        # Update mentions
        for men in self.get_mentions(ent.entity_id):
            self._entity_symbols.remove_alias(ent.entity_id, men)
        for men in ent.mentions:
            self._entity_symbols.add_alias(ent.entity_id, men)
        # Update title
        self._entity_symbols.set_title(ent.entity_id, ent.title)
        # Update types
        for type_sys in self._type_systems:
            for typename in self._type_systems[type_sys].get_types(ent.entity_id):
                self._type_systems[type_sys].remove_type(ent.entity_id, typename)
        for type_sys in ent.types:
            for typename in ent.types[type_sys]:
                self._type_systems[type_sys].add_type(ent.entity_id, typename)
        # Update KG
        for rel in self._kg_symbols.get_relations(ent.entity_id):
            for qid2 in self._kg_symbols.get_connections_by_relation(
                ent.entity_id, rel
            ):
                self._kg_symbols.remove_kg(ent.entity_id, rel, qid2)
        for rel_pair in ent.relations:
            self._kg_symbols.add_kg(
                ent.entity_id, rel_pair["relation"], rel_pair["object"]
            )

    @edit_op
    def prune_to_entities(self, entities_to_keep):
        entities_to_keep = set(entities_to_keep)
        # Check that all entities to keep actually exist
        for qid in entities_to_keep:
            if not self.qid_exists(qid):
                raise ValueError(
                    f"The entity {qid} does not exist in our dump and cannot be kept."
                )
        if self.verbose:
            print("Pruning entity data")
        self._entity_symbols.prune_to_entities(entities_to_keep)
        for type_sys in self._type_systems:
            if self.verbose:
                print(f"Pruning {type_sys} data")
            self._type_systems[type_sys].prune_to_entities(entities_to_keep)
        if self.verbose:
            print(f"Pruning kg data")
        self._kg_symbols.prune_to_entities(entities_to_keep)

    @edit_op
    @check_qid_exists
    def add_type(self, qid, type, type_system):
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        self._type_systems[type_system].add_type(qid, type)

    @edit_op
    @check_qid_exists
    def add_relation(self, qid, relation, qid2):
        self._kg_symbols.add_relation(qid, relation, qid2)

    @edit_op
    @check_qid_exists
    def add_mention(self, qid: str, mention: str, score: float):
        self._entity_symbols.add_mention(qid, mention, score)

    @edit_op
    @check_qid_exists
    def remove_type(self, qid, type, type_system):
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        self._type_systems[type_system].remove_type(qid, type)

    @edit_op
    @check_qid_exists
    def remove_relation(self, qid, relation, qid2):
        self._kg_symbols.remove_relation(qid, relation, qid2)

    @edit_op
    @check_qid_exists
    def remove_mention(self, qid, mention):
        self._entity_symbols.remove_mention(qid, mention)
