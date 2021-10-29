"""Entity profile."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ujson
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from bootleg.symbols.constants import check_qid_exists, edit_op
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.kg_symbols import KGSymbols
from bootleg.symbols.type_symbols import TypeSymbols
from bootleg.utils.utils import get_lnrm

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
    description: str
    types: Optional[Dict[str, List[str]]]
    relations: Optional[List[Dict[str, str]]]


class EntityProfile:
    """Entity Profile object to handle and manage entity, type, and KG metadata."""

    def __init__(
        self,
        entity_symbols,
        type_systems=None,
        kg_symbols=None,
        edit_mode=False,
        verbose=False,
    ):
        """Entity profile initializer."""
        self.edit_mode = edit_mode
        self.verbose = verbose
        self._entity_symbols = entity_symbols
        self._type_systems = type_systems
        self._kg_symbols = kg_symbols

    def save(self, save_dir):
        """Save the profile.

        Args:
            save_dir: save directory
        """
        save_dir = Path(save_dir)
        self._entity_symbols.save(save_dir / ENTITY_SUBFOLDER)
        for type_sys in self._type_systems:
            self._type_systems[type_sys].save(save_dir / TYPE_SUBFOLDER / type_sys)
        if self._kg_symbols is not None:
            self._kg_symbols.save(save_dir / KG_SUBFOLDER)

    @classmethod
    def load_from_cache(
        cls,
        load_dir,
        edit_mode=False,
        verbose=False,
        no_kg=False,
        no_type=False,
        type_systems_to_load=None,
    ):
        """Load a pre-saved profile.

        Args:
            load_dir: load directory
            edit_mode: edit mode flag, default False
            verbose: verbose flag, default False
            no_kg: load kg or not flag, default False
            no_type: load types or not flag, default False. If True, this will ignore type_systems_to_load.
            type_systems_to_load: list of type systems to load, default is None which means all types systems

        Returns: entity profile object
        """
        # Check type system input
        load_dir = Path(load_dir)
        type_subfolder = load_dir / TYPE_SUBFOLDER
        if type_systems_to_load is not None:
            if not isinstance(type_systems_to_load, list):
                raise ValueError(
                    f"`type_systems` must be a list of subfolders in {type_subfolder}"
                )
            for sys in type_systems_to_load:
                if sys not in list([p.name for p in type_subfolder.iterdir()]):
                    raise ValueError(
                        f"`type_systems` must be a list of subfolders in {type_subfolder}. {sys} is not one."
                    )

        if verbose:
            print("Loading Entity Symbols")
        entity_symbols = EntitySymbols.load_from_cache(
            load_dir / ENTITY_SUBFOLDER,
            edit_mode=edit_mode,
            verbose=verbose,
        )
        if no_type:
            print(
                "Not loading type information. We will act as if there is no types associated with any entity "
                "and will not modify the types in any way, even if calling `add`."
            )
        type_sys_dict = {}
        for fold in type_subfolder.iterdir():
            if (
                (not no_type)
                and (type_systems_to_load is None or fold.name in type_systems_to_load)
                and (fold.is_dir())
            ):
                if verbose:
                    print(f"Loading Type Symbols from {fold}")
                type_sys_dict[fold.name] = TypeSymbols.load_from_cache(
                    type_subfolder / fold.name,
                    edit_mode=edit_mode,
                    verbose=verbose,
                )
        if verbose:
            print("Loading KG Symbols")
        if no_kg:
            print(
                "Not loading KG information. We will act as if there is no KG connections between entities. "
                "We will not modify the KG information in any way, even if calling `add`."
            )
        kg_symbols = None
        if not no_kg:
            kg_symbols = KGSymbols.load_from_cache(
                load_dir / KG_SUBFOLDER,
                edit_mode=edit_mode,
                verbose=verbose,
            )
        return cls(entity_symbols, type_sys_dict, kg_symbols, edit_mode, verbose)

    @classmethod
    def load_from_jsonl(
        cls,
        profile_file,
        max_candidates=30,
        max_types=10,
        max_kg_connections=100,
        edit_mode=False,
    ):
        """Load an entity profile from the raw jsonl file.

        Each line is a JSON object with entity metadata.

        Example object::

            {
                "entity_id": "C000",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            }

        Args:
            profile_file: file where jsonl data lives
            max_candidates: maximum entity candidates
            max_types: maximum types per entity
            max_kg_connections: maximum KG connections per entity
            edit_mode: edit mode

        Returns: entity profile object
        """
        (
            qid2title,
            qid2desc,
            alias2qids,
            type_systems,
            qid2relations,
        ) = cls._read_profile_file(profile_file)
        entity_symbols = EntitySymbols(
            alias2qids=alias2qids,
            qid2title=qid2title,
            qid2desc=qid2desc,
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
        """Read profile data helper.

        Args:
            profile_file: file where jsonl data lives

        Returns: Dicts of qid2title, alias2qids, type_systems, qid2relations
        """
        qid2title: Dict[str, str] = {}
        qid2desc: Dict[str, str] = {}
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
                        description=line.get("description", ""),
                        types=line.get("types", {}),
                        relations=line.get("relations", []),
                    )
                except ValidationError as e:
                    print(e.json())
                    raise e
                if ent.entity_id in qid2title:
                    raise ValueError(f"{ent.entity_id} is already in our dump")
                qid2title[ent.entity_id] = ent.title
                qid2desc[ent.entity_id] = ent.description
                # For each [mention, score] value, create a value of mention -> [qid, score] in the alias2qid dict
                for men_pair in ent.mentions:
                    # Lower case mentions for mention extraction
                    new_men = get_lnrm(men_pair[0], strip=True, lower=True)
                    if new_men not in alias2qids:
                        alias2qids[new_men] = []
                    alias2qids[new_men].append([ent.entity_id, men_pair[1]])
                # Add type systems of type_sys -> QID -> list of type names
                for type_sys in ent.types:
                    if type_sys not in type_systems:
                        type_systems[type_sys] = {}
                    type_systems[type_sys][ent.entity_id] = ent.types[type_sys]
                # Add kg relations QID -> relation -> list of object QIDs
                for rel_pair in ent.relations:
                    if "relation" not in rel_pair or "object" not in rel_pair:
                        raise ValueError(
                            "For each value in relations, it must be a JSON with keys relation and object"
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
        return qid2title, qid2desc, alias2qids, type_systems, qid2relations

    # To quickly get the mention scores, the object must be in edit mode
    @edit_op
    def save_to_jsonl(self, profile_file):
        """Dump the entity dump to jsonl format.

        Args:
            profile_file: file to save the data
        """
        with open(profile_file, "w") as out_f:
            for qid in tqdm(self.get_all_qids(), disable=not self.verbose):
                mentions = self.get_mentions_with_scores(qid)
                title = self.get_title(qid)
                desc = self.get_desc(qid)
                ent_type_sys = {}
                for type_sys in self._type_systems:
                    types = self.get_types(qid, type_sys)
                    if len(types) > 0:
                        ent_type_sys[type_sys] = types
                relations = []
                all_connections = self.get_all_connections(qid)
                for rel in all_connections:
                    for qid2 in all_connections[rel]:
                        relations.append({"relation": rel, "object": qid2})
                ent_obj = {
                    "entity_id": qid,
                    "mentions": mentions,
                    "title": title,
                }
                # Add description if nonempty
                if len(desc) > 0:
                    ent_obj["description"] = desc
                if len(ent_type_sys) > 0:
                    ent_obj["types"] = ent_type_sys
                if len(relations) > 0:
                    ent_obj["relations"] = relations
                out_f.write(ujson.dumps(ent_obj) + "\n")

    # ============================================================
    # GETTERS
    # ============================================================
    def qid_exists(self, qid):
        """Check if QID exists.

        Args:
            qid: entity QID

        Returns: Boolean
        """
        return self._entity_symbols.qid_exists(qid)

    def mention_exists(self, mention):
        """Check if mention exists.

        Args:
            mention: mention

        Returns: Boolean
        """
        return self._entity_symbols.alias_exists(mention)

    def get_all_qids(self):
        """Return all entity QIDs.

        Returns: List of strings
        """
        return self._entity_symbols.get_all_qids()

    def get_all_mentions(self):
        """Return list of all mentions.

        Returns: List of strings
        """
        return self._entity_symbols.get_all_aliases()

    def get_all_typesystems(self):
        """Return list of all type systems.

        Returns: List of strings
        """
        return list(self._type_systems.keys())

    def get_all_types(self, type_system):
        """Return list of all type names for a type system.

        Args:
            type_system: type system

        Returns: List of strings
        """
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        return self._type_systems[type_system].get_all_types()

    def get_type_typeid(self, type, type_system):
        """Get the type type id for the type of the ``type_system`` system.

        Args:
            type: type
            type_system: type system

        Returns: type id
        """
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        return self._type_systems[type_system].get_type_typeid(type)

    @check_qid_exists
    def get_title(self, qid):
        """Get the title of an entity QID.

        Args:
            qid: entity QID

        Returns: string
        """
        return self._entity_symbols.get_title(qid)

    @check_qid_exists
    def get_desc(self, qid):
        """Get the description of an entity QID.

        Args:
            qid: entity QID

        Returns: string
        """
        return self._entity_symbols.get_desc(qid)

    @check_qid_exists
    def get_eid(self, qid):
        """Get the entity EID (internal number) of an entity QID.

        Args:
            qid: entity QID

        Returns: integer
        """
        return self._entity_symbols.get_eid(qid)

    def get_qid_cands(self, mention):
        """Get the entity QID candidates of the mention.

        Args:
            mention: mention

        Returns: List of QIDs
        """
        return self._entity_symbols.get_qid_cands(mention)

    def get_qid_count_cands(self, mention):
        """Get the entity QID candidates with their scores of the mention.

        Args:
            mention: mention

        Returns: List of tuples [QID, score]
        """
        return self._entity_symbols.get_qid_count_cands(mention)

    @property
    def num_entities_with_pad_and_nocand(self):
        """Get the number of entities including a PAD and UNK entity.

        Returns: integer
        """
        return self._entity_symbols.num_entities_with_pad_and_nocand

    @check_qid_exists
    def get_types(self, qid, type_system):
        """Get the type names associated with the given QID for the ``type_system`` system.

        Args:
            qid: QID
            type_system: type system

        Returns: list of typename strings
        """
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        return self._type_systems[type_system].get_types(qid)

    @check_qid_exists
    def get_connections_by_relation(self, qid, relation):
        """Return list of other_qids connected to ``qid`` by relation.

        Args:
            qid: QID
            relation: relation

        Returns: List
        """
        if self._kg_symbols is None:
            return []
        return self._kg_symbols.get_connections_by_relation(qid, relation)

    @check_qid_exists
    def get_all_connections(self, qid):
        """Return dictionary of relation -> list of other_qids connected to ``qid`` by relation.

        Args:
            qid: QID

        Returns: Dict
        """
        if self._kg_symbols is None:
            return {}
        return self._kg_symbols.get_all_connections(qid)

    @check_qid_exists
    def is_connected(self, qid, qid2):
        """Check if two QIDs are connected in KG.

        Args:
            qid: QID one
            qid2: QID two

        Returns: boolean
        """
        if self._kg_symbols is None:
            return False
        return self._kg_symbols.is_connected(qid, qid2)

    # ============================================================
    # EDIT MODE OPERATIONS
    # ============================================================
    # GETTERS
    # get_mentions is in edit mode due to needing the qid->mention dict
    @edit_op
    @check_qid_exists
    def get_mentions(self, qid):
        """Get the mentions for the QID.

        Args:
            qid: QID

        Returns: List of mentions
        """
        return self._entity_symbols.get_mentions(qid)

    @edit_op
    @check_qid_exists
    def get_mentions_with_scores(self, qid):
        """Get the mentions with thier scores associated with the QID.

        Args:
            qid: QID

        Returns: List of tuples [mention, score]
        """
        return self._entity_symbols.get_mentions_with_scores(qid)

    @edit_op
    def get_entities_of_type(self, typename, type_system):
        """Get all entities of type ``typename`` for type system ``type_system``.

        Args:
            typename: type name
            type_system: type system

        Returns: List of QIDs
        """
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        return self._type_systems[type_system].get_entities_of_type(typename)

    # UPDATES
    @edit_op
    def add_entity(self, entity_obj):
        """Add entity to our dump.

        Args:
            entity_obj: JSON object of entity metadata
        """
        if (
            type(entity_obj) is not dict
            or "entity_id" not in entity_obj
            or "mentions" not in entity_obj
        ):
            raise ValueError(
                "The input to update_entity needs to be a dictionary with an entity_id key and mentions key as "
                "you are replacing the entity information in bulk."
            )
        try:
            ent = EntityObj(
                entity_id=entity_obj["entity_id"],
                mentions=entity_obj["mentions"],
                title=entity_obj.get("title", entity_obj["entity_id"]),
                description=entity_obj.get("description", ""),
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
                    f"Error {entity_obj}. When adding a new entity, you must use the same type system. "
                    f"We don't support new type systems."
                )
        # Add kg relations QID -> relation -> list of object QIDs
        parsed_rels = {}
        for rel_pair in ent.relations:
            if "relation" not in rel_pair or "object" not in rel_pair:
                raise ValueError(
                    "For each value in relations, it must be a JSON with keys relation and object"
                )
            if (
                self._kg_symbols is not None
                and rel_pair["relation"] not in self._kg_symbols.get_all_relations()
            ):
                raise ValueError(
                    f"Error {entity_obj}. When adding a new entity, you must use the same set of relations. "
                    f"We don't support new relations."
                )
            if rel_pair["relation"] not in parsed_rels:
                parsed_rels[rel_pair["relation"]] = []
            parsed_rels[rel_pair["relation"]].append(rel_pair["object"])
        # Lower case mentions for mention extraction
        mentions = [
            [get_lnrm(men[0], strip=True, lower=True), men[1]] for men in ent.mentions
        ]
        self._entity_symbols.add_entity(
            ent.entity_id, mentions, ent.title, ent.description
        )
        for type_sys in self._type_systems:
            self._type_systems[type_sys].add_entity(
                ent.entity_id, ent.types.get(type_sys, [])
            )
        if self._kg_symbols is not None:
            self._kg_symbols.add_entity(ent.entity_id, parsed_rels)

    @edit_op
    @check_qid_exists
    def reidentify_entity(self, qid, new_qid):
        """Rename ``qid`` to ``new_qid``.

        Args:
            qid: old QID
            new_qid: new QID
        """
        # We assume this is a new entity
        if self._entity_symbols.qid_exists(new_qid):
            raise ValueError(
                f"The entity {new_qid} already exists. Please call update_entity instead."
            )
        self._entity_symbols.reidentify_entity(qid, new_qid)
        for type_sys in self._type_systems:
            self._type_systems[type_sys].reidentify_entity(qid, new_qid)
        if self._kg_symbols is not None:
            self._kg_symbols.reidentify_entity(qid, new_qid)

    @edit_op
    def update_entity(self, entity_obj):
        """Update the metadata associated with the entity.

        The entity must already be in our dump to be updated.

        Args:
            entity_obj: JSON of entity metadata.
        """
        if (
            type(entity_obj) is not dict
            or "entity_id" not in entity_obj
            or "mentions" not in entity_obj
        ):
            raise ValueError(
                "The input to update_entity needs to be a dictionary with an entity_id key and mentions key as "
                "you are replacing the entity information in bulk."
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
            # Lower case mentions for mention extraction
            men = [get_lnrm(men[0], strip=True, lower=True), men[1]]
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
        if self._kg_symbols is not None:
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
        """Remove all entities except those in ``entities_to_keep``.

        Args:
            entities_to_keep: List or Set of entities to keep
        """
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
            print("Pruning kg data")
        if self._kg_symbols is not None:
            self._kg_symbols.prune_to_entities(entities_to_keep)

    @edit_op
    @check_qid_exists
    def add_type(self, qid, type, type_system):
        """Add type to QID in for the given type system.

        Args:
            qid: QID
            type: type name
            type_system: type system
        """
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        self._type_systems[type_system].add_type(qid, type)

    @edit_op
    @check_qid_exists
    def add_relation(self, qid, relation, qid2):
        """Add the relation triple.

        Args:
            qid: head QID
            relation: relation
            qid2: tail QID
        """
        if self._kg_symbols is not None:
            self._kg_symbols.add_relation(qid, relation, qid2)

    @edit_op
    @check_qid_exists
    def add_mention(self, qid: str, mention: str, score: float):
        """Add the mention with its score to the QID.

        Args:
            qid: QID
            mention: mention
            score: score
        """
        self._entity_symbols.add_mention(qid, mention, score)

    @edit_op
    @check_qid_exists
    def remove_type(self, qid, type, type_system):
        """Remove the type from QID in the given type system.

        Args:
            qid: QID
            type: type to remove
            type_system: type system
        """
        if type_system not in self._type_systems:
            raise ValueError(
                f"The type system {type_system} is not one of {self._type_systems.keys()}"
            )
        self._type_systems[type_system].remove_type(qid, type)

    @edit_op
    @check_qid_exists
    def remove_relation(self, qid, relation, qid2):
        """Remove the relation triple.

        Args:
            qid: head QID
            relation: relation
            qid2: tail QID
        """
        if self._kg_symbols is not None:
            self._kg_symbols.remove_relation(qid, relation, qid2)

    @edit_op
    @check_qid_exists
    def remove_mention(self, qid, mention):
        """Remove the mention from being associated with the QID.

        Args:
            qid: QID
            mention: mention
        """
        self._entity_symbols.remove_mention(qid, mention)
