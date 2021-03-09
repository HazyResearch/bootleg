import os
import shutil
import unittest
from pathlib import Path

import torch
import ujson
from pydantic import ValidationError

from bootleg.layers.alias_to_ent_encoder import AliasEntityTable
from bootleg.symbols.entity_profile import EntityProfile
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.kg_symbols import KGSymbols
from bootleg.symbols.type_symbols import TypeSymbols
from bootleg.utils import utils
from bootleg.utils.classes.dotted_dict import DottedDict


class TypeSymbolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.save_dir = Path("test/data/entity_loader/entity_db_save")
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def tearDown(self) -> None:
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_type_init(self):
        qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        max_types = 2
        type_symbols = TypeSymbols(qid2typenames, max_types=max_types)

        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall"],
            "Q789": [],
        }
        gold_qid2typeid = {"Q123": [1], "Q345": [3], "Q567": [1, 2], "Q789": []}
        gold_type_vocab = {"animal": 1, "animall": 2, "dog": 3, "drop": 4}
        gold_type_vocab_inv = {1: "animal", 2: "animall", 3: "dog", 4: "drop"}

        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertIsNone(type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)
        self.assertDictEqual(gold_type_vocab, type_symbols._type_vocab)
        self.assertDictEqual(gold_type_vocab_inv, type_symbols._type_vocab_inv)

        max_types = 4
        type_symbols = TypeSymbols(qid2typenames, max_types=max_types, edit_mode=True)

        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        gold_typename2qids = {
            "animal": {"Q123", "Q567"},
            "animall": {"Q567"},
            "dog": {"Q345"},
            "drop": {"Q567"},
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)

    def test_type_load_and_save(self):
        qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        max_types = 2
        type_symbols = TypeSymbols(qid2typenames, max_types=max_types)
        type_symbols.save(self.save_dir, prefix="test")
        type_symbols_2 = TypeSymbols.load_from_cache(self.save_dir, prefix="test")

        self.assertEqual(type_symbols_2.max_types, type_symbols.max_types)
        self.assertDictEqual(type_symbols_2._qid2typenames, type_symbols._qid2typenames)
        self.assertIsNone(type_symbols._typename2qids)
        self.assertIsNone(type_symbols_2._typename2qids)
        self.assertDictEqual(type_symbols_2._qid2typeid, type_symbols._qid2typeid)
        self.assertDictEqual(type_symbols_2._type_vocab, type_symbols._type_vocab)
        self.assertDictEqual(
            type_symbols_2._type_vocab_inv, type_symbols._type_vocab_inv
        )

    def test_type_add_remove_typemap(self):
        qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        max_types = 3
        type_symbols = TypeSymbols(qid2typenames, max_types=max_types, edit_mode=False)
        # Check if fails with edit_mode = False
        with self.assertRaises(AttributeError) as context:
            type_symbols.add_type("Q789", "animal")
        assert type(context.exception) is AttributeError

        type_symbols = TypeSymbols(qid2typenames, max_types=max_types, edit_mode=True)
        # Check the invalid type fails (this type isn't in our set of types)
        with self.assertRaises(ValueError) as context:
            type_symbols.add_type("Q789", "annnimal")
        assert type(context.exception) is ValueError

        # Add to a previously empty QID
        type_symbols.add_type("Q789", "animal")
        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": ["animal"],
        }
        gold_qid2typeid = {"Q123": [1], "Q345": [3], "Q567": [1, 2, 4], "Q789": [1]}
        gold_typename2qids = {
            "animal": {"Q123", "Q567", "Q789"},
            "animall": {"Q567"},
            "dog": {"Q345"},
            "drop": {"Q567"},
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)

        # Check that nothing happens with relation pair that doesn't exist and the operation goes through
        type_symbols.remove_type("Q345", "animal")
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)

        # Now actually remove something
        type_symbols.remove_type("Q789", "animal")
        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        gold_qid2typeid = {"Q123": [1], "Q345": [3], "Q567": [1, 2, 4], "Q789": []}
        gold_typename2qids = {
            "animal": {"Q123", "Q567"},
            "animall": {"Q567"},
            "dog": {"Q345"},
            "drop": {"Q567"},
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)

        # Add to a full QID where we must replace. We do not bring back the old type if we remove the replace one.
        type_symbols.add_type("Q567", "dog")
        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "dog"],
            "Q789": [],
        }
        gold_qid2typeid = {"Q123": [1], "Q345": [3], "Q567": [1, 2, 3], "Q789": []}
        gold_typename2qids = {
            "animal": {"Q123", "Q567"},
            "animall": {"Q567"},
            "dog": {"Q345", "Q567"},
            "drop": set(),
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)

        type_symbols.remove_type("Q567", "dog")
        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall"],
            "Q789": [],
        }
        gold_qid2typeid = {"Q123": [1], "Q345": [3], "Q567": [1, 2], "Q789": []}
        gold_typename2qids = {
            "animal": {"Q123", "Q567"},
            "animall": {"Q567"},
            "dog": {"Q345"},
            "drop": set(),
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)

    def test_add_entity(self):
        qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        max_types = 3
        type_symbols = TypeSymbols(qid2typenames, max_types=max_types, edit_mode=True)
        # Check the invalid type fails (this type isn't in our set of types)
        with self.assertRaises(ValueError) as context:
            type_symbols.add_entity("Q910", ["annnimal", "animal", "dog", "drop"])
        assert type(context.exception) is ValueError

        # Add to a previously empty QID
        type_symbols.add_entity("Q910", ["animall", "animal", "dog", "drop"])
        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
            "Q910": ["animall", "animal", "dog"],  # Max types limits new types added
        }
        gold_qid2typeid = {
            "Q123": [1],
            "Q345": [3],
            "Q567": [1, 2, 4],
            "Q789": [],
            "Q910": [2, 1, 3],
        }
        gold_typename2qids = {
            "animal": {"Q123", "Q567", "Q910"},
            "animall": {"Q567", "Q910"},
            "dog": {"Q345", "Q910"},
            "drop": {"Q567"},
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)

    def test_reidentify_entity(self):
        qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        max_types = 3
        type_symbols = TypeSymbols(qid2typenames, max_types=max_types, edit_mode=True)
        type_symbols.reidentify_entity("Q567", "Q911")
        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q911": ["animal", "animall", "drop"],
            "Q789": [],
        }
        gold_qid2typeid = {
            "Q123": [1],
            "Q345": [3],
            "Q911": [1, 2, 4],
            "Q789": [],
        }
        gold_typename2qids = {
            "animal": {"Q123", "Q911"},
            "animall": {"Q911"},
            "dog": {"Q345"},
            "drop": {"Q911"},
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)

    def test_prune_to_entities(self):
        qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
            "Q567": ["animal", "animall", "drop"],
            "Q789": [],
        }
        max_types = 3
        type_symbols = TypeSymbols(qid2typenames, max_types=max_types, edit_mode=True)
        type_symbols.prune_to_entities({"Q123", "Q345"})
        gold_qid2typenames = {
            "Q123": ["animal"],
            "Q345": ["dog"],
        }
        gold_qid2typeid = {
            "Q123": [1],
            "Q345": [3],
        }
        gold_typename2qids = {
            "animal": {"Q123"},
            "animall": set(),
            "dog": {"Q345"},
            "drop": set(),
        }
        self.assertDictEqual(gold_qid2typenames, type_symbols._qid2typenames)
        self.assertDictEqual(gold_typename2qids, type_symbols._typename2qids)
        self.assertDictEqual(gold_qid2typeid, type_symbols._qid2typeid)


class KGSymbolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.save_dir = Path("test/data/entity_loader/entity_db_save")
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def tearDown(self) -> None:
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_kg_init(self):
        qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        max_connections = 1
        kg_symbols = KGSymbols(qid2relations, max_connections=max_connections)

        gold_qid2relations = {
            "Q123": {"sibling": ["Q345"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        gold_allrelations = {"sibling"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertIsNone(kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelations, kg_symbols._all_relations)

    def test_kg_load_and_save(self):
        qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        max_connections = 1
        kg_symbols = KGSymbols(qid2relations, max_connections=max_connections)
        kg_symbols.save(self.save_dir, prefix="test")
        kg_symbols_2 = KGSymbols.load_from_cache(self.save_dir, prefix="test")

        self.assertEqual(kg_symbols_2.max_connections, kg_symbols.max_connections)
        self.assertDictEqual(kg_symbols_2._qid2relations, kg_symbols._qid2relations)
        self.assertIsNone(kg_symbols_2._obj2head)

        # Test that adjacency matrix was saved correctly
        gold_qid2qid_adj = [
            tuple(["Q123", "Q345"]),
            tuple(["Q345", "Q123"]),
            tuple(["Q567", "Q123"]),
        ]
        qid2qid_adj = []
        with open(os.path.join(self.save_dir, f"testqid2qid_adj.txt")) as in_f:
            for line in in_f:
                qid2qid_adj.append(tuple(line.split()))
        self.assertListEqual(gold_qid2qid_adj, qid2qid_adj)

    def test_relation_add_remove_kgmapping(self):
        qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        max_connections = 2
        kg_symbols = KGSymbols(qid2relations, max_connections=max_connections)

        # Check if fails with edit_mode = False
        with self.assertRaises(AttributeError) as context:
            kg_symbols.add_relation("Q789", "sibling", "Q123")
        assert type(context.exception) is AttributeError

        kg_symbols = KGSymbols(
            qid2relations, max_connections=max_connections, edit_mode=True
        )
        kg_symbols.add_relation("Q789", "sibling", "Q123")
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {"sibling": ["Q123"]},
        }
        gold_obj2head = {
            "Q123": {"Q789", "Q567", "Q345"},
            "Q345": {"Q123"},
            "Q567": {"Q123"},
        }
        gold_allrelationes = {"sibling"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

        kg_symbols.add_relation("Q123", "sibling", "Q789")
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345", "Q789"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {"sibling": ["Q123"]},
        }
        gold_obj2head = {
            "Q123": {"Q789", "Q567", "Q345"},
            "Q345": {"Q123"},
            "Q789": {"Q123"},
        }
        gold_allrelationes = {"sibling"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

        kg_symbols.remove_relation("Q123", "sibling", "Q789")
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {"sibling": ["Q123"]},
        }
        gold_obj2head = {"Q123": {"Q789", "Q567", "Q345"}, "Q345": {"Q123"}}
        gold_allrelationes = {"sibling"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

        # Check nothing changes with bad remove that doesn't exist
        kg_symbols.remove_relation("Q789", "siblinggg", "Q123")
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

        # Check fails with relation that doesn't exist
        with self.assertRaises(ValueError) as context:
            kg_symbols.add_relation("Q789", "siblinggg", "Q567")
        assert type(context.exception) is ValueError

        # Check nothing changes with relation pair that doesn't exist
        kg_symbols.remove_relation("Q567", "siblinggg", "Q789")
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

        kg_symbols.remove_relation("Q789", "sibling", "Q123")
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        gold_obj2head = {"Q123": {"Q567", "Q345"}, "Q345": {"Q123"}}
        gold_allrelationes = {"sibling"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

    def test_add_entity(self):
        qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        max_connections = 2

        kg_symbols = KGSymbols(
            qid2relations, max_connections=max_connections, edit_mode=True
        )

        # Check if fails with relation that wasn't seen before
        with self.assertRaises(ValueError) as context:
            kg_symbols.add_entity("Q910", {"siblinggg": ["Q567", "Q123", "Q345"]})
        assert type(context.exception) is ValueError

        # Add kg
        kg_symbols.add_entity("Q910", {"sibling": ["Q567", "Q123", "Q345"]})
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
            "Q910": {"sibling": ["Q567", "Q123"]},  # Max connections limits to 2
        }
        gold_obj2head = {
            "Q123": {"Q910", "Q567", "Q345"},
            "Q345": {"Q123"},
            "Q567": {"Q123", "Q910"},
        }
        gold_allrelationes = {"sibling"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

    def test_reidentify_entities(self):
        qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"], "sib": ["Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"], "sib": ["Q123", "Q567"]},
            "Q789": {},
        }
        max_connections = 2

        kg_symbols = KGSymbols(
            qid2relations, max_connections=max_connections, edit_mode=True
        )
        kg_symbols.reidentify_entity("Q567", "Q911")
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345", "Q911"], "sib": ["Q911"]},
            "Q345": {"sibling": ["Q123"]},
            "Q911": {"sibling": ["Q123"], "sib": ["Q123", "Q911"]},
            "Q789": {},
        }
        gold_obj2head = {
            "Q123": {"Q911", "Q345"},
            "Q345": {"Q123"},
            "Q911": {"Q123", "Q911"},
        }
        gold_allrelationes = {"sibling", "sib"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

        kg_symbols.reidentify_entity("Q789", "Q912")
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345", "Q911"], "sib": ["Q911"]},
            "Q345": {"sibling": ["Q123"]},
            "Q911": {"sibling": ["Q123"], "sib": ["Q123", "Q911"]},
            "Q912": {},
        }
        gold_obj2head = {
            "Q123": {"Q911", "Q345"},
            "Q345": {"Q123"},
            "Q911": {"Q123", "Q911"},
        }
        gold_allrelationes = {"sibling", "sib"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)

    def test_prune_to_entities(self):
        qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q567"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        max_connections = 2

        kg_symbols = KGSymbols(
            qid2relations, max_connections=max_connections, edit_mode=True
        )
        kg_symbols.prune_to_entities({"Q345", "Q567"})
        gold_qid2relations = {
            "Q345": {"sibling": ["Q567"]},
            "Q567": {},
        }
        gold_obj2head = {"Q567": {"Q345"}}
        gold_allrelationes = {"sibling"}
        self.assertDictEqual(gold_qid2relations, kg_symbols._qid2relations)
        self.assertDictEqual(gold_obj2head, kg_symbols._obj2head)
        self.assertSetEqual(gold_allrelationes, kg_symbols._all_relations)


class EntitySymbolTest(unittest.TestCase):
    def setUp(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        self.entity_symbols = EntitySymbols.load_from_cache(load_dir=entity_dump_dir)

    def test_create_entities(self):
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }

        trueqid2title = {
            "Q1": "alias1",
            "Q2": "multi alias2",
            "Q3": "word alias3",
            "Q4": "nonalias4",
        }

        # the non-candidate class is included in entity_dump
        trueqid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        truealiastrie = {"multi word alias2": 0, "alias1": 1, "alias3": 2, "alias4": 3}

        entity_symbols = EntitySymbols(
            max_candidates=3,
            alias2qids=truealias2qids,
            qid2title=trueqid2title,
        )
        tri_as_dict = {}
        for k in entity_symbols._alias_trie:
            tri_as_dict[k] = entity_symbols._alias_trie[k]

        self.assertEqual(entity_symbols.max_candidates, 3)
        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertDictEqual(entity_symbols._qid2title, trueqid2title)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(tri_as_dict, truealiastrie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertIsNone(entity_symbols._qid2aliases)

        entity_symbols = EntitySymbols(
            max_candidates=3,
            alias2qids=truealias2qids,
            qid2title=trueqid2title,
            edit_mode=True,
        )
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3"},
            "Q2": {"multi word alias2", "alias4"},
            "Q3": {"alias4"},
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }

        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)

    def test_load_entites_keep_noncandidate(self):
        truealias2qids = {
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }

        trueqid2title = {
            "Q1": "alias1",
            "Q2": "multi alias2",
            "Q3": "word alias3",
            "Q4": "nonalias4",
        }

        # the non-candidate class is included in entity_dump
        trueqid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

        truealiastrie = {"multi word alias2": 0, "alias1": 1, "alias3": 2, "alias4": 3}

        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}

        tri_as_dict = {}
        for k in self.entity_symbols._alias_trie:
            tri_as_dict[k] = self.entity_symbols._alias_trie[k]

        self.assertEqual(self.entity_symbols.max_candidates, 3)
        self.assertEqual(self.entity_symbols.max_eid, 4)
        self.assertEqual(self.entity_symbols.max_alid, 3)
        self.assertDictEqual(self.entity_symbols._alias2qids, truealias2qids)
        self.assertDictEqual(self.entity_symbols._qid2title, trueqid2title)
        self.assertDictEqual(self.entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(tri_as_dict, truealiastrie)
        self.assertDictEqual(self.entity_symbols._alias2id, truealias2id)

    def test_getters(self):
        self.assertEqual(self.entity_symbols.get_qid(1), "Q1")
        self.assertSetEqual(
            set(self.entity_symbols.get_all_aliases()),
            {"alias1", "multi word alias2", "alias3", "alias4"},
        )
        self.assertEqual(self.entity_symbols.get_eid("Q3"), 3)
        self.assertListEqual(self.entity_symbols.get_qid_cands("alias1"), ["Q1", "Q4"])
        self.assertListEqual(
            self.entity_symbols.get_qid_cands("alias1", max_cand_pad=True),
            ["Q1", "Q4", "-1"],
        )
        self.assertListEqual(
            self.entity_symbols.get_eid_cands("alias1", max_cand_pad=True), [1, 4, -1]
        )
        self.assertEqual(self.entity_symbols.get_title("Q1"), "alias1")
        self.assertEqual(self.entity_symbols.get_alias_idx("alias1"), 0)
        self.assertEqual(self.entity_symbols.get_alias_from_idx(1), "alias3")
        self.assertEqual(self.entity_symbols.alias_exists("alias3"), True)
        self.assertEqual(self.entity_symbols.alias_exists("alias5"), False)

    def test_add_remove_mention(self):
        alias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2], ["Q3", 1]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }

        qid2title = {
            "Q1": "alias1",
            "Q2": "multi alias2",
            "Q3": "word alias3",
            "Q4": "nonalias4",
        }

        max_candidates = 3

        # the non-candidate class is included in entity_dump
        trueqid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        truealiastrie = {"multi word alias2": 0, "alias1": 1, "alias3": 2, "alias4": 3}
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }

        entity_symbols = EntitySymbols(
            max_candidates=max_candidates,
            alias2qids=alias2qids,
            qid2title=qid2title,
        )
        tri_as_dict = {}
        for k in entity_symbols._alias_trie:
            tri_as_dict[k] = entity_symbols._alias_trie[k]

        self.assertEqual(entity_symbols.max_candidates, 3)
        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertDictEqual(entity_symbols._qid2title, qid2title)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(tri_as_dict, truealiastrie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertIsNone(entity_symbols._qid2aliases)

        # Check if fails with edit_mode = False
        with self.assertRaises(AttributeError) as context:
            entity_symbols.add_mention("Q2", "alias3", 31.0)
        assert type(context.exception) is AttributeError

        entity_symbols = EntitySymbols(
            max_candidates=max_candidates,
            alias2qids=alias2qids,
            qid2title=qid2title,
            edit_mode=True,
        )

        # Check nothing changes if pair doesn't exist
        entity_symbols.remove_mention("Q3", "alias1")

        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3"},
            "Q2": {"multi word alias2", "alias4"},
            "Q3": {"alias4"},
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }

        self.assertEqual(entity_symbols.max_candidates, 3)
        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._qid2title, qid2title)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)

        # ADD Q2 ALIAS 3
        entity_symbols.add_mention("Q2", "alias3", 31.0)
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3"},
            "Q2": {"multi word alias2", "alias4", "alias3"},
            "Q3": {"alias4"},
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q2", 31.0], ["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        trueid2alias = {0: "alias1", 1: "alias3", 2: "alias4", 3: "multi word alias2"}

        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)

        # ADD Q1 ALIAS 4
        entity_symbols.add_mention("Q1", "alias4", 31.0)
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3", "alias4"},
            "Q2": {"multi word alias2", "alias3"},
            "Q3": {"alias4"},
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q2", 31.0], ["Q1", 30.0]],
            "alias4": [["Q1", 31.0], ["Q4", 20], ["Q3", 15.0]],
        }
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        trueid2alias = {0: "alias1", 1: "alias3", 2: "alias4", 3: "multi word alias2"}

        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)

        # REMOVE Q3 ALIAS 4
        entity_symbols.remove_mention("Q3", "alias4")
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3", "alias4"},
            "Q2": {"multi word alias2", "alias3"},
            "Q3": set(),
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q2", 31.0], ["Q1", 30.0]],
            "alias4": [["Q1", 31.0], ["Q4", 20]],
        }
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        trueid2alias = {0: "alias1", 1: "alias3", 2: "alias4", 3: "multi word alias2"}

        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)

        # REMOVE Q4 ALIAS 4
        entity_symbols.remove_mention("Q4", "alias4")
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3", "alias4"},
            "Q2": {"multi word alias2", "alias3"},
            "Q3": set(),
            "Q4": {"alias1", "multi word alias2"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q2", 31.0], ["Q1", 30.0]],
            "alias4": [["Q1", 31.0]],
        }
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        trueid2alias = {0: "alias1", 1: "alias3", 2: "alias4", 3: "multi word alias2"}

        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)

        # REMOVE Q1 ALIAS 4
        entity_symbols.remove_mention("Q1", "alias4")
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3"},
            "Q2": {"multi word alias2", "alias3"},
            "Q3": set(),
            "Q4": {"alias1", "multi word alias2"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q2", 31.0], ["Q1", 30.0]],
        }
        truealias2id = {"alias1": 0, "alias3": 1, "multi word alias2": 3}
        trueid2alias = {0: "alias1", 1: "alias3", 3: "multi word alias2"}

        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 3)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)

        # ADD Q1 BLIAS 0
        entity_symbols.add_mention("Q1", "blias0", 11)
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3", "blias0"},
            "Q2": {"multi word alias2", "alias3"},
            "Q3": set(),
            "Q4": {"alias1", "multi word alias2"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q2", 31.0], ["Q1", 30.0]],
            "blias0": [["Q1", 11.0]],
        }
        truealias2id = {"alias1": 0, "alias3": 1, "multi word alias2": 3, "blias0": 4}
        trueid2alias = {0: "alias1", 1: "alias3", 3: "multi word alias2", 4: "blias0"}

        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 4)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)

        # SET SCORE Q2 ALIAS3
        # Check if fails not a pair
        with self.assertRaises(ValueError) as context:
            entity_symbols.set_score("Q2", "alias1", 2)
        assert type(context.exception) is ValueError

        entity_symbols.set_score("Q2", "alias3", 2)
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3", "blias0"},
            "Q2": {"multi word alias2", "alias3"},
            "Q3": set(),
            "Q4": {"alias1", "multi word alias2"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q1", 30.0], ["Q2", 2]],
            "blias0": [["Q1", 11.0]],
        }
        truealias2id = {"alias1": 0, "alias3": 1, "multi word alias2": 3, "blias0": 4}
        trueid2alias = {0: "alias1", 1: "alias3", 3: "multi word alias2", 4: "blias0"}

        self.assertEqual(entity_symbols.max_eid, 4)
        self.assertEqual(entity_symbols.max_alid, 4)
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)

    def test_add_entity(self):
        alias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2], ["Q3", 1]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        qid2title = {
            "Q1": "alias1",
            "Q2": "multi alias2",
            "Q3": "word alias3",
            "Q4": "nonalias4",
        }
        max_candidates = 3

        entity_symbols = EntitySymbols(
            max_candidates=max_candidates,
            alias2qids=alias2qids,
            qid2title=qid2title,
            edit_mode=True,
        )

        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3"},
            "Q2": {"multi word alias2", "alias4"},
            "Q3": {"alias4"},
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        trueqid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        trueid2alias = {0: "alias1", 1: "alias3", 2: "alias4", 3: "multi word alias2"}
        truemax_eid = 4
        truemax_alid = 3
        truenum_entities = 4
        truenum_entities_with_pad_and_nocand = 6
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(
            entity_symbols._eid2qid, {v: i for i, v in trueqid2eid.items()}
        )
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)
        self.assertEqual(entity_symbols.max_eid, truemax_eid)
        self.assertEqual(entity_symbols.max_alid, truemax_alid)
        self.assertEqual(entity_symbols.num_entities, truenum_entities)
        self.assertEqual(
            entity_symbols.num_entities_with_pad_and_nocand,
            truenum_entities_with_pad_and_nocand,
        )

        # Add entity
        entity_symbols.add_entity(
            "Q5", [["multi word alias2", 1.5], ["alias5", 20.0]], "Snake"
        )
        trueqid2aliases = {
            "Q1": {"alias1", "multi word alias2", "alias3"},
            "Q2": {"multi word alias2", "alias4"},
            "Q3": {"alias4"},
            "Q4": {"alias1", "alias4"},
            "Q5": {"multi word alias2", "alias5"},
        }
        truealias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [
                ["Q2", 5.0],
                ["Q1", 3],
                ["Q5", 1.5],
            ],  # adding new entity-mention pair - we override scores to add it. Hence Q4 is removed
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
            "alias5": [["Q5", 20]],
        }
        trueqid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}
        truealias2id = {
            "alias1": 0,
            "alias3": 1,
            "alias4": 2,
            "multi word alias2": 3,
            "alias5": 4,
        }
        trueid2alias = {
            0: "alias1",
            1: "alias3",
            2: "alias4",
            3: "multi word alias2",
            4: "alias5",
        }
        truemax_eid = 5
        truemax_alid = 4
        truenum_entities = 5
        truenum_entities_with_pad_and_nocand = 7
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(
            entity_symbols._eid2qid, {v: i for i, v in trueqid2eid.items()}
        )
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)
        self.assertEqual(entity_symbols.max_eid, truemax_eid)
        self.assertEqual(entity_symbols.max_alid, truemax_alid)
        self.assertEqual(entity_symbols.num_entities, truenum_entities)
        self.assertEqual(
            entity_symbols.num_entities_with_pad_and_nocand,
            truenum_entities_with_pad_and_nocand,
        )

    def test_reidentify_entity(self):
        alias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2], ["Q3", 1]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        qid2title = {
            "Q1": "alias1",
            "Q2": "multi alias2",
            "Q3": "word alias3",
            "Q4": "nonalias4",
        }
        max_candidates = 3

        entity_symbols = EntitySymbols(
            max_candidates=max_candidates,
            alias2qids=alias2qids,
            qid2title=qid2title,
            edit_mode=True,
        )
        entity_symbols.reidentify_entity("Q1", "Q7")
        trueqid2aliases = {
            "Q7": {"alias1", "multi word alias2", "alias3"},
            "Q2": {"multi word alias2", "alias4"},
            "Q3": {"alias4"},
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }
        truealias2qids = {
            "alias1": [["Q7", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q7", 3], ["Q4", 2]],
            "alias3": [["Q7", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        trueqid2eid = {"Q7": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        truealias2id = {"alias1": 0, "alias3": 1, "alias4": 2, "multi word alias2": 3}
        trueid2alias = {0: "alias1", 1: "alias3", 2: "alias4", 3: "multi word alias2"}
        truemax_eid = 4
        truenum_entities = 4
        truenum_entities_with_pad_and_nocand = 6
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(
            entity_symbols._eid2qid, {v: i for i, v in trueqid2eid.items()}
        )
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)
        self.assertEqual(entity_symbols.max_eid, truemax_eid)
        self.assertEqual(entity_symbols.num_entities, truenum_entities)
        self.assertEqual(
            entity_symbols.num_entities_with_pad_and_nocand,
            truenum_entities_with_pad_and_nocand,
        )

    def test_prune_to_entities(self):
        alias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2], ["Q3", 1]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        qid2title = {
            "Q1": "alias1",
            "Q2": "multi alias2",
            "Q3": "word alias3",
            "Q4": "nonalias4",
        }
        max_candidates = 3

        entity_symbols = EntitySymbols(
            max_candidates=max_candidates,
            alias2qids=alias2qids,
            qid2title=qid2title,
            edit_mode=True,
        )
        entity_symbols.prune_to_entities({"Q3", "Q4"})
        trueqid2aliases = {
            "Q3": {"alias4"},
            "Q4": {"alias1", "multi word alias2", "alias4"},
        }
        truealias2qids = {
            "alias1": [["Q4", 6]],
            "multi word alias2": [["Q4", 2]],
            "alias4": [["Q4", 20], ["Q3", 15.0]],
        }
        trueqid2eid = {"Q3": 1, "Q4": 2}
        truealias2id = {"alias1": 0, "alias4": 1, "multi word alias2": 2}
        trueid2alias = {0: "alias1", 1: "alias4", 2: "multi word alias2"}
        truemax_eid = 2
        truemax_alid = 2
        truenum_entities = 2
        truenum_entities_with_pad_and_nocand = 4
        self.assertDictEqual(entity_symbols._qid2aliases, trueqid2aliases)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(
            entity_symbols._eid2qid, {v: i for i, v in trueqid2eid.items()}
        )
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertIsNone(entity_symbols._alias_trie)
        self.assertDictEqual(entity_symbols._alias2id, truealias2id)
        self.assertDictEqual(entity_symbols._id2alias, trueid2alias)
        self.assertEqual(entity_symbols.max_eid, truemax_eid)
        self.assertEqual(entity_symbols.max_alid, truemax_alid)
        self.assertEqual(entity_symbols.num_entities, truenum_entities)
        self.assertEqual(
            entity_symbols.num_entities_with_pad_and_nocand,
            truenum_entities_with_pad_and_nocand,
        )


class AliasTableTest(unittest.TestCase):
    def setUp(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        self.entity_symbols = EntitySymbols.load_from_cache(
            entity_dump_dir, alias_cand_map_file="alias2qids.json"
        )
        self.config = {
            "data_config": {
                "train_in_candidates": False,
                "entity_dir": "test/data/entity_loader/entity_data",
                "entity_prep_dir": "prep",
                "alias_cand_map": "alias2qids.json",
                "max_aliases": 3,
                "data_dir": "test/data/entity_loader",
                "overwrite_preprocessed_data": True,
            },
            "run_config": {"distributed": False},
        }

    def tearDown(self) -> None:
        dir = os.path.join(
            self.config["data_config"]["entity_dir"],
            self.config["data_config"]["entity_prep_dir"],
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir)

    def test_setup_notincand(self):
        self.alias_entity_table = AliasEntityTable(
            DottedDict(self.config["data_config"]), self.entity_symbols
        )
        gold_alias2entity_table = torch.tensor(
            [
                [0, 1, 4, -1],
                [0, 1, -1, -1],
                [0, 4, 3, 2],
                [0, 2, 1, 4],
                [-1, -1, -1, -1],
            ]
        )
        assert torch.equal(
            gold_alias2entity_table.long(),
            self.alias_entity_table.alias2entity_table.long(),
        )

    def test_setup_incand(self):
        self.config["data_config"]["train_in_candidates"] = True
        self.alias_entity_table = AliasEntityTable(
            DottedDict(self.config["data_config"]), self.entity_symbols
        )

        gold_alias2entity_table = torch.tensor(
            [[1, 4, -1], [1, -1, -1], [4, 3, 2], [2, 1, 4], [-1, -1, -1]]
        )
        assert torch.equal(
            gold_alias2entity_table.long(),
            self.alias_entity_table.alias2entity_table.long(),
        )

    def test_forward(self):
        self.alias_entity_table = AliasEntityTable(
            DottedDict(self.config["data_config"]), self.entity_symbols
        )
        # idx 0 is multi word alias 2, idx 1 is alias 1
        actual_indices = self.alias_entity_table.forward(torch.tensor([[[0, 1]]]))
        # 0 is for non-candidate, -1 is for padded value
        expected_tensor = torch.tensor([[[[0, 1, 4, -1], [0, 1, -1, -1]]]])
        assert torch.equal(actual_indices.long(), expected_tensor.long())


if __name__ == "__main__":
    unittest.main()
