"""Test entity profile."""
import os
import shutil
import unittest
from pathlib import Path

import emmental
import numpy as np
import torch
import ujson
from pydantic import ValidationError

from bootleg.run import run_model
from bootleg.symbols.entity_profile import EntityProfile
from bootleg.utils.parser import parser_utils


class EntityProfileTest(unittest.TestCase):
    """Entity profile test."""

    def setUp(self) -> None:
        """Set up."""
        self.dir = Path("tests/data/entity_profile_test")
        self.save_dir = Path(self.dir / "entity_db_save")
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir2 = Path(self.dir / "entity_db_save2")
        self.save_dir2.mkdir(exist_ok=True, parents=True)
        self.profile_file = Path(self.dir / "raw_data/entity_profile.jsonl")
        self.profile_file.parent.mkdir(exist_ok=True, parents=True)
        self.data_dir = self.dir / "data"
        self.train_data = self.data_dir / "train.jsonl"
        self.train_data.parent.mkdir(exist_ok=True, parents=True)
        self.arg_file = self.dir / "args.json"

    def tearDown(self) -> None:
        """Tear down."""
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)

    def write_data(self, file, data):
        """Write data to file."""
        with open(file, "w") as out_f:
            for d in data:
                out_f.write(ujson.dumps(d) + "\n")

    def test_profile_load_simple(self):
        """Test profile load simple."""
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "description": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
            # Missing type system
            {
                "entity_id": "Q567",
                "mentions": [["catt", 6.5], ["animal", 3.3]],
                "title": "Catt",
                "description": "Catt",
                "types": {"hyena": ["animal", "animall"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
            # No KG/Types
            {
                "entity_id": "Q789",
                "mentions": [["animal", 12.2]],
                "title": "Dogg",
            },
        ]
        self.write_data(self.profile_file, data)
        gold_qid2title = {"Q123": "Dog", "Q345": "Cat", "Q567": "Catt", "Q789": "Dogg"}
        gold_qid2desc = {"Q123": "Dog", "Q345": "Cat", "Q567": "Catt", "Q789": ""}
        gold_alias2qids = {
            "dog": [["Q123", 10.0]],
            "dogg": [["Q123", 7.0]],
            "cat": [["Q345", 10.0]],
            "catt": [["Q345", 7.0], ["Q567", 6.5]],
            "animal": [["Q789", 12.2], ["Q123", 4.0], ["Q567", 3.3], ["Q345", 3.0]],
        }
        gold_type_systems = {
            "hyena": {
                "Q123": ["animal"],
                "Q345": ["animal"],
                "Q567": ["animal", "animall"],
                "Q789": [],
            },
            "wiki": {"Q123": ["dog"], "Q345": ["cat"], "Q567": [], "Q789": []},
        }
        gold_qid2relations = {
            "Q123": {"sibling": ["Q345", "Q567"]},
            "Q345": {"sibling": ["Q123"]},
            "Q567": {"sibling": ["Q123"]},
            "Q789": {},
        }
        (
            qid2title,
            qid2desc,
            alias2qids,
            type_systems,
            qid2relations,
        ) = EntityProfile._read_profile_file(self.profile_file)

        self.assertDictEqual(gold_qid2title, qid2title)
        self.assertDictEqual(gold_qid2desc, qid2desc)
        self.assertDictEqual(gold_alias2qids, alias2qids)
        self.assertDictEqual(gold_type_systems, type_systems)
        self.assertDictEqual(gold_qid2relations, qid2relations)

        # Test loading/saving from jsonl
        ep = EntityProfile.load_from_jsonl(self.profile_file, edit_mode=True)
        ep.save_to_jsonl(self.profile_file)
        read_in_data = [ujson.loads(li) for li in open(self.profile_file)]

        assert len(read_in_data) == len(data)

        for qid_obj in data:
            found_other_obj = None
            for possible_match in read_in_data:
                if qid_obj["entity_id"] == possible_match["entity_id"]:
                    found_other_obj = possible_match
                    break
            assert found_other_obj is not None
            self.assertDictEqual(qid_obj, found_other_obj)

        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            # Extra QID
            {
                "entity_id": "Q123",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "description": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
        ]
        self.write_data(self.profile_file, data)
        with self.assertRaises(ValueError) as context:
            EntityProfile._read_profile_file(self.profile_file)
        assert type(context.exception) is ValueError
        assert "is already in our dump" in str(context.exception)

        data = [
            # Relation in wrong format
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relationnn": "sibling", "objject": "Q345"},
                ],
            }
        ]
        self.write_data(self.profile_file, data)
        with self.assertRaises(ValueError) as context:
            EntityProfile._read_profile_file(self.profile_file)
        assert type(context.exception) is ValueError
        assert "it must be a JSON with keys relation and object" in str(
            context.exception
        )

    def test_profile_load_jsonl_errors(self):
        """Test profile load from jsonl."""
        data = [
            {
                "entity_id": 123,
                "mentions": [["dog"], ["dogg"], ["animal"]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
        ]
        self.write_data(self.profile_file, data)
        with self.assertRaises(ValidationError) as context:
            EntityProfile._read_profile_file(self.profile_file)
        assert type(context.exception) is ValidationError

    def test_profile_dump_load(self):
        """Test profile load from dump."""
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
        ]
        self.write_data(self.profile_file, data)
        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, max_candidates=5, edit_mode=True
        )
        entity_profile.save(self.save_dir2)

        # Test load correctly
        entity_profile2 = EntityProfile.load_from_cache(self.save_dir2)

        self.assertSetEqual(
            set(entity_profile.get_all_qids()), set(entity_profile2.get_all_qids())
        )
        self.assertSetEqual(
            set(entity_profile.get_all_typesystems()),
            set(entity_profile2.get_all_typesystems()),
        )
        for type_sys in entity_profile.get_all_typesystems():
            self.assertSetEqual(
                set(entity_profile.get_all_types(type_sys)),
                set(entity_profile2.get_all_types(type_sys)),
            )
        for qid in entity_profile.get_all_qids():
            self.assertSetEqual(
                set(entity_profile.get_all_connections(qid)),
                set(entity_profile2.get_all_connections(qid)),
            )

        # Test load with no types or kgs
        entity_profile2 = EntityProfile.load_from_cache(
            self.save_dir2, no_type=True, no_kg=True
        )

        self.assertSetEqual(
            set(entity_profile.get_all_qids()), set(entity_profile2.get_all_qids())
        )
        assert len(entity_profile2.get_all_typesystems()) == 0
        self.assertIsNone(entity_profile2._kg_symbols)

        # Testing that the functions still work despite not loading them
        assert len(entity_profile2.get_all_connections("Q123")) == 0

        # Test load with no types or kgs
        entity_profile2 = EntityProfile.load_from_cache(
            self.save_dir2, no_kg=True, type_systems_to_load=["wiki"]
        )

        self.assertSetEqual(
            set(entity_profile.get_all_qids()), set(entity_profile2.get_all_qids())
        )
        assert entity_profile2.get_all_typesystems() == ["wiki"]
        self.assertSetEqual(
            set(entity_profile.get_all_types("wiki")),
            set(entity_profile2.get_all_types("wiki")),
        )
        self.assertIsNone(entity_profile2._kg_symbols)

        # Assert error loading type system that is not there
        with self.assertRaises(ValueError) as context:
            entity_profile2.get_all_types("hyena")
        assert type(context.exception) is ValueError
        assert "type system hyena is not one" in str(context.exception)

    def test_checks(self):
        """Test checks."""
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
        ]
        self.write_data(self.profile_file, data)
        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, max_candidates=5
        )

        with self.assertRaises(AttributeError) as context:
            entity_profile.add_relation("Q345", "sibling", "Q123")
        assert type(context.exception) is AttributeError

        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, max_candidates=5, edit_mode=True
        )

        with self.assertRaises(ValueError) as context:
            entity_profile.add_relation("Q789", "sibling", "Q123")
        assert type(context.exception) is ValueError
        assert "is not in our dump" in str(context.exception)

        with self.assertRaises(ValueError) as context:
            entity_profile.add_relation(qid="Q789", relation="sibling", qid2="Q123")
        assert type(context.exception) is ValueError
        assert "is not in our dump" in str(context.exception)

        with self.assertRaises(ValueError) as context:
            entity_profile.add_type(qid="Q345", type="sibling", type_system="blah")
        assert type(context.exception) is ValueError
        assert "type system blah is not one" in str(context.exception)

        with self.assertRaises(ValueError) as context:
            entity_profile.get_types(qid="Q345", type_system="blah")
        assert type(context.exception) is ValueError
        assert "type system blah is not one" in str(context.exception)

    def test_getters(self):
        """Test getters."""
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
        ]
        self.write_data(self.profile_file, data)
        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, max_candidates=3, edit_mode=True
        )
        self.assertEqual(entity_profile.get_eid("Q345"), 2)
        self.assertTrue(entity_profile.mention_exists("cat"))
        self.assertFalse(entity_profile.mention_exists("dat"))
        self.assertListEqual(entity_profile.get_qid_cands("cat"), ["Q345"])
        self.assertListEqual(
            entity_profile.get_qid_count_cands("cat"), [["Q345", 10.0]]
        )
        self.assertSetEqual(
            set(entity_profile.get_all_mentions()),
            {"dog", "dogg", "animal", "cat", "catt"},
        )
        self.assertSetEqual(
            set(entity_profile.get_mentions("Q345")), {"animal", "cat", "catt"}
        )
        self.assertSetEqual(
            set(entity_profile.get_entities_of_type("cat", "wiki")), {"Q345"}
        )
        self.assertEqual(entity_profile.num_entities_with_pad_and_nocand, 4)
        self.assertTrue(entity_profile.is_connected("Q123", "Q345"))

    def test_add_entity(self):
        """Test add entity."""
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
        ]
        self.write_data(self.profile_file, data)
        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, max_candidates=3, edit_mode=True
        )
        entity_profile.save(self.save_dir2)

        # Test bad format
        with self.assertRaises(ValueError) as context:
            entity_profile.add_entity(["bad format"])
        assert type(context.exception) is ValueError
        assert "The input to update_entity needs to be a dictionary" in str(
            context.exception
        )

        new_entity = {
            "entity_id": "Q345",
            "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
            "title": "Cat",
            "types": {"hyena": ["animal"], "wiki": ["cat"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }

        # Test already existing entity
        with self.assertRaises(ValueError) as context:
            entity_profile.add_entity(new_entity)
        assert type(context.exception) is ValueError
        assert "The entity Q345 already exists" in str(context.exception)

        new_entity = {
            "entity_id": "Q789",
            "mentions": [["snake", 10.0], ["animal", 3.0]],
            "title": "Snake",
            "description": "Snake",
            "types": {"hyena": ["animal"], "new_sys": ["snakey"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }

        # Test can't update qid not in dump
        with self.assertRaises(ValueError) as context:
            entity_profile.update_entity(new_entity)
        assert type(context.exception) is ValueError
        assert "The entity Q789 is not in our dump" in str(context.exception)

        # Test new type system
        with self.assertRaises(ValueError) as context:
            entity_profile.add_entity(new_entity)
        assert type(context.exception) is ValueError
        assert "When adding a new entity, you must use the same type system" in str(
            context.exception
        )

        new_entity = {
            "entity_id": "Q789",
            "mentions": [["snake", 10.0], ["animal", 3.0]],
            "title": "Snake",
            "description": "Snake",
            "types": {"hyena": ["animal"]},
            "relations": [{"relatiion": "sibbbling", "object": "Q123"}],
        }

        # Test new bad relation format
        with self.assertRaises(ValueError) as context:
            entity_profile.add_entity(new_entity)
        assert type(context.exception) is ValueError
        assert (
            "For each value in relations, it must be a JSON with keys relation and object"
            in str(context.exception)
        )

        new_entity = {
            "entity_id": "Q789",
            "mentions": [["snake", 10.0], ["animal", 3.0]],
            "title": "Snake",
            "description": "Snake",
            "types": {"hyena": ["animal"]},
            "relations": [{"relation": "sibbbling", "object": "Q123"}],
        }

        # Test new relation
        with self.assertRaises(ValueError) as context:
            entity_profile.add_entity(new_entity)
        assert type(context.exception) is ValueError
        assert (
            "When adding a new entity, you must use the same set of relations."
            in str(context.exception)
        )

        new_entity = {
            "entity_id": "Q789",
            "mentions": [["snakke", 10.0], ["animal", 3.0]],
            "title": "Snake",
            "description": "Snake",
            "types": {"hyena": ["animal"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }
        # Assert it is added
        entity_profile.add_entity(new_entity)
        self.assertTrue(entity_profile.qid_exists("Q789"))
        self.assertEqual(entity_profile.get_title("Q789"), "Snake")
        self.assertEqual(entity_profile.get_desc("Q789"), "Snake")
        self.assertListEqual(
            entity_profile.get_mentions_with_scores("Q789"),
            [["snakke", 10.0], ["animal", 3.0]],
        )
        self.assertListEqual(entity_profile.get_types("Q789", "hyena"), ["animal"])
        self.assertListEqual(entity_profile.get_types("Q789", "wiki"), [])
        self.assertListEqual(
            entity_profile.get_connections_by_relation("Q789", "sibling"), ["Q123"]
        )

        # Update entity
        new_entity = {
            "entity_id": "Q789",
            "mentions": [["snake", 10.0], ["animal", 3.0]],
            "title": "Snake",
            "description": "Snake",
            "types": {"hyena": ["animal"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }
        # Assert it is added
        entity_profile.update_entity(new_entity)
        self.assertTrue(entity_profile.qid_exists("Q789"))
        self.assertEqual(entity_profile.get_title("Q789"), "Snake")
        self.assertEqual(entity_profile.get_desc("Q789"), "Snake")
        self.assertListEqual(
            entity_profile.get_mentions_with_scores("Q789"),
            [["snake", 10.0], ["animal", 3.0]],
        )
        self.assertListEqual(entity_profile.get_types("Q789", "hyena"), ["animal"])
        self.assertListEqual(entity_profile.get_types("Q789", "wiki"), [])
        self.assertListEqual(
            entity_profile.get_connections_by_relation("Q789", "sibling"), ["Q123"]
        )

        # Check that no_kg still works with load_from_cache
        entity_profile2 = EntityProfile.load_from_cache(
            self.save_dir2, no_kg=True, edit_mode=True
        )
        entity_profile2.add_entity(new_entity)
        self.assertTrue(entity_profile2.qid_exists("Q789"))
        self.assertEqual(entity_profile2.get_title("Q789"), "Snake")
        self.assertEqual(entity_profile2.get_desc("Q789"), "Snake")
        self.assertListEqual(
            entity_profile2.get_mentions_with_scores("Q789"),
            [["snake", 10.0], ["animal", 3.0]],
        )
        self.assertListEqual(entity_profile2.get_types("Q789", "hyena"), ["animal"])
        self.assertListEqual(entity_profile2.get_types("Q789", "wiki"), [])
        self.assertListEqual(
            entity_profile2.get_connections_by_relation("Q789", "sibling"), []
        )

    def test_reindentify_entity(self):
        """Test reindentify entity."""
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
        ]
        self.write_data(self.profile_file, data)
        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, max_candidates=5, edit_mode=True
        )
        entity_profile.save(self.save_dir2)

        with self.assertRaises(ValueError) as context:
            entity_profile.reidentify_entity("Q123", "Q345")
        assert type(context.exception) is ValueError
        assert "The entity Q345 already exists" in str(context.exception)

        with self.assertRaises(ValueError) as context:
            entity_profile.reidentify_entity("Q125", "Q911")
        assert type(context.exception) is ValueError
        assert "The entity Q125 is not in our dump" in str(context.exception)
        entity_profile.reidentify_entity("Q123", "Q911")
        self.assertTrue(entity_profile.qid_exists("Q911"))
        self.assertFalse(entity_profile.qid_exists("Q123"))
        self.assertEqual(entity_profile.get_title("Q911"), "Dog")
        self.assertEqual(entity_profile.get_desc("Q911"), "Dog")
        self.assertListEqual(
            entity_profile.get_mentions_with_scores("Q911"),
            [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
        )
        self.assertListEqual(entity_profile.get_types("Q911", "hyena"), ["animal"])
        self.assertListEqual(entity_profile.get_types("Q911", "wiki"), ["dog"])
        self.assertListEqual(
            entity_profile.get_connections_by_relation("Q911", "sibling"),
            ["Q345", "Q567"],
        )

        # Check that no_kg still works with load_from_cache
        entity_profile2 = EntityProfile.load_from_cache(
            self.save_dir2, no_kg=True, edit_mode=True
        )
        entity_profile2.reidentify_entity("Q123", "Q911")
        self.assertTrue(entity_profile2.qid_exists("Q911"))
        self.assertFalse(entity_profile2.qid_exists("Q123"))
        self.assertEqual(entity_profile2.get_title("Q911"), "Dog")
        self.assertEqual(entity_profile2.get_desc("Q911"), "Dog")
        self.assertListEqual(
            entity_profile2.get_mentions_with_scores("Q911"),
            [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
        )
        self.assertListEqual(entity_profile2.get_types("Q911", "hyena"), ["animal"])
        self.assertListEqual(entity_profile2.get_types("Q911", "wiki"), ["dog"])
        self.assertListEqual(
            entity_profile2.get_connections_by_relation("Q911", "sibling"),
            [],
        )

    def test_prune_to_entities(self):
        """Test prune to entities."""
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
        ]
        self.write_data(self.profile_file, data)
        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, max_candidates=5, edit_mode=True
        )
        entity_profile.save(self.save_dir2)

        with self.assertRaises(ValueError) as context:
            entity_profile.prune_to_entities({"Q123", "Q567"})
        assert type(context.exception) is ValueError
        assert "The entity Q567 does not exist" in str(context.exception)

        entity_profile.prune_to_entities({"Q123"})
        self.assertTrue(entity_profile.qid_exists("Q123"))
        self.assertFalse(entity_profile.qid_exists("Q345"))
        self.assertListEqual(
            entity_profile.get_mentions_with_scores("Q123"),
            [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
        )
        self.assertListEqual(entity_profile.get_types("Q123", "hyena"), ["animal"])
        self.assertListEqual(entity_profile.get_types("Q123", "wiki"), ["dog"])
        self.assertListEqual(
            entity_profile.get_connections_by_relation("Q123", "sibling"),
            [],
        )

        # Check that no_kg still works with load_from_cache
        entity_profile2 = EntityProfile.load_from_cache(
            self.save_dir2, no_kg=True, edit_mode=True
        )
        entity_profile2.prune_to_entities({"Q123"})
        self.assertTrue(entity_profile2.qid_exists("Q123"))
        self.assertFalse(entity_profile2.qid_exists("Q345"))
        self.assertListEqual(
            entity_profile2.get_mentions_with_scores("Q123"),
            [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
        )
        self.assertListEqual(entity_profile2.get_types("Q123", "hyena"), ["animal"])
        self.assertListEqual(entity_profile2.get_types("Q123", "wiki"), ["dog"])
        self.assertListEqual(
            entity_profile2.get_connections_by_relation("Q123", "sibling"),
            [],
        )

    def test_end2end(self):
        """Test end2end."""
        # ======================
        # PART 1: TRAIN A SMALL MODEL WITH ONE PROFILE DUMP
        # ======================
        # Generate entity profile data
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
                "description": "Dog",
                "types": {"hyena": ["animal"], "wiki": ["dog"]},
                "relations": [
                    {"relation": "sibling", "object": "Q345"},
                    {"relation": "sibling", "object": "Q567"},
                ],
            },
            {
                "entity_id": "Q345",
                "mentions": [["cat", 10.0], ["catt", 7.0], ["animal", 3.0]],
                "title": "Cat",
                "description": "Cat",
                "types": {"hyena": ["animal"], "wiki": ["cat"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
            # Missing type system
            {
                "entity_id": "Q567",
                "mentions": [["catt", 6.5], ["animal", 3.3]],
                "title": "Catt",
                "description": "Catt",
                "types": {"hyena": ["animal", "animall"]},
                "relations": [{"relation": "sibling", "object": "Q123"}],
            },
            # No KG/Types
            {
                "entity_id": "Q789",
                "mentions": [["animal", 12.2]],
                "title": "Dogg",
            },
        ]
        # Generate train data
        train_data = [
            {
                "sent_idx_unq": 1,
                "sentence": "I love animals and dogs",
                "qids": ["Q567", "Q123"],
                "aliases": ["animal", "dog"],
                "gold": [True, True],
                "spans": [[2, 3], [4, 5]],
            }
        ]
        self.write_data(self.profile_file, data)
        self.write_data(self.train_data, train_data)
        entity_profile = EntityProfile.load_from_jsonl(self.profile_file)
        # Dump profile data in format for model
        entity_profile.save(self.save_dir)

        # Setup model args to read the data/new profile data
        raw_args = {
            "emmental": {
                "n_epochs": 1,
            },
            "run_config": {
                "dataloader_threads": 0,
                "dataset_threads": 1,
            },
            "model_config": {"hidden_size": 10},
            "train_config": {"batch_size": 2},
            "data_config": {
                "entity_dir": str(self.save_dir),
                "max_seq_len": 7,
                "data_dir": str(self.data_dir),
                "word_embedding": {
                    "context_layers": 1,
                    "entity_layers": 1,
                    "cache_dir": str(self.save_dir / "retrained_bert_model"),
                },
                "train_dataset": {"file": "train.jsonl"},
                "dev_dataset": {"file": "train.jsonl"},
                "test_dataset": {"file": "train.jsonl"},
            },
        }
        with open(self.arg_file, "w") as out_f:
            ujson.dump(raw_args, out_f)

        args = parser_utils.parse_boot_and_emm_args(str(self.arg_file))
        # This _MUST_ get passed the args so it gets a random seed set
        emmental.init(log_dir=str(self.dir / "temp_log"), config=args)
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)

        # ======================
        # PART 2: RUN MODEL
        # ======================
        scores = run_model(mode="train", config=args)
        saved_model_path1 = f"{emmental.Meta.log_path}/last_model.pth"
        assert type(scores) is dict

        # ======================
        # PART 3: MODIFY PROFILE AND LOAD PRETRAINED MODEL AND TRAIN FOR MORE
        # ======================
        entity_profile = EntityProfile.load_from_jsonl(
            self.profile_file, edit_mode=True
        )
        entity_profile.add_type("Q123", "cat", "wiki")
        entity_profile.remove_type("Q123", "dog", "wiki")
        entity_profile.add_mention("Q123", "cat", 100.0)
        # Dump profile data in format for model
        entity_profile.save(self.save_dir2)

        # Modify arg paths
        args["data_config"]["entity_dir"] = str(self.save_dir2)

        # Load pretrained model
        args["model_config"]["model_path"] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"]["model_path"] = saved_model_path1

        # Init another run
        emmental.init(log_dir=str(self.dir / "temp_log"), config=args)
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)
        scores = run_model(mode="train", config=args)
        assert type(scores) is dict

        # ======================
        # PART 4: VERIFY CHANGES IN THE MODEL WERE AS EXPECTED
        # ======================
        # Check that the alias mappings are different
        alias2entity_table1 = torch.from_numpy(
            np.memmap(
                self.save_dir / "prep" / "alias2entity_table_alias2qids_InC1.pt",
                dtype="int64",
                mode="r",
                shape=(5, 30),
            )
        )
        gold_alias2entity_table1 = torch.tensor(
            [
                [
                    1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    4,
                    1,
                    3,
                    2,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    2,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    2,
                    3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
            ]
        )

        assert torch.equal(alias2entity_table1, gold_alias2entity_table1)
        # The change is the "cat" alias has entity 1 added to the beginning
        # It used to only point to Q345 which is entity 2
        alias2entity_table2 = torch.from_numpy(
            np.memmap(
                self.save_dir2 / "prep" / "alias2entity_table_alias2qids_InC1.pt",
                dtype="int64",
                mode="r",
                shape=(5, 30),
            )
        )
        gold_alias2entity_table2 = torch.tensor(
            [
                [
                    1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    4,
                    1,
                    3,
                    2,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    1,
                    2,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    2,
                    3,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
            ]
        )

        assert torch.equal(alias2entity_table2, gold_alias2entity_table2)


if __name__ == "__main__":
    unittest.main()
