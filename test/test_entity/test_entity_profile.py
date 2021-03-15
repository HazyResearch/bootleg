import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch
import ujson
from pydantic import ValidationError

import emmental
from bootleg.run import run_model
from bootleg.symbols.entity_profile import (
    ENTITY_SUBFOLDER,
    KG_SUBFOLDER,
    TYPE_SUBFOLDER,
    EntityProfile,
)
from bootleg.utils.parser import parser_utils


class EntityProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path("test/data/entity_profile_test")
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
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)

    def write_data(self, file, data):
        with open(file, "w") as out_f:
            for d in data:
                out_f.write(ujson.dumps(d) + "\n")

    def test_profile_load_simple(self):
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
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
            # Missing type system
            {
                "entity_id": "Q567",
                "mentions": [["catt", 6.5], ["animal", 3.3]],
                "title": "Catt",
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
            alias2qids,
            type_systems,
            qid2relations,
        ) = EntityProfile._read_profile_file(self.profile_file)

        self.assertDictEqual(gold_qid2title, qid2title)
        self.assertDictEqual(gold_alias2qids, alias2qids)
        self.assertDictEqual(gold_type_systems, type_systems)
        self.assertDictEqual(gold_qid2relations, qid2relations)

        # Test loading/saving from jsonl
        ep = EntityProfile.load_from_jsonl(self.profile_file, edit_mode=True)
        ep.save_to_jsonl(self.profile_file)
        read_in_data = [ujson.loads(l) for l in open(self.profile_file)]

        assert len(read_in_data) == len(data)

        for qid_obj in data:
            found_other_obj = None
            for possible_match in read_in_data:
                if qid_obj["entity_id"] == possible_match["entity_id"]:
                    found_other_obj = possible_match
                    break
            assert found_other_obj is not None
            self.assertDictEqual(qid_obj, found_other_obj)

    def test_profile_load_jsonl_errors(self):
        data = [
            {
                "entity_id": 123,
                "mentions": [["dog"], ["dogg"], ["animal"]],
                "title": "Dog",
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
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
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
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
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

    def test_add_entity(self):
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
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
            "types": {"hyena": ["animal"], "new_sys": ["snakey"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }

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
            "mentions": [["snake", 10.0], ["animal", 3.0]],
            "title": "Snake",
            "types": {"hyena": ["animal"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }
        # Assert it is added
        entity_profile.add_entity(new_entity)
        self.assertTrue(entity_profile.qid_exists("Q789"))
        self.assertEqual(entity_profile.get_title("Q789"), "Snake")
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
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
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
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
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
        # ======================
        # PART 1: TRAIN A SMALL MODEL WITH ONE PROFILE DUMP
        # ======================
        # Generate entity profile data
        data = [
            {
                "entity_id": "Q123",
                "mentions": [["dog", 10.0], ["dogg", 7.0], ["animal", 4.0]],
                "title": "Dog",
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
            # Missing type system
            {
                "entity_id": "Q567",
                "mentions": [["catt", 6.5], ["animal", 3.3]],
                "title": "Catt",
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
                "dataloader_threads": 1,
                "dataset_threads": 1,
            },
            "train_config": {"batch_size": 2},
            "model_config": {"hidden_size": 20, "num_heads": 1},
            "data_config": {
                "entity_dir": str(self.save_dir),
                "max_seq_len": 7,
                "max_aliases": 2,
                "data_dir": str(self.data_dir),
                "emb_dir": str(self.save_dir),
                "word_embedding": {
                    "layers": 1,
                    "freeze": True,
                    "cache_dir": str(self.save_dir / "retrained_bert_model"),
                },
                "ent_embeddings": [
                    {
                        "key": "learned",
                        "freeze": False,
                        "load_class": "LearnedEntityEmb",
                        "args": {"learned_embedding_size": 10},
                    },
                    {
                        "key": "learned_type",
                        "load_class": "LearnedTypeEmb",
                        "freeze": True,
                        "args": {
                            "type_labels": f"{TYPE_SUBFOLDER}/wiki/qid2typeids.json",
                            "type_vocab": f"{TYPE_SUBFOLDER}/wiki/type_vocab.json",
                            "max_types": 2,
                            "type_dim": 10,
                        },
                    },
                    {
                        "key": "kg_adj",
                        "load_class": "KGIndices",
                        "batch_on_the_fly": True,
                        "normalize": False,
                        "args": {"kg_adj": f"{KG_SUBFOLDER}/kg_adj.txt"},
                    },
                ],
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
        args["data_config"]["emb_dir"] = str(self.save_dir2)

        # Load pretrained model
        args["model_config"]["model_path"] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"]["model_path"] = saved_model_path1

        # Init another run
        emmental.init(log_dir=str(self.dir / "temp_log"), config=args)
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)
        scores = run_model(mode="train", config=args)
        saved_model_path2 = f"{emmental.Meta.log_path}/last_model.pth"
        assert type(scores) is dict

        # ======================
        # PART 4: VERIFY CHANGES IN THE MODEL WERE AS EXPECTED
        # ======================
        # Check that type mappings are different in the right way...we remove "dog" from EID 1 and added "cat". "dog" is not longer a type.
        eid2typeids_table1, type2row_dict1, num_types_with_unk1 = torch.load(
            self.save_dir / "prep" / "type_table_type_mappings_wiki_qid2typeids_2.pt"
        )
        eid2typeids_table2, type2row_dict2, num_types_with_unk2 = torch.load(
            self.save_dir2 / "prep" / "type_table_type_mappings_wiki_qid2typeids_2.pt"
        )
        # Modify mapping 2 to become mapping 1
        # Row 1 is Q123, Col 0 is type (this was "cat")
        eid2typeids_table2[1][0] = entity_profile._type_systems["wiki"]._type_vocab[
            "dog"
        ]
        self.assertEqual(num_types_with_unk1, num_types_with_unk2)
        self.assertDictEqual({1: 1, 2: 2}, type2row_dict1)
        self.assertDictEqual({1: 1}, type2row_dict2)
        assert torch.equal(eid2typeids_table1, eid2typeids_table2)
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
            ]
        )
        assert torch.equal(alias2entity_table2, gold_alias2entity_table2)

        # The type embeddings were frozen so they should be the same
        model1 = torch.load(saved_model_path1)
        model2 = torch.load(saved_model_path2)
        assert torch.equal(
            model1["model"]["module_pool"]["learned_type"]["type_emb.weight"],
            model2["model"]["module_pool"]["learned_type"]["type_emb.weight"],
        )


if __name__ == "__main__":
    unittest.main()
