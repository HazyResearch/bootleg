import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch
import ujson
from transformers import BertModel, BertTokenizer

from bootleg.symbols.entity_profile import EntityProfile
from bootleg.utils.entity_profile.fit_to_profile import (
    match_entities,
    refit_titles,
    refit_weights,
)
from bootleg.utils.preprocessing.build_static_embeddings import average_titles


class TestFitToProfile(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path("test/data/fit_to_profile_test")
        self.train_db = Path(self.dir / "train_entity_db")
        self.train_db.mkdir(exist_ok=True, parents=True)
        self.new_db = Path(self.dir / "entity_db_save2")
        self.new_db.mkdir(exist_ok=True, parents=True)
        self.profile_file = Path(self.dir / "raw_data/entity_profile.jsonl")
        self.profile_file.parent.mkdir(exist_ok=True, parents=True)
        # Dump train profile data
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
        ep = EntityProfile.load_from_jsonl(self.profile_file)
        ep.save(self.train_db)

    def tearDown(self) -> None:
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)

    def write_data(self, file, data):
        with open(file, "w") as out_f:
            for d in data:
                out_f.write(ujson.dumps(d) + "\n")

    def test_match_entities(self):
        # TEST ADD
        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
            edit_mode=True,
        )
        # Modify train profle
        new_entity1 = {
            "entity_id": "Q910",
            "mentions": [["cobra", 10.0], ["animal", 3.0]],
            "title": "Cobra",
            "types": {"hyena": ["animal"], "wiki": ["dog"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }
        new_entity2 = {
            "entity_id": "Q101",
            "mentions": [["snake", 10.0], ["snakes", 7.0], ["animal", 3.0]],
            "title": "Snake",
            "types": {"hyena": ["animal"], "wiki": ["dog"]},
        }
        train_entity_profile.add_entity(new_entity1)
        train_entity_profile.add_entity(new_entity2)
        train_entity_profile.reidentify_entity("Q123", "Q321")
        # Save new profile
        train_entity_profile.save(self.new_db)
        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
        )
        new_entity_profile = EntityProfile.load_from_cache(load_dir=self.new_db)

        oldqid2newqid = {"Q123": "Q321"}
        newqid2oldqid = {v: k for k, v in oldqid2newqid.items()}
        np_removed_ents, np_same_ents, np_new_ents, oldeid2neweid = match_entities(
            train_entity_profile, new_entity_profile, oldqid2newqid, newqid2oldqid
        )
        gold_removed_ents = set()
        gold_same_ents = {"Q321", "Q345", "Q567", "Q789"}
        gold_new_ents = {"Q910", "Q101"}
        gold_oldeid2neweid = {1: 1, 2: 2, 3: 3, 4: 4}
        self.assertSetEqual(gold_removed_ents, np_removed_ents)
        self.assertSetEqual(gold_same_ents, np_same_ents)
        self.assertSetEqual(gold_new_ents, np_new_ents)
        self.assertDictEqual(gold_oldeid2neweid, oldeid2neweid)

        # TEST PRUNE - this profile already has 910 and 101
        new_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.new_db, edit_mode=True
        )
        new_entity_profile.prune_to_entities(
            {"Q321", "Q910", "Q101"}
        )  # These now get eids 1, 2, 3
        # Manually set the eids for the test
        new_entity_profile._entity_symbols._qid2eid = {"Q321": 3, "Q910": 2, "Q101": 1}
        new_entity_profile._entity_symbols._eid2qid = {3: "Q321", 2: "Q910", 1: "Q101"}
        # Save new profile
        new_entity_profile.save(self.new_db)
        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
        )
        new_entity_profile = EntityProfile.load_from_cache(load_dir=self.new_db)
        oldqid2newqid = {"Q123": "Q321"}
        newqid2oldqid = {v: k for k, v in oldqid2newqid.items()}
        np_removed_ents, np_same_ents, np_new_ents, oldeid2neweid = match_entities(
            train_entity_profile, new_entity_profile, oldqid2newqid, newqid2oldqid
        )
        gold_removed_ents = {"Q345", "Q567", "Q789"}
        gold_same_ents = {"Q321"}
        gold_new_ents = {"Q910", "Q101"}
        gold_oldeid2neweid = {1: 3}
        self.assertSetEqual(gold_removed_ents, np_removed_ents)
        self.assertSetEqual(gold_same_ents, np_same_ents)
        self.assertSetEqual(gold_new_ents, np_new_ents)
        self.assertDictEqual(gold_oldeid2neweid, oldeid2neweid)

    def test_fit_entities(self):
        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
            edit_mode=True,
        )
        # Modify train profle
        new_entity1 = {
            "entity_id": "Q910",
            "mentions": [["cobra", 10.0], ["animal", 3.0]],
            "title": "Cobra",
            "types": {"hyena": ["animal"], "wiki": ["dog"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }
        new_entity2 = {
            "entity_id": "Q101",
            "mentions": [["snake", 10.0], ["snakes", 7.0], ["animal", 3.0]],
            "title": "Snake",
            "types": {"hyena": ["animal"], "wiki": ["dog"]},
        }
        train_entity_profile.add_entity(new_entity1)
        train_entity_profile.add_entity(new_entity2)
        train_entity_profile.reidentify_entity("Q123", "Q321")
        # Save new profile
        train_entity_profile.save(self.new_db)
        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
        )
        new_entity_profile = EntityProfile.load_from_cache(load_dir=self.new_db)
        neweid2oldeid = {1: 1, 2: 2, 3: 3, 4: 4}
        np_same_ents = {"Q321", "Q345", "Q567", "Q789"}
        state_dict = {
            "module_pool": {
                "learned": {
                    # 4 entities + 1 UNK + 1 PAD for train data
                    "learned_entity_embedding.weight": torch.tensor(
                        [
                            [1.0, 2, 3, 4, 5],
                            [2, 2, 2, 2, 2],
                            [3, 3, 3, 3, 3],
                            [4, 4, 4, 4, 4],
                            [5, 5, 5, 5, 5],
                            [0, 0, 0, 0, 0],
                        ]
                    )
                }
            }
        }
        vector_for_new_ent = np.arange(5)
        new_state_dict = refit_weights(
            np_same_ents,
            neweid2oldeid,
            train_entity_profile,
            new_entity_profile,
            vector_for_new_ent,
            state_dict,
        )

        gold_state_dict = {
            "module_pool": {
                "learned": {
                    "learned_entity_embedding.weight": torch.tensor(
                        [
                            [1.0, 2, 3, 4, 5],
                            [2, 2, 2, 2, 2],
                            [3, 3, 3, 3, 3],
                            [4, 4, 4, 4, 4],
                            [5, 5, 5, 5, 5],
                            [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4],
                            [0, 0, 0, 0, 0],
                        ]
                    )
                }
            }
        }
        gld = gold_state_dict
        nsd = new_state_dict
        keys_to_check = ["module_pool", "learned", "learned_entity_embedding.weight"]
        for k in keys_to_check:
            assert k in nsd
            assert k in gld
            if type(gld[k]) is dict:
                gld = gld[k]
                nsd = nsd[k]
                continue
            else:
                assert torch.equal(nsd[k], gld[k])

        # TEST WITH EIDREG
        state_dict = {
            "module_pool": {
                "learned": {
                    # 4 entities + 1 UNK + 1 PAD for train data
                    "learned_entity_embedding.weight": torch.tensor(
                        [
                            [1.0, 2, 3, 4, 5],
                            [2, 2, 2, 2, 2],
                            [3, 3, 3, 3, 3],
                            [4, 4, 4, 4, 4],
                            [5, 5, 5, 5, 5],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    "eid2reg": torch.tensor(
                        [0.0, 0.2, 0.3, 0.4, 0.5, 0.0],
                    ),
                }
            }
        }
        vector_for_new_ent = np.arange(5)
        new_state_dict = refit_weights(
            np_same_ents,
            neweid2oldeid,
            train_entity_profile,
            new_entity_profile,
            vector_for_new_ent,
            state_dict,
        )

        gold_state_dict = {
            "module_pool": {
                "learned": {
                    "learned_entity_embedding.weight": torch.tensor(
                        [
                            [1.0, 2, 3, 4, 5],
                            [2, 2, 2, 2, 2],
                            [3, 3, 3, 3, 3],
                            [4, 4, 4, 4, 4],
                            [5, 5, 5, 5, 5],
                            [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    "eid2reg": torch.tensor(
                        [0.0, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.0],
                    ),
                }
            }
        }
        gld = gold_state_dict
        nsd = new_state_dict
        keys_to_check = ["module_pool", "learned", "learned_entity_embedding.weight"]
        for k in keys_to_check:
            assert k in nsd
            assert k in gld
            if type(gld[k]) is dict:
                gld = gld[k]
                nsd = nsd[k]
                continue
            else:
                assert torch.equal(nsd[k], gld[k])
        gld = gold_state_dict
        nsd = new_state_dict
        keys_to_check = ["module_pool", "learned", "eid2reg"]
        for k in keys_to_check:
            assert k in nsd
            assert k in gld
            if type(gld[k]) is dict:
                gld = gld[k]
                nsd = nsd[k]
                continue
            else:
                assert torch.equal(nsd[k], gld[k])

        # TEST WITH PRUNE
        # TEST PRUNE - this profile already has 910 and 101
        new_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.new_db, edit_mode=True
        )
        new_entity_profile.prune_to_entities(
            {"Q321", "Q910", "Q101"}
        )  # These now get eids 1, 2, 3
        # Manually set the eids for the test
        new_entity_profile._entity_symbols._qid2eid = {"Q321": 3, "Q910": 2, "Q101": 1}
        new_entity_profile._entity_symbols._eid2qid = {3: "Q321", 2: "Q910", 1: "Q101"}
        new_entity_profile.save(self.new_db)
        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
        )
        new_entity_profile = EntityProfile.load_from_cache(load_dir=self.new_db)
        neweid2oldeid = {3: 1}
        np_same_ents = {"Q321"}
        state_dict = {
            "module_pool": {
                "learned": {
                    # 4 entities + 1 UNK + 1 PAD for train data
                    "learned_entity_embedding.weight": torch.tensor(
                        [
                            [1.0, 2, 3, 4, 5],
                            [2, 2, 2, 2, 2],
                            [3, 3, 3, 3, 3],
                            [4, 4, 4, 4, 4],
                            [5, 5, 5, 5, 5],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    "eid2reg": torch.tensor(
                        [0.0, 0.2, 0.3, 0.4, 0.5, 0.0],
                    ),
                }
            }
        }
        vector_for_new_ent = np.arange(5)
        new_state_dict = refit_weights(
            np_same_ents,
            neweid2oldeid,
            train_entity_profile,
            new_entity_profile,
            vector_for_new_ent,
            state_dict,
        )

        gold_state_dict = {
            "module_pool": {
                "learned": {
                    "learned_entity_embedding.weight": torch.tensor(
                        [
                            [1.0, 2, 3, 4, 5],
                            [0, 1, 2, 3, 4],
                            [0, 1, 2, 3, 4],
                            [2, 2, 2, 2, 2],
                            [0, 0, 0, 0, 0],
                        ]
                    ),
                    "eid2reg": torch.tensor(
                        [0.0, 0.5, 0.5, 0.2, 0.0],
                    ),
                }
            }
        }
        gld = gold_state_dict
        nsd = new_state_dict
        keys_to_check = ["module_pool", "learned", "learned_entity_embedding.weight"]
        for k in keys_to_check:
            assert k in nsd
            assert k in gld
            if type(gld[k]) is dict:
                gld = gld[k]
                nsd = nsd[k]
                continue
            else:
                assert torch.equal(nsd[k], gld[k])
        gld = gold_state_dict
        nsd = new_state_dict
        keys_to_check = ["module_pool", "learned", "eid2reg"]
        for k in keys_to_check:
            assert k in nsd
            assert k in gld
            if type(gld[k]) is dict:
                gld = gld[k]
                nsd = nsd[k]
                continue
            else:
                assert torch.equal(nsd[k], gld[k])

    def test_fit_titles(self):
        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
            edit_mode=True,
        )
        # Modify train profle
        new_entity1 = {
            "entity_id": "Q910",
            "mentions": [["cobra", 10.0], ["animal", 3.0]],
            "title": "Cobra",
            "types": {"hyena": ["animal"], "wiki": ["dog"]},
            "relations": [{"relation": "sibling", "object": "Q123"}],
        }
        new_entity2 = {
            "entity_id": "Q101",
            "mentions": [["snake", 10.0], ["snakes", 7.0], ["animal", 3.0]],
            "title": "Snake",
            "types": {"hyena": ["animal"], "wiki": ["dog"]},
        }
        train_entity_profile.add_entity(new_entity1)
        train_entity_profile.add_entity(new_entity2)
        train_entity_profile.reidentify_entity("Q123", "Q321")
        # Save new profile
        train_entity_profile.save(self.new_db)
        # Create old title embs
        title_embs = np.zeros(
            (
                6,
                768,
            )
        )
        for i in range(len(title_embs)):
            title_embs[i] = np.ones(768) * i

        train_entity_profile = EntityProfile.load_from_cache(
            load_dir=self.train_db,
        )
        new_entity_profile = EntityProfile.load_from_cache(load_dir=self.new_db)
        neweid2oldeid = {1: 1, 2: 2, 3: 3, 4: 4}
        np_same_ents = {"Q321", "Q345", "Q567", "Q789"}
        np_new_ents = {"Q910", "Q101"}

        new_title_embs = refit_titles(
            np_same_ents,
            np_new_ents,
            neweid2oldeid,
            train_entity_profile,
            new_entity_profile,
            title_embs,
            str(self.dir / "temp_title.npy"),
            "bert-base-uncased",
            str(self.dir / "temp_bert_models"),
            True,
        )
        # Compute gold BERT titles for new entities
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
            cache_dir=str(self.dir / "temp_bert_models"),
        )
        model = BertModel.from_pretrained(
            "bert-base-uncased",
            cache_dir=str(self.dir / "temp_bert_models"),
            output_attentions=False,
            output_hidden_states=False,
        )
        model.eval()
        input_ids = tokenizer(
            ["Cobra", "Snake"], padding=True, truncation=True, return_tensors="pt"
        )
        inputs = input_ids["input_ids"]
        attention_mask = input_ids["attention_mask"]
        outputs = model(inputs, attention_mask=attention_mask)[0]
        outputs[inputs == 0] = 0
        avgtitle = average_titles(inputs, outputs).to("cpu").detach().numpy()

        gold_title_embs = np.zeros(
            (
                8,
                768,
            )
        )
        gold_title_embs[0] = np.ones(768) * 0
        gold_title_embs[1] = np.ones(768) * 1
        gold_title_embs[2] = np.ones(768) * 2
        gold_title_embs[3] = np.ones(768) * 3
        gold_title_embs[4] = np.ones(768) * 4
        gold_title_embs[5] = avgtitle[0]
        gold_title_embs[6] = avgtitle[1]
        gold_title_embs[7] = np.ones(768) * 0  # Last row is set to zero in fit method
        np.testing.assert_array_almost_equal(new_title_embs, gold_title_embs)


if __name__ == "__main__":
    unittest.main()
