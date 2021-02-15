import os
import shutil
import unittest

import torch

from bootleg.layers.alias_to_ent_encoder import AliasEntityTable
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
from bootleg.utils.classes.dotted_dict import DottedDict

# subclass to only instantiate what we need for testing


class EntitySymbolTest(unittest.TestCase):
    def setUp(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        self.entity_symbols = EntitySymbols(load_dir=entity_dump_dir)

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

        truealiastrie = {"multi word alias2": 0, "alias1": 1, "alias3": 2, "alias4": 3}
        entity_symbols = EntitySymbols(
            load_dir=None,
            max_candidates=3,
            max_alias_len=3,
            alias2qids=truealias2qids,
            qid2title=trueqid2title,
        )
        tri_as_dict = {}
        for k in entity_symbols._alias_trie:
            tri_as_dict[k] = entity_symbols._alias_trie[k]

        self.assertEqual(entity_symbols.max_candidates, 3)
        self.assertEqual(entity_symbols.max_alias_len, 3)
        self.assertDictEqual(entity_symbols._alias2qids, truealias2qids)
        self.assertDictEqual(entity_symbols._qid2title, trueqid2title)
        self.assertDictEqual(entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(tri_as_dict, truealiastrie)

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

        tri_as_dict = {}
        for k in self.entity_symbols._alias_trie:
            tri_as_dict[k] = self.entity_symbols._alias_trie[k]

        self.assertEqual(self.entity_symbols.max_candidates, 3)
        self.assertEqual(self.entity_symbols.max_alias_len, 3)
        self.assertDictEqual(self.entity_symbols._alias2qids, truealias2qids)
        self.assertDictEqual(self.entity_symbols._qid2title, trueqid2title)
        self.assertDictEqual(self.entity_symbols._qid2eid, trueqid2eid)
        self.assertDictEqual(tri_as_dict, truealiastrie)

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
        self.assertEqual(self.entity_symbols.get_alias_idx("alias1"), 1)
        self.assertEqual(self.entity_symbols.get_alias_from_idx(2), "alias3")
        self.assertEqual(self.entity_symbols.alias_exists("alias3"), True)
        self.assertEqual(self.entity_symbols.alias_exists("alias5"), False)


class AliasTableTest(unittest.TestCase):
    def setUp(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        self.entity_symbols = EntitySymbols(
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
                [0, 2, 1, 4],
                [0, 1, 4, -1],
                [0, 1, -1, -1],
                [0, 4, 3, 2],
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
            [[2, 1, 4], [1, 4, -1], [1, -1, -1], [4, 3, 2], [-1, -1, -1]]
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
        expected_tensor = torch.tensor([[[[0, 2, 1, 4], [0, 1, 4, -1]]]])
        assert torch.equal(actual_indices.long(), expected_tensor.long())


if __name__ == "__main__":
    unittest.main()
