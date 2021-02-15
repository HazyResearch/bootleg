import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
import ujson
from transformers import BertTokenizer

from bootleg.datasets.dataset import BootlegDataset, read_in_types
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils


class DataTypeLoader(unittest.TestCase):
    def setUp(self):
        # tests that the sampling is done correctly on indices
        # load data from directory
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_type_data.json"
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased",
            do_lower_case=False,
            cache_dir="test/data/emb_data/pretrained_bert_models",
        )
        self.is_bert = True
        self.entity_symbols = EntitySymbols(
            os.path.join(
                self.args.data_config.entity_dir, self.args.data_config.entity_map_dir
            ),
            alias_cand_map_file=self.args.data_config.alias_cand_map,
        )
        self.temp_file_name = "test/data/data_loader/test_data.jsonl"
        self.guid_dtype = lambda max_aliases: np.dtype(
            [
                ("sent_idx", "i8", 1),
                ("subsent_idx", "i8", 1),
                ("alias_orig_list_pos", "i8", max_aliases),
            ]
        )

    def tearDown(self) -> None:
        dir = os.path.join(
            self.args.data_config.data_dir, self.args.data_config.data_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir)
        dir = os.path.join(
            self.args.data_config.entity_dir, self.args.data_config.entity_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir)
        if os.path.exists(self.temp_file_name):
            os.remove(self.temp_file_name)

    def test_load_type_data(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        TYPE LABELS
        {
          "Q1": [0, 1],
          "Q2": [2],
          "Q3": [],
          "Q4": [1]
        }
        """
        eid2type_gold = {"1": 0 + 1, "2": 2 + 1, "3": -1 + 1, "4": 1 + 1}
        eid2type = read_in_types(self.args.data_config)
        self.assertDictEqual(eid2type, eid2type_gold)

    def test_load_type_data_extra_entity(self):
        # Test that we only add entities in our dump
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        TYPE LABELS
        {
          "Q1": [0, 1],
          "Q2": [2],
          "Q3": [],
          "Q4": [1],
          "Q5": [1]
        }
        """
        temp_type_data = {"Q1": [0, 1], "Q2": [2], "Q3": [], "Q4": [1], "Q5": [1]}
        file = "test/data/emb_data/temp_type_mapping.json"
        with open(file, "w") as out_f:
            ujson.dump(temp_type_data, out_f)

        self.args.data_config.type_prediction.file = "temp_type_mapping.json"

        eid2type_gold = {"1": 0 + 1, "2": 2 + 1, "3": -1 + 1, "4": 1 + 1}
        eid2type = read_in_types(self.args.data_config)

        self.assertDictEqual(eid2type, eid2type_gold)
        if os.path.exists(file):
            os.remove(file)

    def test_simple_type_data(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        TYPE LABELS
        {
          "Q1": [0, 1],
          "Q2": [2],
          "Q3": [],
          "Q4": [1]
        }
        """
        max_seq_len = 7
        max_aliases = 4
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        Y_dict = {
            "gold_type_id": torch.tensor([[0 + 1, 1 + 1, -1, -1]]),
        }

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert torch.equal(Y_dict["gold_type_id"], dataset.Y_dict["gold_type_id"])

    def test_single_mention_dataset(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        TYPE LABELS
        {
          "Q1": [0, 1],
          "Q2": [2],
          "Q3": [],
          "Q4": [1]
        }
        """
        max_seq_len = 7
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        Y_dict = {
            "gold_type_id": torch.tensor([[0 + 1], [1 + 1]]),
        }

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert torch.equal(Y_dict["gold_type_id"], dataset.Y_dict["gold_type_id"])

    def test_subsent_data(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        TYPE LABELS
        {
          "Q1": [0, 1],
          "Q2": [2],
          "Q3": [],
          "Q4": [1]
        }
        """
        # Test 1: the sentence is long and has far apart aliases so it gets split up into two subsentences; the types should follow
        max_seq_len = 7
        max_aliases = 4
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias3", "alias4"],
                "qids": ["Q1", "Q2"],
                "sent_idx_unq": 0,
                "sentence": "alias3 cat cat cat cat cat cat alias4",
                "spans": [[0, 1], [7, 8]],
                "gold": [True, True],
            }
        ]
        Y_dict = {
            "gold_type_id": torch.tensor([[0 + 1, -1, -1, -1], [2 + 1, -1, -1, -1]]),
        }

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert torch.equal(Y_dict["gold_type_id"], dataset.Y_dict["gold_type_id"])

    def test_masked_aliases(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        TYPE LABELS
        {
          "Q1": [0, 1],
          "Q2": [2],
          "Q3": [],
          "Q4": [1]
        }
        """
        # Test 1: this sentence gets split into two with two aliases each (the first alias of second split masked out). The types should
        # follow this trend. Since split is train, they also ignore the gold.
        max_seq_len = 7
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "gold": [True, False, False],
            }
        ]
        Y_dict = {
            "gold_type_id": torch.tensor([[0 + 1, 1 + 1], [-1, 0 + 1]]),
        }
        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert torch.equal(Y_dict["gold_type_id"], dataset.Y_dict["gold_type_id"])

        # Test 2: with the split of "dev", the subsentences should remain unchanged but the true index in Y_dict should be -1
        max_seq_len = 7
        max_aliases = 2
        split = "dev"
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "gold": [True, False, False],
            }
        ]
        Y_dict = {
            "gold_type_id": torch.tensor([[0 + 1, -1], [-1, -1]]),
        }

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split=split,
            is_bert=True,
        )
        assert torch.equal(Y_dict["gold_type_id"], dataset.Y_dict["gold_type_id"])


if __name__ == "__main__":
    unittest.main()
