"""Test entity dataset."""
import os
import shutil
import unittest

import ujson
from transformers import AutoTokenizer

from bootleg.dataset import BootlegDataset
from bootleg.symbols.constants import SPECIAL_TOKENS
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
from bootleg.utils.data_utils import read_in_types
from bootleg.utils.parser import parser_utils


class DataEntityLoader(unittest.TestCase):
    """Entity data loader."""

    def setUp(self):
        """Set up."""
        # tests that the sampling is done correctly on indices
        # load data from directory
        self.args = parser_utils.parse_boot_and_emm_args(
            "tests/run_args/test_entity_data.json"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-cased",
            do_lower_case=False,
            use_fast=True,
            cache_dir="tests/data/emb_data/pretrained_bert_models",
        )
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.is_bert = True
        self.entity_symbols = EntitySymbols.load_from_cache(
            os.path.join(
                self.args.data_config.entity_dir, self.args.data_config.entity_map_dir
            ),
            alias_cand_map_file=self.args.data_config.alias_cand_map,
            alias_idx_file=self.args.data_config.alias_idx_map,
        )
        self.entity_temp_dir = "tests/data/entity_loader/entity_data_test"
        self.temp_file_name = "tests/data/data_loader/test_data.jsonl"

    def tearDown(self) -> None:
        """Tear down."""
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
        if os.path.exists(self.entity_temp_dir):
            shutil.rmtree(self.entity_temp_dir)

    def test_load_type_data(self):
        """
        Test load type data.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        ENTITY TITLE
        {
          "Q1":"alias1",
          "Q2":"multi alias2",
          "Q3":"word alias3",
          "Q4":"nonalias4"
        }
        TYPE LABELS
        {
          "Q1": [1, 2],
          "Q2": [3],
          "Q3": [],
          "Q4": [2]
        }
        TYPE VOCAB
        {
          "T1": 1,
          "T2": 2,
          "T3": 3
        }
        """
        qid2typename_gold = {"Q1": ["T1", "T2"], "Q2": ["T3"], "Q3": [], "Q4": ["T2"]}
        qid2typename = read_in_types(self.args.data_config, self.entity_symbols)
        self.assertDictEqual(qid2typename, qid2typename_gold)

    def test_load_type_data_extra_entity(self):
        """
        Test load type data extra entity.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        ENTITY TITLE
        {
          "Q1":"alias1",
          "Q2":"multi alias2",
          "Q3":"word alias3",
          "Q4":"nonalias4"
        }
        TYPE LABELS
        {
          "Q1": [1, 2],
          "Q2": [3],
          "Q3": [],
          "Q4": [2],
          "Q5": [3]
        }
        """
        temp_type_data = {"Q1": [1, 2], "Q2": [3], "Q3": [], "Q4": [2], "Q5": [2]}
        file = "tests/data/emb_data/temp_type_mapping.json"
        with open(file, "w") as out_f:
            ujson.dump(temp_type_data, out_f)

        self.args.data_config.entity_type_data.type_file = "temp_type_mapping.json"

        qid2typename_gold = {"Q1": ["T1", "T2"], "Q2": ["T3"], "Q3": [], "Q4": ["T2"]}
        qid2typename = read_in_types(self.args.data_config, self.entity_symbols)

        self.assertDictEqual(qid2typename, qid2typename_gold)
        if os.path.exists(file):
            os.remove(file)

    def test_simple_entity_data(self):
        """
        Test simple entity data.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        ENTITY TITLE
        {
          "Q1":"alias1",
          "Q2":"multi alias2",
          "Q3":"word alias3",
          "Q4":"nonalias4"
        }
        TYPE LABELS
        {
          "Q1": [1, 2],
          "Q2": [3],
          "Q3": [],
          "Q4": [2]
        }
        """
        max_seq_len = 7
        max_ent_len = 10
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_ent_len = max_ent_len
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

        # THERE ARE NO DESCRIPTIONS BUT THE SEP TOKEN IS STILL ADDED WITH EMPTY DESC
        X_entity_dict = self.tokenizer(
            [
                "[SEP]",
                "alias1 [ent_type] T1 [ent_type] T2",
                "multi alias2 [ent_type] T3",
                "word alias3 [ent_type]",
                "nonalias4 [ent_type] T2",
                "[SEP]",
            ],
            max_length=max_ent_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        gold_entity_to_mask = [
            [0 for _ in range(len(inp))] for inp in X_entity_dict["input_ids"]
        ]
        gold_entity_to_mask[1][1:3] = [1, 1]
        gold_entity_to_mask[2][1:4] = [1, 1, 1]
        gold_entity_to_mask[3][1:4] = [1, 1, 1]
        gold_entity_to_mask[4][1:5] = [1, 1, 1, 1]
        utils.write_jsonl(self.temp_file_name, input_data)
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=True,
            load_entity_data=True,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        self.assertListEqual(
            X_entity_dict["input_ids"],
            dataset.X_entity_dict["entity_input_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["token_type_ids"],
            dataset.X_entity_dict["entity_token_type_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["attention_mask"],
            dataset.X_entity_dict["entity_attention_mask"].tolist(),
        )
        self.assertListEqual(
            gold_entity_to_mask,
            dataset.X_entity_dict["entity_to_mask"].tolist(),
        )

    def test_max_ent_type_len(self):
        """
        Test max entity type length.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        ENTITY TITLE
        {
          "Q1":"alias1",
          "Q2":"multi alias2",
          "Q3":"word alias3",
          "Q4":"nonalias4"
        }
        TYPE LABELS
        {
          "Q1": [1, 2],
          "Q2": [3],
          "Q3": [],
          "Q4": [2]
        }
        """
        max_seq_len = 7
        max_ent_len = 10
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_ent_len = max_ent_len
        self.args.data_config.entity_type_data.max_ent_type_len = 1
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

        # THERE ARE NO DESCRIPTIONS
        X_entity_dict = self.tokenizer(
            [
                "[SEP]",
                "alias1 [ent_type] T1",
                "multi alias2 [ent_type] T3",
                "word alias3 [ent_type]",
                "nonalias4 [ent_type] T2",
                "[SEP]",
            ],
            max_length=max_ent_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        gold_entity_to_mask = [
            [0 for _ in range(len(inp))] for inp in X_entity_dict["input_ids"]
        ]
        gold_entity_to_mask[1][1:3] = [1, 1]
        gold_entity_to_mask[2][1:4] = [1, 1, 1]
        gold_entity_to_mask[3][1:4] = [1, 1, 1]
        gold_entity_to_mask[4][1:5] = [1, 1, 1, 1]
        utils.write_jsonl(self.temp_file_name, input_data)
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=True,
            load_entity_data=True,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )

        self.assertListEqual(
            X_entity_dict["input_ids"],
            dataset.X_entity_dict["entity_input_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["token_type_ids"],
            dataset.X_entity_dict["entity_token_type_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["attention_mask"],
            dataset.X_entity_dict["entity_attention_mask"].tolist(),
        )
        self.assertListEqual(
            gold_entity_to_mask,
            dataset.X_entity_dict["entity_to_mask"].tolist(),
        )

    def test_desc_entity_data(self):
        """
        Test entity description data.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        ENTITY TITLE
        {
          "Q1":"alias1",
          "Q2":"multi alias2",
          "Q3":"word alias3",
          "Q4":"nonalias4"
        }
        TYPE LABELS
        {
          "Q1": [1, 2],
          "Q2": [3],
          "Q3": [],
          "Q4": [2]
        }
        """
        self.args.data_config.use_entity_desc = True
        # For this test, we make a new entity_mappings directory to prevent multiprocessing tests from pytest to
        # also read in the qid2desc file
        shutil.copytree("tests/data/entity_loader/entity_data", self.entity_temp_dir)
        self.args.data_config.entity_dir = self.entity_temp_dir
        qid2desc = {
            "Q1": "testing desc",
            "Q3": "words",
        }
        out_file = (
            "tests/data/entity_loader/entity_data_test/entity_mappings/qid2desc.json"
        )
        with open(out_file, "w") as out_f:
            ujson.dump(qid2desc, out_f)

        entity_symbols = EntitySymbols.load_from_cache(
            os.path.join(
                self.args.data_config.entity_dir, self.args.data_config.entity_map_dir
            ),
            alias_cand_map_file=self.args.data_config.alias_cand_map,
            alias_idx_file=self.args.data_config.alias_idx_map,
        )

        max_seq_len = 7
        max_ent_len = 7
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_ent_len = max_ent_len
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

        # THERE ARE DESCRIPTIONS
        X_entity_dict = self.tokenizer(
            [
                "[SEP]",
                "alias1 [ent_type] T1 [ent_type] T2 [ent_desc] testing desc",
                "multi alias2 [ent_type] T3 [ent_desc]",
                "word alias3 [ent_type] [ent_desc] words",
                "nonalias4 [ent_type] T2 [ent_desc]",
                "[SEP]",
            ],
            max_length=max_ent_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        gold_entity_to_mask = [
            [0 for _ in range(len(inp))] for inp in X_entity_dict["input_ids"]
        ]
        gold_entity_to_mask[1][1:3] = [1, 1]
        gold_entity_to_mask[2][1:4] = [1, 1, 1]
        gold_entity_to_mask[3][1:4] = [1, 1, 1]
        gold_entity_to_mask[4][1:5] = [1, 1, 1, 1]
        utils.write_jsonl(self.temp_file_name, input_data)
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=True,
            load_entity_data=True,
            tokenizer=self.tokenizer,
            entity_symbols=entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        self.assertListEqual(
            X_entity_dict["input_ids"],
            dataset.X_entity_dict["entity_input_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["token_type_ids"],
            dataset.X_entity_dict["entity_token_type_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["attention_mask"],
            dataset.X_entity_dict["entity_attention_mask"].tolist(),
        )
        self.assertListEqual(
            gold_entity_to_mask,
            dataset.X_entity_dict["entity_to_mask"].tolist(),
        )

    def test_entity_kg_data(self):
        """
        Test entity KG data.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        ENTITY TITLE
        {
          "Q1":"alias1",
          "Q2":"multi alias2",
          "Q3":"word alias3",
          "Q4":"nonalias4"
        }
        TYPE LABELS
        {
          "Q1": [1, 2],
          "Q2": [3],
          "Q3": [],
          "Q4": [2]
        }
        KG LABELS
        {
          "Q1": {"rel1": ["Q2"]},
          "Q3": {"rel2": ["Q2"]}
        }
        """
        max_seq_len = 7
        max_ent_len = 10
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_ent_len = max_ent_len
        self.args.data_config.entity_kg_data.use_entity_kg = True
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

        # THERE ARE NO DESCRIPTIONS BUT THE SEP TOKEN IS STILL ADDED WITH EMPTY DESC
        X_entity_dict = self.tokenizer(
            [
                "[SEP]",
                "alias1 [ent_type] T1 [ent_type] T2 [ent_kg] rel1 multi alias2",
                "multi alias2 [ent_type] T3 [ent_kg]",
                "word alias3 [ent_type] [ent_kg] rel2 multi alias2",
                "nonalias4 [ent_type] T2 [ent_kg]",
                "[SEP]",
            ],
            max_length=max_ent_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        gold_entity_to_mask = [
            [0 for _ in range(len(inp))] for inp in X_entity_dict["input_ids"]
        ]
        gold_entity_to_mask[1][1:3] = [1, 1]
        gold_entity_to_mask[2][1:4] = [1, 1, 1]
        gold_entity_to_mask[3][1:4] = [1, 1, 1]
        gold_entity_to_mask[4][1:5] = [1, 1, 1, 1]
        utils.write_jsonl(self.temp_file_name, input_data)
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=True,
            load_entity_data=True,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        self.assertListEqual(
            X_entity_dict["input_ids"],
            dataset.X_entity_dict["entity_input_ids"].tolist(),
        )

        self.assertListEqual(
            X_entity_dict["token_type_ids"],
            dataset.X_entity_dict["entity_token_type_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["attention_mask"],
            dataset.X_entity_dict["entity_attention_mask"].tolist(),
        )
        self.assertListEqual(
            gold_entity_to_mask,
            dataset.X_entity_dict["entity_to_mask"].tolist(),
        )

    def test_multiprocess_entity_data(self):
        """
        Test multiprocessing entity data.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        ENTITY TITLE
        {
          "Q1":"alias1",
          "Q2":"multi alias2",
          "Q3":"word alias3",
          "Q4":"nonalias4"
        }
        TYPE LABELS
        {
          "Q1": [1, 2],
          "Q2": [3],
          "Q3": [],
          "Q4": [2]
        }
        """
        max_seq_len = 7
        max_ent_len = 10
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_ent_len = max_ent_len
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

        # THERE ARE NO DESCRIPTIONS
        X_entity_dict = self.tokenizer(
            [
                "[SEP]",
                "alias1 [ent_type] T1 [ent_type] T2",
                "multi alias2 [ent_type] T3",
                "word alias3 [ent_type]",
                "nonalias4 [ent_type] T2",
                "[SEP]",
            ],
            max_length=max_ent_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )
        gold_entity_to_mask = [
            [0 for _ in range(len(inp))] for inp in X_entity_dict["input_ids"]
        ]
        gold_entity_to_mask[1][1:3] = [1, 1]
        gold_entity_to_mask[2][1:4] = [1, 1, 1]
        gold_entity_to_mask[3][1:4] = [1, 1, 1]
        gold_entity_to_mask[4][1:5] = [1, 1, 1, 1]
        utils.write_jsonl(self.temp_file_name, input_data)
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=True,
            load_entity_data=True,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=3,
            split="train",
            is_bert=True,
        )
        self.assertListEqual(
            X_entity_dict["input_ids"],
            dataset.X_entity_dict["entity_input_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["token_type_ids"],
            dataset.X_entity_dict["entity_token_type_ids"].tolist(),
        )
        self.assertListEqual(
            X_entity_dict["attention_mask"],
            dataset.X_entity_dict["entity_attention_mask"].tolist(),
        )
        self.assertListEqual(
            gold_entity_to_mask,
            dataset.X_entity_dict["entity_to_mask"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
