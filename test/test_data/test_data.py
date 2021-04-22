import os
import shutil
import unittest

import numpy as np
import torch
from transformers import BertTokenizer

from bootleg.datasets.dataset import BootlegDataset
from bootleg.symbols.constants import PAD_ID
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils
from bootleg.utils.sentence_utils import pad_sentence


def adjust_sentence(sentence, max_len, is_bert, tokenizer, offset=0):
    encoded = tokenizer.encode(sentence)
    if len(encoded) > max_len:
        if is_bert:
            # Want to keep max_len tokens, ensuring 102 is last
            encoded = [101] + encoded[offset + 1 : max_len - 2 + offset + 1] + [102]
        else:
            encoded = encoded[offset : max_len + offset]
    if is_bert:
        # BERT pad is 0
        return pad_sentence(encoded, 0, max_len)
    else:
        return pad_sentence(encoded, PAD_ID, max_len)


def assert_data_dicts_equal(dict_l, dict_r):
    for k in dict_l:
        assert k in dict_r, f"Key is {k}"
        if type(dict_l[k]) is torch.Tensor:
            assert torch.equal(dict_l[k].float(), dict_r[k].float()), f"Key is {k}"
        elif type(dict_l[k]) is np.ndarray:
            np.testing.assert_array_equal(dict_l[k], dict_r[k])
        elif type(dict_l[k]) is list:
            assert len(dict_l[k]) == len(dict_r[k]), f"Key is {k}"
            for item_l, item_r in zip(dict_l[k], dict_r[k]):
                if (
                    type(item_l) is np.ndarray
                ):  # special case with lists of UIDs being loaded from mmap file being slightly different
                    for subitem_l, subitem_r in zip(
                        item_l.tolist()[0], item_r.tolist()
                    ):
                        if type(subitem_l) is np.ndarray:
                            assert all(subitem_l == subitem_r), f"Key is {k}"
                        else:
                            assert subitem_l == subitem_r, f"Key is {k}"
                else:
                    assert item_l == item_r, f"Key is {k}"
        else:
            assert dict_l[k] == dict_r[k], f"Key is {k}"
    for k in dict_r:
        assert k in dict_l, f"Key is {k}"


class DataLoader(unittest.TestCase):
    def setUp(self):
        # tests that the sampling is done correctly on indices
        # load data from directory
        self.args = parser_utils.parse_boot_and_emm_args("test/run_args/test_data.json")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased",
            do_lower_case=False,
            cache_dir="test/data/emb_data/pretrained_bert_models",
        )
        self.is_bert = True
        self.entity_symbols = EntitySymbols.load_from_cache(
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
                ("alias_orig_list_pos", "i8", (max_aliases,)),
            ]
        )

    def tearDown(self) -> None:
        dir = os.path.join(
            self.args.data_config.data_dir, self.args.data_config.data_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        dir = os.path.join(
            self.args.data_config.entity_dir, self.args.data_config.entity_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        if os.path.exists(self.temp_file_name):
            os.remove(self.temp_file_name)

    def test_simple_dataset(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
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
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id = np.array([(0, 0, [0, 1, -1, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id],
                "sent_idx": torch.tensor([0]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, 4, -1, -1]]),
                "end_span_idx": torch.tensor(
                    [[2, 5, -1, -1]]
                ),  # the end span gets -1 to be inclusive
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias1"),
                            self.entity_symbols.get_alias_idx("multi word alias2"),
                            -1,
                            -1,
                        ]
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias1 or multi word alias2",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1, -1, -1]]),
                "gold_eid": torch.tensor([[1, 4, -1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 2, -1, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, 2, -1, -1]]),
            },
        )
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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_in_candidate_flag(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_seq_len = 10
        max_aliases = 4
        split = "train"
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        # Test 1: the code fails because it's training and Q3 is not a candidate of multi word alias2
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q3"],
                "sent_idx_unq": 0,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        with self.assertRaises(Exception) as context:
            dataset = BootlegDataset(
                self.args.data_config,
                name="Bootleg_test",
                dataset=self.temp_file_name,
                use_weak_label=use_weak_label,
                tokenizer=self.tokenizer,
                entity_symbols=self.entity_symbols,
                dataset_threads=1,
                split=split,
                is_bert=True,
            )
            self.assertTrue(type(context.exception) == AssertionError)

        # Test 2: the code passes because it's split is dev and Q3 is not a candidate of multi word alias2
        split = "dev"
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q3"],
                "sent_idx_unq": 0,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id = np.array([(0, 0, [0, 1, -1, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id],
                "sent_idx": torch.tensor([0]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, 4, -1, -1]]),
                "end_span_idx": torch.tensor(
                    [[2, 7, -1, -1]]
                ),  # the end span gets -1 to be inclusive
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias1"),
                            self.entity_symbols.get_alias_idx("multi word alias2"),
                            -1,
                            -1,
                        ]
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias1 or multi word alias2",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1, -1, -1]]),
                "gold_eid": torch.tensor([[1, 3, -1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, -2, -1, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, -2, -1, -1]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 3: the code passes because it's training but train in candidates is False
        # and Q3 is not a candidate of multi word alias2
        split = "train"
        self.args.data_config.train_in_candidates = False
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q3"],
                "sent_idx_unq": 0,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id = np.array([(0, 0, [0, 1, -1, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id],
                "sent_idx": torch.tensor([0]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, 4, -1, -1]]),
                "end_span_idx": torch.tensor(
                    [[2, 7, -1, -1]]
                ),  # the end span gets -1 to be inclusive
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias1"),
                            self.entity_symbols.get_alias_idx("multi word alias2"),
                            -1,
                            -1,
                        ]
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias1 or multi word alias2",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1, -1, -1]]),
                "gold_eid": torch.tensor([[1, 3, -1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[1, 0, -1, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[1, 0, -1, -1]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_single_mention_dataset(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
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
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id_arr = [
            np.array([(0, 0, [0])], dtype=self.guid_dtype(max_aliases)),
            np.array([(0, 1, [1])], dtype=self.guid_dtype(max_aliases)),
        ]
        X_dict, Y_dict = (
            {
                "guids": uniq_id_arr,
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1], [4]]),
                "end_span_idx": torch.tensor(
                    [[2], [5]]
                ),  # the end span gets -1 to be inclusive
                "alias_idx": torch.tensor(
                    [
                        [self.entity_symbols.get_alias_idx("alias1")],
                        [self.entity_symbols.get_alias_idx("multi word alias2")],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias1 or multi word alias2",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "alias1 or multi word alias2",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0], [1]]),
                "gold_eid": torch.tensor([[1], [4]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0], [2]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0], [2]]),
            },
        )
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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_nonmatch_alias(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_seq_len = 7
        max_aliases = 4
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias0", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias0 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id = np.array([(0, 0, [0, 1, -1, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id],
                "sent_idx": torch.tensor([0]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, 4, -1, -1]]),
                "end_span_idx": torch.tensor(
                    [[2, 5, -1, -1]]
                ),  # the end span gets -1 to be inclusive
                "alias_idx": torch.tensor(
                    [
                        [
                            -2,
                            self.entity_symbols.get_alias_idx("multi word alias2"),
                            -1,
                            -1,
                        ]
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias0 or multi word alias2",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1, -1, -1]]),
                "gold_eid": torch.tensor([[1, 4, -1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[-2, 2, -1, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[-2, 2, -1, -1]]),
            },
        )
        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True

        with self.assertRaises(Exception) as context:
            dataset = BootlegDataset(
                self.args.data_config,
                name="Bootleg_test",
                dataset=self.temp_file_name,
                use_weak_label=use_weak_label,
                tokenizer=self.tokenizer,
                entity_symbols=self.entity_symbols,
                dataset_threads=1,
                split="train",
                is_bert=True,
            )
            self.assertTrue(type(context.exception) == AssertionError)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="test",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Assert the -2 stays even though train_in_cands is False
        self.args.data_config.train_in_candidates = False
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="test",
            is_bert=True,
        )
        X_dict["for_dump_gold_cand_K_idx_train"] = torch.tensor([[-2, 3, -1, -1]])
        Y_dict["gold_cand_K_idx"] = torch.tensor([[-2, 3, -1, -1]])
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_long_sentences(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: the sentence is long and has far apart aliases so it gets split up into two subsentences
        max_seq_len = 7
        max_aliases = 4
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias3", "alias4"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias3 cat cat cat cat cat cat alias4",
                "spans": [[0, 1], [7, 8]],
                "gold": [True, True],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array(
            [(0, 0, [0, -1, -1, -1])], dtype=self.guid_dtype(max_aliases)
        )
        uniq_id2 = np.array(
            [(0, 1, [1, -1, -1, -1])], dtype=self.guid_dtype(max_aliases)
        )
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1, uniq_id2],
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1, -1, -1, -1], [4, -1, -1, -1]]),
                "end_span_idx": torch.tensor([[2, -1, -1, -1], [5, -1, -1, -1]]),
                "alias_idx": torch.tensor(
                    [
                        [self.entity_symbols.get_alias_idx("alias3"), -1, -1, -1],
                        [self.entity_symbols.get_alias_idx("alias4"), -1, -1, -1],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 cat cat cat",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "cat cat cat alias4",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, -1, -1, -1], [1, -1, -1, -1]]),
                "gold_eid": torch.tensor([[1, -1, -1, -1], [4, -1, -1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor(
                    [[0, -1, -1, -1], [0, -1, -1, -1]]
                ),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, -1, -1, -1], [0, -1, -1, -1]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 1: the sentence is long but there is only one alias, so the sentence gets windowed
        max_seq_len = 7
        max_aliases = 4
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias3", "alias4"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias3 cat alias4 cat cat cat cat",
                "spans": [[0, 1], [2, 3]],
                "gold": [True, True],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id = np.array([(0, 0, [0, 1, -1, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id],
                "sent_idx": torch.tensor([0]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, 4, -1, -1]]),
                "end_span_idx": torch.tensor([[2, 5, -1, -1]]),
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias3"),
                            self.entity_symbols.get_alias_idx("alias4"),
                            -1,
                            -1,
                        ]
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 cat alias4 cat cat",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1, -1, -1]]),
                "gold_eid": torch.tensor([[1, 4, -1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 0, -1, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, 0, -1, -1]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_long_aliases(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: there are more than max_aliases aliases in the sentence and it should be split into subparts
        # the second subpart will have a repeat alias that should _not_ be
        # backpropped (as it already is in the first subpart)
        # Therefore, the true entity idx is -1
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
                "gold": [True, True, True],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, 1])], dtype=self.guid_dtype(max_aliases))
        uniq_id2 = np.array([(0, 1, [1, 2])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1, uniq_id2],
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1, 3], [2, 4]]),
                "end_span_idx": torch.tensor([[2, 4], [3, 5]]),
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias3"),
                            self.entity_symbols.get_alias_idx("alias4"),
                        ],
                        [
                            self.entity_symbols.get_alias_idx("alias4"),
                            self.entity_symbols.get_alias_idx("alias3"),
                        ],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                            offset=1,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1], [1, 2]]),
                "gold_eid": torch.tensor([[1, 4], [-1, 1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 0], [-1, 0]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, 0], [-1, 0]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_non_gold_aliases(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: the gold of False should be untouched for train, with only one True gold
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
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, 1])], dtype=self.guid_dtype(max_aliases))
        uniq_id2 = np.array([(0, 1, [1, 2])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1, uniq_id2],
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1, 3], [2, 4]]),
                "end_span_idx": torch.tensor([[2, 4], [3, 5]]),
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias3"),
                            self.entity_symbols.get_alias_idx("alias4"),
                        ],
                        [
                            self.entity_symbols.get_alias_idx("alias4"),
                            self.entity_symbols.get_alias_idx("alias3"),
                        ],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                            offset=1,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1], [1, 2]]),
                "gold_eid": torch.tensor([[1, 4], [-1, 1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 0], [-1, 0]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, 0], [-1, 0]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 2: the gold of False should be untouched for train, with all False golds
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
                "gold": [False, False, False],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, 1])], dtype=self.guid_dtype(max_aliases))
        uniq_id2 = np.array([(0, 1, [1, 2])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1, uniq_id2],
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1, 3], [2, 4]]),
                "end_span_idx": torch.tensor([[2, 4], [3, 5]]),
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias3"),
                            self.entity_symbols.get_alias_idx("alias4"),
                        ],
                        [
                            self.entity_symbols.get_alias_idx("alias4"),
                            self.entity_symbols.get_alias_idx("alias3"),
                        ],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                            offset=1,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1], [1, 2]]),
                "gold_eid": torch.tensor([[1, 4], [-1, 1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 0], [-1, 0]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, 0], [-1, 0]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 3: with the split of "dev", the subsentences should remain unchanged
        # but the true index in Y_dict should be -1
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
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, 1])], dtype=self.guid_dtype(max_aliases))
        uniq_id2 = np.array([(0, 1, [1, 2])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1, uniq_id2],
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1, 3], [2, 4]]),
                "end_span_idx": torch.tensor([[2, 4], [3, 5]]),
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias3"),
                            self.entity_symbols.get_alias_idx("alias4"),
                        ],
                        [
                            self.entity_symbols.get_alias_idx("alias4"),
                            self.entity_symbols.get_alias_idx("alias3"),
                        ],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                            offset=1,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1], [1, 2]]),
                "gold_eid": torch.tensor(
                    [
                        [
                            1,
                            -1,
                        ],
                        [-1, -1],
                    ]
                ),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 0], [-1, 0]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, -1], [-1, -1]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 4: with the split of dev, all true indices should be -1 but the sentences should still be used
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
                "gold": [False, False, False],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, 1])], dtype=self.guid_dtype(max_aliases))
        uniq_id2 = np.array([(0, 1, [1, 2])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1, uniq_id2],
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1, 3], [2, 4]]),
                "end_span_idx": torch.tensor([[2, 4], [3, 5]]),
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias3"),
                            self.entity_symbols.get_alias_idx("alias4"),
                        ],
                        [
                            self.entity_symbols.get_alias_idx("alias4"),
                            self.entity_symbols.get_alias_idx("alias3"),
                        ],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                            offset=1,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1], [1, 2]]),
                "gold_eid": torch.tensor([[-1, -1], [-1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 0], [-1, 0]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[-1, -1], [-1, -1]]),
            },
        )

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_non_gold_no_weak_label_aliases(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 0: with all TRUE golds, use weak label of FALSE doesn't change anything
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
                "gold": [True, True, True],
            }
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, 1])], dtype=self.guid_dtype(max_aliases))
        uniq_id2 = np.array([(0, 1, [1, 2])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1, uniq_id2],
                "sent_idx": torch.tensor([0, 0]),
                "subsent_idx": torch.tensor([0, 1]),
                "start_span_idx": torch.tensor([[1, 3], [2, 4]]),
                "end_span_idx": torch.tensor([[2, 4], [3, 5]]),
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias3"),
                            self.entity_symbols.get_alias_idx("alias4"),
                        ],
                        [
                            self.entity_symbols.get_alias_idx("alias4"),
                            self.entity_symbols.get_alias_idx("alias3"),
                        ],
                    ]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        ),
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                            offset=1,
                        ),
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1], [1, 2]]),
                "gold_eid": torch.tensor([[1, 4], [-1, 1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 0], [-1, 0]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, 0], [-1, 0]]),
            },
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 1: now that weak label is set to False, the golds of False should be removed for split of "train"
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
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1],
                "sent_idx": torch.tensor([0]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, -1]]),
                "end_span_idx": torch.tensor([[2, -1]]),
                "alias_idx": torch.tensor(
                    [[self.entity_symbols.get_alias_idx("alias3"), -1]]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, -1]]),
                "gold_eid": torch.tensor([[1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, -1]]),
            },
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 2: now that weak label is set to False, the sentence with all golds of False
        # should be removed for "train".
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
                "gold": [False, False, False],
            },
            {
                "aliases": ["alias3"],
                "qids": ["Q1"],
                "sent_idx_unq": 1,
                "sentence": "alias3",
                "spans": [[0, 1]],
                "gold": [True],
            },
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(1, 0, [0, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1],
                "sent_idx": torch.tensor([1]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, -1]]),
                "end_span_idx": torch.tensor([[2, -1]]),
                "alias_idx": torch.tensor(
                    [[self.entity_symbols.get_alias_idx("alias3"), -1]]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3", max_seq_len, self.is_bert, self.tokenizer
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, -1]]),
                "gold_eid": torch.tensor([[1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, -1]]),
            },
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 3: with the split of "dev", nothing should change from test 1 above where we were using "train"
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
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(0, 0, [0, -1])], dtype=self.guid_dtype(max_aliases))
        # uniq_id2 = np.array([(0, 1, [1, 2])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1],
                "sent_idx": torch.tensor([0]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, -1]]),
                "end_span_idx": torch.tensor([[2, -1]]),
                "alias_idx": torch.tensor(
                    [[self.entity_symbols.get_alias_idx("alias3"), -1]]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3 alias4 alias3",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, -1]]),
                "gold_eid": torch.tensor([[1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, -1]]),
            },
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 4: with the split of dev, all true indices should be -1 but the sentences should still be used
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
                "gold": [False, False, False],
            },
            {
                "aliases": ["alias3"],
                "qids": ["Q1"],
                "sent_idx_unq": 1,
                "sentence": "alias3",
                "spans": [[0, 1]],
                "gold": [True],
            },
        ]
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id1 = np.array([(1, 0, [0, -1])], dtype=self.guid_dtype(max_aliases))
        X_dict, Y_dict = (
            {
                "guids": [uniq_id1],
                "sent_idx": torch.tensor([1]),
                "subsent_idx": torch.tensor([0]),
                "start_span_idx": torch.tensor([[1, -1]]),
                "end_span_idx": torch.tensor([[2, -1]]),
                "alias_idx": torch.tensor(
                    [[self.entity_symbols.get_alias_idx("alias3"), -1]]
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias3", max_seq_len, self.is_bert, self.tokenizer
                        )
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, -1]]),
                "gold_eid": torch.tensor([[1, -1]]),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, -1]]),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, -1]]),
            },
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False

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
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_multiple_sentences(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: the gold of False should be untouched for train
        max_seq_len = 7
        max_aliases = 4
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": i,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
            for i in range(53)
        ]
        assert len(input_data) == 53
        # UNIQ_ID is sent_id, subsent_idx, and alias_orig_list_pos
        uniq_id_get = lambda s: np.array(
            [(s, 0, [0, 1, -1, -1])], dtype=self.guid_dtype(max_aliases)
        )
        X_dict, Y_dict = (
            {
                "guids": [uniq_id_get(i) for i in range(53)],
                "sent_idx": torch.arange(53),
                "subsent_idx": torch.tensor([0] * 53),
                "start_span_idx": torch.tensor([[1, 4, -1, -1]] * 53),
                "end_span_idx": torch.tensor(
                    [[2, 5, -1, -1]] * 53
                ),  # the end span gets -1 to be inclusive
                "alias_idx": torch.tensor(
                    [
                        [
                            self.entity_symbols.get_alias_idx("alias1"),
                            self.entity_symbols.get_alias_idx("multi word alias2"),
                            -1,
                            -1,
                        ]
                    ]
                    * 53
                ),
                "token_ids": torch.tensor(
                    [
                        adjust_sentence(
                            "alias1 or multi word alias2",
                            max_seq_len,
                            self.is_bert,
                            self.tokenizer,
                        )
                        for _ in range(53)
                    ]
                ),
                "alias_orig_list_pos": torch.tensor([[0, 1, -1, -1]] * 53),
                "gold_eid": torch.tensor([[1, 4, -1, -1]] * 53),
                "for_dump_gold_cand_K_idx_train": torch.tensor([[0, 2, -1, -1]] * 53),
            },
            {
                "gold_cand_K_idx": torch.tensor([[0, 2, -1, -1]] * 53),
            },
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=2,
            split="train",
            is_bert=True,
        )
        # Using multiple threads will make data in random sorted chunk order
        sort_idx = np.argsort(dataset.X_dict["sent_idx"])
        for key in list(dataset.X_dict.keys()):
            dataset.X_dict[key] = dataset.X_dict[key][sort_idx]
        for key in list(dataset.Y_dict.keys()):
            dataset.Y_dict[key] = dataset.Y_dict[key][sort_idx]
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)


if __name__ == "__main__":
    unittest.main()
