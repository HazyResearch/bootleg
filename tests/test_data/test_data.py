"""Test data."""
import os
import shutil
import unittest
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer

from bootleg.dataset import BootlegDataset, extract_context_windows
from bootleg.symbols.constants import SPECIAL_TOKENS
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils


def adjust_sentence(sentence, max_len, max_window_len, span, tokenizer):
    """Tokenize and adjust sentence for max length."""
    tokens = sentence.split()
    prev_context, next_context = extract_context_windows(span, tokens, max_window_len)
    context_tokens = (
        prev_context
        + ["[ent_start]"]
        + tokens[span[0] : span[1]]
        + ["[ent_end]"]
        + next_context
    )
    new_span = [
        context_tokens.index("[ent_start]"),
        context_tokens.index("[ent_end]") + 1,
    ]
    encoded = tokenizer(
        context_tokens,
        is_split_into_words=True,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        return_overflowing_tokens=False,
    )
    return encoded, new_span


def get_uniq_ids(sent_i, num_aliases, guid_dtype, max_aliases=1):
    """Get unique ids."""
    res = []
    for i in range(num_aliases):
        res.append(np.array((sent_i, i, [i]), dtype=guid_dtype(max_aliases)))
    return res


def assert_data_dicts_equal(dict_l, dict_r):
    """Assert data dicts are equals."""
    for k in dict_l:
        assert k in dict_r, f"Key is {k}"
        if type(dict_l[k]) is torch.Tensor:
            assert torch.allclose(dict_l[k].float(), dict_r[k].float()), f"Key is {k}"
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
    """Data test."""

    def setUp(self):
        """Set up."""
        # tests that the sampling is done correctly on indices
        # load data from directory
        self.args = parser_utils.parse_boot_and_emm_args(
            "tests/run_args/test_data.json"
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
        )
        self.temp_file_name = "tests/data/data_loader/test_data.jsonl"
        self.guid_dtype = lambda max_aliases: np.dtype(
            [
                ("sent_idx", "i8", 1),
                ("subsent_idx", "i8", 1),
                ("alias_orig_list_pos", "i8", (max_aliases,)),
            ]
        )

    def tearDown(self) -> None:
        """Tear down."""
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

    def prep_dicts(
        self,
        max_seq_len,
        max_window_len,
        gold_cand_idx,
        gold_cand_idx_train,
        use_weak,
        input_data,
    ):
        """Prep data dicts."""
        X_dict, Y_dict = defaultdict(list), defaultdict(list)
        for i, inp in enumerate(input_data):
            guids = get_uniq_ids(i, len(inp["aliases"]), self.guid_dtype)
            for j in range(len(inp["aliases"])):
                if use_weak is False and inp["gold"][j] is False:
                    continue
                X_dict["guids"].append(guids[j])
                tok_sent, new_span = adjust_sentence(
                    inp["sentence"],
                    max_seq_len,
                    max_window_len,
                    inp["spans"][j],
                    self.tokenizer,
                )
                for k in tok_sent:
                    X_dict[k].append(tok_sent[k])
                X_dict["sent_idx"].append(i)
                X_dict["subsent_idx"].append(j)
                if inp["aliases"][j] not in self.entity_symbols.get_all_aliases():
                    alias_idx = -2
                else:
                    alias_idx = self.entity_symbols.get_alias_idx(inp["aliases"][j])
                X_dict["alias_idx"].append(alias_idx)
                X_dict["alias_orig_list_pos"].append(j)
                if gold_cand_idx[i][j] != -1:
                    gold_eid = self.entity_symbols.get_eid(inp["qids"][j])
                else:
                    gold_eid = -1
                X_dict["gold_eid"].append(gold_eid)
                X_dict["for_dump_gold_eid"].append(
                    self.entity_symbols.get_eid(inp["qids"][j])
                )

                word_mask_scores = [-1 for _ in range(len(tok_sent["input_ids"]))]
                if tok_sent.word_to_tokens(new_span[0]) is None:
                    import pdb

                    pdb.set_trace()
                new_span_start = tok_sent.word_to_tokens(new_span[0]).start + 1
                # -1 to index the [ent_end] token, not the token after
                if tok_sent.word_to_tokens(new_span[1] - 1) is None:
                    # -1 for CLS token
                    new_span_end = len(tok_sent["input_ids"]) - 1
                else:
                    new_span_end = tok_sent.word_to_tokens(new_span[1] - 1).start
                word_mask_scores[new_span_start:new_span_end] = [
                    1 for _ in range(new_span_start, new_span_end)
                ]
                X_dict["word_qid_cnt_mask_score"].append(word_mask_scores)

                X_dict["for_dump_gold_cand_K_idx_train"].append(
                    gold_cand_idx_train[i][j]
                )
                Y_dict["gold_cand_K_idx"].append(gold_cand_idx[i][j])
        for k in X_dict:
            if k == "guids":
                X_dict[k] = np.array(X_dict[k])
                continue
            X_dict[k] = torch.tensor(X_dict[k])
        for k in Y_dict:
            Y_dict[k] = torch.tensor(Y_dict[k])
        return X_dict, Y_dict

    def test_simple_dataset(self):
        """
        Test simple dataset.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias'-1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        gold_cand_idx_train = [[0, 2]]
        gold_cand_idx = [[0, 2]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_in_candidate_flag(self):
        """
        Test in candidates.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_seq_len = 10
        max_aliases = 1
        max_window_len = 7
        split = "train"
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
        # Test 1: the code fails because it's training and Q3 is not a candidate of multi word alias2
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q3"],
                "sent_idx_unq": 0,
                "sentence": "alias'-1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]
        use_weak_label = True
        utils.write_jsonl(self.temp_file_name, input_data)
        with self.assertRaises(Exception) as context:
            BootlegDataset(
                self.args.data_config,
                name="Bootleg_test",
                dataset=self.temp_file_name,
                use_weak_label=use_weak_label,
                load_entity_data=False,
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
                "sentence": "alias'-1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]

        gold_cand_idx_train = [[0, -2]]
        gold_cand_idx = [[0, -2]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
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
                "sentence": "alias'-1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
        ]

        gold_cand_idx_train = [[1, 0]]
        gold_cand_idx = [[1, 0]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)
        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split=split,
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_nonmatch_alias(self):
        """
        Test aliases not in dict.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[-2, 2]]
        gold_cand_idx = [[-2, 2]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        with self.assertRaises(Exception) as context:
            dataset = BootlegDataset(
                self.args.data_config,
                name="Bootleg_test",
                dataset=self.temp_file_name,
                use_weak_label=use_weak_label,
                load_entity_data=False,
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
            load_entity_data=False,
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
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="test",
            is_bert=True,
        )
        X_dict["for_dump_gold_cand_K_idx_train"] = torch.tensor([-2, 3])
        Y_dict["gold_cand_K_idx"] = torch.tensor([-2, 3])
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_long_sentences(self):
        """
        Test long sentences.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: the sentence is long and has far apart aliases so it gets split up into two subsentences
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0]]
        gold_cand_idx = [[0, 0]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 1: the sentence is long but there is only one alias, so the sentence gets windowed
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0]]
        gold_cand_idx = [[0, 0]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_long_aliases(self):
        """
        Test large number aliases.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0, 0]]
        gold_cand_idx = [[0, 0, 0]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_non_gold_aliases(self):
        """
        Test non-gold aliases.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: the gold of False should be untouched for train, with only one True gold
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0, 0]]
        gold_cand_idx = [[0, 0, 0]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 2: the gold of False should be untouched for train, with all False golds
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0, 0]]
        gold_cand_idx = [[0, 0, 0]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
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
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        split = "dev"
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0, 0]]
        gold_cand_idx = [[0, -1, -1]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split=split,
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 4: with the split of dev, all true indices should be -1 but the sentences should still be used
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        split = "dev"
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0, 0]]
        gold_cand_idx = [[-1, -1, -1]]
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split=split,
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_non_gold_no_weak_label_aliases(self):
        """
        Test non gold aliases without weak labels.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 0: with all TRUE golds, use weak label of FALSE doesn't change anything
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, 0, 0]]
        gold_cand_idx = [[0, 0, 0]]
        use_weak_label = False
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 1: now that weak label is set to False, the golds of False should be removed for split of "train"
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, -1, -1]]
        gold_cand_idx = [[0, -1, -1]]
        use_weak_label = False
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
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
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[-1, -1, -1], [0]]
        gold_cand_idx = [[-1, -1, -1], [0]]
        use_weak_label = False
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split="train",
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 3: with the split of "dev", nothing should change from test 1 above where we were using "train"
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        split = "dev"
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[0, -1, -1]]
        gold_cand_idx = [[0, -1, -1]]
        use_weak_label = False
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split=split,
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

        # Test 4: with the split of dev, all true indices should be -1 but the sentences should still be used
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        split = "dev"
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
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
        gold_cand_idx_train = [[-1, -1, -1], [0]]
        gold_cand_idx = [[-1, -1, -1], [0]]
        use_weak_label = False
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=1,
            split=split,
            is_bert=True,
        )
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)

    def test_multiple_sentences(self):
        """
        Test multiple sentences at once with multiprocessing.

        ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: the gold of False should be untouched for train
        max_seq_len = 15
        max_window_len = 4
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.max_seq_window_len = max_window_len
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": i,
                "sentence": "alias'-1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "gold": [True, True],
            }
            for i in range(53)
        ]
        assert len(input_data) == 53
        gold_cand_idx_train = [[0, 2]] * 53
        gold_cand_idx = [[0, 2]] * 53
        use_weak_label = True
        X_dict, Y_dict = self.prep_dicts(
            max_seq_len,
            max_window_len,
            gold_cand_idx,
            gold_cand_idx_train,
            use_weak_label,
            input_data,
        )

        utils.write_jsonl(self.temp_file_name, input_data)

        dataset = BootlegDataset(
            self.args,
            name="Bootleg_test",
            dataset=self.temp_file_name,
            use_weak_label=use_weak_label,
            load_entity_data=False,
            tokenizer=self.tokenizer,
            entity_symbols=self.entity_symbols,
            dataset_threads=2,
            split="train",
            is_bert=True,
        )
        # Using multiple threads will make data in random sorted chunk order
        sort_arr = np.array(np.zeros(53 * 2), dtype=[("x", "<i8"), ("y", "<i8")])
        sort_arr["x"] = dataset.X_dict["sent_idx"]
        sort_arr["y"] = dataset.X_dict["subsent_idx"]
        sort_idx = np.argsort(sort_arr, order=["x", "y"])
        for key in list(dataset.X_dict.keys()):
            dataset.X_dict[key] = dataset.X_dict[key][sort_idx]
        for key in list(dataset.Y_dict.keys()):
            dataset.Y_dict[key] = dataset.Y_dict[key][sort_idx]
        assert_data_dicts_equal(X_dict, dataset.X_dict)
        assert_data_dicts_equal(Y_dict, dataset.Y_dict)


if __name__ == "__main__":
    unittest.main()
