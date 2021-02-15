import os
import unittest

import marisa_trie
import torch
from transformers import BertModel, BertTokenizer

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils.preprocessing.build_static_embeddings import build_title_table


class EntitySymbolsSubclass(EntitySymbols):
    def __init__(self):
        self.max_candidates = 3
        # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
        self.max_alias_len = 1
        self._qid2title = {"Q1": "a b c d e", "Q2": "f", "Q3": "dd a b", "Q4": "x y z"}
        self._qid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        self._alias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2


class TestStaticEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        self.save_file = "test/data/static_emb.pt"
        self.cache = "test/data/emb_data/pretrained_bert_models"
        self.entity_symbols = EntitySymbolsSubclass()

    def tearDown(self) -> None:
        if os.path.exists(self.save_file):
            os.remove(self.save_file)

    def test_static_titles(self):
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased",
            do_lower_case="uncased" in "bert-base-cased",
            cache_dir=self.cache,
        )
        model = BertModel.from_pretrained(
            "bert-base-cased",
            cache_dir=self.cache,
            output_attentions=False,
            output_hidden_states=False,
        )
        model.eval()
        cpu = True
        batch_size = 5
        entity2avgtitle = build_title_table(
            cpu, batch_size, model, tokenizer, self.entity_symbols
        )
        entity2avgtitle_gold = torch.zeros(6, 768)
        # The [0] gets the last hidden state (model returns tuple)
        entity2avgtitle_gold[1] = (
            model(
                tokenizer(
                    "a b c d e", padding=True, truncation=True, return_tensors="pt"
                )["input_ids"]
            )[0]
            .squeeze(0)
            .mean(0)
        )
        entity2avgtitle_gold[2] = (
            model(
                tokenizer("f", padding=True, truncation=True, return_tensors="pt")[
                    "input_ids"
                ]
            )[0]
            .squeeze(0)
            .mean(0)
        )
        entity2avgtitle_gold[3] = (
            model(
                tokenizer("dd a b", padding=True, truncation=True, return_tensors="pt")[
                    "input_ids"
                ]
            )[0]
            .squeeze(0)
            .mean(0)
        )
        entity2avgtitle_gold[4] = (
            model(
                tokenizer("x y z", padding=True, truncation=True, return_tensors="pt")[
                    "input_ids"
                ]
            )[0]
            .squeeze(0)
            .mean(0)
        )

        assert torch.allclose(entity2avgtitle, entity2avgtitle_gold, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
