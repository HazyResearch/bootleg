import unittest
import os, sys
import torch

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils, parser_utils
from bootleg.utils.classes.record_trie_collection import RecordTrieCollection
from bootleg.utils.postprocessing.compress_topk_entity_embeddings import filter_qids, filter_embs
from bootleg.utils.sentence_utils import split_sentence


class TestTopKEmbsCompression(unittest.TestCase):

    def test_filter_qids(self):
        entity_dump_dir = "test/data/preprocessing/base/entity_data/entity_mappings"
        entity_db = EntitySymbols(load_dir=entity_dump_dir,
            alias_cand_map_file="alias2qids.json")
        qid2count = {"Q1":10,"Q2":20,"Q3":2,"Q4":4}
        perc_emb_drop = 0.8

        gold_qid2topk_eid = {"Q1":2,"Q2":1,"Q3":2,"Q4":2}
        gold_old2new_eid = {0:0,-1:-1,2:1,3:2}
        gold_new_toes_eid = 2
        gold_num_topk_entities = 2

        qid2topk_eid, old2new_eid, new_toes_eid, num_topk_entities = filter_qids(perc_emb_drop, entity_db, qid2count)
        self.assertEqual(gold_qid2topk_eid, qid2topk_eid)
        self.assertEqual(gold_old2new_eid, old2new_eid)
        self.assertEqual(gold_new_toes_eid, new_toes_eid)
        self.assertEqual(gold_num_topk_entities, num_topk_entities)

    def test_filter_embs(self):
        entity_dump_dir = "test/data/preprocessing/base/entity_data/entity_mappings"
        entity_db = EntitySymbols(load_dir=entity_dump_dir,
            alias_cand_map_file="alias2qids.json")
        num_topk_entities = 2
        old2new_eid = {0:0,-1:-1,2:1,3:2}
        qid2topk_eid = {"Q1":2,"Q2":1,"Q3":2,"Q4":2}
        toes_eid = 2
        state_dict = {}
        state_dict["emb_layer.entity_embs.learned.learned_entity_embedding.weight"] = torch.Tensor([
            [1.0,2,3,4,5],
            [2,2,2,2,2],
            [3,3,3,3,3],
            [4,4,4,4,4],
            [5,5,5,5,5],
            [0,0,0,0,0]
        ])

        gold_state_dict = {}
        gold_state_dict["emb_layer.entity_embs.learned.learned_entity_embedding.weight"] = torch.Tensor([
            [1.0,2,3,4,5],
            [3,3,3,3,3],
            [4,4,4,4,4],
            [0,0,0,0,0]
        ])

        new_state_dict = filter_embs(num_topk_entities, entity_db, old2new_eid, qid2topk_eid, toes_eid, state_dict)
        for k in gold_state_dict:
            assert k in new_state_dict
            assert torch.equal(new_state_dict[k], gold_state_dict[k])


if __name__ == '__main__':
    unittest.main()
