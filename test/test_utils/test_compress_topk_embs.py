import unittest

import torch

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils.entity_profile.compress_topk_entity_embeddings import (
    filter_embs,
    filter_qids,
)


class TestTopKEmbsCompression(unittest.TestCase):
    def test_filter_qids(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        entity_db = EntitySymbols.load_from_cache(
            load_dir=entity_dump_dir, alias_cand_map_file="alias2qids.json"
        )
        qid2count = {"Q1": 10, "Q2": 20, "Q3": 2, "Q4": 4}
        perc_emb_drop = 0.8

        gold_qid2topk_eid = {"Q1": 2, "Q2": 1, "Q3": 2, "Q4": 2}
        gold_old2new_eid = {0: 0, -1: -1, 2: 1, 3: 2}
        gold_new_toes_eid = 2
        gold_num_topk_entities = 2

        (
            qid2topk_eid,
            old2new_eid,
            new_toes_eid,
            num_topk_entities,
        ) = filter_qids(perc_emb_drop, entity_db, qid2count)
        self.assertEqual(gold_qid2topk_eid, qid2topk_eid)
        self.assertEqual(gold_old2new_eid, old2new_eid)
        self.assertEqual(gold_new_toes_eid, new_toes_eid)
        self.assertEqual(gold_num_topk_entities, num_topk_entities)

    def test_filter_embs(self):
        entity_dump_dir = "test/data/entity_loader/entity_data/entity_mappings"
        entity_db = EntitySymbols.load_from_cache(
            load_dir=entity_dump_dir,
            alias_cand_map_file="alias2qids.json",
            alias_idx_file="alias2id.json",
        )
        num_topk_entities = 2
        old2new_eid = {0: 0, -1: -1, 2: 1, 3: 2}
        qid2topk_eid = {"Q1": 2, "Q2": 1, "Q3": 2, "Q4": 2}
        toes_eid = 2
        state_dict = {
            "module_pool": {
                "learned": {
                    "learned_entity_embedding.weight": torch.Tensor(
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

        gold_state_dict = {
            "module_pool": {
                "learned": {
                    "learned_entity_embedding.weight": torch.Tensor(
                        [
                            [1.0, 2, 3, 4, 5],
                            [3, 3, 3, 3, 3],
                            [4, 4, 4, 4, 4],
                            [0, 0, 0, 0, 0],
                        ]
                    )
                }
            }
        }

        new_state_dict = filter_embs(
            num_topk_entities,
            entity_db,
            old2new_eid,
            qid2topk_eid,
            toes_eid,
            state_dict,
        )
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


if __name__ == "__main__":
    unittest.main()
