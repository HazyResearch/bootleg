import shutil

import jsonlines
import marisa_trie
import numpy as np
import os
import tempfile
import unittest

import ujson

from bootleg import DottedDict
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import eval_utils
from bootleg.utils.eval_utils import write_data_labels

class EntitySymbolsSubclass(EntitySymbols):
    def __init__(self):
        self.max_candidates = 2
        # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
        self.max_alias_len = 1
        self._qid2title = {"Q1": "a b c d e", "Q2": "f", "Q3": "dd a b", "Q4": "x y z"}
        self._qid2eid = {"Q1" : 1, "Q2": 2, "Q3": 3, "Q4": 4}
        self._alias2qids = {
              "a":[["Q1",10.0],["Q4",6]],
              "b":[["Q2",5.0],["Q1",3]],
              "c":[["Q1",30.0],["Q2",3]],
              "d":[["Q4",20],["Q3",15.0]],
              "e":[["Q1",10.0],["Q4",6]],
              "f":[["Q2",5.0],["Q1",3]],
              "g":[["Q1",30.0],["Q2",3]]
            }
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2
        self.alias_cand_map_file = "alias2qids.json"

class EmbeddingExtractionTest(unittest.TestCase):

    def test_select_embs(self):
        # final_entity_embs is batch x M x K x hidden_size, pred_cands in batch x M
        # batch size 2, M is 2, K is 3, hidden size is 4
        final_entity_embs = np.array([
                  [
                    [[0,1,2,3],     [4,5,6,7],     [8,9,10,11]],
                    [[11,12,13,14], [15,16,17,18], [19,20,21,22]]
                  ],

                  [
                    [[10,11,12,13],     [14,15,16,17],     [18,19,20,21]],
                    [[110,120,130,140], [150,160,170,180], [190,200,210,220]]
                  ]
                ])

        pred_cands = np.array([[0,1], [2,0]])
        gold_embs = np.array([[
                                [0,1,2,3],
                                [15,16,17,18]
                              ],
                              [
                                [18,19,20,21],
                                [110,120,130,140]
                              ]])
        # compare outputs
        selected_embs = eval_utils.select_embs(final_entity_embs, pred_cands, batch_size=2, M=2)
        assert np.array_equal(selected_embs, gold_embs)

    def test_merge_subsentences(self):

        test_full_emb_file = tempfile.NamedTemporaryFile()
        test_merged_emb_file = tempfile.NamedTemporaryFile()
        gold_merged_emb_file = tempfile.NamedTemporaryFile()

        num_examples = 3
        total_num_mentions = 7
        M = 3
        K = 2
        hidden_size = 2

        # create full embedding file
        storage_type_full = np.dtype([('M', int), ('K', int), ('hidden_size', int), ('sent_idx', int), ('subsent_idx', int),
            ('alias_list_pos', int, M), ('entity_emb', float, M*hidden_size),
            ('final_loss_true', int, M), ('final_loss_pred', int, M),
            ('final_loss_prob', float, M), ('final_loss_cand_probs', float, M*K)])
        full_emb = np.memmap(test_full_emb_file.name, dtype=storage_type_full, mode='w+', shape=(num_examples,))

        # 2 sentences, 1st sent has 1 subsentence, 2nd sentence has 2 subsentences
        # first sentence
        full_emb['hidden_size'] = hidden_size
        full_emb['M'] = M
        full_emb['K'] = K
        full_emb[0]['sent_idx'] = 0
        full_emb[0]['subsent_idx'] = 0
        # last alias is padded
        full_emb[0]['alias_list_pos'] = np.array([0, 1, -1])
        full_emb[0]['final_loss_true'] = np.array([0, 1, -1])
        # entity embs are flattened
        full_emb[0]['entity_emb'] = np.array([0, 1, 2, 3, 0, 0])

        full_emb[1]['sent_idx'] = 1
        full_emb[1]['subsent_idx'] = 0
        full_emb[1]['alias_list_pos'] = np.array([0, 1, 2])
        # last alias goes with next subsentence
        full_emb[1]['final_loss_true'] = np.array([1, 1, -1])
        full_emb[1]['entity_emb'] = np.array([4, 5, 6, 7, 8, 9])

        full_emb[2]['sent_idx'] = 1
        full_emb[2]['subsent_idx'] = 1
        full_emb[2]['alias_list_pos'] = np.array([2, 3, 4])
        full_emb[2]['final_loss_true'] = np.array([1, 1, 1])
        full_emb[2]['entity_emb'] = np.array([10, 11, 12, 13, 14, 15])

        # create merged embedding file
        storage_type_merged = np.dtype([('hidden_size', int),
                             ('sent_idx', int),
                             ('alias_list_pos', int),
                             ('entity_emb', float, hidden_size),
                             ('final_loss_pred', int),
                             ('final_loss_prob', float),
                             ('final_loss_cand_probs', float, K)])
        merged_emb_gold = np.memmap(gold_merged_emb_file.name, dtype=storage_type_merged, mode="w+", shape=(total_num_mentions,))
        merged_emb_gold['entity_emb'] = np.array([[0, 1],
                                             [2, 3],
                                             [4, 5],
                                             [6, 7],
                                             [10, 11],
                                             [12, 13],
                                             [14, 15]])

        # create data file -- just needs aliases and sentence indices
        data = [{'aliases': ['a', 'b'], 'sent_idx_unq': 0},
                {'aliases': ['c', 'd', 'e', 'f', 'g'], 'sent_idx_unq': 1}]

        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        with jsonlines.open(temp_file, 'w') as f:
            for row in data:
                f.write(row)

        # assert that output of merge_subsentences is correct
        num_processes = 2
        eval_utils.merge_subsentences(
            num_processes,
            temp_file,
            test_merged_emb_file.name,
            storage_type_merged,
            test_full_emb_file.name,
            storage_type_full,
            dump_embs=True)
        bootleg_merged_emb = np.memmap(test_merged_emb_file.name, dtype=storage_type_merged, mode="r+")
        merged_emb_gold = np.memmap(gold_merged_emb_file.name, dtype=storage_type_merged, mode="r+")
        assert len(bootleg_merged_emb) == total_num_mentions
        for i in range(len(bootleg_merged_emb)):
            assert np.array_equal(bootleg_merged_emb[i]['entity_emb'], merged_emb_gold[i]['entity_emb'])

        # clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        test_full_emb_file.close()
        test_merged_emb_file.close()
        gold_merged_emb_file.close()

    def test_write_out_subsentences(self):

        merged_entity_emb_file = tempfile.NamedTemporaryFile()
        out_file = tempfile.NamedTemporaryFile()
        data_file = tempfile.NamedTemporaryFile()

        entity_dir = "test/entity_db"
        entity_map_dir = "entity_mappings"
        alias_cand_map = "alias2qids.json"
        data_config = DottedDict(
            entity_dir = entity_dir,
            entity_map_dir = entity_map_dir,
            alias_cand_map = alias_cand_map
        )
        entitiy_symbols = EntitySymbolsSubclass()
        entitiy_symbols.dump(save_dir=os.path.join(entity_dir, entity_map_dir))

        num_examples = 3
        total_num_mentions = 7
        M = 3
        K = 2
        hidden_size = 2

        # create data file -- just needs aliases and sentence indices
        data = [{'aliases': ['a', 'b'], 'sent_idx_unq': 0},
                {'aliases': ['c', 'd', 'e', 'f', 'g'], 'sent_idx_unq': 1}]

        with jsonlines.open(data_file.name, 'w') as f:
            for row in data:
                f.write(row)

        merged_storage_type = np.dtype([('hidden_size', int),
                             ('sent_idx', int),
                             ('alias_list_pos', int),
                             ('entity_emb', float, hidden_size),
                             ('final_loss_pred', int),
                             ('final_loss_prob', float),
                             ('final_loss_cand_probs', float, K)])

        merged_entity_emb = np.memmap(merged_entity_emb_file.name, dtype=merged_storage_type, mode="w+", shape=(total_num_mentions,))
        # 2 sentences, 1st sent has 1 subsentence, 2nd sentence has 2 subsentences - 7 mentions total
        merged_entity_emb['hidden_size'] = hidden_size
        # first men
        merged_entity_emb[0]['sent_idx'] = 0
        merged_entity_emb[0]['alias_list_pos'] = 0
        merged_entity_emb[0]['entity_emb'] = np.array([0, 1])
        merged_entity_emb[0]['final_loss_pred'] = 1
        merged_entity_emb[0]['final_loss_prob'] = 0.9
        merged_entity_emb[0]['final_loss_cand_probs'] = np.array([0.1, 0.9])
        # second men
        merged_entity_emb[1]['sent_idx'] = 0
        merged_entity_emb[1]['alias_list_pos'] = 1
        merged_entity_emb[1]['entity_emb'] = np.array([2, 3])
        merged_entity_emb[1]['final_loss_pred'] = 1
        merged_entity_emb[1]['final_loss_prob'] = 0.9
        merged_entity_emb[1]['final_loss_cand_probs'] = np.array([0.1, 0.9])
        # third men
        merged_entity_emb[2]['sent_idx'] = 1
        merged_entity_emb[2]['alias_list_pos'] = 0
        merged_entity_emb[2]['entity_emb'] = np.array([4, 5])
        merged_entity_emb[2]['final_loss_pred'] = 0
        merged_entity_emb[2]['final_loss_prob'] = 0.9
        merged_entity_emb[2]['final_loss_cand_probs'] = np.array([0.9, 0.1])
        # fourth men
        merged_entity_emb[3]['sent_idx'] = 1
        merged_entity_emb[3]['alias_list_pos'] = 1
        merged_entity_emb[3]['entity_emb'] = np.array([6, 7])
        merged_entity_emb[3]['final_loss_pred'] = 0
        merged_entity_emb[3]['final_loss_prob'] = 0.9
        merged_entity_emb[3]['final_loss_cand_probs'] = np.array([0.9, 0.1])
        # fifth men
        merged_entity_emb[4]['sent_idx'] = 1
        merged_entity_emb[4]['alias_list_pos'] = 2
        merged_entity_emb[4]['entity_emb'] = np.array([10, 11])
        merged_entity_emb[4]['final_loss_pred'] = 1
        merged_entity_emb[4]['final_loss_prob'] = 0.9
        merged_entity_emb[4]['final_loss_cand_probs'] = np.array([0.1, 0.9])
        # sixth men
        merged_entity_emb[5]['sent_idx'] = 1
        merged_entity_emb[5]['alias_list_pos'] = 3
        merged_entity_emb[5]['entity_emb'] = np.array([12, 13])
        merged_entity_emb[5]['final_loss_pred'] = 1
        merged_entity_emb[5]['final_loss_prob'] = 0.9
        merged_entity_emb[5]['final_loss_cand_probs'] = np.array([0.1, 0.9])
        # seventh men
        merged_entity_emb[6]['sent_idx'] = 1
        merged_entity_emb[6]['alias_list_pos'] = 4
        merged_entity_emb[6]['entity_emb'] = np.array([14, 15])
        merged_entity_emb[6]['final_loss_pred'] = 1
        merged_entity_emb[6]['final_loss_prob'] = 0.9
        merged_entity_emb[6]['final_loss_cand_probs'] = np.array([0.1, 0.9])

        num_processes = 2
        train_in_candidates = True
        dump_embs = True

        write_data_labels(num_processes, merged_entity_emb_file.name,
                      merged_storage_type, data_file.name,
                      out_file.name, train_in_candidates, dump_embs, data_config)

        '''
          "a":[["Q1",10.0],["Q4",6]],
          "b":[["Q2",5.0],["Q1",3]],
          "c":[["Q1",30.0],["Q2",3]],
          "d":[["Q4",20],["Q3",15.0]],
          "e":[["Q1",10.0],["Q4",6]],
          "f":[["Q2",5.0],["Q1",3]],
          "g":[["Q1",30.0],["Q2",3]]
        '''
        all_lines = []
        with open(out_file.name) as check_f:
            for line in check_f:
                all_lines.append(ujson.loads(line))

        gold_lines = [
            {
                'sent_idx_unq': 0,
                'aliases': ['a', 'b'],
                'qids' : ["Q4", "Q1"],
                'probs' : [0.9, 0.9],
                'cands' : [["Q4", "Q1"], ["Q1", "Q2"]],
                'cand_probs' : [[0.9, 0.1], [0.9, 0.1]],
                'entity_ids' : [4, 1],
                'ctx_emb_ids' : [0, 1],
                'cand_entity_ids': [[4, 1], [1, 2]]
            },
            {
                'sent_idx_unq': 1,
                'aliases': ['c', 'd', 'e', 'f', 'g'],
                'qids' : ["Q1", "Q4", "Q4", "Q1", "Q2"],
                'probs' : [0.9, 0.9, 0.9, 0.9, 0.9],
                'cands' : [["Q1", "Q2"], ["Q4", "Q3"], ["Q4", "Q1"], ["Q1", "Q2"], ["Q2", "Q1"]],
                'cand_probs' : [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
                'entity_ids' : [1, 4, 4, 1, 2],
                'ctx_emb_ids' : [2, 3, 4, 5, 6],
                'cand_entity_ids': [[1, 2], [4, 3], [4, 1], [1, 2], [2, 1]]
            }
        ]

        assert len(all_lines) == len(gold_lines)
        for i in range(len(gold_lines)):
            self.assertDictEqual(gold_lines[i], all_lines[i], f"{ujson.dumps(gold_lines[i], indent=4)} VS {ujson.dumps(all_lines[i], indent=4)}")

        # clean up
        if os.path.exists(entity_dir):
            shutil.rmtree(entity_dir)
        merged_entity_emb_file.close()
        out_file.close()
        data_file.close()

if __name__ == "__main__":
    unittest.main()