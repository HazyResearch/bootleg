import jsonlines
import numpy as np
import os
import tempfile
import unittest

from bootleg.utils import eval_utils

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
        num_examples = 3
        total_num_mentions = 7
        M = 3
        K = 2
        hidden_size = 2

        # create full embedding file
        storage_type = np.dtype([('M', int), ('K', int), ('hidden_size', int), ('sent_idx', int), ('subsent_idx', int),
            ('alias_list_pos', int, M), ('entity_emb', float, M*hidden_size),
            ('final_loss_true', int, M), ('final_loss_pred', int, M),
            ('final_loss_prob', float, M)])
        full_emb = np.memmap('tmp.pt', dtype=storage_type, mode='w+', shape=(num_examples,))

        # 2 sentences, 1st sent has 1 subsentence, 2nd sentence has 2 subsentences
        # first sentence
        full_emb['hidden_size'] = hidden_size
        full_emb['M'] = M
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
        storage_type = np.dtype([('hidden_size', int),
                             ('sent_idx', int),
                             ('alias_list_pos', int),
                             ('entity_emb', float, hidden_size),
                             ('final_loss_pred', int),
                             ('final_loss_prob', float)])
        merged_emb = np.zeros(total_num_mentions, dtype=storage_type)
        merged_emb['entity_emb'] = np.array([[0, 1],
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
        bootleg_merged_emb = eval_utils.merge_subsentences(temp_file, full_emb, dump_embs=True)
        for i in range(len(bootleg_merged_emb)):
            assert np.array_equal(bootleg_merged_emb[i]['entity_emb'], merged_emb[i]['entity_emb'])

        # clean up
        os.remove(temp_file)

if __name__ == "__main__":
    unittest.main()