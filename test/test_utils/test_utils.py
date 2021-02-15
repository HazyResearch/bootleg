import copy
import unittest

import networkx as nx
import numpy as np
import torch

from bootleg.utils import embedding_utils, model_utils


class Utils(unittest.TestCase):
    def test_kg_embedding_utils(self):
        # M = 2, K = 3, 4 entities total
        entity_indices = torch.tensor([[1, 3, -1], [2, -1, -1]])

        # The border of the adj is all 0s to account for -1 (pad) and 0 (unk) eids
        # 1-2, 1-4, 2-3
        adj = nx.adjacency_matrix(
            nx.Graph(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0],
                        [0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                )
            )
        )

        res = embedding_utils.prep_kg_feature_matrix(entity_indices, adj)
        res_gold = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        np.testing.assert_array_equal(res_gold, res)

    def test_emb_dropout_tensor(self):
        tensor = torch.randn(5, 4, 3)
        reg_ten = torch.ones(5, 4)
        reg_ten[0:2, 1:3] = 0
        reg_ten[2:5, 3:4] = 0
        res = model_utils.emb_dropout_by_tensor(
            training=True, regularization_tensor=reg_ten, tensor=tensor
        )

        res_gold = copy.deepcopy(tensor)
        res_gold[reg_ten == 1] = 0

        assert torch.equal(res_gold, res)

    def test_select_alias_word(self):
        # batch = 3, M = 2, N = 5, H = 4
        sent_embedding = torch.randn(3, 5, 4)
        alias_pos_in_sent = torch.tensor([[2, 1], [1, -1], [0, -1]])
        res = model_utils.select_alias_word_sent(alias_pos_in_sent, sent_embedding)
        res_gold = torch.zeros(3, 2, 4)
        res_gold[0][0] = sent_embedding[0][2]
        res_gold[0][1] = sent_embedding[0][1]
        res_gold[1][0] = sent_embedding[1][1]
        res_gold[2][0] = sent_embedding[2][0]

        assert torch.equal(res_gold, res)
