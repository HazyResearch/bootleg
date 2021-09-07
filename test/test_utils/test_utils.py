import unittest

import torch

from bootleg.utils import model_utils


class Utils(unittest.TestCase):
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
