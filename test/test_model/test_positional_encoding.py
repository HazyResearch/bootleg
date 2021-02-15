import copy
import unittest

import torch

from bootleg.layers.helper_modules import PositionalEncoding


class PostionalEncodingTest(unittest.TestCase):
    def test_pe_forward(self):
        hidden_dim = 4
        max_len = 5
        batch_size = 2

        pe = PositionalEncoding(hidden_dim=hidden_dim, max_len=max_len)
        # this is like PE for a sentence
        # x is batch x sent_len x hidden_dim
        x = torch.ones(batch_size, 4, hidden_dim)
        spans = [[0, 1, 2, 3], [0, 1, 2, 3]]
        # pe.pe is size 5+1 x 3
        expected_out = copy.deepcopy(pe.pe)
        # +1 is for the padded position for unk aliases
        expected_out = torch.cat(batch_size * [expected_out]) + 1
        expected_out = expected_out[:, :4, :]
        out = pe.forward(x, spans)
        assert torch.allclose(out, expected_out)

        # size batch x M x K x hidden_dim
        # represents entity indexing for two batches with 3 aliases with 3 candidates each
        # batch one: alias one is at pos 1, alias two is at pos 2, and alias three does not exist
        # batch two: alias one is at pos 3, alias two and three do not exist
        # each candidate for alias i gets assigned the same positional encoding
        x = torch.ones(2, 3, 3, 4)
        # the position is the same for each of the 3 candidates, hence three numbers in a row
        spans = [
            [[1, 1, 1], [2, 2, 2], [-1, -1, -1]],
            [[3, 3, 3], [-1, -1, -1], [-1, -1, -1]],
        ]
        # pe.pe is size 6 x 3
        expected_out = copy.deepcopy(pe.pe)
        expected_out = torch.cat(batch_size * [expected_out])
        # expected_out = expected_out.unsqueeze(2)
        # expected_out = torch.cat(3*[expected_out], dim=2)
        # the 0s and 1s is just for the batch indexing
        expected_out = (
            expected_out[
                [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
                spans,
                :,
            ]
            + 1
        )
        out = pe.forward(x, spans)
        assert torch.allclose(out, expected_out)


if __name__ == "__main__":
    unittest.main()
