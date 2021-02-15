import unittest

import numpy as np

from bootleg.scorer import BootlegSlicedScorer


class BootlegMockScorer(BootlegSlicedScorer):
    def __init__(self, train_in_candidates):
        self.mock_slices = {
            0: {"all": [1, 1], "slice_1": [0, 0]},
            1: {"all": [1, 1], "slice_1": [1, 0]},
            2: {"all": [1, 1], "slice_1": [0, 1]},
        }
        self.train_in_candidates = train_in_candidates

    def get_slices(self, uid):
        return self.mock_slices[uid]


class TestScorer(unittest.TestCase):
    def test_bootleg_scorer(self):
        # batch = 3, M = 2
        scorer = BootlegMockScorer(train_in_candidates=True)

        golds = np.array([[0, -2], [1, -1], [0, 3]])

        probs = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        preds = np.array([[1, 2], [0, 1], [0, 3]])

        uids = np.array([0, 1, 2])

        res = scorer.bootleg_score(golds, probs, preds, uids)

        gold_res = {}
        slice_name = "all"
        gold_res[f"{slice_name}/total_men"] = 5
        gold_res[f"{slice_name}/total_notNC_men"] = 4
        gold_res[f"{slice_name}/acc_boot"] = 2 / 5
        gold_res[f"{slice_name}/acc_notNC_boot"] = 2 / 4
        gold_res[f"{slice_name}/acc_pop"] = 2 / 5
        gold_res[f"{slice_name}/acc_notNC_pop"] = 2 / 4

        slice_name = "slice_1"
        gold_res[f"{slice_name}/total_men"] = 2
        gold_res[f"{slice_name}/total_notNC_men"] = 2
        gold_res[f"{slice_name}/acc_boot"] = 1 / 2
        gold_res[f"{slice_name}/acc_notNC_boot"] = 1 / 2
        gold_res[f"{slice_name}/acc_pop"] = 0
        gold_res[f"{slice_name}/acc_notNC_pop"] = 0

        self.assertDictEqual(res, gold_res)

    def test_bootleg_scorer_notincand(self):
        # batch = 3, M = 2
        scorer = BootlegMockScorer(train_in_candidates=False)

        golds = np.array([[0, 3], [2, -1], [1, 4]])

        probs = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        preds = np.array([[0, 3], [0, 1], [2, 4]])

        uids = np.array([0, 1, 2])

        res = scorer.bootleg_score(golds, probs, preds, uids)

        gold_res = {}
        slice_name = "all"
        gold_res[f"{slice_name}/total_men"] = 5
        gold_res[f"{slice_name}/total_notNC_men"] = 4
        gold_res[f"{slice_name}/acc_boot"] = 3 / 5
        gold_res[f"{slice_name}/acc_notNC_boot"] = 2 / 4
        gold_res[f"{slice_name}/acc_pop"] = 1 / 5
        gold_res[f"{slice_name}/acc_notNC_pop"] = 1 / 4

        slice_name = "slice_1"
        gold_res[f"{slice_name}/total_men"] = 2
        gold_res[f"{slice_name}/total_notNC_men"] = 2
        gold_res[f"{slice_name}/acc_boot"] = 1 / 2
        gold_res[f"{slice_name}/acc_notNC_boot"] = 1 / 2
        gold_res[f"{slice_name}/acc_pop"] = 0
        gold_res[f"{slice_name}/acc_notNC_pop"] = 0

        self.assertDictEqual(res, gold_res)


if __name__ == "__main__":
    unittest.main()
