import os
import shutil
import unittest

import numpy as np
import torch

from bootleg.slicing.slice_dataset import BootlegSliceDataset
from bootleg.symbols.constants import FINAL_LOSS
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils


def assert_data_dicts_equal(dict_l, dict_r):
    for k in dict_l:
        assert k in dict_r
        if type(dict_l[k]) is torch.Tensor:
            assert torch.equal(dict_l[k].float(), dict_r[k].float())
        elif type(dict_l[k]) is np.ndarray:
            np.testing.assert_array_equal(dict_l[k], dict_r[k])
        else:
            assert dict_l[k] == dict_r[k]
    for k in dict_r:
        assert k in dict_l


def assert_slice_data_equal(gold_data, data):
    assert len(gold_data) == len(data)
    assert len(gold_data[0].tolist()[0]) == len(data[0].tolist()[0])
    assert len(gold_data[0].tolist()[0][0]) == len(data[0].tolist()[0][0])
    for i in range(len(gold_data)):
        for j in range(len(gold_data[i].tolist()[0])):  # number of slices
            for k in range(len(gold_data[i].tolist()[0][0])):  # number of columns
                np.testing.assert_allclose(
                    gold_data[i].tolist()[0][j][k],
                    data[i].tolist()[0][j][k],
                    err_msg=f"i j k {i} {j} {k}",
                )


class DataSlice(unittest.TestCase):
    def setUp(self):
        # tests that the sampling is done correctly on indices
        # load data from directory
        self.args = parser_utils.parse_boot_and_emm_args("test/run_args/test_data.json")
        self.entity_symbols = EntitySymbols(
            os.path.join(
                self.args.data_config.entity_dir, self.args.data_config.entity_map_dir
            ),
            alias_cand_map_file=self.args.data_config.alias_cand_map,
        )
        self.temp_file_name = "test/data/data_loader/test_slice_data.jsonl"

    def tearDown(self) -> None:
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

    def test_simple_dataset(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_aliases = 4
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.eval_slices = ["slice1"]
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "slices": {"slice1": {"0": 0.9, "1": 0.3}},
                "gold": [True, True],
            }
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 2),
                ("prob_labels", float, 2),
            ]
        )
        storage_type = np.dtype(
            [(slice_name, slice_dt, 1) for slice_name in [FINAL_LOSS, "slice1"]]
        )

        ex1 = [
            np.rec.array(
                [0, 0, [1, 1], [1.0, 1.0]], dtype=slice_dt
            ),  # FINAL LOSS SLICE
            np.rec.array([0, 0, [1, 0], [0.9, 0.3]], dtype=slice_dt),  # SLICE1 SLICE
        ]
        gold_data = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        # res = np.vstack((mat1))
        gold_sent_to_row_id_dict = {0: [0]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

    def test_single_mention_dataset(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        max_aliases = 1
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.eval_slices = ["slice1"]
        input_data = [
            {
                "aliases": ["alias1", "multi word alias2"],
                "qids": ["Q1", "Q4"],
                "sent_idx_unq": 0,
                "sentence": "alias1 or multi word alias2",
                "spans": [[0, 1], [2, 5]],
                "slices": {"slice1": {"0": 0.9, "1": 0.3}},
                "gold": [True, True],
            }
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 2),
                ("prob_labels", float, 2),
            ]
        )
        storage_type = np.dtype(
            [(slice_name, slice_dt, 1) for slice_name in [FINAL_LOSS, "slice1"]]
        )
        ex1 = [
            np.rec.array(
                [0, 0, [1, 1], [1.0, 1.0]], dtype=slice_dt
            ),  # FINAL LOSS SLICE
            np.rec.array([0, 0, [1, 0], [0.9, 0.3]], dtype=slice_dt),  # SLICE1 SLICE
        ]
        mat1 = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        gold_data = mat1
        gold_sent_to_row_id_dict = {0: [0]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

    def test_long_aliases(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: even though this sentence was split into multiple parts, the slices remain intact as an entire sentence
        max_seq_len = 5
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.eval_slices = ["slice1", "slice2"]
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "slices": {
                    "slice1": {"0": 0.9, "1": 0.3, "2": 0.5},
                    "slice2": {"0": 0.0, "1": 0.0, "2": 1.0},
                },
                "gold": [True, True, True],
            }
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 3),
                ("prob_labels", float, 3),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array(
                [0, 0, [1, 1, 1], [1.0, 1.0, 1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array([0, 0, [1, 0, 0], [0.9, 0.3, 0.5]], dtype=slice_dt),  # slice1
            np.rec.array([0, 0, [0, 0, 1], [0.0, 0.0, 1.0]], dtype=slice_dt),  # slice2
        ]
        gold_data = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        # res = np.vstack((mat1))
        gold_sent_to_row_id_dict = {0: [0]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

        # Test2: everything should remain the same even if the split is train

        use_weak_label = True
        split = "train"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

        # Test3: when we add another sentence that has fewer aliases, they should be padded to the largest that exist in a sentence, even if it's
        # greater than max aliases

        max_seq_len = 5
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.eval_slices = ["slice1", "slice2"]
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "slices": {
                    "slice1": {"0": 0.9, "1": 0.3, "2": 0.5},
                    "slice2": {"0": 0.0, "1": 0.0, "2": 1.0},
                },
                "gold": [True, True, True],
            },
            {
                "aliases": ["alias3"],
                "qids": ["Q1"],
                "sent_idx_unq": "1",
                "sentence": "alias3",
                "spans": [[0, 1]],
                "slices": {"slice1": {"0": 0.4}, "slice2": {"0": 1.0}},
                "gold": [True],
            },
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 3),
                ("prob_labels", float, 3),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array(
                [0, 0, [1, 1, 1], [1.0, 1.0, 1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array([0, 0, [1, 0, 0], [0.9, 0.3, 0.5]], dtype=slice_dt),  # slice1
            np.rec.array([0, 0, [0, 0, 1], [0.0, 0.0, 1.0]], dtype=slice_dt),  # slice2
        ]
        ex2 = [
            np.rec.array(
                [1, 0, [1, 0, 0], [1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array(
                [1, 0, [0, 0, 0], [0.4, -1.0, -1.0]], dtype=slice_dt
            ),  # slice1
            np.rec.array(
                [1, 0, [1, 0, 0], [1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # slice2
        ]
        mat1 = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        mat2 = np.rec.array(ex2, dtype=storage_type).reshape(1, 1)
        gold_data = np.vstack((mat1, mat2))
        gold_sent_to_row_id_dict = {0: [0], 1: [1]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

    def test_non_gold_aliases(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 1: for a dev split, the False golds will not count as aliases to score and not be in a slice
        max_seq_len = 5
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.eval_slices = ["slice1", "slice2"]
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "slices": {
                    "slice1": {"0": 0.9, "1": 0.3, "2": 0.5},
                    "slice2": {"0": 0.0, "1": 0.0, "2": 1.0},
                },
                "gold": [False, False, True],
            }
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 3),
                ("prob_labels", float, 3),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array(
                [0, 0, [0, 0, 1], [-1.0, -1.0, 1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array(
                [0, 0, [0, 0, 0], [-1.0, -1.0, 0.5]], dtype=slice_dt
            ),  # slice1
            np.rec.array(
                [0, 0, [0, 0, 1], [-1.0, -1.0, 1.0]], dtype=slice_dt
            ),  # slice2
        ]
        gold_data = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        # res = np.vstack((mat1))
        gold_sent_to_row_id_dict = {0: [0]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

        # Test2: everything should remain as it was with a split of train (i.e. FALSE golds are treated as TRUE)

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 3),
                ("prob_labels", float, 3),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array(
                [0, 0, [1, 1, 1], [1.0, 1.0, 1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array([0, 0, [1, 0, 0], [0.9, 0.3, 0.5]], dtype=slice_dt),  # slice1
            np.rec.array([0, 0, [0, 0, 1], [0.0, 0.0, 1.0]], dtype=slice_dt),  # slice2
        ]
        gold_data = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        # res = np.vstack((mat1))
        gold_sent_to_row_id_dict = {0: [0]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        split = "train"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

        # Test3: when we have multiple all FALSE anchors, we keep the slice indices but all aliases are not in a slice

        max_seq_len = 5
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.eval_slices = ["slice1", "slice2"]
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "slices": {
                    "slice1": {"0": 0.9, "1": 0.3, "2": 0.5},
                    "slice2": {"0": 0.0, "1": 0.0, "2": 1.0},
                },
                "gold": [False, False, False],
            },
            {
                "aliases": ["alias3"],
                "qids": ["Q1"],
                "sent_idx_unq": "1",
                "sentence": "alias3",
                "spans": [[0, 1]],
                "slices": {"slice1": {"0": 0.4}, "slice2": {"0": 1.0}},
                "gold": [False],
            },
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 3),
                ("prob_labels", float, 3),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array(
                [0, 0, [0, 0, 0], [-1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array(
                [0, 0, [0, 0, 0], [-1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # slice1
            np.rec.array(
                [0, 0, [0, 0, 0], [-1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # slice2
        ]
        ex2 = [
            np.rec.array(
                [1, 0, [0, 0, 0], [-1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array(
                [1, 0, [0, 0, 0], [-1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # slice1
            np.rec.array(
                [1, 0, [0, 0, 0], [-1.0, -1.0, -1.0]], dtype=slice_dt
            ),  # slice2
        ]
        mat1 = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        mat2 = np.rec.array(ex2, dtype=storage_type).reshape(1, 1)
        gold_data = np.vstack((mat1, mat2))
        gold_sent_to_row_id_dict = {0: [0], 1: [1]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = True
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

    def test_non_gold_no_weak_label_aliases(self):
        """ENTITY SYMBOLS
        {
          "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
          "alias1":[["Q1",10.0],["Q4",6.0]],
          "alias3":[["Q1",30.0]],
          "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
        }
        """
        # Test 0: when use weak labels is FALSE and all golds are TRUE, nothing should change
        max_seq_len = 5
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.eval_slices = ["slice1", "slice2"]
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "slices": {
                    "slice1": {"0": 0.9, "1": 0.3, "2": 0.5},
                    "slice2": {"0": 0.0, "1": 0.0, "2": 1.0},
                },
                "gold": [True, True, True],
            }
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 3),
                ("prob_labels", float, 3),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array(
                [0, 0, [1, 1, 1], [1.0, 1.0, 1.0]], dtype=slice_dt
            ),  # FINAL LOSS
            np.rec.array([0, 0, [1, 0, 0], [0.9, 0.3, 0.5]], dtype=slice_dt),  # slice1
            np.rec.array([0, 0, [0, 0, 1], [0.0, 0.0, 1.0]], dtype=slice_dt),  # slice2
        ]
        gold_data = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        # res = np.vstack((mat1))
        gold_sent_to_row_id_dict = {0: [0]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

        # Test 1: the FALSE golds will be dropped, leaving one alias to score. However, as we have to have at least 2 aliases to predict
        # for memmap to store as an array, we have two in our slice as a minimum.
        max_seq_len = 5
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.eval_slices = ["slice1", "slice2"]
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "slices": {
                    "slice1": {"0": 0.9, "1": 0.3, "2": 0.5},
                    "slice2": {"0": 0.0, "1": 0.0, "2": 1.0},
                },
                "gold": [False, False, True],
            }
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 2),
                ("prob_labels", float, 2),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array([0, 0, [1, 0], [1.0, -1.0]], dtype=slice_dt),  # FINAL LOSS
            np.rec.array([0, 0, [0, 0], [0.5, -1.0]], dtype=slice_dt),  # slice1
            np.rec.array([0, 0, [1, 0], [1.0, -1.0]], dtype=slice_dt),  # slice2
        ]
        gold_data = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        # res = np.vstack((mat1))
        gold_sent_to_row_id_dict = {0: [0]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

        # Test2: nothing should be different for a split of train

        split = "train"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)

        # Test3: when we have multiple all FALSE anchors, we keep the slice indices but all aliases are not in a slice

        max_seq_len = 5
        max_aliases = 2
        self.args.data_config.max_aliases = max_aliases
        self.args.data_config.max_seq_len = max_seq_len
        self.args.data_config.eval_slices = ["slice1", "slice2"]
        input_data = [
            {
                "aliases": ["alias3", "alias4", "alias3"],
                "qids": ["Q1", "Q4", "Q1"],
                "sent_idx_unq": 0,
                "sentence": "alias3 alias4 alias3",
                "spans": [[0, 1], [1, 2], [2, 3]],
                "slices": {
                    "slice1": {"0": 0.9, "1": 0.3, "2": 0.5},
                    "slice2": {"0": 0.0, "1": 0.0, "2": 1.0},
                },
                "gold": [False, False, False],
            },
            {
                "aliases": ["alias3"],
                "qids": ["Q1"],
                "sent_idx_unq": "1",
                "sentence": "alias3",
                "spans": [[0, 1]],
                "slices": {"slice1": {"0": 0.4}, "slice2": {"0": 1.0}},
                "gold": [True],
            },
        ]

        slice_dt = np.dtype(
            [
                ("sent_idx", int),
                ("subslice_idx", int),
                ("alias_slice_incidence", int, 2),
                ("prob_labels", float, 2),
            ]
        )
        storage_type = np.dtype(
            [
                (slice_name, slice_dt, 1)
                for slice_name in [FINAL_LOSS, "slice1", "slice2"]
            ]
        )

        ex1 = [
            np.rec.array([0, 0, [0, 0], [-1.0, -1.0]], dtype=slice_dt),  # FINAL LOSS
            np.rec.array([0, 0, [0, 0], [-1.0, -1.0]], dtype=slice_dt),  # slice1
            np.rec.array([0, 0, [0, 0], [-1.0, -1.0]], dtype=slice_dt),  # slice2
        ]
        ex2 = [
            np.rec.array([1, 0, [1, 0], [1.0, -1.0]], dtype=slice_dt),  # FINAL LOSS
            np.rec.array([1, 0, [0, 0], [0.4, -1.0]], dtype=slice_dt),  # slice1
            np.rec.array([1, 0, [1, 0], [1.0, -1.0]], dtype=slice_dt),  # slice2
        ]
        mat1 = np.rec.array(ex1, dtype=storage_type).reshape(1, 1)
        mat2 = np.rec.array(ex2, dtype=storage_type).reshape(1, 1)
        gold_data = np.vstack((mat1, mat2))
        gold_sent_to_row_id_dict = {0: [0], 1: [1]}

        utils.write_jsonl(self.temp_file_name, input_data)
        use_weak_label = False
        split = "dev"
        dataset = BootlegSliceDataset(
            self.args,
            self.temp_file_name,
            use_weak_label,
            self.entity_symbols,
            dataset_threads=1,
            split=split,
        )
        assert_slice_data_equal(gold_data, dataset.data)
        self.assertDictEqual(gold_sent_to_row_id_dict, dataset.sent_to_row_id_dict)


if __name__ == "__main__":
    unittest.main()
