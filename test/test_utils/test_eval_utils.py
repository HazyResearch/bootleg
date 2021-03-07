import os
import shutil
import tempfile
import unittest

import jsonlines
import marisa_trie
import numpy as np
import torch
import ujson

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import eval_utils
from bootleg.utils.classes.dotted_dict import DottedDict
from bootleg.utils.eval_utils import check_and_create_alias_cand_trie, write_data_labels
from bootleg.utils.utils import create_single_item_trie


class EntitySymbolsSubclass(EntitySymbols):
    def __init__(self):
        self.max_candidates = 2
        # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
        self.max_alias_len = 1
        self._qid2title = {"Q1": "a b c d e", "Q2": "f", "Q3": "dd a b", "Q4": "x y z"}
        self._qid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        self._alias2qids = {
            "a": [["Q1", 10.0], ["Q4", 6]],
            "b": [["Q2", 5.0], ["Q1", 3]],
            "c": [["Q1", 30.0], ["Q2", 3]],
            "d": [["Q4", 20], ["Q3", 15.0]],
            "e": [["Q1", 10.0], ["Q4", 6]],
            "f": [["Q2", 5.0], ["Q1", 3]],
            "g": [["Q1", 30.0], ["Q2", 3]],
        }
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2
        self.alias_cand_map_file = "alias2qids.json"


class EvalUtils(unittest.TestCase):

    # tests if we match standard torch fns where expected
    def test_masked_class_logsoftmax_basic(self):
        # shape batch x M x K
        # model outputs
        preds = torch.tensor([[[2.0, 2.0, 1.0], [3.0, 5.0, 4.0]]])
        # all that matters for this test is that the below is non-negative
        # since negative indicates masking
        entity_ids = torch.tensor([[[1, 3, 4], [5, 3, 1]]])
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds, mask=mask)
        torch_logsoftmax = torch.nn.LogSoftmax(dim=2)
        torch_log_preds = torch_logsoftmax(preds)
        assert torch.allclose(torch_log_preds, pred_log_preds)

        # if we mask one of the candidates, we should no longer
        # get the same result as torch fn which doesn't mask
        entity_ids = torch.tensor([[[1, 3, 4], [5, 3, -1]]])
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds, mask=mask)
        assert not torch.allclose(torch_log_preds, pred_log_preds)
        # make sure masked values are approximately zero before log (when exponented)
        assert torch.allclose(
            torch.tensor([[[0.422319, 0.422319, 0.155362], [0.119203, 0.880797, 0.0]]]),
            torch.exp(pred_log_preds),
        )

    # combines with loss fn to see if we match torch cross entropy where expected
    def test_masked_class_logsoftmax_with_loss(self):
        # shape batch x M x K
        # model outputs
        preds = torch.tensor([[[2.0, 2.0, 1.0], [3.0, 5.0, 4.0]]])
        # all that matters for this test is that the below is non-negative
        # since negative indicates masking
        entity_ids = torch.tensor([[[1, 3, 4], [5, 3, 1]]])
        true_entity_class = torch.tensor([[0, 1]])
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        pred_log_preds = eval_utils.masked_class_logsoftmax(
            pred=preds, mask=mask
        ).transpose(1, 2)
        pred_loss = torch.nn.NLLLoss(ignore_index=-1)(pred_log_preds, true_entity_class)
        torch_loss_fn = torch.nn.CrossEntropyLoss()
        # predictions need to be batch_size x K x M
        torch_loss = torch_loss_fn(preds.transpose(1, 2), true_entity_class)
        assert torch.allclose(torch_loss, pred_loss)

    # tests if masking is done correctly
    def test_masked_class_logsoftmax_masking(self):
        preds = torch.tensor([[[2.0, 4.0, 1.0], [3.0, 5.0, 4.0]]])
        entity_ids = torch.tensor([[[1, 3, -1], [5, -1, -1]]])
        first_sample = torch.tensor([[2.0, 4.0]])
        denom_0 = torch.log(torch.sum(torch.exp(first_sample)))
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        # we only need to match on non-masked values
        expected_log_probs = torch.tensor(
            [
                [
                    [first_sample[0][0] - denom_0, first_sample[0][1] - denom_0, 0],
                    [0, 0, 0],
                ]
            ]
        )
        pred_log_preds = (
            eval_utils.masked_class_logsoftmax(pred=preds, mask=mask) * mask
        )
        assert torch.allclose(expected_log_probs, pred_log_preds)

    # check the case where the entire row is masked out
    def test_masked_class_logsoftmax_grads_full_mask(self):
        preds = torch.tensor([[[2.0, 4.0], [3.0, 5.0], [1.0, 4.0]]], requires_grad=True)
        # batch x M x K
        entity_ids = torch.tensor([[[1, -1], [-1, -1], [4, 5]]])
        # batch x M
        true_entity_class = torch.tensor([[0, -1, 1]])
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        pred_log_preds = eval_utils.masked_class_logsoftmax(
            pred=preds, mask=mask
        ).transpose(1, 2)
        pred_loss = torch.nn.NLLLoss(ignore_index=-1)(pred_log_preds, true_entity_class)
        pred_loss.backward()
        actual_grad = preds.grad
        true_entity_class_expanded = true_entity_class.unsqueeze(-1).expand_as(
            entity_ids
        )
        masked_actual_grad = torch.where(
            (entity_ids != -1) & (true_entity_class_expanded != -1),
            torch.ones_like(preds),
            actual_grad,
        )
        # just put 1's where we want non-zeros and use mask above to only compare padded gradients
        expected_grad = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]])
        assert torch.allclose(expected_grad, masked_actual_grad)

    # check the case where the entire row is masked out
    def test_masked_class_logsoftmax_grads_excluded_alias(self):
        preds = torch.tensor([[[2.0, 4.0], [1.0, 4.0], [8.0, 2.0]]], requires_grad=True)
        # batch x M x K
        entity_ids = torch.tensor([[[1, -1], [4, 5], [8, 9]]])
        # batch x M
        true_entity_class = torch.tensor([[0, -1, 1]])
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        pred_log_preds = eval_utils.masked_class_logsoftmax(
            pred=preds, mask=mask
        ).transpose(1, 2)
        pred_loss = torch.nn.NLLLoss(ignore_index=-1)(pred_log_preds, true_entity_class)
        pred_loss.backward()
        actual_grad = preds.grad
        true_entity_class_expanded = true_entity_class.unsqueeze(-1).expand_as(
            entity_ids
        )
        masked_actual_grad = torch.where(
            (entity_ids != -1) & (true_entity_class_expanded != -1),
            torch.ones_like(preds),
            actual_grad,
        )
        # just put 1's where we want non-zeros and use mask above to only compare padded gradients
        expected_grad = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]])
        assert torch.allclose(expected_grad, masked_actual_grad)

    # compare grads with and without masking
    def test_masked_class_logsoftmax_grads(self):
        # check gradients on preds since that will go back into the rest of the network
        preds = torch.tensor(
            [[[2.0, 4.0, 1.0], [3.0, 5.0, 4.0], [1.0, 4.0, 6.0]]], requires_grad=True
        )
        entity_ids = torch.tensor([[[1, 3, -1], [5, -1, -1], [4, 5, 6]]])
        true_entity_class = torch.tensor([[1, 0, 2]])
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        pred_log_preds = eval_utils.masked_class_logsoftmax(
            pred=preds, mask=mask
        ).transpose(1, 2)
        pred_loss = torch.nn.NLLLoss(ignore_index=-1)(pred_log_preds, true_entity_class)
        pred_loss.backward()
        actual_grad = preds.grad

        # we want zero grads on masked candidates
        masked_actual_grad = torch.where(
            entity_ids > 0, torch.ones_like(preds), actual_grad
        )
        # just put 1's where we want non-zeros and use mask above to only compare padded gradients
        expected_grad = torch.tensor(
            [[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
        )
        assert torch.allclose(expected_grad, masked_actual_grad)

        # we want to match pytorch when NOT using masking
        # zero out the gradient to call backward again
        preds.grad.zero_()

        # no masking now
        entity_ids = torch.tensor([[[1, 3, 1], [5, 4, 8], [4, 5, 6]]])
        true_entity_class = torch.tensor([[1, 0, 2]])
        mask = torch.where(
            entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds)
        )
        pred_log_preds = eval_utils.masked_class_logsoftmax(
            pred=preds, mask=mask
        ).transpose(1, 2)
        pred_loss = torch.nn.NLLLoss(ignore_index=-1)(pred_log_preds, true_entity_class)
        pred_loss.backward()
        # clone so we can call backward again and zero out the grad
        actual_grad = preds.grad.clone()
        preds.grad.zero_()

        torch_loss_fn = torch.nn.CrossEntropyLoss()
        torch_loss = torch_loss_fn(preds.transpose(1, 2), true_entity_class)
        torch_loss.backward()
        torch_grad = preds.grad
        assert torch.allclose(torch_grad, actual_grad)

    def test_select_embs(self):
        # final_entity_embs is batch x M x K x hidden_size, pred_cands in batch x M
        # batch size 2, M is 2, K is 3, hidden size is 4
        final_entity_embs = np.array(
            [
                [
                    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                    [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]],
                ],
                [
                    [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]],
                    [[110, 120, 130, 140], [150, 160, 170, 180], [190, 200, 210, 220]],
                ],
            ]
        )

        pred_cands = np.array([[0, 1], [2, 0]])
        gold_embs = np.array(
            [[[0, 1, 2, 3], [15, 16, 17, 18]], [[18, 19, 20, 21], [110, 120, 130, 140]]]
        )
        # compare outputs
        selected_embs = np.vstack(
            [
                eval_utils.select_embs(final_entity_embs[i], pred_cands[i], M=2)
                for i in range(gold_embs.shape[0])
            ]
        )
        assert np.array_equal(selected_embs, gold_embs)

    def test_merge_subsentences(self):

        test_full_emb_file = tempfile.NamedTemporaryFile()
        test_merged_emb_file = tempfile.NamedTemporaryFile()
        gold_merged_emb_file = tempfile.NamedTemporaryFile()
        cache_folder = tempfile.TemporaryDirectory()

        num_examples = 3
        total_num_mentions = 7
        M = 3
        K = 2
        hidden_size = 2

        # create full embedding file
        storage_type_full = np.dtype(
            [
                ("M", int),
                ("K", int),
                ("hidden_size", int),
                ("sent_idx", int),
                ("subsent_idx", int),
                ("alias_list_pos", int, M),
                ("entity_emb", float, M * hidden_size),
                ("final_loss_true", int, M),
                ("final_loss_pred", int, M),
                ("final_loss_prob", float, M),
                ("final_loss_cand_probs", float, M * K),
            ]
        )
        full_emb = np.memmap(
            test_full_emb_file.name,
            dtype=storage_type_full,
            mode="w+",
            shape=(num_examples,),
        )

        # 2 sentences, 1st sent has 1 subsentence, 2nd sentence has 2 subsentences
        # first sentence
        full_emb["hidden_size"] = hidden_size
        full_emb["M"] = M
        full_emb["K"] = K
        full_emb[0]["sent_idx"] = 0
        full_emb[0]["subsent_idx"] = 0
        # last alias is padded
        full_emb[0]["alias_list_pos"] = np.array([0, 1, -1])
        full_emb[0]["final_loss_true"] = np.array([0, 1, -1])
        # entity embs are flattened
        full_emb[0]["entity_emb"] = np.array([0, 1, 2, 3, 0, 0])

        full_emb[1]["sent_idx"] = 1
        full_emb[1]["subsent_idx"] = 0
        full_emb[1]["alias_list_pos"] = np.array([0, 1, 2])
        # last alias goes with next subsentence
        full_emb[1]["final_loss_true"] = np.array([1, 1, -1])
        full_emb[1]["entity_emb"] = np.array([4, 5, 6, 7, 8, 9])

        full_emb[2]["sent_idx"] = 1
        full_emb[2]["subsent_idx"] = 1
        full_emb[2]["alias_list_pos"] = np.array([2, 3, 4])
        full_emb[2]["final_loss_true"] = np.array([1, 1, 1])
        full_emb[2]["entity_emb"] = np.array([10, 11, 12, 13, 14, 15])

        # create merged embedding file
        storage_type_merged = np.dtype(
            [
                ("hidden_size", int),
                ("sent_idx", int),
                ("alias_list_pos", int),
                ("entity_emb", float, hidden_size),
                ("final_loss_pred", int),
                ("final_loss_prob", float),
                ("final_loss_cand_probs", float, K),
            ]
        )
        merged_emb_gold = np.memmap(
            gold_merged_emb_file.name,
            dtype=storage_type_merged,
            mode="w+",
            shape=(total_num_mentions,),
        )
        merged_emb_gold["entity_emb"] = np.array(
            [[0, 1], [2, 3], [4, 5], [6, 7], [10, 11], [12, 13], [14, 15]]
        )

        # create data file -- just needs aliases and sentence indices
        data = [
            {"aliases": ["a", "b"], "sent_idx_unq": 0},
            {"aliases": ["c", "d", "e", "f", "g"], "sent_idx_unq": 1},
        ]
        # Keys are string for trie
        sent_idx2num_mentions = {"0": 2, "1": 5}
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        with jsonlines.open(temp_file, "w") as f:
            for row in data:
                f.write(row)

        # assert that output of merge_subsentences is correct
        num_processes = 1

        eval_utils.merge_subsentences(
            num_processes,
            sent_idx2num_mentions,
            cache_folder.name,
            test_merged_emb_file.name,
            storage_type_merged,
            test_full_emb_file.name,
            storage_type_full,
            dump_embs=True,
        )
        bootleg_merged_emb = np.memmap(
            test_merged_emb_file.name, dtype=storage_type_merged, mode="r+"
        )
        merged_emb_gold = np.memmap(
            gold_merged_emb_file.name, dtype=storage_type_merged, mode="r+"
        )
        assert len(bootleg_merged_emb) == total_num_mentions
        for i in range(len(bootleg_merged_emb)):
            assert np.array_equal(
                bootleg_merged_emb[i]["entity_emb"], merged_emb_gold[i]["entity_emb"]
            )

        # Try with multiprocessing
        num_processes = 2
        eval_utils.merge_subsentences(
            num_processes,
            sent_idx2num_mentions,
            cache_folder.name,
            test_merged_emb_file.name,
            storage_type_merged,
            test_full_emb_file.name,
            storage_type_full,
            dump_embs=True,
        )
        bootleg_merged_emb = np.memmap(
            test_merged_emb_file.name, dtype=storage_type_merged, mode="r+"
        )
        merged_emb_gold = np.memmap(
            gold_merged_emb_file.name, dtype=storage_type_merged, mode="r+"
        )
        assert len(bootleg_merged_emb) == total_num_mentions
        for i in range(len(bootleg_merged_emb)):
            assert np.array_equal(
                bootleg_merged_emb[i]["entity_emb"], merged_emb_gold[i]["entity_emb"]
            )

        # clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        test_full_emb_file.close()
        test_merged_emb_file.close()
        gold_merged_emb_file.close()
        cache_folder.cleanup()

    def test_write_out_subsentences(self):

        merged_entity_emb_file = tempfile.NamedTemporaryFile()
        out_file = tempfile.NamedTemporaryFile()
        data_file = tempfile.NamedTemporaryFile()
        cache_folder = tempfile.TemporaryDirectory()

        entity_dir = "test/entity_db"
        entity_map_dir = "entity_mappings"
        alias_cand_map = "alias2qids.json"
        data_config = DottedDict(
            entity_dir=entity_dir,
            entity_map_dir=entity_map_dir,
            alias_cand_map=alias_cand_map,
        )
        entity_symbols = EntitySymbolsSubclass()
        entity_symbols.dump(save_dir=os.path.join(entity_dir, entity_map_dir))

        num_examples = 3
        total_num_mentions = 7
        M = 3
        K = 2
        hidden_size = 2

        # create data file -- just needs aliases and sentence indices
        data = [
            {"aliases": ["a", "b"], "sent_idx_unq": 0},
            {"aliases": ["c", "d", "e", "f", "g"], "sent_idx_unq": 1},
        ]
        # Dict is a string key for trie
        sent_idx2rows = {"0": data[0], "1": data[1]}
        with jsonlines.open(data_file.name, "w") as f:
            for row in data:
                f.write(row)

        merged_storage_type = np.dtype(
            [
                ("hidden_size", int),
                ("sent_idx", int),
                ("alias_list_pos", int),
                ("entity_emb", float, hidden_size),
                ("final_loss_pred", int),
                ("final_loss_prob", float),
                ("final_loss_cand_probs", float, K),
            ]
        )

        merged_entity_emb = np.memmap(
            merged_entity_emb_file.name,
            dtype=merged_storage_type,
            mode="w+",
            shape=(total_num_mentions,),
        )
        # 2 sentences, 1st sent has 1 subsentence, 2nd sentence has 2 subsentences - 7 mentions total
        merged_entity_emb["hidden_size"] = hidden_size
        # first men
        merged_entity_emb[0]["sent_idx"] = 0
        merged_entity_emb[0]["alias_list_pos"] = 0
        merged_entity_emb[0]["entity_emb"] = np.array([0, 1])
        merged_entity_emb[0]["final_loss_pred"] = 1
        merged_entity_emb[0]["final_loss_prob"] = 0.9
        merged_entity_emb[0]["final_loss_cand_probs"] = np.array([0.1, 0.9])
        # second men
        merged_entity_emb[1]["sent_idx"] = 0
        merged_entity_emb[1]["alias_list_pos"] = 1
        merged_entity_emb[1]["entity_emb"] = np.array([2, 3])
        merged_entity_emb[1]["final_loss_pred"] = 1
        merged_entity_emb[1]["final_loss_prob"] = 0.9
        merged_entity_emb[1]["final_loss_cand_probs"] = np.array([0.1, 0.9])
        # third men
        merged_entity_emb[2]["sent_idx"] = 1
        merged_entity_emb[2]["alias_list_pos"] = 0
        merged_entity_emb[2]["entity_emb"] = np.array([4, 5])
        merged_entity_emb[2]["final_loss_pred"] = 0
        merged_entity_emb[2]["final_loss_prob"] = 0.9
        merged_entity_emb[2]["final_loss_cand_probs"] = np.array([0.9, 0.1])
        # fourth men
        merged_entity_emb[3]["sent_idx"] = 1
        merged_entity_emb[3]["alias_list_pos"] = 1
        merged_entity_emb[3]["entity_emb"] = np.array([6, 7])
        merged_entity_emb[3]["final_loss_pred"] = 0
        merged_entity_emb[3]["final_loss_prob"] = 0.9
        merged_entity_emb[3]["final_loss_cand_probs"] = np.array([0.9, 0.1])
        # fifth men
        merged_entity_emb[4]["sent_idx"] = 1
        merged_entity_emb[4]["alias_list_pos"] = 2
        merged_entity_emb[4]["entity_emb"] = np.array([10, 11])
        merged_entity_emb[4]["final_loss_pred"] = 1
        merged_entity_emb[4]["final_loss_prob"] = 0.9
        merged_entity_emb[4]["final_loss_cand_probs"] = np.array([0.1, 0.9])
        # sixth men
        merged_entity_emb[5]["sent_idx"] = 1
        merged_entity_emb[5]["alias_list_pos"] = 3
        merged_entity_emb[5]["entity_emb"] = np.array([12, 13])
        merged_entity_emb[5]["final_loss_pred"] = 1
        merged_entity_emb[5]["final_loss_prob"] = 0.9
        merged_entity_emb[5]["final_loss_cand_probs"] = np.array([0.1, 0.9])
        # seventh men
        merged_entity_emb[6]["sent_idx"] = 1
        merged_entity_emb[6]["alias_list_pos"] = 4
        merged_entity_emb[6]["entity_emb"] = np.array([14, 15])
        merged_entity_emb[6]["final_loss_pred"] = 1
        merged_entity_emb[6]["final_loss_prob"] = 0.9
        merged_entity_emb[6]["final_loss_cand_probs"] = np.array([0.1, 0.9])

        num_processes = 1
        train_in_candidates = True
        dump_embs = True
        max_candidates = 2

        """
          "a":[["Q1",10.0],["Q4",6]],
          "b":[["Q2",5.0],["Q1",3]],
          "c":[["Q1",30.0],["Q2",3]],
          "d":[["Q4",20],["Q3",15.0]],
          "e":[["Q1",10.0],["Q4",6]],
          "f":[["Q2",5.0],["Q1",3]],
          "g":[["Q1",30.0],["Q2",3]]
        """

        gold_lines = [
            {
                "sent_idx_unq": 0,
                "aliases": ["a", "b"],
                "qids": ["Q4", "Q1"],
                "probs": [0.9, 0.9],
                "cands": [["Q1", "Q4"], ["Q2", "Q1"]],
                "cand_probs": [[0.1, 0.9], [0.1, 0.9]],
                "entity_ids": [4, 1],
                "ctx_emb_ids": [0, 1],
            },
            {
                "sent_idx_unq": 1,
                "aliases": ["c", "d", "e", "f", "g"],
                "qids": ["Q1", "Q4", "Q4", "Q1", "Q2"],
                "probs": [0.9, 0.9, 0.9, 0.9, 0.9],
                "cands": [
                    ["Q1", "Q2"],
                    ["Q4", "Q3"],
                    ["Q1", "Q4"],
                    ["Q2", "Q1"],
                    ["Q1", "Q2"],
                ],
                "cand_probs": [
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.1, 0.9],
                    [0.1, 0.9],
                    [0.1, 0.9],
                ],
                "entity_ids": [1, 4, 4, 1, 2],
                "ctx_emb_ids": [2, 3, 4, 5, 6],
            },
        ]

        write_data_labels(
            num_processes=num_processes,
            merged_entity_emb_file=merged_entity_emb_file.name,
            merged_storage_type=merged_storage_type,
            sent_idx2row=sent_idx2rows,
            cache_folder=cache_folder.name,
            out_file=out_file.name,
            entity_dump=entity_symbols,
            train_in_candidates=train_in_candidates,
            max_candidates=max_candidates,
            dump_embs=dump_embs,
            trie_candidate_map_folder=None,
            trie_qid2eid_file=None,
        )
        all_lines = []
        with open(out_file.name) as check_f:
            for line in check_f:
                all_lines.append(ujson.loads(line))

        assert len(all_lines) == len(gold_lines)

        all_lines_sent_idx_map = {line["sent_idx_unq"]: line for line in all_lines}
        gold_lines_sent_idx_map = {line["sent_idx_unq"]: line for line in gold_lines}

        assert len(all_lines_sent_idx_map) == len(gold_lines_sent_idx_map)
        for sent_idx in all_lines_sent_idx_map:
            self.assertDictEqual(
                gold_lines_sent_idx_map[sent_idx],
                all_lines_sent_idx_map[sent_idx],
                f"{ujson.dumps(gold_lines_sent_idx_map[sent_idx], indent=4)} VS {ujson.dumps(all_lines_sent_idx_map[sent_idx], indent=4)}",
            )

        # TRY MULTIPROCESSING
        num_processes = 2
        # create memmory files for multiprocessing
        trie_candidate_map_folder = tempfile.TemporaryDirectory()
        trie_qid2eid_file = tempfile.NamedTemporaryFile()
        create_single_item_trie(
            entity_symbols.get_qid2eid(), out_file=trie_qid2eid_file.name
        )
        check_and_create_alias_cand_trie(trie_candidate_map_folder.name, entity_symbols)

        write_data_labels(
            num_processes=num_processes,
            merged_entity_emb_file=merged_entity_emb_file.name,
            merged_storage_type=merged_storage_type,
            sent_idx2row=sent_idx2rows,
            cache_folder=cache_folder.name,
            out_file=out_file.name,
            entity_dump=entity_symbols,
            train_in_candidates=train_in_candidates,
            max_candidates=max_candidates,
            dump_embs=dump_embs,
            trie_candidate_map_folder=trie_candidate_map_folder.name,
            trie_qid2eid_file=trie_qid2eid_file.name,
        )

        all_lines = []
        with open(out_file.name) as check_f:
            for line in check_f:
                all_lines.append(ujson.loads(line))

        assert len(all_lines) == len(gold_lines)

        all_lines_sent_idx_map = {line["sent_idx_unq"]: line for line in all_lines}
        gold_lines_sent_idx_map = {line["sent_idx_unq"]: line for line in gold_lines}

        assert len(all_lines_sent_idx_map) == len(gold_lines_sent_idx_map)
        for sent_idx in all_lines_sent_idx_map:
            self.assertDictEqual(
                gold_lines_sent_idx_map[sent_idx],
                all_lines_sent_idx_map[sent_idx],
                f"{ujson.dumps(gold_lines_sent_idx_map[sent_idx], indent=4)} VS {ujson.dumps(all_lines_sent_idx_map[sent_idx], indent=4)}",
            )

        # clean up
        if os.path.exists(entity_dir):
            shutil.rmtree(entity_dir, ignore_errors=True)
        merged_entity_emb_file.close()
        out_file.close()
        data_file.close()
        trie_candidate_map_folder.cleanup()
        cache_folder.cleanup()
        trie_qid2eid_file.close()
