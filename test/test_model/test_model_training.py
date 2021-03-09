import os
import shutil
import unittest

import numpy as np
import pytest
import torch

import emmental
from bootleg.layers.attn_networks import Bootleg
from bootleg.layers.bert_encoder import BertEncoder
from bootleg.symbols.constants import DISAMBIG, FINAL_LOSS, PRED_LAYER
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.tasks.ned_task import disambig_loss
from bootleg.tasks.task_getters import get_embedding_tasks
from bootleg.utils.parser import parser_utils


class ModelTrainTest(unittest.TestCase):
    def setUp(self):
        """ENTITY SYMBOLS
         {
           "multi word alias2":[["Q2",5.0],["Q1",3.0],["Q4",2.0]],
           "alias1":[["Q1",10.0],["Q4",6.0]],
           "alias3":[["Q1",30.0]],
           "alias4":[["Q4",20.0],["Q3",15.0],["Q2",1.0]]
         }
         EMBEDDINGS
         {
             "key": "learned",
             "freeze": false,
             "load_class": "LearnedEntityEmb",
             "args":
             {
               "learned_embedding_size": 10,
             }
         },
         {
            "key": "learned_type",
            "load_class": "LearnedTypeEmb",
            "freeze": false,
            "args": {
                "type_labels": "type_pred_mapping.json",
                "max_types": 1,
                "type_dim": 5,
                "merge_func": "addattn",
                "attn_hidden_size": 5
            }
        }
        """
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_model_training.json"
        )
        self.entity_symbols = EntitySymbols.load_from_cache(
            os.path.join(
                self.args.data_config.entity_dir, self.args.data_config.entity_map_dir
            ),
            alias_cand_map_file=self.args.data_config.alias_cand_map,
        )
        emmental.init(log_dir="test/temp_log")
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)

    def tearDown(self) -> None:
        dir = os.path.join("test/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir)

    def test_model_all_unfrozen(self):
        self.args.data_config.ent_embeddings[0].freeze = False
        self.args.data_config.ent_embeddings[1].freeze = False
        (
            task_flows,
            module_pool,
            module_device_dict,
            extra_bert_embedding_layers,
            to_add_to_payload,
            total_sizes,
        ) = get_embedding_tasks(self.args, self.entity_symbols)
        params_frozen = [
            np[0] for np in module_pool.named_parameters() if not np[1].requires_grad
        ]
        assert len(params_frozen) == 0

    def test_model_type_frozen(self):
        self.args.data_config.ent_embeddings[0].freeze = False
        self.args.data_config.ent_embeddings[1].freeze = True
        (
            task_flows,
            module_pool,
            module_device_dict,
            extra_bert_embedding_layers,
            to_add_to_payload,
            total_sizes,
        ) = get_embedding_tasks(self.args, self.entity_symbols)
        params_frozen = [
            np[0] for np in module_pool.named_parameters() if not np[1].requires_grad
        ]
        params_unfrozen = [
            np[0] for np in module_pool.named_parameters() if np[1].requires_grad
        ]
        assert len(params_unfrozen) > 0
        assert all([p.startswith("learned_type") for p in params_frozen])

    def test_model_word_frozen(self):
        self.args.data_config.word_embedding.freeze = True
        bert_model = BertEncoder(
            self.args.data_config.word_embedding,
            output_size=self.args.model_config.hidden_size,
        )
        # We don't want to freeze the sentence projection layer because it's randomly initialized
        params_unfrozen = [
            np[0]
            for np in bert_model.named_parameters()
            if np[1].requires_grad and "sent_proj" not in np[0]
        ]
        assert len(params_unfrozen) == 0

    # tests that weights are zero for padded words
    def test_sent_entity_attn_mask(self):
        self.args.data_config.ent_embeddings[0].freeze = False
        self.args.data_config.ent_embeddings[1].freeze = False
        self.args.data_config.word_embedding.freeze = False

        batch_size = self.args.train_config.batch_size
        N = self.args.data_config.max_seq_len
        M = self.args.data_config.max_aliases
        K = self.entity_symbols.max_candidates
        H = self.args.model_config.hidden_size

        model = Bootleg(self.args, self.entity_symbols)

        sent_indices = torch.ones(batch_size, N)
        # Pretend that last word of each sentence is padded
        sent_indices[:, -1] = 0
        sent_emb = torch.randn(batch_size, N, H)
        # Generate random mask; it doesn't matter as long as it's a bool
        sent_emb_mask = sent_indices == 0
        entity_embs = torch.randn(batch_size, M, K, H)
        entity_cand_eid_mask = torch.zeros(batch_size, M, K)
        start_span_idx = torch.zeros(batch_size, M)
        end_span_idx = torch.ones(batch_size, M)
        batch_on_the_fly_data = {}

        res = model(
            sent_emb,
            sent_emb_mask,
            entity_embs,
            entity_cand_eid_mask,
            start_span_idx,
            end_span_idx,
            batch_on_the_fly_data,
        )
        sent_entity_attn_weights = model.attention_weights[f"stage_0_entity_word"]
        # K is based on maximum entities seen (and can be smaller than max_entities set through args)
        # sent_entity_attn_weights is shape batch_size x (MxK) x max_sequence_len
        # we only care about the zeros, put 1's in non-zero values
        # 0's represent padded sequences
        expected_weights = torch.tensor(
            [
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                ]
            ]
        )
        expected_nonzero_weight_indices = expected_weights.nonzero()
        actual_nonzero_weight_indices = sent_entity_attn_weights.nonzero()
        assert torch.allclose(
            expected_nonzero_weight_indices, actual_nonzero_weight_indices
        )

    # tests that weights are zero for padded candidates
    # also tests that weights are zero for padded aliases
    def test_entity_attn_mask(self):
        self.args.data_config.ent_embeddings[0].freeze = False
        self.args.data_config.ent_embeddings[1].freeze = False
        self.args.data_config.word_embedding.freeze = False
        batch_size = self.args.train_config.batch_size
        N = self.args.data_config.max_seq_len
        M = self.args.data_config.max_aliases
        K = self.entity_symbols.max_candidates
        H = self.args.model_config.hidden_size

        model = Bootleg(self.args, self.entity_symbols)

        # Test 1: when all entity candidates are not -1, the mask is a block diagonal
        sent_indices = torch.ones(batch_size, N)
        # Pretend that last word of each sentence is padded
        sent_indices[:, -1] = 0
        sent_emb = torch.randn(batch_size, N, H)
        # Generate random mask; it doesn't matter as long as it's a bool
        sent_emb_mask = sent_indices == 0
        entity_embs = torch.randn(batch_size, M, K, H)
        entity_cand_eid_mask = torch.zeros(batch_size, M, K)
        start_span_idx = torch.zeros(batch_size, M)
        end_span_idx = torch.ones(batch_size, M)
        batch_on_the_fly_data = {}

        res = model(
            sent_emb,
            sent_emb_mask,
            entity_embs,
            entity_cand_eid_mask,
            start_span_idx,
            end_span_idx,
            batch_on_the_fly_data,
        )

        entity_attn_weights = model.attention_weights[f"stage_0_self_entity"]
        # K is based on maximum entities seen (and can be smaller than max_entities set through args)
        # entity_attn_weights is shape batch_size x (MxK) x (MxK)
        # we only care about the zeros, put 1's in non-zero values
        # 0's represent padded sequences

        expected_weights = torch.tensor(
            [
                [
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                ]
            ]
        )
        expected_nonzero_weight_indices = expected_weights.nonzero()
        actual_nonzero_weight_indices = entity_attn_weights.nonzero()
        assert torch.allclose(
            expected_nonzero_weight_indices, actual_nonzero_weight_indices
        )

        # Test 1: some candidates are -1, they get masked with a 1
        sent_indices = torch.ones(batch_size, N)
        # Pretend that last word of each sentence is padded
        sent_indices[:, -1] = 0
        sent_emb = torch.randn(batch_size, N, H)
        # Generate random mask; it doesn't matter as long as it's a bool
        sent_emb_mask = sent_indices == 0
        entity_embs = torch.randn(batch_size, M, K, H)
        entity_cand_eid_mask = torch.zeros(batch_size, M, K)
        entity_cand_eid_mask[:, 1, 2] = 1
        start_span_idx = torch.zeros(batch_size, M)
        end_span_idx = torch.ones(batch_size, M)
        batch_on_the_fly_data = {}

        res = model(
            sent_emb,
            sent_emb_mask,
            entity_embs,
            entity_cand_eid_mask.bool(),
            start_span_idx,
            end_span_idx,
            batch_on_the_fly_data,
        )

        entity_attn_weights = model.attention_weights[f"stage_0_self_entity"]
        # K is based on maximum entities seen (and can be smaller than max_entities set through args)
        # entity_attn_weights is shape batch_size x (MxK) x (MxK)
        # we only care about the zeros, put 1's in non-zero values
        # 0's represent padded sequences

        # the last col is 0 to indicate the -1 candidate for alias index 1 will not be added to any other candidate representation in the BMM
        expected_weights = torch.tensor(
            [
                [
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                ]
            ]
        )
        expected_nonzero_weight_indices = expected_weights.nonzero()
        actual_nonzero_weight_indices = entity_attn_weights.nonzero()
        assert torch.allclose(
            expected_nonzero_weight_indices, actual_nonzero_weight_indices
        )

    def test_eval_loss(self):
        # Testing if the disambig loss works if some labels are -2 during eval
        batch = 3
        M = 2
        K = 3
        training = torch.tensor([1]).bool()  # True
        bootleg_pred = {
            "final_scores": {DISAMBIG: {FINAL_LOSS: torch.randn(batch, M, K)}},
            "ent_embs": None,
            "training": training,
        }
        Y = torch.randint(0, 2, (batch, M))
        active = torch.ones(batch).bool()
        mask = torch.zeros(batch, M, K).bool()
        intermediate_output_dict = {
            PRED_LAYER: bootleg_pred,
            "_input_": {"entity_cand_eid_mask": mask},
        }
        # Checking that this passes under normal conditions
        res = disambig_loss(intermediate_output_dict, Y, active)
        assert type(res) is torch.Tensor

        Y[0, 0] = -2
        # Checking that it fails during training
        with pytest.raises(IndexError) as excinfo:
            res = disambig_loss(intermediate_output_dict, Y, active)
        assert "-2" in str(excinfo.value)

        # Checking that this passes under eval
        training = torch.tensor([0]).bool()  # False
        bootleg_pred = {
            "final_scores": {DISAMBIG: {FINAL_LOSS: torch.randn(batch, M, K)}},
            "ent_embs": None,
            "training": training,
        }
        intermediate_output_dict = {
            PRED_LAYER: bootleg_pred,
            "_input_": {"entity_cand_eid_mask": mask},
        }
        res = disambig_loss(intermediate_output_dict, Y, active)
        assert type(res) is torch.Tensor


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
