import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
import copy
import os

from bootleg.dataloaders.wiki_slices import WikiSlices
from bootleg.dataloaders.wiki_dataset import WikiDataset
from bootleg.utils import data_utils, parser_utils, eval_utils, train_utils
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.trainer import Trainer

class ModelTrainTest(unittest.TestCase):

    def setUp(self):
        self.args = parser_utils.get_full_config("test/run_args/test_model_training.json")
        train_utils.setup_train_heads_and_eval_slices(self.args)
        self.word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        self.entity_symbols = EntitySymbols(os.path.join(
            self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir, data_utils.generate_save_data_name(
                data_args=self.args.data_config, use_weak_label=True, split_name="slice_train")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        self.data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir, data_utils.generate_save_data_name(
                data_args=self.args.data_config, use_weak_label=False, split_name="train")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=self.word_symbols,
            entity_symbols=self.entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )
        self.trainer = Trainer(self.args, self.entity_symbols, self.word_symbols)

    def test_load_and_save(self):
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "test.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="test")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )

        self.args.data_config.overwrite_preprocessed_data = False
        slices2 = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "test.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="test")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        np.testing.assert_array_equal(slices.data, slices2.data)


    def test_model_update_weights(self):
        torch.manual_seed(0)
        np.random.seed(0)
        batches = DataLoader(self.data, batch_size=1, shuffle=False, pin_memory=True)
        params = [ np for np in self.trainer.model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]
        # Make sure learned embeddings is one of the ones that should change
        self.assertTrue("emb_layer.entity_embs.learned.learned_entity_embedding.weight" in [p[0] for p in initial_params])
        params_frozen = [ np for np in self.trainer.model.named_parameters() if not np[1].requires_grad ]
        initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]
        for batch in batches:
            self.trainer.update(batch, eval=False)

        # Make sure weights change that should
        params = [ np for np in self.trainer.model.named_parameters() if np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params, params):
            assert not torch.equal(p0, p1)

        # Make sure weights for pretrained embeddings don't change
        params_frozen = [ np for np in self.trainer.model.named_parameters() if not np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
            assert ("word_emb" in name) or ("sent_emb" in name)
            assert torch.equal(p0, p1)


    def test_model_update_words(self):
        torch.manual_seed(0)
        np.random.seed(0)
        args = copy.deepcopy(self.args)
        args.data_config.word_embedding.freeze_word_emb = False
        args.data_config.word_embedding.freeze_sent_emb = False
        batches = DataLoader(self.data, batch_size=1, shuffle=False, pin_memory=True)
        trainer = Trainer(args, self.entity_symbols, self.word_symbols)

        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]

        self.assertTrue("emb_layer.word_emb.word_embedding.weight" in [p[0] for p in initial_params])
        self.assertTrue("emb_layer.sent_emb.attention_modules.stage_0_self_sentence.attn.norm.weight" in [p[0] for p in initial_params])

        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]

        for batch in batches:
            trainer.update(batch, eval=False)
            break

        # Make sure weights change that should
        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params, params):
            print("NAME", name)
            assert not torch.equal(p0, p1)
        # Make sure weights for pretrained embeddings don't change
        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
            assert torch.equal(p0, p1)


    def test_model_update_frozen_words(self):
        torch.manual_seed(0)
        np.random.seed(0)
        args = copy.deepcopy(self.args)
        args.data_config.word_embedding.freeze_word_emb = True
        args.data_config.word_embedding.freeze_sent_emb = False
        batches = DataLoader(self.data, batch_size=1, shuffle=False, pin_memory=True)
        trainer = Trainer(args, self.entity_symbols, self.word_symbols)

        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]
        self.assertTrue("emb_layer.sent_emb.attention_modules.stage_0_self_sentence.attn.norm.weight" in [p[0] for p in initial_params])

        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]
        self.assertTrue("emb_layer.word_emb.word_embedding.weight" in [p[0] for p in initial_params_frozen])

        for batch in batches:
            trainer.update(batch, eval=False)
            break

        # Make sure weights change that should
        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params, params):
            assert not torch.equal(p0, p1)
        # Make sure weights for pretrained embeddings don't change
        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
            assert torch.equal(p0, p1)


    def test_model_update_frozen_sent(self):
        torch.manual_seed(0)
        np.random.seed(0)
        args = copy.deepcopy(self.args)
        args.data_config.word_embedding.freeze_word_emb = False
        args.data_config.word_embedding.freeze_sent_emb = True
        batches = DataLoader(self.data, batch_size=1, shuffle=False, pin_memory=True)
        trainer = Trainer(args, self.entity_symbols, self.word_symbols)

        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]
        self.assertTrue("emb_layer.word_emb.word_embedding.weight" in [p[0] for p in initial_params])

        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]
        self.assertTrue("emb_layer.sent_emb.attention_modules.stage_0_self_sentence.attn.norm.weight" in [p[0] for p in initial_params_frozen])

        for batch in batches:
            trainer.update(batch, eval=False)
            break

        # Make sure weights change that should
        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params, params):
            print("NAME", name)
            assert not torch.equal(p0, p1)
        # Make sure weights for pretrained embeddings don't change
        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
            assert torch.equal(p0, p1)


    def test_model_fine_tune(self):
        torch.manual_seed(0)
        np.random.seed(0)
        args = copy.deepcopy(self.args)
        args.data_config.word_embedding.freeze_word_emb = True
        args.data_config.word_embedding.freeze_sent_emb = True
        # freeze word and entity embeddings
        for ent_emb in args.data_config.ent_embeddings:
            ent_emb.freeze = True

        batches = DataLoader(self.data, batch_size=1, shuffle=False, pin_memory=True)
        trainer = Trainer(args, self.entity_symbols, self.word_symbols)

        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]

        # assert that some parameters are being tuned
        self.assertTrue(len(initial_params) > 0)
        assert all([('attn_network' == p[0].split(".")[0] or 'slice_heads' == p[0].split(".")[0] or 'emb_combiner' == p[0].split(".")[0] or 'emb_layer' == p[0].split(".")[0]) for p in initial_params])
        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]
        # assert key embedding parameters are not being updated
        self.assertTrue("emb_layer.word_emb.word_embedding.weight" in [p[0] for p in initial_params_frozen])
        # this is just asserting a single attention layer weight is in there; if one is in there, they all should be in there
        self.assertTrue("emb_layer.sent_emb.attention_modules.stage_0_self_sentence.attn.norm.weight" in [p[0] for p in initial_params_frozen])
        self.assertTrue("emb_layer.entity_embs.learned.learned_entity_embedding.weight" in [p[0] for p in initial_params_frozen])
        assert all(['emb_layer' == p[0].split(".")[0] for p in initial_params_frozen])

        for batch in batches:
            trainer.update(batch, eval=False)
            break

        # Make sure weights change that should
        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        assert len(params) == len(initial_params)
        for (_, p0), (name, p1) in zip(initial_params, params):
            assert not torch.equal(p0, p1)

        # Make sure weights for pretrained embeddings don't change
        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        assert len(params_frozen) == len(initial_params_frozen)
        for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
            assert torch.equal(p0, p1)


    def test_model_reverse_fine_tune(self):
        torch.manual_seed(0)
        np.random.seed(0)
        args = copy.deepcopy(self.args)
        args.data_config.word_embedding.freeze_word_emb = False
        args.data_config.word_embedding.freeze_sent_emb = False
        for emb in args.data_config.ent_embeddings:
            emb.freeze = False

        batches = DataLoader(self.data, batch_size=1, shuffle=False, pin_memory=True)
        trainer = Trainer(args, self.entity_symbols, self.word_symbols)

        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]

        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]

        self.assertTrue(len(initial_params) > 0)
        assert all(['attn_network' == p[0].split(".")[0] or 'emb_combiner' == p[0].split(".")[0] for p in initial_params_frozen])
        # assert key embedding parameters are not being updated
        self.assertTrue("emb_layer.word_emb.word_embedding.weight" in [p[0] for p in initial_params])
        # this is just asserting a single attention layer weight is in there; if one is in there, they all should be in there
        self.assertTrue("emb_layer.sent_emb.attention_modules.stage_0_self_sentence.attn.norm.weight" in [p[0] for p in initial_params])
        self.assertTrue("emb_layer.entity_embs.learned.learned_entity_embedding.weight" in [p[0] for p in initial_params])
        # slice heads and embedding layer are not frozen
        assert all([('attn_network' == p[0].split(".")[0]) or ('emb_layer' == p[0].split(".")[0]) or ('slice_heads' == p[0].split(".")[0]) or ('emb_combiner' == p[0].split(".")[0]) for p in initial_params])

        for batch in batches:
            trainer.update(batch, eval=False)
            break

        # Make sure weights change that should
        params = [ np for np in trainer.model.named_parameters() if np[1].requires_grad ]
        assert len(params) == len(initial_params)
        for (_, p0), (name, p1) in zip(initial_params, params):
            assert not torch.equal(p0, p1)

        # Make sure weights for pretrained embeddings don't change
        params_frozen = [ np for np in trainer.model.named_parameters() if not np[1].requires_grad ]
        assert len(params_frozen) == len(initial_params_frozen)
        for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
            assert torch.equal(p0, p1)


    # move masked_class_logsoftmax to utils?
    # (doesn't really require train class or other data/model/entity classes)
    # tests if we match standard torch fns where expected
    def test_masked_class_logsoftmax_basic(self):
        # shape batch x M x K
        # model outputs
        preds = torch.tensor([[[2., 2., 1.], [3., 5., 4.]]])
        # all that matters for this test is that the below is non-negative
        # since negative indicates masking
        entity_ids = torch.tensor([[[1, 3, 4], [5, 3, 1]]])
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask)
        torch_logsoftmax = torch.nn.LogSoftmax(dim=2)
        torch_log_preds = torch_logsoftmax(preds)
        assert torch.allclose(torch_log_preds, pred_log_preds)

        # if we mask one of the candidates, we should no longer
        # get the same result as torch fn which doesn't mask
        entity_ids = torch.tensor([[[1, 3, 4], [5, 3, -1]]])
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask)
        assert not torch.allclose(torch_log_preds, pred_log_preds)
        # make sure masked values are approximately zero before log (when exponented)
        assert torch.allclose(torch.tensor([[[0.422319, 0.422319, 0.155362],[0.119203, 0.880797, 0.]]]),
            torch.exp(pred_log_preds))


    # combines with loss fn to see if we match torch cross entropy where expected
    def test_masked_class_logsoftmax_with_loss(self):
        # shape batch x M x K
        # model outputs
        preds = torch.tensor([[[2., 2., 1.], [3., 5., 4.]]])
        # all that matters for this test is that the below is non-negative
        # since negative indicates masking
        entity_ids = torch.tensor([[[1, 3, 4], [5, 3, 1]]])
        true_entity_class = torch.tensor([[0,1]])
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask).transpose(1,2)
        pred_loss = self.trainer.scorer.crit_pred(pred_log_preds, true_entity_class)
        torch_loss_fn = torch.nn.CrossEntropyLoss()
        # predictions need to be batch_size x K x M
        torch_loss = torch_loss_fn(preds.transpose(1,2), true_entity_class)
        assert torch.allclose(torch_loss, pred_loss)


    # tests if masking is done correctly
    def test_masked_class_logsoftmax_masking(self):
        preds = torch.tensor([[[2., 4., 1.], [3., 5., 4.]]])
        entity_ids = torch.tensor([[[1, 3, -1], [5, -1, -1]]])
        first_sample = torch.tensor([[2., 4.]])
        denom_0 = torch.log(torch.sum(torch.exp(first_sample)))
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        # we only need to match on non-masked values
        expected_log_probs = torch.tensor([[[
            first_sample[0][0]-denom_0, first_sample[0][1]-denom_0, 0],
            [0, 0, 0]]])
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask) * mask
        assert torch.allclose(expected_log_probs, pred_log_preds)


    # check the case where the entire row is masked out
    def test_masked_class_logsoftmax_grads_full_mask(self):
        preds = torch.tensor([[[2., 4.], [3., 5.], [1., 4.]]], requires_grad=True)
        # batch x M x K
        entity_ids = torch.tensor([[[1, -1], [-1, -1], [4, 5]]])
        # batch x M
        true_entity_class = torch.tensor([[0, -1, 1]])
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask).transpose(1,2)
        pred_loss = self.trainer.scorer.crit_pred(pred_log_preds, true_entity_class)
        pred_loss.backward()
        actual_grad = preds.grad
        true_entity_class_expanded = true_entity_class.unsqueeze(-1).expand_as(entity_ids)
        masked_actual_grad = torch.where((entity_ids != -1) & (true_entity_class_expanded != -1), torch.ones_like(preds), actual_grad)
        # just put 1's where we want non-zeros and use mask above to only compare padded gradients
        expected_grad = torch.tensor([[[1., 0.], [0., 0.], [1., 1.]]])
        assert torch.allclose(expected_grad, masked_actual_grad)


     # check the case where the entire row is masked out
    def test_masked_class_logsoftmax_grads_excluded_alias(self):
        preds = torch.tensor([[[2., 4.], [1., 4.], [8., 2.]]], requires_grad=True)
        # batch x M x K
        entity_ids = torch.tensor([[[1, -1], [4, 5], [8, 9]]])
        # batch x M
        true_entity_class = torch.tensor([[0, -1, 1]])
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask).transpose(1,2)
        pred_loss = self.trainer.scorer.crit_pred(pred_log_preds, true_entity_class)
        pred_loss.backward()
        actual_grad = preds.grad
        true_entity_class_expanded = true_entity_class.unsqueeze(-1).expand_as(entity_ids)
        masked_actual_grad = torch.where((entity_ids != -1) & (true_entity_class_expanded != -1), torch.ones_like(preds), actual_grad)
        # just put 1's where we want non-zeros and use mask above to only compare padded gradients
        expected_grad = torch.tensor([[[1., 0.], [0., 0.], [1., 1.]]])
        assert torch.allclose(expected_grad, masked_actual_grad)


    # compare grads with and without masking
    def test_masked_class_logsoftmax_grads(self):
        # check gradients on preds since that will go back into the rest of the network
        preds = torch.tensor([[[2., 4., 1.], [3., 5., 4.], [1., 4., 6.]]], requires_grad=True)
        entity_ids = torch.tensor([[[1, 3, -1], [5, -1, -1], [4, 5, 6]]])
        true_entity_class = torch.tensor([[1, 0, 2]])
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask).transpose(1,2)
        pred_loss = self.trainer.scorer.crit_pred(pred_log_preds, true_entity_class)
        pred_loss.backward()
        actual_grad = preds.grad

        # we want zero grads on masked candidates
        masked_actual_grad = torch.where(entity_ids > 0, torch.ones_like(preds), actual_grad)
        # just put 1's where we want non-zeros and use mask above to only compare padded gradients
        expected_grad = torch.tensor([[[1., 1., 0.], [1., 0., 0.], [1., 1., 1.]]])
        assert torch.allclose(expected_grad, masked_actual_grad)

        # we want to match pytorch when NOT using masking
        # zero out the gradient to call backward again
        preds.grad.zero_()

        # no masking now
        entity_ids = torch.tensor([[[1, 3, 1], [5, 4, 8], [4, 5, 6]]])
        true_entity_class = torch.tensor([[1, 0, 2]])
        mask = torch.where(entity_ids < 0, torch.zeros_like(preds), torch.ones_like(preds))
        pred_log_preds = eval_utils.masked_class_logsoftmax(pred=preds,
            mask=mask).transpose(1,2)
        pred_loss = self.trainer.scorer.crit_pred(pred_log_preds, true_entity_class)
        pred_loss.backward()
        # clone so we can call backward again and zero out the grad
        actual_grad = preds.grad.clone()
        preds.grad.zero_()

        torch_loss_fn = torch.nn.CrossEntropyLoss()
        torch_loss = torch_loss_fn(preds.transpose(1,2), true_entity_class)
        torch_loss.backward()
        torch_grad = preds.grad
        assert torch.allclose(torch_grad, actual_grad)


    # tests that weights are zero for padded words
    def test_sent_attn_mask(self):
        batches = DataLoader(self.data, batch_size=len(self.data), shuffle=False, pin_memory=True)
        # only one batch
        batch = iter(batches).next()
        self.trainer.update(batch, eval=False)
        sent_attn_weights = self.trainer.model.emb_layer.sent_emb.attention_weights["layer_0_sent"]
        # sent_attn_weights is shape batch_size x max_sequence_len x max_sequence_len
        # we only care about the zeros, put 1's in non-zero values
        # 0's represent padded sequences
        expected_weights = torch.tensor([[[1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0]],

                                         [[1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0]],

                                         [[1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0]]])
        expected_nonzero_weight_indices = expected_weights.nonzero()
        actual_nonzero_weight_indices = sent_attn_weights.nonzero()
        assert torch.allclose(expected_nonzero_weight_indices, actual_nonzero_weight_indices)


    # tests that weights are zero for padded words
    def test_sent_entity_attn_mask(self):
        batches = DataLoader(self.data, batch_size=len(self.data), shuffle=False, pin_memory=True)
        # only one batch
        batch = iter(batches).next()
        self.trainer.update(batch, eval=False)
        sent_entity_attn_weights = self.trainer.model.attn_network.attention_weights[f"stage_0_entity_word"]
        # K is based on maximum entities seen (and can be smaller than max_entities set through args)
        # sent_entity_attn_weights is shape batch_size x (MxK) x max_sequence_len
        # we only care about the zeros, put 1's in non-zero values
        # 0's represent padded sequences
        expected_weights = torch.tensor([[[1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1, 0]],

                                         [[1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0]],

                                         [[1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0],
                                          [1, 1, 1, 1, 0, 0]]])
        expected_nonzero_weight_indices = expected_weights.nonzero()
        actual_nonzero_weight_indices = sent_entity_attn_weights.nonzero()
        assert torch.allclose(expected_nonzero_weight_indices, actual_nonzero_weight_indices)


    # tests that weights are zero for padded candidates
    # also tests that weights are zero for padded aliases
    def test_entity_attn_mask(self):
        batches = DataLoader(self.data, batch_size=len(self.data), shuffle=False, pin_memory=True)
        # only one batch
        batch = iter(batches).next()
        self.trainer.update(batch, eval=False)
        entity_attn_weights = self.trainer.model.attn_network.attention_weights[f"stage_0_self_entity"]
        # K is based on maximum entities seen (and can be smaller than max_entities set through args)
        # entity_attn_weights is shape batch_size x (MxK) x (MxK)
        # we only care about the zeros, put 1's in non-zero values
        # 0's represent padded sequences
        expected_weights = torch.tensor([[[0, 0, 0, 1, 1, 1], # [:3] correspond to K candidates in "alias1"; [3:6] to "multi word alias2"
                                          [0, 0, 0, 1, 1, 1],
                                          [0, 0, 0, 1, 1, 1],
                                          [1, 1, 0, 0, 0, 0], # [:3] correspond to K candidates in "alias1"; [3:6] to "multi word alias2"
                                          [1, 1, 0, 0, 0, 0],
                                          [1, 1, 0, 0, 0, 0]],

                                         [[0, 0, 0, 1, 1, 1],
                                          [0, 0, 0, 1, 1, 1],
                                          [0, 0, 0, 1, 1, 1],
                                          [1, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0],
                                          [1, 0, 0, 0, 0, 0]],

                                        # there's only a single alias here so weights are evenly distributed across candidates
                                         [[1, 1, 1, 0, 0, 0], # there is only one alias here, so [3:6] should be padded
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0],
                                          [1, 1, 1, 0, 0, 0]]])
        expected_nonzero_weight_indices = expected_weights.nonzero()
        actual_nonzero_weight_indices = entity_attn_weights.nonzero()
        assert torch.allclose(expected_nonzero_weight_indices, actual_nonzero_weight_indices)

# TODO: add tests that check the loss fn with collective disambig dimensions

if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()

'''
'''
