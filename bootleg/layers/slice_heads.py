"""Slice (prediction) heads"""

import torch.nn.functional as F

from bootleg.layers.layers import *
from bootleg.layers.alias_attn import AliasMHA
from bootleg.symbols.constants import *
from bootleg.utils import eval_utils, train_utils, logging_utils, model_utils

class NoSliceHeads(nn.Module):
    """
    No slice heads class. Generates batch x M x K matrix of scores for each mention and candidate to be passed to scorer.
    """
    def __init__(self, args, entity_symbols):
        super(NoSliceHeads, self).__init__()
        self.hidden_size = args.model_config.hidden_size
        self.num_fc_layers = args.model_config.num_fc_layers
        self.dropout = args.train_config.dropout
        self.prediction_head = MLP(self.hidden_size,
                self.hidden_size, 1, 1, self.dropout)

    def forward(self, context_matrix_dict, alias_idx_pair_sent, entity_pack, sent_emb):
        out = {DISAMBIG: {}}
        score = model_utils.max_score_context_matrix(context_matrix_dict, self.prediction_head)
        out[DISAMBIG][FINAL_LOSS] = score
        if "context_matrix_main" not in context_matrix_dict:
            context_matrix_dict["context_matrix_main"] = model_utils.generate_final_context_matrix(context_matrix_dict, ending_key_to_exclude="_nokg")
        return out, context_matrix_dict["context_matrix_main"]


class SliceHeadsSBL(nn.Module):
    """
    Slice heads class (modified from https://github.com/snorkel-team/snorkel/tree/master/snorkel/slicing).

    Implements slice-based learning module which acts as a cheap mixture of experts (https://arxiv.org/abs/1909.06349). Each user defined slice
    gets its own extra representation for specialize on a slice. There is also a base slice that is all examples. The representaitons from
    each head are merged and sent through MLP to be scored for final loss.

    Attributes:
        use_ind_attn: whether to use a modified indicator attention for combining the slice heads that ignores confidences
        remove_final_loss: whether to not use the final loss head (set to to be the base head) --- useful for debugging
    """
    def __init__(self, args, entity_symbols):
        super(SliceHeadsSBL, self).__init__()
        self.logger = logging_utils.get_logger(args)
        self.dropout = args.train_config.dropout
        if "use_ind_attn" in args.model_config.custom_args and args.model_config.custom_args.use_ind_attn:
            self.logger.info('Using attention only over indicator confidences')
            self.use_ind_attn = args.model_config.custom_args.use_ind_attn
        else:
            self.use_ind_attn = False

        # Debugging parameter to see what happens when we remove the "final_loss" merging head
        if "remove_final_loss" in args.model_config.custom_args:
            self.remove_final_loss = args.model_config.custom_args.remove_final_loss
        else:
            self.remove_final_loss = False
        self.logger.info(f"Remove final loss: {self.remove_final_loss} and sen's trick {self.use_ind_attn}")

        # Softmax temperature
        self.temperature = args.train_config.softmax_temp
        self.hidden_size = args.model_config.hidden_size
        self.K = entity_symbols.max_candidates + (not args.data_config.train_in_candidates)
        self.M = args.data_config.max_aliases
        self.train_heads = args.train_config.train_heads

        # Predicts whether or not an example is in the slice
        self.indicator_heads = nn.ModuleDict()
        self.ind_alias_mha = nn.ModuleDict()

        # Creates slice expert representations
        self.transform_modules = nn.ModuleDict()

        for slice_head in self.train_heads:
            # Generates a BxMxH representation that gets fed to the linear layer for predictions
            # This does an attention over the alias word and sentence (with added alias-slice learned embedding) and
            # then does an attention over the alias candidates (with added entity-slice learned embedding)
            self.ind_alias_mha[slice_head] = AliasMHA(args)
            # Binary prediction of in the slice or not
            self.indicator_heads[slice_head] = nn.Linear(self.hidden_size, 2)
            # transform layer for each slice
            self.transform_modules[slice_head] = nn.Linear(self.hidden_size, self.hidden_size)

        # Shared prediction layer to get confidences
        self.shared_slice_pred_head = nn.Linear(self.hidden_size, 1)

        # Embedding for each slice head for the indicator (added to queries in the AliasMHA heads)
        self.slice_emb_ind_alias = nn.Embedding(len(self.train_heads), self.hidden_size)
        self.slice_emb_ind_ent = nn.Embedding(len(self.train_heads), self.hidden_size)

        # Final prediction layer
        self.final_pred_head = nn.Linear(self.hidden_size, 1)

    def forward(self, context_matrix_dict, alias_idx_pair_sent, entity_pack, sent_emb):
        out = {DISAMBIG: {}, INDICATOR: {}}
        indicator_outputs = {}
        expert_slice_repr = {}
        predictor_outputs = {}

        if "context_matrix_main" not in context_matrix_dict:
            context_matrix_dict["context_matrix_main"] = model_utils.generate_final_context_matrix(context_matrix_dict)

        context_matrix = context_matrix_dict["context_matrix_main"]

        batch_size, M, K, H = context_matrix.shape
        assert M == self.M
        assert K == self.K
        assert H == self.hidden_size
        for i, slice_head in enumerate(self.train_heads):
            # Generate slice expert representation per head
            # context_matrix is batch x M x K x H
            expert_slice_repr[slice_head] = self.transform_modules[slice_head](context_matrix)
            # Pass the expert slice representation through the shared prediction layer
            # Predictor_outputs is batch x M x K
            predictor_outputs[slice_head] = self.shared_slice_pred_head(
                expert_slice_repr[slice_head]).squeeze(-1)
            # Get an alias_matrix output (batch x M x H)
            # TODO: remove extra inputs
            alias_matrix, alias_word_weights = self.ind_alias_mha[slice_head](sent_embedding=sent_emb, entity_embedding=context_matrix,
                entity_mask=entity_pack.mask, alias_idx_pair_sent=alias_idx_pair_sent,
                slice_emb_alias=self.slice_emb_ind_alias(torch.tensor(i, device=context_matrix.device)),
                slice_emb_ent=self.slice_emb_ind_ent(torch.tensor(i, device=context_matrix.device)))
            # Get indicator head outputs; size batch x M x 2 per head
            indicator_outputs[slice_head] = self.indicator_heads[slice_head](alias_matrix)

        # Generate predictions via softmax + taking the "positive" class label
        # Output size is batch x M x num_slices
        indicator_preds = torch.cat(
            [
                F.log_softmax(indicator_outputs[slice_head], dim=-1)[:,:,1].unsqueeze(-1)
                for slice_head in self.train_heads
            ],
            dim=-1,
        )
        assert not torch.isnan(indicator_preds).any()
        assert not torch.isinf(indicator_preds).any()
        # Compute the "confidence"
        # Output size should be batch x M x K x num_slices
        predictor_confidences = torch.cat(
            [
                eval_utils.masked_class_logsoftmax(pred=predictor_outputs[slice_head],
                mask=~entity_pack.mask).unsqueeze(-1)
                for slice_head in self.train_heads
            ],
            dim=-1,
        )
        assert not torch.isnan(predictor_confidences).any()
        assert not torch.isinf(predictor_confidences).any()
        if self.use_ind_attn:
            prod = indicator_preds # * margin_confidence
            prod[prod<0.1] = -100000.0 * self.temperature
            attention_weights = F.softmax(prod / self.temperature, dim=-1)
        else:
            # Take margin confidence over K values to generate confidences of batch x M x num_slices
            vals, indices = torch.topk(predictor_confidences, k=2, dim=2)
            margin_confidence = (vals[:, :, 0, :] - vals[:, :, 1, :]) / vals.sum(2)
            assert list(margin_confidence.shape) == [batch_size, self.M, len(self.train_heads)]
            attention_weights = F.softmax(
                    (indicator_preds + margin_confidence) / self.temperature, dim=-1)

        assert not torch.isnan(attention_weights).any()
        assert not torch.isinf(attention_weights).any()

        # attention_weights is batch x M x num_slices
        # slice_representations is batch_size x M x K x num_slices x feat_dim
        slice_representations = torch.stack([expert_slice_repr[slice_head] for slice_head in self.train_heads], dim=3)

        # attention_weights becomes batch_size x M x K x num_slices x H of slice_representations
        attention_weights = attention_weights.unsqueeze(2).unsqueeze(-1).expand_as(slice_representations)
        # Reweight representations with weighted sum across slices
        reweighted_rep = torch.sum(attention_weights * slice_representations, dim=3)
        assert reweighted_rep.shape == context_matrix.shape
        # Pass through the final prediction layer
        for slice_head in self.train_heads:
            out[DISAMBIG][train_utils.get_slice_head_pred_name(slice_head)] = predictor_outputs[slice_head]
        for slice_head in self.train_heads:
            out[INDICATOR][train_utils.get_slice_head_ind_name(slice_head)] = indicator_outputs[slice_head]
        # Used for debugging
        if self.remove_final_loss:
            out[DISAMBIG][FINAL_LOSS] = out[DISAMBIG][train_utils.get_slice_head_pred_name(BASE_SLICE)]
        else:
            out[DISAMBIG][FINAL_LOSS] = self.final_pred_head(reweighted_rep).squeeze(-1)
        return out, reweighted_rep