"""Distributed evaluation"""
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from bootleg.symbols.constants import FINAL_LOSS, DISAMBIG, INDICATOR, BASE_SLICE
from bootleg.utils import eval_utils, train_utils

class EvalWrapper(nn.Module):
    """
    EvalWrapper supports distributed evaluation. For each slice, we collect
    (total_predictions, total_correct_model, total_correct_head).

    For each batch of the master dataset (see wiki_slices), we compute the scores
    using the slice incidence matrix. This matrix says which aliases in the batch
    participate in what slices. The output of the forward is a score for each slice.
    """
    def __init__(self, args, head_key_to_idx, eval_slice_names, train_head_names):
        super(EvalWrapper, self).__init__()
        # Dict mapping from any loss head to unique index
        # Used to get indexes of our buffers
        self.head_key_to_idx = head_key_to_idx
        self.train_in_candidates = int(args.data_config.train_in_candidates)
        self.num_model_stages = args.model_config.num_model_stages
        self.distributed = args.run_config.distributed
        self.train_heads = train_head_names[:]
        self.is_slicing_model = train_utils.is_slicing_model(args)
        self.topk_val = args.run_config.topk
        self.eval_slice_names = eval_slice_names[:]
        # registers a parameter of the model without any gradients
        # will be moved to the same device as the module
        self.register_buffer(f'alias_count',
            torch.zeros(len(self.eval_slice_names), dtype=torch.long))
        self.register_buffer(f'alias_head_correct',
            torch.zeros(len(self.eval_slice_names), dtype=torch.long))
        # we maintain a large matrix of different heads (SBL heads, stage heads, ...) by eval_slice_names
        # to keep track of correct predictions for different slice pred heads
        # # for distributed we need the buffers to not be empty so we require them to have at least 1 in the first dimension
        # # https://discuss.pytorch.org/t/surviving-oom-events-in-distributed-training/36008/5
        # we increment train_heads and models stages by 1 to include the final prediction head
        self.register_buffer(f'alias_pred_correct',
                             torch.zeros(len(head_key_to_idx), len(self.eval_slice_names), dtype=torch.long))
        self.register_buffer(f'alias_pred_topk',
                             torch.zeros(len(head_key_to_idx), len(self.eval_slice_names), dtype=torch.long))
        self.register_buffer(f'alias_pred_correct_pred_in_slice',
                             torch.zeros(len(head_key_to_idx), len(self.eval_slice_names), dtype=torch.long))
        self.register_buffer(f'alias_pred_count_pred_in_slice',
                             torch.zeros(len(head_key_to_idx), len(self.eval_slice_names), dtype=torch.long))

    def compute_scores(self):
        if self.distributed:
            # sync sums across wrappers
            for buffer in self.buffers():
                torch.distributed.all_reduce(buffer, op=dist.reduce_op.SUM)
        # compute scores
        dev_results = defaultdict(dict)
        for out_head_key in self.head_key_to_idx:
            buffer_idx = self.head_key_to_idx[out_head_key]
            # We need to know if the head is from SBL or not and if the head is in train_heads. We first check if the model is a
            # slicing model, if the slice head is a slice:..._pred head, and if the resulting slice name is in train_heads
            # is_slicing_value = (self.is_slicing_model and train_utils.is_name_slice_head_pred(out_head_key) and
                                # train_utils.get_inv_slice_head_pred_name(out_head_key) in self.train_heads)
            is_slicing_value = (train_utils.is_name_slice_head_pred(out_head_key) and
                                train_utils.get_inv_slice_head_pred_name(out_head_key) in self.train_heads)
            if is_slicing_value:
                slice_head = train_utils.get_slice_head_eval_name(train_utils.get_inv_slice_head_pred_name(out_head_key))
            else:
                slice_head = out_head_key
            for j, eval_slice in enumerate(self.eval_slice_names):
                slice_results = self.get_slice_results(buffer_idx, j, is_slicing_value=is_slicing_value)
                dev_results[slice_head][eval_slice] = slice_results
        return dev_results

    def get_slice_results(self, i, j, is_slicing_value):
        # avoid divide by zero errors when there are no samples predicted by the slice head
        # in the eval_slice
        if is_slicing_value:
            if self.alias_pred_count_pred_in_slice[i][j] == 0:
                f1_pred_in_slice = torch.tensor(0).item()
            else:
                f1_pred_in_slice = (self.alias_pred_correct_pred_in_slice[i][j] / float(self.alias_pred_count_pred_in_slice[i][j])).item()
            correct_ent_pred = int(self.alias_pred_correct_pred_in_slice[i][j])
            pred_in_slice = int(self.alias_pred_count_pred_in_slice[i][j])
        else:
            correct_ent_pred = float(np.nan)
            f1_pred_in_slice = float(np.nan)
            pred_in_slice = float(np.nan)
        support = self.alias_count[j]
        f1_micro_ent_head = self.alias_head_correct[j] / float(support)
        f1_micro_ent = self.alias_pred_correct[i][j] / float(support)
        f1_micro_ent_topk = self.alias_pred_topk[i][j] / float(support)
        # print(pred_in_slice, int(support), is_slicing_value)
        slice_results = {
            'correct_ent': int(self.alias_pred_correct[i][j]),
            f'correct_ent_top{self.topk_val}': int(self.alias_pred_topk[i][j]),
            'correct_ent_pred': correct_ent_pred,
            'correct_ent_head': int(self.alias_head_correct[j]),
            'f1_micro_ent': round(f1_micro_ent.item(), 3),
            f'f1_micro_ent_top{self.topk_val}': round(f1_micro_ent_topk.item(), 3),
            'f1_micro_ent_pred': round(f1_pred_in_slice, 3),
            'f1_micro_ent_head': round(f1_micro_ent_head.item(), 3),
            'pred_in_slice': pred_in_slice,
            'true_total_in_slice': int(support)
        }
        return slice_results

    def _get_topk_correct(self, model_preds, slice_indices, padded_entities, true_entity_idx, entity_indices, topk_val):
        # softmax is monotonic, meaning setting pads to -inf and taking max is same as max of masked_log_softmax
        model_preds[entity_indices == -1] = -1 * float("Inf")
        _, pred_idx_topk = model_preds.topk(k=topk_val, dim=2, sorted=True, largest=True)
        # returns batch x M x topk with a True if the prediction equals the gold prediction
        topk_correct_all = pred_idx_topk.eq(true_entity_idx.unsqueeze(-1).expand_as(pred_idx_topk))
        # row will be zero if none of the predictions in the topk match the gold prediction
        topk_correct = topk_correct_all.sum(2)
        # mask out padded entities
        topk_correct[padded_entities] = 0
        topk_correct_slices = topk_correct.unsqueeze(-1)*slice_indices
        summed_topk_correct_slices = topk_correct_slices.sum((0,1))
        return summed_topk_correct_slices, topk_correct_slices

    def _update_pred_counts(self, out_head_key, model_outs, slice_indices, entity_indices, padded_entities, true_entity_idx, true_label):
        buffer_idx = self.head_key_to_idx[out_head_key]
        if out_head_key in model_outs[DISAMBIG]:
            # compute the top1 for the final prediction head over eval slices
            self.alias_pred_correct[buffer_idx] += self._get_topk_correct(
                model_preds=model_outs[DISAMBIG][out_head_key],
                slice_indices=slice_indices,
                true_entity_idx=true_entity_idx,
                padded_entities=padded_entities,
                entity_indices=entity_indices,
                topk_val=1)[0]
            # compute the topk for the final prediction head over eval slices
            self.alias_pred_topk[buffer_idx] += self._get_topk_correct(
                model_preds=model_outs[DISAMBIG][out_head_key],
                slice_indices=slice_indices,
                true_entity_idx=true_entity_idx,
                padded_entities=padded_entities,
                entity_indices=entity_indices,
                topk_val=self.topk_val)[0]

    def forward(self, slice_indices, true_label, entity_indices, model_outs):
        batch_size, M, K = entity_indices.shape
        true_entity_idx = true_label[DISAMBIG][FINAL_LOSS]

        # dummy disambig for when we are only training an end indicator model for weak supervision
        # and only have indicator predictions
        if DISAMBIG not in model_outs:
            model_outs[DISAMBIG] = {}
            for out_head_key in self.head_key_to_idx:
                model_outs[DISAMBIG][out_head_key] = torch.ones(batch_size, M, K).to(entity_indices.device)

        padded_entities = (true_entity_idx==-1)

        # we need to filter out cases where true_entity_idx is -1 bc alias occurs
        # in multiple subsentences to not double count it for a slice
        # it will only be predicted in one of these subsentences
        # where true_entity_idx is not -1
        slice_indices[padded_entities] = 0

        # compute count of head correct across eval slices
        head_values = true_entity_idx.new_full(true_entity_idx.size(),
            fill_value=int(not self.train_in_candidates))
        head_correct = head_values == true_entity_idx
        head_correct[padded_entities] = 0
        head_correct_slices = head_correct.unsqueeze(-1) * slice_indices
        total_head_correct_slices = head_correct_slices.sum((0,1))
        self.alias_head_correct += total_head_correct_slices

        # total number of mentions per slice
        total_count_slices = slice_indices.sum((0,1))
        self.alias_count += total_count_slices

        #===============================================
        # Slicing model predictions (indicators)
        #===============================================
        for i, slice_head in enumerate(self.train_heads):
            out_head_key = train_utils.get_slice_head_pred_name(slice_head)
            self._update_pred_counts(out_head_key, model_outs, slice_indices, entity_indices, padded_entities, true_entity_idx, true_label)

            # compute scores of train head conditions on predicting in slice or not
            buffer_idx = self.head_key_to_idx[out_head_key]
            # If INDICATOR is an output, take it. Otherwise take the ground truth as the model does not have that loss (e.g. HPS model).
            # outs_ind is BxMx2
            if INDICATOR in model_outs:
                outs_ind = model_outs[INDICATOR][train_utils.get_slice_head_ind_name(slice_head)]
                outs_pred_in_slice = torch.argmax(outs_ind, -1).to(entity_indices.device)
            else:
                outs_pred_in_slice = torch.ones(entity_indices.shape[0],entity_indices.shape[1]).to(entity_indices.device).long()

            if out_head_key in model_outs[DISAMBIG]:
                # computes the train_head over each eval slice
                _, pred_correct = self._get_topk_correct(model_preds=model_outs[DISAMBIG][out_head_key],
                                                         slice_indices=slice_indices,
                                                         true_entity_idx=true_entity_idx,
                                                         padded_entities=padded_entities,
                                                         entity_indices=entity_indices,
                                                         topk_val=1)
                # given that it predicted in slice
                # how many did I say were in the slice that were actually in the slice did I get correct
                pred_correct_pred_in_slice = pred_correct * outs_pred_in_slice.unsqueeze(-1) * slice_indices
                total_pred_correct_pred_in_slice = pred_correct_pred_in_slice.sum((0,1))
                # how many did I say were in the slice that were actually in the slice
                total_pred_in_slice = outs_pred_in_slice.unsqueeze(-1) * slice_indices
                total_pred_in_slice = total_pred_in_slice.sum((0,1))

                # print(total_pred_in_slice, total_pred_correct_pred_in_slice, slice_head)
                self.alias_pred_correct_pred_in_slice[buffer_idx] += total_pred_correct_pred_in_slice
                self.alias_pred_count_pred_in_slice[buffer_idx] += total_pred_in_slice

        #===============================================
        # Final prediction head
        #===============================================
        out_head_key = FINAL_LOSS
        self._update_pred_counts(out_head_key, model_outs, slice_indices, entity_indices, padded_entities, true_entity_idx, true_label)

        #===============================================
        # Stage heads
        #===============================================
        # compute the topk for each model stage loss head
        for stage_idx in range(self.num_model_stages-1):
            out_head_key = train_utils.get_stage_head_name(stage_idx)
            self._update_pred_counts(out_head_key, model_outs, slice_indices, entity_indices, padded_entities, true_entity_idx, true_label)

        return None

    def reset_buffers(self):
        self.alias_count.zero_()
        self.alias_pred_correct.zero_()
        self.alias_pred_topk.zero_()
        self.alias_head_correct.zero_()
        self.alias_pred_count_pred_in_slice.zero_()
        self.alias_pred_correct_pred_in_slice.zero_()