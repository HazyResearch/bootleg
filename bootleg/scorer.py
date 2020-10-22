"""Scorer"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bootleg.symbols.constants import DISAMBIG, INDICATOR, FINAL_LOSS, TYPEPRED
from bootleg.utils import logging_utils, eval_utils, train_utils
from bootleg.utils.classes.cross_entropy_with_probs import cross_entropy_with_probs
from bootleg.utils.classes.loss_package import LossPackage

class Scorer:
    """
    Scoring class: aggregates losses on prediction heads for backpropagation.
    """
    def __init__(self, args=None, model_device=None):
        self.model_device = model_device
        self.logger = logging_utils.get_logger(args)
        self.crit_pred = nn.NLLLoss(ignore_index=-1)
        self.type_pred = nn.CrossEntropyLoss(ignore_index=-1)
        self.weights = {train_utils.get_slice_head_ind_name(slice_name):None for slice_name in args.train_config.train_heads}

    def calc_loss(self, training, outs, true_label, entity_pack):
        """
        Accumulates losses across standard prediction heads and indicator heads for slice-based learning.
        """
        loss = LossPackage(self.model_device)
        for loss_key in outs:
            if loss_key == DISAMBIG:
                loss_dis = self.disambig_loss(training, outs[loss_key], true_label[loss_key], entity_pack.mask)
                loss.merge_loss_packages(loss_dis)
            # indicator loss if using SBL
            elif loss_key == INDICATOR:
                loss_ind = self.indicator_loss(training, outs[loss_key], true_label[loss_key])
                loss.merge_loss_packages(loss_ind)
            elif loss_key == TYPEPRED:
                loss_type = self.type_loss(training, outs[loss_key], true_label[loss_key])
                loss.merge_loss_packages(loss_type)
        return loss

    def indicator_loss(self, training, outs, true_label):
        """
        Returns the indicator loss on indicator heads for slice-based learning.
        """
        device = next(iter(outs.values()))[0].device
        loss = LossPackage(device)
        for i, (loss_head, out) in enumerate(outs.items()):
            batch_size, M, _ = out.shape
            # just take the positive label
            log_probs = F.log_softmax(out, dim=-1).reshape(batch_size*M, 2)
            prob_pos_labels = true_label[loss_head].reshape(batch_size*M)
            # we need the negative labels bc the cross_entropy_with_probs takes the number
            # of classes (2) as input
            prob_neg_labels = 1 - prob_pos_labels
            # we need to set to -1 to mask padded aliases
            prob_neg_labels[prob_pos_labels == -1] = -1
            true_labels = torch.stack([prob_neg_labels, prob_pos_labels], dim=1)
            temp = cross_entropy_with_probs(input=log_probs, target=true_labels,
                weight=self.weights[loss_head], ignore_index=-1)
            loss.add_loss(loss_head, temp)
        return loss

    def disambig_loss(self, training, outs, true_label, mask):
        """
        Returns the entity disambiguation loss on prediction heads.
        """
        device = next(iter(outs.values()))[0].device
        loss = LossPackage(device)
        for i, (loss_head, out) in enumerate(outs.items()):
            if FINAL_LOSS in loss_head:
                true_label_head = true_label[FINAL_LOSS]
            else:
                true_label_head = true_label[loss_head]
            # During eval, even if our model does not predict a NIC candidate, we allow for a NIC gold QID
            # This qid gets assigned the label of -2 and is always incorrect in eval_wapper
            # As NLLLoss assumes classes of 0 to #classes-1 except for pad idx, we manually mask
            # the -2 labels for the loss computation only. As this is just for eval, it won't matter.
            if not training:
                label_mask = true_label_head == -2
                true_label_head[label_mask] = -1
            # batch x M x K -> transpose -> swap K classes with M spans for "k-dimensional" NLLloss
            log_probs = eval_utils.masked_class_logsoftmax(pred=out, mask=~mask).transpose(1,2)
            temp = self.crit_pred(log_probs, true_label_head.long().to(device))
            loss.add_loss(loss_head, temp)
            if not training:
                true_label_head[label_mask] = -2
        return loss

    def type_loss(self, training, outs, true_label):
        """
        Returns the type prediction loss.
        """
        device = next(iter(outs.values()))[0].device
        loss = LossPackage(device)
        for i, (loss_head, out) in enumerate(outs.items()):
            batch_size, M, num_types = out.shape
            # just take the positive label
            labels = true_label[loss_head].reshape(batch_size*M)
            input = out.reshape(batch_size*M, num_types)
            temp = self.type_pred(input, labels)
            loss.add_loss(loss_head, temp)
        return loss