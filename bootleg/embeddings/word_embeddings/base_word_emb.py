"""Base word embedding"""
import torch
import torch.nn as nn
import os

from bootleg.utils import logging_utils


class BaseWordEmbedding(nn.Module):
    """
    Base word embedding class. We split the word embedding from the sentence encoder, similar to BERT.

    Attributes:
        pad_id: id of the pad word index
    """
    def __init__(self, args, main_args, word_symbols):
        super(BaseWordEmbedding, self).__init__()
        self.logger = logging_utils.get_logger(main_args)
        self._key = "word"
        self.pad_id = word_symbols.pad_id

    def freeze_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            self.logger.debug(f'Freezing {name}')
        return

    # This mask is for downstream pytorch multiheadattention
    # This assumes that TRUE means MASK (aka IGNORE). For the sentence embedding, the mask therefore is if an index is equal to the pad id
    # Note: This mask cannot be used for a BERT model as they use the reverse mask.
    def get_downstream_mask(self, word_indices):
        return word_indices == self.pad_id

    def forward(self, word_indices):
        raise ValueError("Not implemented.")

    def get_dim(self):
        raise ValueError("Not implemented.")

    def get_key(self):
        raise ValueError("Not implemented.")