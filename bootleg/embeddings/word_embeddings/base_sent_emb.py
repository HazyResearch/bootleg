"""Base sentence embedding class"""

import torch.nn as nn

from bootleg.utils import logging_utils

class BaseSentEmbedding(nn.Module):
    """
    Base sentence embedding class. We split the word embedding from the sentence encoder, similar to BERT.
    """
    def __init__(self, emb_args, main_args, word_emb_dim, word_symbols):
        super(BaseSentEmbedding, self).__init__()
        self.logger = logging_utils.get_logger(main_args)
        self._key = "sentence"
        self._dim = word_emb_dim

    def freeze_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            self.logger.debug(f'Freezing {name}')
        return

    def forward(self, word_package):
        raise ValueError("Not implemented.")

    def get_dim(self):
        return self._dim

    def get_key(self):
        return self._key