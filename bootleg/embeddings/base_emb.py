"""Base embeddings"""

import torch.nn as nn

from bootleg.symbols.constants import MASK_PERC
from bootleg.utils import logging_utils
from bootleg.utils.classes.dotted_dict import DottedDict
from bootleg.utils.classes.required_attributes import RequiredAttributes

class EntityEmb(nn.Module):
    """
    Base embedding that all embedding classes extend.

    Attributes:
        key: unique key for embedding defined in config
        dropout_perc: 1D dropout perc defined in config
        mask_perc: 2D dropout perc defined in config

    Each subclass must defined a self.normalize attribute. This defines if a L2 norm is applied
    to each embedding independently in the forward pass before being concatenated in emb_combine.py.

    If an embedding should be preprocessed in prep phase or in the __get_item__ of the dataset, it must
        - implement batch_prep(alias_indices, entity_indices)
        - set batch_prep: True in config to be processed in prep
        - set batch_on_the_fly: True in config to be process in __get_item__
        - access the prepped embedding in the forward by batch_prepped_data or batch_on_the_fly_data by the embedding key
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(EntityEmb, self).__init__()
        # Metaclasses are types that create classes
        # https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python
        # We use this to enforce that a self.normalize attribute is instantiated in subclasses
        __metaclass__ = RequiredAttributes("normalize")
        self.logger = logging_utils.get_logger(main_args)
        self.entity_symbols = entity_symbols
        self.model_device = model_device
        self.key = key
        self.dropout_perc = 0
        self.mask_perc = 0
        # Used for 2d dropout
        if MASK_PERC in emb_args:
            self.mask_perc = emb_args[MASK_PERC]
            self.logger.debug(f'Setting {self.key} mask perc to {self.mask_perc}')
        # Used for 1d dropout
        if "dropout" in emb_args:
            self.dropout_perc = emb_args.dropout
            self.logger.debug(f'Setting {self.key} dropout to {self.dropout_perc}')

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        raise ValueError("Not implemented")

    def get_dim(self):
        raise ValueError("Not implemented")

    def get_key(self):
        return self.key

    def freeze_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            self.logger.debug(f'Freezing {name}')
        return

    def unfreeze_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
            self.logger.debug(f'Unfreezing {name}')
        return

    # Package up information for forward pass
    def _package(self, tensor, pos_in_sent, alias_indices, mask):
        packed_emb = DottedDict(
            tensor = tensor,
            pos_in_sent = pos_in_sent,
            alias_indices = alias_indices,
            mask = mask,
            normalize = self.normalize,
            key = self.get_key(),
            dim = self.get_dim()
        )
        return packed_emb