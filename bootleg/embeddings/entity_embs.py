"""Entity embeddings"""
import os
import time
import torch.nn as nn
import torch
import scipy.sparse
import numpy as np

from bootleg.embeddings import EntityEmb
from bootleg.utils import model_utils, logging_utils, train_utils, utils, data_utils

class LearnedEntityEmb(EntityEmb):
    """
    Learned entity embeddings class. We support initializing all learned entity embeddings to be the same value. This helps with
    tail generalization as rare embeddings are close together and less prone to errors in embedding noise.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(LearnedEntityEmb, self).__init__(main_args=main_args, emb_args=emb_args, model_device=model_device,
                                               entity_symbols=entity_symbols, word_symbols=word_symbols, word_emb=word_emb, key=key)
        self.logger = logging_utils.get_logger(main_args)
        self.learned_embedding_size = emb_args.learned_embedding_size
        self.normalize = True
        self.learned_entity_embedding = nn.Embedding(
            entity_symbols.num_entities_with_pad_and_nocand,
            self.learned_embedding_size,
            padding_idx=-1, sparse=True)
        # If tail_init is false, all embeddings are randomly intialized.
        # If tail_init is true, we initialize all embeddings to be the same.
        tail_init = True
        if "tail_init" in emb_args:
            tail_init = emb_args.tail_init
        if tail_init:
            qid_count_dict = {}
            self.logger.debug(f"All learned entity embeddings are initialized to the same value.")
            init_vec = model_utils.init_tail_embeddings(self.learned_entity_embedding, qid_count_dict, entity_symbols, pad_idx=-1)
        else:
            self.logger.debug(f"All learned embeddings are randomly initialized.")
        self._dim = main_args.model_config.hidden_size
        self.dropout = nn.Dropout(self.dropout_perc)

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        tensor = self.learned_entity_embedding(entity_package.tensor)
        tensor = self.dropout(tensor)
        emb = self._package(tensor=tensor, pos_in_sent=entity_package.pos_in_sent, alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return emb

    def get_dim(self):
        return self.learned_embedding_size