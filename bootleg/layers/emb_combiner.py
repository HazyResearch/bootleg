"""Embedding combine layer"""

import torch.nn as nn
from bootleg.layers.layers import *
from bootleg.utils import model_utils, logging_utils
from bootleg.utils.classes.dotted_dict import DottedDict


class EmbCombinerProj(nn.Module):
    """
    Embedding combiner + projection layer.

    Generates entity payloads. Projects entity embeddings to hidden size
    for input into attention network. Adds positional encodings to entity embeddings.
    """
    def __init__(self, args, emb_sizes, sent_emb_size, word_symbols, entity_symbols):
        super(EmbCombinerProj, self).__init__()
        self.logger = logging_utils.get_logger(args)
        self.K = entity_symbols.max_candidates + (not args.data_config.train_in_candidates)
        self.M = args.data_config.max_aliases
        self.hidden_size = args.model_config.hidden_size
        self.sent_len = args.data_config.max_word_token_len + 2*word_symbols.is_bert
        self.sent_emb_size = sent_emb_size
        # Don't include sizes from the relation indices
        total_dim = sum([v for k,v in emb_sizes.items()])
        if args.data_config.type_prediction.use_type_pred:
            total_dim += args.data_config.type_prediction.dim
        self.position_enc = nn.ModuleDict()
        # These are NOT learned parameters so we will use it for first and last positional tokens
        self.position_enc['alias'] = PositionalEncoding(self.hidden_size)
        self.position_enc['alias_position_cat'] = CatAndMLP(size=2*self.hidden_size, num_hidden_units=total_dim, output_size=self.hidden_size, num_layers=2)
        self.linear_layers = nn.ModuleDict()
        self.linear_layers['project_embedding'] = CatAndMLP(size=total_dim,
            num_hidden_units=total_dim, output_size=self.hidden_size, num_layers=1)

    def forward(self, sent_embedding, alias_idx_pair_sent, entity_embedding, entity_mask):
        batch_size = sent_embedding.tensor.shape[0]
        # Create list of all entity tensors
        alias_list = []
        alias_indices = None
        for embedding in entity_embedding:
            # Entity shape: batch_size x M x K x embedding_dim
            assert(embedding.tensor.shape[0] == batch_size)
            assert(embedding.tensor.shape[1] == self.M)
            assert(embedding.tensor.shape[2] == self.K)
            emb = embedding.tensor
            if alias_indices is not None:
                assert torch.equal(alias_indices, embedding.alias_indices), "Alias indices should not be different between embeddings in embCombiner"
            alias_indices = embedding.alias_indices
            # Normalize input embeddings
            if embedding.normalize:
                emb = model_utils.normalize_matrix(emb, dim=3)
                assert not torch.isnan(emb).any()
                assert not torch.isinf(emb).any()
            alias_list.append(emb)
        alias_tensor = self.linear_layers['project_embedding'](alias_list)
        alias_tensor_first = self.position_enc['alias'](alias_tensor, alias_idx_pair_sent[0].transpose(0,1).repeat(self.K, 1, 1).transpose(0, 2).long())
        alias_tensor_last = self.position_enc['alias'](alias_tensor, alias_idx_pair_sent[1].transpose(0,1).repeat(self.K, 1, 1).transpose(0, 2).long())
        alias_tensor = self.position_enc['alias_position_cat']([alias_tensor_first, alias_tensor_last])

        proj_ent_embedding = DottedDict(
            tensor=alias_tensor,
            # Position of entities in sentence
            pos_in_sent=alias_idx_pair_sent,
            # Indexes of aliases
            alias_indices=alias_indices,
            # All entity embeddings have the same mask currently
            mask=embedding.mask,
            # Do not normalize this embedding if normalized is called
            normalize=False,
            dim=alias_tensor.shape[-1])
        return sent_embedding, proj_ent_embedding