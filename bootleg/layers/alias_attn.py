"""Alias attention model"""

import torch.nn as nn

from bootleg.layers.layers import *
from bootleg.utils import model_utils


class AliasMHA(nn.Module):
    """
    Alias multi-headed attention.

    Takes in a sentence embedding and the entity embeddings and returns
    an alias representation for each alias in the sentence by computing a
    weighted average of entity candidates.
    """

    def __init__(self, args):
        super(AliasMHA, self).__init__()
        self.ff_inner_size = args.model_config.ff_inner_size
        self.num_heads = args.model_config.num_heads
        self.hidden_size = args.model_config.hidden_size
        self.dropout = args.train_config.dropout
        self.attention_module = AttnBlock(size=self.hidden_size,
            ff_inner_size=args.model_config.ff_inner_size,
            dropout=self.dropout, num_heads=self.num_heads)
        self.sent_alias_attn = AttnBlock(size=self.hidden_size,
            ff_inner_size=args.model_config.ff_inner_size,
            dropout=self.dropout, num_heads=self.num_heads)

    def forward(self, sent_embedding, entity_embedding, entity_mask, alias_idx_pair_sent, slice_emb_alias, slice_emb_ent):
        batch_size, M, K, _ = entity_embedding.shape
        # Index is which word to select
        alias_word_tensor = model_utils.select_alias_word_sent(alias_idx_pair_sent, sent_embedding, index=0)

        # Slice emb is hidden_size x 1 -> batch x M x hidden_size
        # Add in slice alias embedding
        slice_emb_alias = slice_emb_alias.unsqueeze(0).unsqueeze(1).expand(batch_size, M, self.hidden_size)
        alias_word_tensor = alias_word_tensor + slice_emb_alias
        alias_word_tensor = alias_word_tensor.transpose(0,1)
        assert alias_word_tensor.shape[1] == batch_size

        # Sentence attn between alias vector and full sentence
        alias_word_tensor, alias_word_weights = self.sent_alias_attn(q=alias_word_tensor, x=sent_embedding.tensor.transpose(0,1),
            key_mask=sent_embedding.mask, attn_mask=None)

        # Add in slice entity embedding
        slice_emb_ent = slice_emb_ent.unsqueeze(0).unsqueeze(1).expand(batch_size, M, self.hidden_size)
        alias_word_tensor = alias_word_tensor.transpose(0,1) + slice_emb_ent
        alias_word_tensor = alias_word_tensor.transpose(0,1)
        entity_attn_tensor = entity_embedding.contiguous().view(batch_size, M*K, self.hidden_size).transpose(0,1)
        key_padding_mask_entities = entity_mask.contiguous().view(batch_size, M*K)
        # Each M alias should ONLY pay attention to it's OWN candidates
        entity_mask = torch.ones((M, K*M)).to(key_padding_mask_entities.device)
        # M x (M*K)
        # TODO: move this to init
        for i in range(M):
            entity_mask[i, i*K:(i+1)*K] = 0.0
            # Must manually move this to the device as it's not part of a module
            entity_mask = entity_mask.masked_fill((entity_mask == 1), float(-1e9))
        alias_entity_attn_context, alias_entity_attn_weights = self.attention_module(
                q=alias_word_tensor,
                x=entity_attn_tensor,
                key_mask=key_padding_mask_entities,
                attn_mask=entity_mask
        )
        # Returns batch x M x hidden
        return alias_entity_attn_context.transpose(0,1), alias_word_weights