"""Standard sentence embedding class"""

import torch.nn as nn
import torch

from bootleg.embeddings.word_embeddings import BaseSentEmbedding
from bootleg.layers.layers import MLP, SelfAttnBlock
from bootleg.utils.classes.dotted_dict import DottedDict
from bootleg.utils import model_utils

class StandardSentEmb(BaseSentEmbedding):
    """Self-attention stack over word embeddings."""
    def __init__(self, emb_args, main_args, word_emb_dim, word_symbols):
        super(StandardSentEmb, self).__init__(emb_args=emb_args, main_args=main_args, word_emb_dim=word_emb_dim,
            word_symbols=word_symbols)
        # Even if sent_emb is frozen, we may need to keep track of gradients if word_emb is NOT frozen
        # If both are frozen, we can use torch.no_grad in the forward pass to save memory
        self.requires_grad = not emb_args.freeze_sent_emb or not emb_args.freeze_word_emb
        self.num_layers = emb_args.layers
        num_heads = main_args.model_config.num_heads
        self.attention_modules = nn.ModuleDict()
        for i in range(self.num_layers):
            self.attention_modules[f"stage_{i}_self_sentence"] = \
                SelfAttnBlock(word_emb_dim,
                              main_args.model_config.ff_inner_size,
                              main_args.train_config.dropout,
                              num_heads)
        self._key = "sentence"
        self._dim = word_emb_dim
        self.attention_weights = {}
        self.attention_modules.apply(model_utils.init_weights)

    def forward(self, word_package):
        attention_mask = word_package.mask
        word_vectors = word_package.tensor
        batch_size = word_vectors.shape[0]
        word_vectors = word_vectors.transpose(0,1)
        out = word_vectors
        if self.requires_grad:
            for i in range(self.num_layers):
                out, weights = self.attention_modules[f"stage_{i}_self_sentence"](out, key_mask=attention_mask, attn_mask=None)
                self.attention_weights[f"layer_{i}_sent"] = weights
        else:
            with torch.no_grad():
                for i in range(self.num_layers):
                    out, weights = self.attention_modules[f"stage_{i}_self_sentence"](out, key_mask=attention_mask, attn_mask=None)
                    self.attention_weights[f"layer_{i}_sent"] = weights
        out = out.transpose(0,1)
        assert out.shape[0] == batch_size
        emb = DottedDict(
            tensor=out,
            # Make the main mask also be the downstream mask as this is post sentence embedding
            downstream_mask=word_package.downstream_mask,
            key=self.get_key(),
            dim=self.get_dim()
        )
        return emb