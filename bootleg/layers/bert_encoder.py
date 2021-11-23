"""BERT encoder."""
import torch
from torch import nn


class Encoder(nn.Module):
    """
    Encoder module.

    Return the CLS token of Transformer.

    Args:
        transformer: transformer
        out_dim: out dimension to project to
    """

    def __init__(self, transformer, out_dim):
        """BERT Encoder initializer."""
        super(Encoder, self).__init__()
        transformer_output_dim = transformer.embeddings.word_embeddings.weight.size(1)
        self.linear = nn.Linear(transformer_output_dim, out_dim)
        self.activation = nn.Tanh()
        self.transformer = transformer

    def forward(self, token_ids, segment_ids=None, attention_mask=None):
        """BERT Encoder forward."""
        encoded_layers, pooled_output = self.transformer(
            input_ids=token_ids.reshape(-1, token_ids.shape[-1]),
            token_type_ids=segment_ids.reshape(-1, segment_ids.shape[-1]),
            attention_mask=attention_mask.reshape(-1, attention_mask.shape[-1]),
            return_dict=False,
        )
        full_embs = pooled_output.reshape(*token_ids.shape[:-1], -1)
        embs = self.activation(self.linear(full_embs))
        training_bool = (
            torch.tensor([1], device=token_ids.device) * self.training
        ).bool()
        return embs, training_bool
