import torch
from torch import nn as nn
from torch.nn.init import xavier_normal_

from bootleg.layers.helper_modules import MLP
from bootleg.utils import model_utils


class TypePred(nn.Module):
    """Basic type prediction class using MLP over mention embedding.

    For each M mention, predictions with type that mention is. Extracts
    that type embedding and concats to each candidate's entity
    embedding.

    Args:
        input_size: input dim
        emb_size: embedding dim
        num_types: number of types to predict
        num_candidates: number of entity candidates
    """

    def __init__(self, input_size, emb_size, num_types, num_candidates):
        super(TypePred, self).__init__()
        self.K = num_candidates
        # num_types x emb_size
        self.emb_size = emb_size
        self.type_embedding = torch.nn.Parameter(torch.Tensor(num_types, emb_size))
        xavier_normal_(self.type_embedding)
        self.type_embedding.requires_grad = True
        # Two layer MLP
        self.prediction = MLP(
            input_size=input_size,
            num_hidden_units=input_size,
            output_size=num_types,
            num_layers=2,
        )
        self.type_softmax = nn.Softmax(dim=2)

    def forward(self, sent_emb, start_span_idx):
        """Model forward.

        Args:
            sent_emb: sentence embedding (B x N x L)
            start_span_idx: span index into sentence embedding (B x M)

        Returns: type embeding tensor (B x M x K x dim), type weight prediction (B x M x num_types)
        """
        batch, M = start_span_idx.shape
        alias_mask = start_span_idx == -1
        # Get alias tensor and expand to be for each candidate for soft attn
        alias_word_tensor = model_utils.select_alias_word_sent(start_span_idx, sent_emb)

        # batch x M x num_types
        batch_type_pred = self.prediction(alias_word_tensor)
        batch_type_weights = self.type_softmax(batch_type_pred)
        # batch x M x emb_size
        batch_type_embs = torch.matmul(
            batch_type_weights, self.type_embedding.unsqueeze(0)
        )
        # mask out unk alias embeddings
        batch_type_embs[alias_mask] = 0
        batch_type_embs = batch_type_embs.unsqueeze(2).expand(
            batch, M, self.K, self.emb_size
        )
        # normalize the output before being concatenated
        batch_type_embs = model_utils.normalize_matrix(batch_type_embs, dim=3)
        return batch_type_embs, batch_type_pred
