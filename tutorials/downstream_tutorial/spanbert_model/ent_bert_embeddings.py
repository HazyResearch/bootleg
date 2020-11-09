import numpy as np
import torch
from torch import nn


class EntBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type and ent embeddings."""

    def __init__(self, bert_embeddings, ent_emb_file, static_ent_emb_file, freeze=True):
        super(EntBertEmbeddings, self).__init__()
        self.word_embeddings = bert_embeddings.word_embeddings

        self.position_embeddings = bert_embeddings.position_embeddings

        self.token_type_embeddings = bert_embeddings.token_type_embeddings

        if ent_emb_file is not None:
            ent_emb_matrix = torch.from_numpy(np.load(ent_emb_file))
            self.ent_embeddings = nn.Embedding(
                ent_emb_matrix.size()[0], ent_emb_matrix.size()[1], padding_idx=0
            )
            self.ent_embeddings.weight.data.copy_(ent_emb_matrix)

            if freeze:
                for param in self.ent_embeddings.parameters():
                    param.requires_grad = False

            self.ent_proj = nn.Linear(
                ent_emb_matrix.size()[1], self.word_embeddings.embedding_dim, bias=False
            )
        else:
            self.ent_embeddings = None

        if static_ent_emb_file is not None:
            static_ent_emb_matrix = torch.from_numpy(np.load(static_ent_emb_file))
            self.static_ent_embeddings = nn.Embedding(
                static_ent_emb_matrix.size()[0],
                static_ent_emb_matrix.size()[1],
                padding_idx=0,
            )
            self.static_ent_embeddings.weight.data.copy_(static_ent_emb_matrix)

            if freeze:
                for param in self.static_ent_embeddings.parameters():
                    param.requires_grad = False

            self.static_ent_proj = nn.Linear(
                static_ent_emb_matrix.size()[1],
                self.word_embeddings.embedding_dim,
                bias=False,
            )
        else:
            self.static_ent_embeddings = None

        self.LayerNorm = bert_embeddings.LayerNorm
        self.dropout = bert_embeddings.dropout

    def forward(
        self, input_ids, input_ent_ids, input_static_ent_ids, token_type_ids=None
    ):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        ents_embeddings = None
        if self.ent_embeddings is not None:
            ents_embeddings = self.ent_embeddings(input_ent_ids)
            ents_embeddings = self.ent_proj(ents_embeddings)

        static_ents_embeddings = None
        if self.static_ent_embeddings is not None:
            static_ents_embeddings = self.static_ent_embeddings(input_static_ent_ids)
            static_ents_embeddings = self.static_ent_proj(static_ents_embeddings)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if ents_embeddings is not None:
            embeddings += ents_embeddings
        if static_ents_embeddings is not None:
            embeddings += static_ents_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
