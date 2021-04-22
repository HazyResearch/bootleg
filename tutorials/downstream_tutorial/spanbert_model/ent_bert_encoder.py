import numpy as np
import torch
from pytorch_pretrained_bert.modeling import (
    BertEncoder,
    BertLayerNorm,
    BertPooler,
    BertPreTrainedModel,
)
from torch import nn


class EntBertEncoder(BertPreTrainedModel):
    def __init__(
        self,
        config,
        input_dim,
        output_dim,
        ent_emb_file,
        static_ent_emb_file,
        type_ent_emb_file,
        rel_ent_emb_file,
        tanh=False,
        norm=False,
        freeze=True,
    ):
        super(EntBertEncoder, self).__init__(config)
        if (
            ent_emb_file is not None
            or static_ent_emb_file is not None
            or type_ent_emb_file is not None
            or rel_ent_emb_file is not None
        ):
            self.encoder = BertEncoder(config)
        else:
            self.encoder = None
        self.pooler = BertPooler(config)

        self.apply(self.init_bert_weights)

        if ent_emb_file is not None:
            ent_emb_matrix = torch.from_numpy(np.load(ent_emb_file))
            self.ent_embeddings = nn.Embedding(
                ent_emb_matrix.size()[0], ent_emb_matrix.size()[1], padding_idx=0
            )
            self.ent_embeddings.weight.data.copy_(ent_emb_matrix)
            input_dim += ent_emb_matrix.size()[1]
            if freeze:
                for param in self.ent_embeddings.parameters():
                    param.requires_grad = False
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
            input_dim += static_ent_emb_matrix.size()[1]
            if freeze:
                for param in self.static_ent_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.static_ent_embeddings = None

        if type_ent_emb_file is not None:
            type_ent_emb_matrix = torch.from_numpy(np.load(type_ent_emb_file))
            self.type_ent_embeddings = nn.Embedding(
                type_ent_emb_matrix.size()[0],
                type_ent_emb_matrix.size()[1],
                padding_idx=0,
            )
            self.type_ent_embeddings.weight.data.copy_(type_ent_emb_matrix)
            input_dim += type_ent_emb_matrix.size()[1]
            if freeze:
                for param in self.type_ent_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.type_ent_embeddings = None

        if rel_ent_emb_file is not None:
            rel_ent_emb_matrix = torch.from_numpy(np.load(rel_ent_emb_file))
            self.rel_ent_embeddings = nn.Embedding(
                rel_ent_emb_matrix.size()[0],
                rel_ent_emb_matrix.size()[1],
                padding_idx=0,
            )
            self.rel_ent_embeddings.weight.data.copy_(rel_ent_emb_matrix)
            input_dim += rel_ent_emb_matrix.size()[1]
            if freeze:
                for param in self.rel_ent_embeddings.parameters():
                    param.requires_grad = False
        else:
            self.rel_ent_embeddings = None

        self.proj = nn.Linear(input_dim, output_dim)

        if tanh is True:
            self.proj_activation = nn.Tanh()
        else:
            self.proj_activation = None

        self.norm = norm
        if self.norm is True:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        bert_output,
        input_ent_ids,
        input_static_ent_ids,
        input_type_ent_ids,
        input_rel_ent_ids,
        attention_mask=None,
        output_all_encoded_layers=True,
    ):

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal
        # attention used in OpenAI GPT, we just need to prepare the broadcast dimension
        # here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = [bert_output[-1]]

        ents_embeddings = None
        if self.ent_embeddings is not None:
            ents_embeddings = self.ent_embeddings(input_ent_ids)
            embedding_output.append(ents_embeddings)

        static_ents_embeddings = None
        if self.static_ent_embeddings is not None:
            static_ents_embeddings = self.static_ent_embeddings(input_static_ent_ids)
            embedding_output.append(static_ents_embeddings)

        type_ents_embeddings = None
        if self.type_ent_embeddings is not None:
            type_ents_embeddings = self.type_ent_embeddings(input_type_ent_ids)
            embedding_output.append(type_ents_embeddings)

        rel_ents_embeddings = None
        if self.rel_ent_embeddings is not None:
            rel_ents_embeddings = self.rel_ent_embeddings(input_rel_ent_ids)
            embedding_output.append(rel_ents_embeddings)

        if self.encoder is not None:
            embedding_output = torch.cat(embedding_output, dim=-1)
            embedding_output = self.proj(embedding_output)

            if self.proj_activation:
                embedding_output = self.proj_activation(embedding_output)

            if self.norm:
                embedding_output = self.LayerNorm(embedding_output)
                embedding_output = self.dropout(embedding_output)

            encoded_layers = self.encoder(
                embedding_output,
                extended_attention_mask,
                output_all_encoded_layers=output_all_encoded_layers,
            )
        else:
            assert len(embedding_output) == 1
            encoded_layers = embedding_output

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
