"""Embedding combine layer."""
import logging

import torch.nn as nn

from bootleg.layers.helper_modules import CatAndMLP, PositionalEncoding

logger = logging.getLogger(__name__)


class EmbeddingPayloadBase(nn.Module):
    """Embedding query + concat + projection layer.

    Queries entity embeddings to generate entity payloads. Projects
    entity embeddings to hidden size for input into attention network.

    Args:
        args: args
        entity_symbols: entity symbols
        total_sizes: total entity embedding sizes
    """

    def __init__(self, args, entity_symbols, total_sizes):
        super(EmbeddingPayloadBase, self).__init__()

        self.K = entity_symbols.max_candidates + (
            not args.data_config.train_in_candidates
        )
        self.M = args.data_config.max_aliases
        self.hidden_size = args.model_config.hidden_size

        self.total_dim = sum([v for k, v in total_sizes.items()])
        # Don't include sizes from the relation indices
        if args.data_config.type_prediction.use_type_pred:
            self.total_dim += args.data_config.type_prediction.dim

        self.position_enc = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.linear_layers["project_embedding"] = CatAndMLP(
            size=self.total_dim,
            num_hidden_units=self.total_dim,
            output_size=self.hidden_size,
            num_layers=1,
        )

    def forward(self, start_span_idx, end_span_idx, *embedding_args):
        """Model forward.

        Args:
            start_span_idx: start span into sentence embedding for mention
            end_span_idx: end span into sentence embedding for mention
            *embedding_args: all entity embeddings (list of B x M x K x dim)

        Returns: entity payload B x M x K x H
        """
        # Create list of all entity tensors
        batch_size = start_span_idx.shape[0]
        emb_list = []
        for embedding_tensor in embedding_args:
            # If some embeddings do not want to be added to the payload, they will return an None
            # This happens for the kg bias (KGIndices) class for our kg attention network
            if embedding_tensor is None:
                continue
            # Entity shape: batch_size x M x K x embedding_dim
            assert embedding_tensor.shape[0] == batch_size
            assert embedding_tensor.shape[1] == self.M
            assert embedding_tensor.shape[2] == self.K
            emb_list.append(embedding_tensor)
        if len(emb_list) > 1:
            embedding_payload = self.linear_layers["project_embedding"](emb_list)
        else:
            embedding_payload = emb_list[0]
        return embedding_payload


class EmbeddingPayload(EmbeddingPayloadBase):
    """Embedding query + concat + adds positional embedding + projection layer.

    Queries entity embeddings to generate entity payloads. Projects
    entity embeddings to hidden size and adds positional embedding information for each
     candidate about where it is in the sentence. Used as input into attention network.

    Args:
        args: args
        entity_symbols: entity symbols
        total_sizes: total entity embedding sizes
    """

    def __init__(self, args, entity_symbols, total_sizes):
        super(EmbeddingPayload, self).__init__(args, entity_symbols, total_sizes)
        self.position_enc = nn.ModuleDict()
        # These are NOT learned parameters so we will use it for first and last positional tokens
        self.position_enc["alias"] = PositionalEncoding(self.hidden_size)
        self.position_enc["alias_position_cat"] = CatAndMLP(
            size=2 * self.hidden_size,
            num_hidden_units=self.total_dim,
            output_size=self.hidden_size,
            num_layers=2,
        )

    def forward(self, start_span_idx, end_span_idx, *embedding_args):
        """Model forward.

        Args:
            start_span_idx: start span into sentence embedding for mention
            end_span_idx: end span into sentence embedding for mention
            *embedding_args: all entity embeddings (list of B x M x K x dim)

        Returns: entity payload B x M x K x H
        """
        # Create list of all entity tensors
        batch_size = start_span_idx.shape[0]
        emb_list = []
        for embedding_tensor in embedding_args:
            # If some embeddings do not want to be added to the payload, they will return an None
            # This happens for the kg bias (KGIndices) class for our kg attention network
            if embedding_tensor is None:
                continue
            # Entity shape: batch_size x M x K x embedding_dim
            assert embedding_tensor.shape[0] == batch_size
            assert embedding_tensor.shape[1] == self.M
            assert embedding_tensor.shape[2] == self.K
            emb_list.append(embedding_tensor)
        emb_tensor = self.linear_layers["project_embedding"](emb_list)
        emb_tensor_first = self.position_enc["alias"](
            emb_tensor,
            start_span_idx.transpose(0, 1).repeat(self.K, 1, 1).transpose(0, 2).long(),
        )
        emb_tensor_last = self.position_enc["alias"](
            emb_tensor,
            end_span_idx.transpose(0, 1).repeat(self.K, 1, 1).transpose(0, 2).long(),
        )
        embedding_payload = self.position_enc["alias_position_cat"](
            [emb_tensor_first, emb_tensor_last]
        )
        return embedding_payload
