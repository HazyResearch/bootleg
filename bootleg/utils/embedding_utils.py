import logging

import numpy as np

from bootleg.symbols.constants import (
    DROPOUT_1D,
    DROPOUT_2D,
    FREEZE,
    NORMALIZE,
    SEND_THROUGH_BERT,
)

logger = logging.getLogger(__name__)


def get_max_candidates(entity_symbols, data_config):
    """Returns the maximum number of candidates used in the model, taking into
    account train_in_candidates If train_in_canddiates is False, we add a NC
    entity candidate (for null candidate)

    Args:
        entity_symbols: entity symbols
        data_config: data config

    Returns:
    """
    return entity_symbols.max_candidates + int(not data_config.train_in_candidates)


def get_embedding_args(emb):
    """Extract the embedding arguments that are the same for _all_ embedding
    objects (see base_emb.py). These are defined in the upper level of the
    config.

    Allowed arguments:
        - cpu: True/False (whether embedding on CPU or not)
        - freeze: True/False (freeze parameters or not)
        - dropout1d: float between 0, 1
        - dropout2d: float between 0, 1
        - normalize: True/False
        - sent_through_bert: True/False (whether this embedding outputs indices for BERT encoder -- see bert_encoder.py)

    Args:
        emb: embedding dictionary arguments from config

    Returns: parsed arguments with defaults
    """
    emb_args = emb.get("args", None)
    assert (
        "load_class" in emb
    ), "You must specify a load_class in the embedding config: {load_class: ..., key: ...}"
    assert (
        "key" in emb
    ), "You must specify a key in the embedding config: {load_class: ..., key: ...}"
    # Add cpu
    cpu = emb.get("cpu", False)
    assert type(cpu) is bool
    # Add freeze
    freeze = emb.get(FREEZE, False)
    assert type(freeze) is bool
    # Add 1D dropout
    dropout1d_perc = emb.get(DROPOUT_1D, 0.0)
    assert 1.0 >= dropout1d_perc >= 0.0
    # Add 2D dropout
    dropout2d_perc = emb.get(DROPOUT_2D, 0.0)
    assert 1.0 >= dropout2d_perc >= 0.0
    # Add normalize
    normalize = emb.get(NORMALIZE, True)
    assert type(normalize) is bool
    # Add through BERT
    through_bert = emb.get(SEND_THROUGH_BERT, False)
    assert type(through_bert) is bool
    return (
        cpu,
        dropout1d_perc,
        dropout2d_perc,
        emb_args,
        freeze,
        normalize,
        through_bert,
    )


def prep_kg_feature_sum(entity_indices, adj):
    """Given matrix of entity indices (M x K) and an adjacent matrix (M*K x
    M*K), returns the sum of the values in adj between two entity indexes that
    are not part of the same mention.

    Args:
        entity_indices: entity EIDs (M x K) - each value EID < E
        adj: adjacency matrix (E x E) - E is total number of entities in our world

    Returns: sum of all values in adj connected to some entity index by other indices in entity_indices
    """
    subset_adj = prep_kg_feature_matrix(entity_indices, adj)
    # do sum over all candidates for MxK candidates
    kg_feat = np.squeeze(subset_adj.sum(1))
    # return summed MxK feature indicating a candidates relatedness to other aliases' candidates
    return kg_feat


def prep_kg_feature_matrix(entity_indices, adj):
    """Given matrix of entity indices (M x K) and an adjacent matrix (M*K x
    M*K), returns matrix of the values in adj between two entity indexes that
    are not part of the same mention.

    Args:
        entity_indices: entity EIDs (M x K) - each value EID < E
        adj: adjacency matrix (E x E) - E is total number of entities in our world

    Returns: (M*K x M*K) matrix of values in adj connected to some entity index by other indices in entity_indices
    """
    M, K = entity_indices.shape
    # ensure we are using numpy (code is different for torch versus numpy)
    entity_indices = np.array(entity_indices)
    entity_indices = entity_indices.flatten()
    # use entity ids to extract MxK
    # format for CSR matrix must be like: [[0,2],[[0],[2]]]
    # subset_adj = self.kg_adj[entity_indices, entity_indices.unsqueeze(1)]
    subset_adj = adj[entity_indices, np.expand_dims(entity_indices, 1)]
    subset_adj = subset_adj.toarray()
    # mask out noisy connectivity to candidates of same alias
    single_mask = np.array([[True] * K] * K)
    # https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    full_mask = np.kron(np.eye(M, dtype=bool), single_mask)
    subset_adj[full_mask] = 0
    return subset_adj
