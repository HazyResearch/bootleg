"""Knowledge graph embeddings."""
import logging
import os
import pickle
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse
import ujson as json

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.embeddings import EntityEmb
from bootleg.layers.helper_modules import *
from bootleg.utils import data_utils, embedding_utils, utils
from bootleg.utils.model_utils import selective_avg

logger = logging.getLogger(__name__)


class KGAdjEmb(EntityEmb):
    """KG adjaceny base embedding class that stores a statistical feature for
    pairs of entities. Uses numpy sparse matrices to store the features. It
    must define self.kg_adj_process_func so we can postprocessing the KG
    embeddings in any desired way to be used downstream.

    Add to your config via::

        ent_embeddings:
            - key: adj_index
             load_class: KGIndices
             batch_on_the_fly: true
             normalize: false
             args:
               kg_adj: <path to kg adj file>

    Args:
        main_args: main args
        emb_args: specific embedding args
        entity_symbols: entity symbols
        key: unique embedding key
        cpu: bool of if one cpu or not
        normalize: bool if normalize embeddings or not
        dropout1d_perc: 1D dropout percent
        dropout2d_perc: 2D dropout percent
    """

    def __init__(
        self,
        main_args,
        emb_args,
        entity_symbols,
        key,
        cpu,
        normalize,
        dropout1d_perc,
        dropout2d_perc,
    ):
        super(KGAdjEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        assert "kg_adj" in emb_args, f"KG embedding requires kg_adj to be set in args"
        assert (
            self.normalize is False
        ), f"We can't normalize a KGAdjEmb as it has hidden dim 1"
        assert (
            self.dropout1d_perc == 0.0
        ), f"We can't dropout 1d a KGAdjEmb as it has hidden dim 1"
        assert (
            self.dropout2d_perc == 0.0
        ), f"We can't dropout 2d a KGAdjEmb as it has hidden dim 1"
        # This determines that, when prepping the embeddings, we will query the kg_adj matrix and sum the results - generating
        # one value per entity candidate
        self.kg_adj_process_func = embedding_utils.prep_kg_feature_sum
        self._dim = 1

        self.threshold_weight = 0
        if "threshold" in emb_args:
            self.threshold_weight = float(emb_args.threshold)
        self.log_weight = False
        if "log_weight" in emb_args:
            self.log_weight = emb_args.log_weight
            assert type(self.log_weight) is bool
        log_rank_0_debug(
            logger,
            f"Setting log_weight to be {self.log_weight} and threshold to be {self.threshold_weight} in {key}",
        )

        self.kg_adj, self.prep_file = self.prep(
            data_config=main_args.data_config,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            threshold=self.threshold_weight,
            log_weight=self.log_weight,
        )

    @classmethod
    def prep(
        cls,
        data_config,
        emb_args,
        entity_symbols,
        threshold,
        log_weight,
    ):
        """Preps the KG information.

        Args:
            data_config: data config
            emb_args: embedding args
            entity_symbols: entity symbols
            threshold: weight threshold for counting an edge
            log_weight: whether to take the log of the weight value after the threshold

        Returns: numpy sparce KG adjacency matrix, prep file
        """
        file_tag = os.path.splitext(emb_args.kg_adj)[0]
        prep_dir = data_utils.get_emb_prep_dir(data_config)
        prep_file = os.path.join(prep_dir, f"kg_adj_file_{file_tag}.npz")
        utils.ensure_dir(os.path.dirname(prep_file))
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(logger, f"Loading existing KG adj from {prep_file}")
            start = time.time()
            kg_adj = scipy.sparse.load_npz(prep_file)
            log_rank_0_debug(
                logger, f"Loaded existing KG adj in {round(time.time() - start, 2)}s"
            )
        else:
            start = time.time()
            kg_adj_file = os.path.join(data_config.emb_dir, emb_args.kg_adj)
            log_rank_0_info(logger, f"Building KG adj from {kg_adj_file}")
            kg_adj = cls.build_kg_adj(
                kg_adj_file, entity_symbols, threshold, log_weight
            )
            scipy.sparse.save_npz(prep_file, kg_adj)
            log_rank_0_debug(
                logger,
                f"Finished building and saving KG adj in {round(time.time() - start, 2)}s.",
            )
        return kg_adj, prep_file

    @classmethod
    def build_kg_adj(cls, kg_adj_file, entity_symbols, threshold, log_weight):
        """Builds the KG adjacency matrix from inputs.

        Args:
            kg_adj_file: KG adjacency file
            entity_symbols: entity symbols
            threshold: weight threshold to count as an edge
            log_weight: whether to log the weight after the threshold

        Returns: KG adjacency
        """
        G = nx.Graph()
        qids = set(entity_symbols.get_all_qids())
        edges_to_add = []
        with open(kg_adj_file) as f:
            for line in f:
                head, tail = line.strip().split()
                # head and tail must be in list of qids
                if head in qids and tail in qids:
                    edges_to_add.append((head, tail))
        G.add_edges_from(edges_to_add)
        # convert to entityids
        G = nx.relabel_nodes(G, entity_symbols.get_qid2eid())
        # create adjacency matrix
        adj = nx.adjacency_matrix(
            G, nodelist=range(entity_symbols.num_entities_with_pad_and_nocand)
        )
        assert (
            adj.sum() > 0
        ), f"Your KG Adj matrix has all 0 values. Something was likely parsed wrong."
        # assert that the padded entity has no connections
        assert all(adj.toarray()[:, -1] == 0)
        assert all(adj.toarray()[-1, :] == 0)
        # assert that the unk entity has no connections
        assert all(adj.toarray()[:, 0] == 0)
        assert all(adj.toarray()[0, :] == 0)
        return adj

    # this is called inside the model and packages up the data for the model
    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)
            batch_on_the_fly_data: dict of batch on the fly embeddings

        Returns: B x M x K x 1 tensor of summed KG connection values
        """
        orig_shape = entity_cand_eid.shape
        # needs to be size: batch x m x k x hidden_dim (in this case hidden dim is 1)
        assert self.key in batch_on_the_fly_data, (
            f"KGAdjEmb missing from batch prepped data. It must be there for KGEmb."
            f' Check if you KGEmb looks like "load_class": "<load_class>","batch_prep": true,"args":...'
        )
        kg_feat = (
            torch.as_tensor(batch_on_the_fly_data[self.key])
            .reshape(orig_shape)
            .unsqueeze(-1)
            .float()
        )
        # As hidden dim is 1, we do not normalize or dropout this embedding
        return kg_feat

    def get_dim(self):
        return self._dim

    def __getstate__(self):
        state = self.__dict__.copy()
        # Not picklable
        del state["logger"]
        # we never want to have to pickle entity_symbols since it's so large
        # this module will have to copy attributes during dataloader creation
        # we won't need it again bc we only need it for prep of KG which was done at this point
        del state["entity_symbols"]
        del state["kg_adj"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # we can assume the adjacency matrix has already been built and saved
        self.kg_adj = scipy.sparse.load_npz(self.prep_file)


class KGWeightedAdjEmb(KGAdjEmb):
    """A modified KG adjaceny class that stores a weighted value in the
    adjaceny matrix.

    Args:
        main_args: main args
        emb_args: specific embedding args
        entity_symbols: entity symbols
        key: unique embedding key
        cpu: bool of if one cpu or not
        normalize: bool if normalize embeddings or not
        dropout1d_perc: 1D dropout percent
        dropout2d_perc: 2D dropout percent
    """

    def __init__(
        self,
        main_args,
        emb_args,
        entity_symbols,
        key,
        cpu,
        normalize,
        dropout1d_perc,
        dropout2d_perc,
    ):
        super(KGWeightedAdjEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )

    @classmethod
    def prep(
        cls,
        data_config,
        emb_args,
        entity_symbols,
        threshold,
        log_weight,
    ):
        """Preps the KG information.

        Args:
            data_config: data config
            emb_args: embedding args
            entity_symbols: entity symbols
            threshold: weight threshold for counting an edge
            log_weight: whether to take the log of the weight value after the threshold

        Returns: numpy sparce KG adjacency matrix, prep file
        """
        file_tag = os.path.splitext(emb_args.kg_adj)[0]
        prep_dir = data_utils.get_emb_prep_dir(data_config)
        prep_file = os.path.join(
            prep_dir, f"kg_adj_file_{file_tag}_{threshold}_{log_weight}.npz"
        )
        utils.ensure_dir(os.path.dirname(prep_file))
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(logger, f"Loading existing KG adj from {prep_file}")
            start = time.time()
            kg_adj = scipy.sparse.load_npz(prep_file)
            log_rank_0_debug(
                logger, f"Loaded existing KG adj in {round(time.time() - start, 2)}s"
            )
        else:
            start = time.time()
            kg_adj_file = os.path.join(data_config.emb_dir, emb_args.kg_adj)
            log_rank_0_debug(logger, f"Building KG adj from {kg_adj_file}")
            kg_adj = cls.build_kg_adj(
                kg_adj_file, entity_symbols, threshold, log_weight
            )
            scipy.sparse.save_npz(prep_file, kg_adj)
            log_rank_0_debug(
                logger,
                f"Finished building and saving KG adj in {round(time.time() - start, 2)}s.",
            )
        return kg_adj, prep_file

    @classmethod
    def build_kg_adj(cls, kg_adj_file, entity_symbols, threshold, log_weight):
        """Builds the KG adjacency matrix from inputs.

        Args:
            kg_adj_file: KG adjacency file
            entity_symbols: entity symbols
            threshold: weight threshold to count as an edge
            log_weight: whether to log the weight after the threshold

        Returns: KG adjacency
        """
        G = nx.Graph()
        qids = set(entity_symbols.get_all_qids())
        edges_to_add = []
        num_added = 0
        num_total = 0
        file_ending = os.path.splitext(kg_adj_file)[1][1:]
        # Get ending to determine if txt or json file
        assert file_ending in [
            "json",
            "txt",
        ], f"We only support loading txt or json files for edge weights. You provided {file_ending}"
        with open(kg_adj_file) as f:
            if file_ending == "json":
                all_edges = json.load(f)
                for head in all_edges:
                    for tail in all_edges[head]:
                        weight = all_edges[head][tail]
                        num_total += 1
                        if head in qids and tail in qids and weight > threshold:
                            num_added += 1
                            if log_weight:
                                edges_to_add.append((head, tail, np.log(weight)))
                            else:
                                edges_to_add.append((head, tail, weight))
            else:
                for line in f:
                    splt = line.strip().split()
                    if len(splt) == 2:
                        head, tail = splt
                        weight = 1.0
                    elif len(splt) == 3:
                        head, tail, weight = splt
                    else:
                        raise ValueError(
                            f"A line {line} in {kg_adj_file} has not 2 or 3 values after called split()."
                        )
                    num_total += 1
                    # head and tail must be in list of qids
                    if head in qids and tail in qids and weight > threshold:
                        num_added += 1
                        if log_weight:
                            edges_to_add.append((head, tail, np.log(weight)))
                        else:
                            edges_to_add.append((head, tail, weight))
        log_rank_0_debug(
            logger, f"Adding {num_added} out of {num_total} items from {kg_adj_file}"
        )
        G.add_weighted_edges_from(edges_to_add)
        # convert to entityids
        G = nx.relabel_nodes(G, entity_symbols.get_qid2eid())
        # create adjacency matrix
        adj = nx.adjacency_matrix(
            G, nodelist=range(entity_symbols.num_entities_with_pad_and_nocand)
        )
        assert (
            adj.sum() > 0
        ), f"Your KG Adj matrix has all 0 values. Something was likely parsed wrong."
        # assert that the padded entity has no connections
        assert (adj[:, -1] != 0).sum() == 0
        assert (adj[-1, :] != 0).sum() == 0
        # assert that the unk entity has no connections
        assert (adj[:, 0] != 0).sum() == 0
        assert (adj[0, :] != 0).sum() == 0
        return adj


class KGIndices(KGWeightedAdjEmb):
    """KG indices that is _not_ appended to the entity payload embedding. This
    is used in the KG attention module.

    Forward returns an empty dict. A tensor of batch x M x K x M x K denoting which entity candidates are connected to each other in the KG is stored
    in batch_on_the_fly (see dataset.py).

    The attn_network looks for this component by the key REL_INDICES_KEY (in constants.py) to pull out this kg component. This _must_ be the key in the embedding
    list in the config or it will be ignored. If the attn_network class is extended, then the key used must match that in the config.

    Args:
        main_args: main args
        emb_args: specific embedding args
        entity_symbols: entity symbols
        key: unique embedding key
        cpu: bool of if one cpu or not
        normalize: bool if normalize embeddings or not
        dropout1d_perc: 1D dropout percent
        dropout2d_perc: 2D dropout percent
    """

    def __init__(
        self,
        main_args,
        emb_args,
        entity_symbols,
        key,
        cpu,
        normalize,
        dropout1d_perc,
        dropout2d_perc,
    ):
        super(KGIndices, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        self._dim = 0
        # Weight for the diagonal addition to the KG indices - allows for summing an entity with other connections
        self.kg_bias_weight = torch.nn.Parameter(torch.tensor(2.0))
        self.kg_softmax = nn.Softmax(dim=2)
        # This determines that, when prepping the embeddings, we will query the kg_adj matrix - generating
        # M*K values per entity candidate
        self.kg_adj_process_func = embedding_utils.prep_kg_feature_matrix
        log_rank_0_debug(
            logger,
            f"You are using the KGIndices class with key {key}."
            f" This key need to be used in the attention network to access the kg bias matrix."
            f" This embedding is not appended to the payload.",
        )

    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        """This is the forward of the kg_adjacency matrix that gets used in the
        bias term of our kg attention.

        We do NOT want this appended to the embedding payload. So we do
        not return a tensor here. Instead we add to batch_on_the_fly data
        a B x (M*K) x (M*K) tensor of KG values which is accessed by the neural model.

        Args:
            entity_cand_eid: entity candidate tensor
            batch_on_the_fly_data: Dict of KG embedding key and tensor values

        Returns: None
        """
        # return M x K x hidden size embedding (to be appended)
        batch, M, K = entity_cand_eid.shape
        # needs to be size: batch x m x k x hidden_dim
        assert (
            self.key in batch_on_the_fly_data
        ), f"{self.key} missing from preprocessed data (batch_prep or batch_on_the_fly). It must be there for KGIndices."
        # preprocess matrix to be ready for bmm
        kg_bias = (
            batch_on_the_fly_data[self.key]
            .float()
            .to(entity_cand_eid.device)
            .reshape(batch, M * K, M * K)
        )
        kg_bias_diag = kg_bias + self.kg_bias_weight * torch.eye(M * K).repeat(
            batch, 1, 1
        ).view(batch, M * K, M * K).to(kg_bias.device)
        kg_bias_norm = self.kg_softmax(
            kg_bias_diag.masked_fill((kg_bias_diag == 0), float(-1e9))
        )
        batch_on_the_fly_data[self.key] = kg_bias_norm
        # do not return anything
        return None

    def get_dim(self):
        return self._dim
