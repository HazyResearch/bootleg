"""Knowledge graph embeddings"""
import os
import pickle
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse
import ujson as json

from bootleg.embeddings import EntityEmb
from bootleg.layers.layers import *
from bootleg.utils import logging_utils, data_utils, utils
from bootleg.utils.model_utils import selective_avg


class KGAdjEmb(EntityEmb):
    """
    KG adjaceny base embedding class that stores a statistical feature for pairs of entities. Uses numpy sparse matrices
    to store the features.

    Attributes:
        mask_candidates: to mask (set to 0) the connections between candidates of a single mention
        _dim: dimension of the embedding

    Implements batch_prep method that queries the sparse KG matrix.
    Forward pass returns batch x M x K x _dim feature
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols,
        word_symbols=None, word_emb=None, key=""):
        super(KGAdjEmb, self).__init__(main_args=main_args, emb_args=emb_args,
            model_device=model_device, entity_symbols=entity_symbols,
            word_symbols=word_symbols, word_emb=word_emb, key=key)
        # Needed to recreate logger
        self.main_args = main_args
        # Set normalize attribute to be False. This embedding is a single statistical value and should not be
        # normalized as it would become 1
        self.normalize = False
        self.logger = logging_utils.get_logger(main_args)
        self.mask_candidates = True
        self.logger.debug(f'{key}: Mask candidates value {self.mask_candidates}, normalize {self.normalize}')
        self._dim = 1
        self.model_device = model_device
        self.kg_adj, self.prep_file = self.prep(main_args=main_args, emb_args=emb_args,
            entity_symbols=entity_symbols, log_func=self.logger.debug)

    @classmethod
    def prep(cls, main_args, emb_args, entity_symbols, log_func=print, word_symbols=None):
        """Loads KG information into matrix format or loads from saved state"""
        file_tag = os.path.splitext(emb_args.kg_adj)[0]
        prep_dir = data_utils.get_emb_prep_dir(main_args)
        prep_file = os.path.join(prep_dir,
            f'kg_adj_file_{file_tag}.npz')
        utils.ensure_dir(os.path.dirname(prep_file))
        if (not main_args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file)):
            log_func(f'Loading existing KG adj from {prep_file}')
            start = time.time()
            kg_adj = scipy.sparse.load_npz(prep_file)
            log_func(f'Loaded existing KG adj in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            kg_adj_file = os.path.join(main_args.data_config.emb_dir, emb_args.kg_adj)
            log_func(f'Building KG adj from {kg_adj_file}')
            kg_adj = cls.build_kg_adj(kg_adj_file, entity_symbols, emb_args)
            scipy.sparse.save_npz(prep_file, kg_adj)
            log_func(f"Finished building and saving KG adj in {round(time.time() - start, 2)}s.")
        return kg_adj, prep_file

    @classmethod
    def build_kg_adj(cls, kg_adj_file, entity_symbols, emb_args):
        """Constructs the KG adjacency matrix from file. Build a binary adjacency matrix if two entity are connected"""
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
        adj = nx.adjacency_matrix(G, nodelist=range(entity_symbols.num_entities_with_pad_and_nocand))
        return adj

    def preprocess_adj(self, adj, entity_indices):
        """For each entity, sums the adjacency value of all other candidate in the sentence"""
        M, K = entity_indices.shape
        # ensure we are using numpy (code is different for torch versus numpy)
        entity_indices = np.array(entity_indices)
        entity_indices = entity_indices.flatten()
        # use entity ids to extract MxK
        # format for CSR matrix must be like: [[0,2],[[0],[2]]]
        # subset_adj = self.kg_adj[entity_indices, entity_indices.unsqueeze(1)]
        subset_adj = adj[entity_indices, np.expand_dims(entity_indices,1)]
        subset_adj = subset_adj.toarray()
        if self.mask_candidates:
            # mask out noisy connectivity to candidates of same alias
            single_mask = np.array([[True]*K]*K)
            # https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
            full_mask = np.kron(np.eye(M,dtype=bool),single_mask)
            subset_adj[full_mask] = 0
        # do sum over all candidates for MxK candidates
        kg_feat = np.squeeze(subset_adj.sum(1))
        # return summed MxK feature indicating a candidates relatedness to other aliases' candidates
        return kg_feat

    # this is defines what the dataloader calls to prepare data for the batch
    def batch_prep(self, alias_indices, entity_indices):
        """Called from prep or dataloader"""
        return self.preprocess_adj(self.kg_adj, entity_indices)

    # this is called inside the model and packages up the data for the model
    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        orig_shape = entity_package.tensor.shape
        # needs to be size: batch x m x k x hidden_dim (in this case hidden dim is 1)
        assert self.key in batch_prepped_data or self.key in batch_on_the_fly_data, f'KGAdjEmb missing from batch prepped data. It must be there for KGEmb.' \
                                                             f' Check if you KGEmb looks like "load_class": "<load_class>","batch_prep": true,"args":...'
        if self.key in batch_prepped_data:
            kg_feat = batch_prepped_data[self.key].reshape(orig_shape).unsqueeze(-1).float()
        else:
            kg_feat = torch.as_tensor(batch_on_the_fly_data[self.key]).reshape(orig_shape).unsqueeze(-1).float()
        res = self._package(tensor=kg_feat.to(self.model_device),
                            pos_in_sent=entity_package.pos_in_sent,
                            alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return res

    def get_dim(self):
        return self._dim

    def __getstate__(self):
        state = self.__dict__.copy()
        # Not picklable
        del state['logger']
        # we never want to have to pickle entity_symbols since it's so large
        # this module will have to copy attributes during dataloader creation
        # we won't need it again bc we only need it for prep of KG which was done at this point
        del state['entity_symbols']
        del state['kg_adj']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = logging_utils.get_logger(self.main_args)
        # we can assume the adjacency matrix has already been built and saved
        self.kg_adj = scipy.sparse.load_npz(self.prep_file)


class KGIndices(KGAdjEmb):
    """
    KG indices that is _not_ appended to the entity payload embedding. This is used in the KG attention module.

    Attributes:
        _dim: 0 as it is not part of the payload

    Forward returns an empty dict. A tensor of batch x M x K x M x K denoting which entity candidates are connected to each other in the KG is stored
    in batch_prep data or batch_on_the_fly (see wiki_dataset.py).

    The attn_network looks for this component by the key REL_INDICES_KEY (in constants.py) to pull out this kg component. This _must_ be the key in the embedding
    list in the config or it will be ignored. If the attn_network class is extended, then the key used must match that in the config.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols,
        word_symbols=None, word_emb=None, key=""):
        super(KGIndices, self).__init__(main_args=main_args, emb_args=emb_args,
            model_device=model_device, entity_symbols=entity_symbols,
            word_symbols=word_symbols, word_emb=word_emb, key=key)
        # needed to recreate logger
        self.main_args = main_args
        self.logger = logging_utils.get_logger(main_args)
        self.model_device = model_device
        self.mask_candidates = True
        self.kg_adj, self.prep_file = self.prep(main_args=main_args, emb_args=emb_args,
            entity_symbols=entity_symbols, log_func=self.logger.debug)
        self._dim = 0
        self.logger.debug(f"You are using the KGIndices class with key {key}."
                         f" This key need to be used in the attention network to access the kg bias matrix."
                         f" This embedding is not appended to the payload.")

    @classmethod
    def build_kg_adj(cls, kg_adj_file, entity_symbols, emb_args):
        """This class sets the adjacency value to be the log of the value stored in the value if that value is > 10. This can be overwritten and
        demonstrates how to overwrite the adjacency class."""
        G = nx.Graph()
        qids = set(entity_symbols.get_all_qids())
        edges_to_add = []
        num_added = 0
        num_total = 0
        file_ending = os.path.splitext(kg_adj_file)[1][1:]
        # Get ending to determine if txt or json file
        assert file_ending in ["json", "txt"], f"We only support loading txt or json files for edge weights. You provided {file_ending}"
        with open(kg_adj_file) as f:
            if file_ending == "json":
                all_edges = json.load(f)
                for head in all_edges:
                    for tail in all_edges[head]:
                        weight = all_edges[head][tail]
                        num_total += 1
                        if head in qids and tail in qids and weight > 10:
                            num_added += 1
                            edges_to_add.append((head, tail, np.log(weight)))
            else:
                for line in f:
                    head, tail = line.strip().split()
                    num_total += 1
                    # head and tail must be in list of qids
                    if head in qids and tail in qids:
                        num_added += 1
                        edges_to_add.append((head, tail, 1.0))
        print(f"Adding {num_added} out of {num_total} cooccurrence items from {kg_adj_file}")
        G.add_weighted_edges_from(edges_to_add)
        # convert to entityids
        G = nx.relabel_nodes(G, entity_symbols.get_qid2eid())
        # create adjacency matrix
        adj = nx.adjacency_matrix(G, nodelist=range(entity_symbols.num_entities_with_pad_and_nocand))
        return adj

    def batch_prep(self, alias_indices, entity_indices):
        """Queries the adjacency matrix"""
        M, K = entity_indices.shape
        # ensure we are using numpy (code is different for torch versus numpy)
        entity_indices = np.array(entity_indices)
        entity_indices = entity_indices.flatten()
        # use entity ids to extract MxK
        # format for CSR matrix must be like: [[0,2],[[0],[2]]]
        # subset_adj = self.kg_adj[entity_indices, entity_indices.unsqueeze(1)]
        subset_adj = self.kg_adj[entity_indices, np.expand_dims(entity_indices,1)]
        subset_adj = subset_adj.toarray()
        if self.mask_candidates:
            # mask out noisy connectivity to candidates of same alias
            single_mask = np.array([[True]*K]*K)
            # https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
            full_mask = np.kron(np.eye(M,dtype=bool),single_mask)
            subset_adj[full_mask] = 0
        return subset_adj.flatten()

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        """This is the forward of the kg_adjacency matrix that gets used in the bias term of our kg attention. We do NOT want this
        appended to the embedding payload. So we do not return a tensor here. As this must be in batch_prep/batch_on_the_fly, the model
        will access the bias through those dicts"""
        # return M x K x hidden size embedding (to be appended)
        batch, M, K = entity_package.tensor.shape
        # needs to be size: batch x m x k x hidden_dim
        assert self.key in batch_prepped_data or self.key in batch_on_the_fly_data, f'{self.key} missing from preprocessed data (batch_prep or batch_on_the_fly). It must be there for KGIndices.'
        # makes sure we can load the right shape
        if self.key in batch_prepped_data:
            subset_adj = torch.as_tensor(batch_prepped_data[self.key]).reshape(batch, M, K, M*K)
        else:
            subset_adj = torch.as_tensor(batch_on_the_fly_data[self.key]).reshape(batch, M, K, M*K)
        # do not return anything to be appended
        res = {}
        return res

    def get_dim(self):
        return self._dim


class KGRelEmbBase(EntityEmb):
    """
    KG relation embedding base class. This generates contextualized relation embeddings based on the candidates in the sentence.

    Attributes:
        merge_func: determines how the relations shared across candidate will be merged for a single candidate.
                    Support average, softattn, and addattn. Specified in config.

    Forward returns batch x M x K x dim relation embedding.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols,
        word_symbols=None, word_emb=None, key=""):
        super(KGRelEmbBase, self).__init__(main_args=main_args, emb_args=emb_args,
            model_device=model_device, entity_symbols=entity_symbols,
            word_symbols=word_symbols, word_emb=word_emb, key=key)

    @classmethod
    def prep(cls, main_args, emb_args, entity_symbols, log_func=print, word_symbols=None):
        # prep adjacency matrix
        # weights in adjacency matrix correspond to relation
        file_tag = os.path.splitext(os.path.basename(emb_args.kg_triples))[0]
        prep_dir = data_utils.get_emb_prep_dir(main_args)
        prep_file_adj = os.path.join(prep_dir,
            f'kg_triples_file_{file_tag}.npz')
        prep_file_relations = os.path.join(prep_dir,
            f'kg_relations_{file_tag}.pkl')
        prep_file_table = os.path.join(prep_dir,
            f'rel_table_{file_tag}.pkl')
        if (not main_args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file_adj) and os.path.exists(prep_file_relations)
            and os.path.exists(prep_file_table)):
            log_func(f'Loading existing KG adj from {prep_file_adj} and {prep_file_relations}')
            start = time.time()
            kg_triples = scipy.sparse.load_npz(prep_file_adj)
            kg_relations = pickle.load(open(prep_file_relations, 'rb'))
            rel_table = pickle.load(open(prep_file_table, 'rb'))
            log_func(f'Loaded existing KG adj in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            kg_triples_file = os.path.join(main_args.data_config.emb_dir, emb_args.kg_triples)
            log_func(f'Building KG adj from {kg_triples_file}')
            kg_triples, kg_relations, rel_table = cls.build_kg_triples(kg_triples_file, entity_symbols,
                relations=emb_args.relations, log_func=log_func)
            scipy.sparse.save_npz(prep_file_adj, kg_triples)
            pickle.dump(kg_relations, open(prep_file_relations, 'wb'))
            pickle.dump(rel_table, open(prep_file_table, 'wb'))
            log_func(f"Finished building and saving KG adj in {round(time.time() - start, 2)}s.")
        return kg_triples, kg_relations, rel_table, prep_file_adj, prep_file_table

    # This method appends multiple relations for pair of QIDs
    @classmethod
    def build_kg_triples(cls, kg_adj_file, entity_symbols, relations, log_func):
        """Reads in file KG related entity pairs and the relations between them. The adjaceny matrix stores the index into a table with the multiple
        relation ids."""
        G = nx.Graph()
        qids = set(entity_symbols.get_all_qids())
        rel_map = defaultdict(set)
        max_rels = -1
        # build map
        all_relations = set()
        with open(kg_adj_file) as f:
            for line in f:
                head, rel, tail = line.strip().split()
                # TODO: remove this when we don't want to constrain relations
                if len(relations) > 0 and (rel not in set(relations)):
                    continue
                # head and tail must be in list of qids
                if head in qids and tail in qids:
                    all_relations.add(rel)
                    first, second = sorted([head, tail])
                    rel_map[(first, second)].add(rel)
                    if len(rel_map[(first, second)]) > max_rels:
                        max_rels = len(rel_map[(first, second)])
        log_func(f"Max Rels {max_rels}")
        # 0 means no relation exist and will be sparse so we add one
        relation_mapping = {r:i+1 for i,r in enumerate(sorted(list(all_relations)))}
        # store in table for fast access
        # leave the first row zero for no relation
        rel_table = np.zeros((len(rel_map)+1, max_rels), np.int16)
        weighted_edges_to_add = []
        for i, ((u,v), vals) in enumerate(rel_map.items()):
            idx = i+1
            rel_table[idx][:len(vals)] = np.array([relation_mapping[r] for r in vals])
            # the weight is the id into the table of triplets
            weighted_edges_to_add.append((u, v, idx))

        G.add_weighted_edges_from(weighted_edges_to_add)
        # convert to entityids
        G = nx.relabel_nodes(G, entity_symbols.get_qid2eid())
        # create adjacency matrix
        adj = nx.adjacency_matrix(G, nodelist=range(entity_symbols.num_entities_with_pad_and_nocand))
        return adj, all_relations, rel_table

    def batch_prep(self, alias_indices, entity_indices):
        """Generates the matrix of the relation ids for the set of entity_indices."""
        M, K = entity_indices.shape
        # ensure we are using numpy (code is different for torch versus numpy)
        entity_indices = np.array(entity_indices)
        entity_indices = entity_indices.flatten()
        # use entity ids to extract MxK
        # format for CSR matrix must be like: [[0,2],[[0],[2]]]
        # subset_adj = self.kg_adj[entity_indices, entity_indices.unsqueeze(1)]
        subset_adj = self.kg_adj[entity_indices, np.expand_dims(entity_indices,1)]
        subset_adj = subset_adj.toarray()
        if self.mask_candidates:
            # mask out noisy connectivity to candidates of same alias
            single_mask = np.array([[True]*K]*K)
            # https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
            full_mask = np.kron(np.eye(M,dtype=bool),single_mask)
            subset_adj[full_mask] = 0
        # KG_feat should be (MxK) x (MxK)
        # return MxK for each candidate but K corresponding to self should be masked out
        return subset_adj

    def average_rels(self, entity_package, rel_ids, batch_rel_emb, sent_emb):
        """For each candidate, average the relation embs it shares with other candidates in the sentence."""
        return selective_avg(ids=rel_ids, mask=rel_ids>0, embeds=batch_rel_emb)

    def soft_attn_merge(self, entity_package, rel_ids, batch_rel_emb, sent_emb):
        """For each candidate, use a weighted average of the relation embs it shares with the other candidates based on relation embedding
        similarity to the contextualized mention embedding."""
        # st2 = time.time()
        # Get alias tensor and expand to be for each candidate for soft attn
        batch, M, K, _ = rel_ids.shape
        alias_word_tensor = model_utils.select_alias_word_sent(entity_package.pos_in_sent, sent_emb, index=0)
        _, _, sent_dim = alias_word_tensor.shape
        alias_word_tensor = alias_word_tensor.unsqueeze(2).expand(batch, M, K, sent_dim)
        alias_word_tensor = alias_word_tensor.contiguous().reshape(batch*M*K, sent_dim)

        rel_ids = rel_ids.reshape(batch*M*K, M*K)

        # Need to reduce memory. As we only need to softly "choose" over the nonzero relations in each list, we can subselect the rows/cols based on
        # nonzezro rel_ids. As the process merges relations in the soft attention, we do not need to reconstruct them at the end. Hence
        # we do not store indices.
        full_mask = (rel_ids>0) # True mask are kept
        rows_to_keep = (full_mask.sum(-1) > 0)
        cols_to_keep = (full_mask.sum(0) > 0)
        non_zero_rel_ids = rel_ids[:,cols_to_keep]
        mask = (non_zero_rel_ids>0)
        # Get embeddings
        batch_rel_emb = batch_rel_emb.reshape(batch*M*K, M*K, -1)
        batch_rel_emb = batch_rel_emb[:,cols_to_keep]
        assert batch_rel_emb.size(0) == batch*M*K
        dim = batch_rel_emb.size(2)
        # If no rows to keep (i.e., all embeddings are zero)
        if rows_to_keep.sum() == 0:
            assert batch_rel_emb.sum() == 0
            return torch.zeros(batch, M, K, dim).to(self.model_device)
        # Remove rows that have no relation id to save memory
        batch_rel_emb_subset = batch_rel_emb[rows_to_keep]
        alias_word_tensor_subset = alias_word_tensor[rows_to_keep]
        mask_subset = mask[rows_to_keep]
        # Get soft attn
        # Break it into span chunks to try to control the memory usage in the case of an outlier batch with many connections
        batch_rel_emb_subset_list = []
        for idx in range(0, batch_rel_emb_subset.shape[0], self.sub_chunk):
            batch_rel_emb_averaged = self.soft_attn(batch_rel_emb_subset[idx:idx+self.sub_chunk,:], alias_word_tensor_subset[idx:idx+self.sub_chunk,:], mask=mask_subset[idx:idx+self.sub_chunk,:])
            batch_rel_emb_subset_list.append(batch_rel_emb_averaged)

        new_batch_rel_emb = torch.zeros(batch*M*K, dim).to(self.model_device)
        new_batch_rel_emb[rows_to_keep] = torch.cat(batch_rel_emb_subset_list, dim=0)

        # Convert batch back to original shape
        new_batch_rel_emb = new_batch_rel_emb.reshape(batch, M, K, dim)
        return new_batch_rel_emb

    def add_attn_merge(self, entity_package, rel_ids, batch_rel_emb, sent_emb):
        """For each candidate, use a weighted average of the relation embs it shares with the other candidates based on a simple additive attention."""
        batch, M, K, num_rels, dim = batch_rel_emb.shape
        # we don't want to compute probabilities over padded rels
        mask = (rel_ids>0) # True mask are kept
        # Reshape for add attn
        batch_rel_emb = batch_rel_emb.contiguous().reshape(batch*M*K, num_rels, dim)
        # Get add attn
        batch_rel_emb = self.add_attn(batch_rel_emb, mask=mask)
        # Convert batch back to original shape
        batch_rel_emb = batch_rel_emb.reshape(batch, M, K, dim)
        return batch_rel_emb

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        raise ValueError('Not implemented')

    def get_dim(self):
        raise ValueError('Not implemented')

    def __getstate__(self):
        state = self.__dict__.copy()
        # Not picklable
        del state['logger']
        # we never want to have to pickle entity_symbols since it's so large
        # this module will have to copy attributes during dataloader creation
        # we won't need it again bc we only need it for prep of KG which was done at this point
        del state['entity_symbols']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = logging_utils.get_logger(self.main_args)


class KGRelEmb(KGRelEmbBase):
    """
    KG relation embedding class.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols,
        word_symbols=None, word_emb=None, key=""):
        super(KGRelEmb, self).__init__(main_args=main_args, emb_args=emb_args,
            model_device=model_device, entity_symbols=entity_symbols,
            word_symbols=word_symbols, word_emb=word_emb, key=key)
        # needed to recreate logger
        self.main_args = main_args
        self.logger = logging_utils.get_logger(main_args)
        self.model_device = model_device
        self.mask_candidates = True
        self.kg_adj, self.kg_relations, self.rel2rowid, self.prep_file_adj = self.prep(main_args=main_args, emb_args=emb_args,
            entity_symbols=entity_symbols, log_func=self.logger.debug)
        # initialize learned relation embedding
        self.num_relations_with_pad = len(self.kg_relations) + 1
        self._dim = emb_args.rel_dim
        # Sparse cannot be true for the relation embedding or we get distributed Gloo errors depending on the batch (Gloo EnforceNotMet...)
        self.relation_emb = torch.nn.Embedding(self.num_relations_with_pad, self._dim, padding_idx=0, sparse=False).to(model_device)
        self.merge_func = self.average_rels
        if "merge_func" in emb_args:
            if emb_args.merge_func not in ["average", "softattn", "addattn"]:
                self.logger.warning(f"{key}: You have set the type merge_func to be {emb_args.merge_func} but that is not in the allowable list of [average, sofftattn]")
            elif emb_args.merge_func == "softattn":
                if "attn_hidden_size" in emb_args:
                    attn_hidden_size = emb_args.attn_hidden_size
                else:
                    attn_hidden_size = 100
                self.sub_chunk = 3000
                self.logger.debug(f"{key}: Setting merge_func to be soft_attn in relation emb with context size {attn_hidden_size} and sub_chunk of {self.sub_chunk}")
                # Softmax of types using the sentence context
                self.soft_attn = SoftAttn(emb_dim=self._dim, context_dim=main_args.model_config.hidden_size, size=attn_hidden_size)
                self.merge_func = self.soft_attn_merge
            elif emb_args.merge_func == "addattn":
                if "attn_hidden_size" in emb_args:
                    attn_hidden_size = emb_args.attn_hidden_size
                else:
                    attn_hidden_size = 100
                self.logger.info(f"{key}: Setting merge_func to be add_attn in relation emb with context size {attn_hidden_size}")
                # Softmax of types using the sentence context
                self.add_attn = PositionAwareAttention(input_size=self._dim, attn_size=attn_hidden_size, feature_size=0)
                self.merge_func = self.add_attn_merge
        self.normalize = True
        self.logger.debug(f'{key}: Using {self._dim}-dim relation embedding for {len(self.kg_relations)} relations with 1 pad relation. Normalize is {self.normalize}')

    @classmethod
    def prep(cls, main_args, emb_args, entity_symbols, log_func=print, word_symbols=None):
        # prep adjacency matrix
        # weights in adjacency matrix correspond to relation
        file_tag = os.path.splitext(os.path.basename(emb_args.kg_triples))[0]
        prep_dir = data_utils.get_emb_prep_dir(main_args)
        prep_file_adj = os.path.join(prep_dir,
            f'kg_triples_file_{file_tag}.npz')
        prep_file_relations = os.path.join(prep_dir,
            f'kg_relations_{file_tag}.pkl')
        prep_file_relidmap = os.path.join(prep_dir,
            f'kg_relid2rowid_{file_tag}.pkl')
        if (not main_args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file_adj) and os.path.exists(prep_file_relations) and os.path.exists(prep_file_relidmap)):
            log_func(f'Loading existing KG adj from {prep_file_adj} and {prep_file_relations}')
            start = time.time()
            kg_triples = scipy.sparse.load_npz(prep_file_adj)
            kg_relations = pickle.load(open(prep_file_relations, 'rb'))
            rel2rowid = pickle.load(open(prep_file_relidmap, 'rb'))
            log_func(f'Loaded existing KG adj in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            kg_triples_file = os.path.join(main_args.data_config.emb_dir, emb_args.kg_triples)
            log_func(f'Building KG adj from {kg_triples_file}')
            kg_triples, rel2rowid, kg_relations = cls.build_kg_triples(kg_triples_file, entity_symbols,
                relations=emb_args.relations, log_func=log_func)
            scipy.sparse.save_npz(prep_file_adj, kg_triples)
            pickle.dump(kg_relations, open(prep_file_relations, 'wb'))
            pickle.dump(rel2rowid, open(prep_file_relidmap, 'wb'))
            log_func(f"Finished building and saving KG adj in {round(time.time() - start, 2)}s.")
        return kg_triples, kg_relations, rel2rowid, prep_file_adj

    # This methods takes a single relation per pair of related QIDs
    @classmethod
    def build_kg_triples(cls, kg_adj_file, entity_symbols, relations, log_func):
        """Reads in file KG related entity pairs and a _single_ relation between them (a simplifying assumption). The adjaceny matrix stores the _single_ relation index."""
        G = nx.Graph()
        qids = set(entity_symbols.get_all_qids())
        # Only store one relation
        rel_map = defaultdict(int)
        # build map
        all_relations = set()
        with open(kg_adj_file) as f:
            for line in f:
                head, rel, tail = line.strip().split()
                # TODO: remove this when we don't want to constrain relations
                if len(relations) > 0 and (rel not in set(relations)):
                    continue
                # head and tail must be in list of qids
                all_relations.add(rel)
                if head in qids and tail in qids:
                    first, second = sorted([head, tail])
                    # Takes first relation
                    if (first, second) not in rel_map:
                        rel_map[(first, second)] = rel
        # 0 means no relation exist
        relation_mapping = {r:i+1 for i,r in enumerate(sorted(list(all_relations)))}
        weighted_edges_to_add = []
        for i, ((u,v), val) in enumerate(rel_map.items()):
            # the weight is the id into the table of triplets
            weighted_edges_to_add.append((u, v, relation_mapping[val]))
        G.add_weighted_edges_from(weighted_edges_to_add)
        # convert to entityids
        G = nx.relabel_nodes(G, entity_symbols.get_qid2eid())
        # create adjacency matrix
        adj = nx.adjacency_matrix(G, nodelist=range(entity_symbols.num_entities_with_pad_and_nocand))
        return adj, relation_mapping, all_relations


    def batch_prep(self, alias_indices, entity_indices):
        M, K = entity_indices.shape
        # rel_ids is (MxK) x (MxK) where the value is the relation id
        rel_ids = super(KGRelEmb, self).batch_prep(alias_indices, entity_indices)
        assert list(rel_ids.shape) == [(M*K), (M*K)]
        return rel_ids.astype(np.int16).flatten()

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        # return M x K x hidden size embedding (to be appended)
        batch, M, K = entity_package.tensor.shape
        # needs to be size: batch x m x k x hidden_dim
        assert self.key in batch_prepped_data or self.key in batch_on_the_fly_data, f"Embedding {self.key} either needs to have batch_prep true or batch_on_the_fly true"
        if self.key in batch_prepped_data:
            rel_ids = batch_prepped_data[self.key].clone().detach().reshape(batch, M, K, M*K).long()
        else:
            rel_ids = batch_on_the_fly_data[self.key].clone().detach().reshape(batch, M, K, M*K).long()
        batch_rel_emb = self.relation_emb(rel_ids)
        batch_rel_emb = self.merge_func(entity_package, rel_ids, batch_rel_emb, sent_emb)
        res = self._package(tensor=batch_rel_emb.to(self.model_device),
                            pos_in_sent=entity_package.pos_in_sent,
                            alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return res

    def get_dim(self):
        return self._dim


class KGWeightedAdjEmb(KGAdjEmb):
    """
    A modified KG adjaceny class that stores a weighted value in the adjaceny matrix.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols,
        word_symbols=None, word_emb=None, key=""):
        super(KGWeightedAdjEmb, self).__init__(main_args=main_args, emb_args=emb_args,
            model_device=model_device, entity_symbols=entity_symbols,
            word_symbols=word_symbols, word_emb=word_emb, key=key)
        # needed to recreate logger
        self.main_args = main_args
        self.logger = logging_utils.get_logger(main_args)
        self._dim = 1
        self.model_device = model_device
        self.kg_adj, self.prep_file = self.prep(main_args=main_args, emb_args=emb_args,
            entity_symbols=entity_symbols, log_func=self.logger.debug)

    @classmethod
    def build_kg_adj(cls, kg_adj_file, entity_symbols, emb_args):
        """This class sets the adjacency value to be the log of the value stored in the value if that value is > 10. This can be overwritten and
        demonstrates how to overwrite the adjacency class."""
        G = nx.Graph()
        qids = set(entity_symbols.get_all_qids())
        edges_to_add = []
        num_added = 0
        num_total = 0
        with open(kg_adj_file) as f:
            all_edges = json.load(f)
            for head in all_edges:
                for tail in all_edges[head]:
                    weight = all_edges[head][tail]
                    num_total += 1
                    if head in qids and tail in qids and weight > 10:
                        num_added += 1
                        edges_to_add.append((head, tail, np.log(weight)))
        print(f"Adding {num_added} out of {num_total} cooccurrence items")
        G.add_weighted_edges_from(edges_to_add)
        # convert to entityids
        G = nx.relabel_nodes(G, entity_symbols.get_qid2eid())
        # create adjacency matrix
        adj = nx.adjacency_matrix(G, nodelist=range(entity_symbols.num_entities_with_pad_and_nocand))
        return adj
