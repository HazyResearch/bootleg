"""Entity embeddings"""
import os
import time
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from bootleg.embeddings import EntityEmb
from bootleg.utils import model_utils, logging_utils, utils, data_utils, train_utils


class LearnedEntityEmb(EntityEmb):
    """
    Learned entity embeddings class. We support initializing all learned entity embeddings to be the same value. This helps with
    tail generalization as rare embeddings are close together and less prone to errors in embedding noise.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(LearnedEntityEmb, self).__init__(main_args=main_args, emb_args=emb_args, model_device=model_device,
                                               entity_symbols=entity_symbols, word_symbols=word_symbols, word_emb=word_emb, key=key)
        self.logger = logging_utils.get_logger(main_args)
        self.learned_embedding_size = emb_args.learned_embedding_size
        self.normalize = True
        self.learned_entity_embedding = nn.Embedding(
            entity_symbols.num_entities_with_pad_and_nocand,
            self.learned_embedding_size,
            padding_idx=-1, sparse=True)
        # If tail_init is false, all embeddings are randomly intialized.
        # If tail_init is true, we initialize all embeddings to be the same.
        self.tail_init = True
        self.tail_init_zeros = False
        # None init vec will be random
        init_vec = None
        if "tail_init" in emb_args:
            self.tail_init = emb_args.tail_init
        if "tail_init_zeros" in emb_args:
            self.tail_init_zeros = emb_args.tail_init_zeros
            self.tail_init = False
            init_vec = torch.zeros(1, self.learned_embedding_size)
        assert not (self.tail_init and self.tail_init_zeros), f"Can only have one of tail_init or tail_init_zeros set"
        if self.tail_init or self.tail_init_zeros:
            if self.tail_init_zeros:
                self.logger.debug(f"All learned entity embeddings are initialized to zero.")
            else:
                self.logger.debug(f"All learned entity embeddings are initialized to the same value.")
            init_vec = model_utils.init_embeddings_to_vec(self.learned_entity_embedding, pad_idx=-1, vec=init_vec)
            vec_save_file = os.path.join(train_utils.get_save_folder(main_args.run_config), "init_vec_entity_embs.npy")
            self.logger.debug(f"Saving init vector to {vec_save_file}")
            np.save(vec_save_file, init_vec)
        else:
            self.logger.debug(f"All learned embeddings are randomly initialized.")
        self._dim = main_args.model_config.hidden_size
        self.eid2reg = None
        # Regularization mapping goes from eid to 2d dropout percent
        if "regularize_mapping" in emb_args:
            self.logger.debug(f"Using regularization mapping in enity embedding from {emb_args.regularize_mapping}")
            self.eid2reg = self.load_regularization_mapping(main_args, entity_symbols, emb_args.regularize_mapping, self.logger.debug)
            self.eid2reg = self.eid2reg.to(model_device)


    def load_regularization_mapping(cls, main_args, entity_symbols, reg_file, log_func):
        """
        Reads in a csv file with columns [qid, regularization].
        In the forward pass, the entity id with associated qid will be regularized with probability regularization.
        """
        reg_str = reg_file.split(".csv")[0]
        prep_dir = data_utils.get_data_prep_dir(main_args)
        prep_file = os.path.join(prep_dir,
            f'entity_regularization_mapping_{reg_str}.pt')
        utils.ensure_dir(os.path.dirname(prep_file))
        log_func(f"Looking for regularization mapping in {prep_file}")
        if (not main_args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file)):
            log_func(f'Loading existing entity regularization mapping from {prep_file}')
            start = time.time()
            eid2reg = torch.load(prep_file)
            log_func(f'Loaded existing entity regularization mapping in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            reg_file = os.path.join(main_args.data_config.data_dir, reg_file)
            log_func(f'Building entity regularization mapping from {reg_file}')
            qid2reg = pd.read_csv(reg_file)
            assert "qid" in qid2reg.columns and "regularization" in qid2reg.columns, f"Expected qid and regularization as the column names for {reg_file}"
            # default of no mask
            eid2reg_arr = [0.0]*entity_symbols.num_entities_with_pad_and_nocand
            for row_idx, row in qid2reg.iterrows():
                eid = entity_symbols.get_eid(row["qid"])
                eid2reg_arr[eid] = row["regularization"]
            eid2reg = torch.tensor(eid2reg_arr)
            torch.save(eid2reg, prep_file)
            log_func(f"Finished building and saving entity regularization mapping in {round(time.time() - start, 2)}s.")
        return eid2reg

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        tensor = self.learned_entity_embedding(entity_package.tensor)
        if self.eid2reg is not None:
            regularization_tensor = self.eid2reg[entity_package.tensor.long()]
            tensor = model_utils.emb_dropout_by_tensor(self.training, regularization_tensor, tensor)
        emb = self._package(tensor=tensor, pos_in_sent=entity_package.pos_in_sent, alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return emb

    def get_dim(self):
        return self.learned_embedding_size


class StaticEmb(EntityEmb):
    """Loads a static embedding for each entity."""
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(StaticEmb, self).__init__(main_args=main_args, emb_args=emb_args, model_device=model_device,
                                          entity_symbols=entity_symbols, word_symbols=word_symbols, word_emb=word_emb,
                                          key=key)
        self.logger = logging_utils.get_logger(main_args)
        self.model_device = model_device
        entity2static_table = self.prep(main_args=main_args, emb_args=emb_args, word_symbols=word_symbols, entity_symbols=entity_symbols,
            log_func=self.logger.debug)
        self.orig_dim = entity2static_table.shape[1]
        self.entity2static_embedding = torch.nn.Embedding(
            entity_symbols.num_entities_with_pad_and_nocand,
            self.orig_dim,
            padding_idx=-1,
            sparse=True)
        self.normalize = False
        self.logger.debug(f"{key}: StaticEmb with {self.orig_dim} dimension embedding")
        if self.orig_dim > 1:
            self.normalize = True
        self.logger.debug(f"{key}: Setting normalize to True")
        self._dim = self.orig_dim
        self.entity2static_embedding.weight.data[:].copy_(entity2static_table)
        self.entity2static_embedding.weight.requires_grad = False

    @classmethod
    def prep(cls, main_args, emb_args, word_symbols, entity_symbols, log_func=print):
        static_str = os.path.splitext(emb_args.emb_file)[0]
        prep_dir = data_utils.get_emb_prep_dir(main_args)
        prep_file = os.path.join(prep_dir,
            f'static_table_{static_str}.pt')
        log_func(f"Looking for static embedding saved at {prep_file}")
        if (not main_args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file)):
            log_func(f'Loading existing static embedding table from {prep_file}')
            start = time.time()
            entity2staticemb_table = torch.load(prep_file)
            log_func(f'Loaded existing static embedding table in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            emb_file = os.path.join(main_args.data_config.emb_dir, emb_args.emb_file)
            log_func(f'Building static table from file {emb_file}')
            entity2staticemb_table = cls.build_static_embeddings(emb_file=emb_file, entity_symbols=entity_symbols)
            torch.save(entity2staticemb_table, prep_file)
            log_func(f"Finished building and saving static embedding table in {round(time.time() - start, 2)}s.")
        return entity2staticemb_table

    @classmethod
    def build_static_embeddings(cls, emb_file, entity_symbols):
        """Builds the table of the embedding associated with each entity."""
        ending = os.path.splitext(emb_file)[1]
        if ending == ".glove":
            embeddings, embedding_size = data_utils.load_glove(emb_file, log_func=print)
        elif ending == ".json":
            dct = utils.load_json_file(emb_file)
            val = next(iter(dct.values()))
            if type(val) is int or type(val) is float:
                embedding_size = 1
                conver_func = lambda x: np.array([x])
            elif type(val) is list:
                embedding_size = len(val)
                conver_func = lambda x: np.array([y for y in x])
            else:
                raise ValueError(f"Unrecognized type for the array value of {type(val)}")
            embeddings = {}
            for k in dct:
                embeddings[k] = conver_func(dct[k])
                assert len(embeddings[k]) == embedding_size
        else:
            raise ValueError(f"We do not support static embeddings from {ending}. We only support .glove or .json")
        entity2staticemb_table = torch.zeros(entity_symbols.num_entities_with_pad_and_nocand, embedding_size)
        found = 0
        for qid in tqdm(entity_symbols.get_all_qids()):
            if qid in embeddings:
                found += 1
                emb = embeddings[qid]
                eid = entity_symbols.get_eid(qid)
                entity2staticemb_table[eid, :embedding_size] = torch.from_numpy(emb)
        print(f"Found {found/len(entity_symbols.get_all_qids())} percent of all entities have a static embedding")
        return entity2staticemb_table


    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        tensor = self.entity2static_embedding(entity_package.tensor)
        emb = self._package(tensor=tensor, pos_in_sent=entity_package.pos_in_sent, alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return emb

    def get_dim(self):
        return self._dim