"""Entity embeddings."""
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import emmental
from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.embeddings import EntityEmb
from bootleg.layers.helper_modules import MLP
from bootleg.utils import data_utils, model_utils, utils

logger = logging.getLogger(__name__)


class LearnedEntityEmb(EntityEmb):
    """Learned entity embeddings class.

    We support initializing all learned entity embeddings to be the same
    value. This helps with tail generalization as rare embeddings are
    close together and less prone to errors in embedding noise.

    Add to your config via::

        ent_embeddings:
           - key: learned
             load_class: LearnedEntityEmb
             freeze: false
             cpu: false
             args:
               learned_embedding_size: 256

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
        super(LearnedEntityEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        assert (
            "learned_embedding_size" in emb_args
        ), f"LearnedEntityEmb must have learned_embedding_size in args"
        self.learned_embedding_size = emb_args.learned_embedding_size

        # Set sparsity based on optimizer. The None optimizer is Bootleg's SparseDenseAdam
        optimiz = main_args.learner_config.optimizer_config.optimizer
        if optimiz in [None, "sparse_adam"]:
            sparse = True
        else:
            sparse = False

        if (
            torch.distributed.is_initialized()
            and main_args.model_config.distributed_backend == "nccl"
        ):
            sparse = False
        log_rank_0_debug(
            logger, f"Setting sparsity for entity embeddings to be {sparse}"
        )
        self.learned_entity_embedding = nn.Embedding(
            entity_symbols.num_entities_with_pad_and_nocand,
            self.learned_embedding_size,
            padding_idx=-1,
            sparse=sparse,
        )
        self._dim = main_args.model_config.hidden_size

        if "regularize_mapping" in emb_args:
            eid2reg = torch.zeros(entity_symbols.num_entities_with_pad_and_nocand)
        else:
            eid2reg = None

        # If tail_init is false, all embeddings are randomly intialized.
        # If tail_init is true, we initialize all embeddings to be the same.
        self.tail_init = True
        self.tail_init_zeros = False
        # None init vec will be random
        init_vec = None
        if not self.from_pretrained:
            if "tail_init" in emb_args:
                self.tail_init = emb_args.tail_init
            if "tail_init_zeros" in emb_args:
                self.tail_init_zeros = emb_args.tail_init_zeros
                self.tail_init = False
                init_vec = torch.zeros(1, self.learned_embedding_size)
            assert not (
                self.tail_init and self.tail_init_zeros
            ), f"Can only have one of tail_init or tail_init_zeros set"
            if self.tail_init or self.tail_init_zeros:
                if self.tail_init_zeros:
                    log_rank_0_debug(
                        logger,
                        f"All learned entity embeddings are initialized to zero.",
                    )
                else:
                    log_rank_0_debug(
                        logger,
                        f"All learned entity embeddings are initialized to the same value.",
                    )
                init_vec = model_utils.init_embeddings_to_vec(
                    self.learned_entity_embedding, pad_idx=-1, vec=init_vec
                )
                vec_save_file = os.path.join(
                    emmental.Meta.log_path, "init_vec_entity_embs.npy"
                )
                log_rank_0_debug(logger, f"Saving init vector to {vec_save_file}")
                if (
                    torch.distributed.is_initialized()
                    and torch.distributed.get_rank() == 0
                ):
                    np.save(vec_save_file, init_vec)
            else:
                log_rank_0_debug(
                    logger, f"All learned embeddings are randomly initialized."
                )
            # Regularization mapping goes from eid to 2d dropout percent
            if "regularize_mapping" in emb_args:
                if self.dropout1d_perc > 0 or self.dropout2d_perc > 0:
                    log_rank_0_debug(
                        logger,
                        f"You have 1D or 2D regularization set with a regularize_mapping. Do you mean to do this?",
                    )
                log_rank_0_debug(
                    logger,
                    f"Using regularization mapping in enity embedding from {emb_args.regularize_mapping}",
                )
                eid2reg = self.load_regularization_mapping(
                    main_args.data_config,
                    entity_symbols,
                    emb_args.regularize_mapping,
                )
        self.register_buffer("eid2reg", eid2reg)

    @classmethod
    def load_regularization_mapping(cls, data_config, entity_symbols, reg_file):
        """Reads in a csv file with columns [qid, regularization].

        In the forward pass, the entity id with associated qid will be
        regularized with probability regularization.

        Args:
            data_config: data config
            qid2topk_eid: Dict from QID to eid in the entity embedding
            num_entities_with_pad_and_nocand: number of entities including pad and null candidate option
            reg_file: regularization csv file

        Returns: Tensor where each value is the regularization value for EID
        """
        reg_str = os.path.splitext(os.path.basename(reg_file))[0]
        prep_dir = data_utils.get_data_prep_dir(data_config)
        prep_file = os.path.join(
            prep_dir, f"entity_regularization_mapping_{reg_str}.pt"
        )
        utils.ensure_dir(os.path.dirname(prep_file))
        log_rank_0_debug(logger, f"Looking for regularization mapping in {prep_file}")
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(
                logger,
                f"Loading existing entity regularization mapping from {prep_file}",
            )
            start = time.time()
            eid2reg = torch.load(prep_file)
            log_rank_0_debug(
                logger,
                f"Loaded existing entity regularization mapping in {round(time.time() - start, 2)}s",
            )
        else:
            start = time.time()
            log_rank_0_info(
                logger, f"Building entity regularization mapping from {reg_file}"
            )
            qid2reg = pd.read_csv(reg_file)
            assert (
                "qid" in qid2reg.columns and "regularization" in qid2reg.columns
            ), f"Expected qid and regularization as the column names for {reg_file}"
            # default of no mask
            eid2reg_arr = [0.0] * entity_symbols.num_entities_with_pad_and_nocand
            for row_idx, row in qid2reg.iterrows():
                eid = entity_symbols.get_eid(row["qid"])
                eid2reg_arr[eid] = row["regularization"]
            eid2reg = torch.tensor(eid2reg_arr)
            torch.save(eid2reg, prep_file)
            log_rank_0_debug(
                logger,
                f"Finished building and saving entity regularization mapping in {round(time.time() - start, 2)}s.",
            )
        return eid2reg

    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)
            batch_on_the_fly_data: dict of batch on the fly embeddings

        Returns: B x M x K x dim tensor of entity embeddings
        """
        tensor = model_utils.move_to_device(
            self.learned_entity_embedding(entity_cand_eid)
        )
        if self.eid2reg is not None:
            regularization_tensor = model_utils.move_to_device(
                self.eid2reg[entity_cand_eid.long()]
            )
            tensor = model_utils.emb_dropout_by_tensor(
                self.training, regularization_tensor, tensor
            )
        tensor = self.normalize_and_dropout_emb(tensor)
        return tensor

    def get_dim(self):
        return self.learned_embedding_size


class TopKEntityEmb(EntityEmb):
    """Top K learned entity embeddings class.

    Class for loading the learned embeddings of only the top K entities,
    ranked by occurrence in training data.

    To use this embedding to compress a pretrained model or to train a model, you will need to run
    utils.postprocessing.compress_topk_entity_embeddings.py

    Once run, add the following to your config::

        ent_embeddings:
           - key: learned_topk
             load_class: TopKEntityEmb
             freeze: false
             cpu: false
             args:
               learned_embedding_size: 256
               perc_emb_drop: 0.95 # This MUST match the percent given to this method
               qid2topk_eid: <path to output json file>

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
        super(TopKEntityEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        assert (
            "learned_embedding_size" in emb_args
        ), f"TopKEntityEmb must have learned_embedding_size in args"
        assert "perc_emb_drop" in emb_args, (
            f"To use TopKEntityEmb we need perc_emb_drop to be in the args. This gives the percentage of embeddings"
            f" removed."
        )
        self.learned_embedding_size = emb_args.learned_embedding_size
        # We remove perc_emb_drop percent of the embeddings and add one to represent the new toes embedding
        num_topk_entities_with_pad_and_nocand = (
            entity_symbols.num_entities_with_pad_and_nocand
            - int(emb_args.perc_emb_drop * entity_symbols.num_entities)
            + 1
        )
        # Mapping of entity to the new eid mapping
        eid2topkeid = torch.arange(0, entity_symbols.num_entities_with_pad_and_nocand)
        # There are issues with using -1 index into the embeddings; so we manually set it to be the last value
        eid2topkeid[-1] = num_topk_entities_with_pad_and_nocand - 1
        if "qid2topk_eid" not in emb_args:
            assert self.from_pretrained, (
                f"If you don't provide the qid2topk_eid mapping as an argument to TopKEntityEmb, "
                f"you must be loading a model from a checkpoint to build this index mapping"
            )

        self.learned_entity_embedding = nn.Embedding(
            num_topk_entities_with_pad_and_nocand,
            self.learned_embedding_size,
            padding_idx=-1,
            sparse=False,
        )

        self._dim = main_args.model_config.hidden_size

        if "regularize_mapping" in emb_args:
            eid2reg = torch.zeros(num_topk_entities_with_pad_and_nocand)
        else:
            eid2reg = None
        # If tail_init is false, all embeddings are randomly intialized.
        # If tail_init is true, we initialize all embeddings to be the same.
        self.tail_init = True
        self.tail_init_zeros = False
        # None init vec will be random
        init_vec = None
        if not self.from_pretrained:
            qid2topk_eid = utils.load_json_file(emb_args.qid2topk_eid)
            assert (
                len(qid2topk_eid) == entity_symbols.num_entities
            ), f"You must have an item in qid2topk_eid for each qid in entity_symbols"
            for qid in entity_symbols.get_all_qids():
                old_eid = entity_symbols.get_eid(qid)
                new_eid = qid2topk_eid[qid]
                eid2topkeid[old_eid] = new_eid
            assert eid2topkeid[0] == 0, f"The 0 eid shouldn't be changed"
            assert (
                eid2topkeid[-1] == num_topk_entities_with_pad_and_nocand - 1
            ), "The -1 eid should still map to -1"

            if "tail_init" in emb_args:
                self.tail_init = emb_args.tail_init
            if "tail_init_zeros" in emb_args:
                self.tail_init_zeros = emb_args.tail_init_zeros
                self.tail_init = False
                init_vec = torch.zeros(1, self.learned_embedding_size)
            assert not (
                self.tail_init and self.tail_init_zeros
            ), f"Can only have one of tail_init or tail_init_zeros set"
            if self.tail_init or self.tail_init_zeros:
                if self.tail_init_zeros:
                    log_rank_0_debug(
                        logger,
                        f"All learned entity embeddings are initialized to zero.",
                    )
                else:
                    log_rank_0_debug(
                        logger,
                        f"All learned entity embeddings are initialized to the same value.",
                    )
                init_vec = model_utils.init_embeddings_to_vec(
                    self.learned_entity_embedding, pad_idx=-1, vec=init_vec
                )
                vec_save_file = os.path.join(
                    emmental.Meta.log_path, "init_vec_entity_embs.npy"
                )
                log_rank_0_debug(logger, f"Saving init vector to {vec_save_file}")
                if (
                    torch.distributed.is_initialized()
                    and torch.distributed.get_rank() == 0
                ):
                    np.save(vec_save_file, init_vec)
            else:
                log_rank_0_debug(
                    logger, f"All learned embeddings are randomly initialized."
                )
            # Regularization mapping goes from eid to 2d dropout percent
            if "regularize_mapping" in emb_args:
                log_rank_0_debug(
                    logger,
                    f"You are using regularization mapping with a topK entity embedding. This means all QIDs that are mapped to the same"
                    f" EID will get the same regularization value.",
                )
                if self.dropout1d_perc > 0 or self.dropout2d_perc > 0:
                    log_rank_0_debug(
                        logger,
                        f"You have 1D or 2D regularization set with a regularize_mapping. Do you mean to do this?",
                    )
                log_rank_0_debug(
                    logger,
                    f"Using regularization mapping in enity embedding from {emb_args.regularize_mapping}",
                )
                eid2reg = self.load_regularization_mapping(
                    main_args.data_config,
                    qid2topk_eid,
                    num_topk_entities_with_pad_and_nocand,
                    emb_args.regularize_mapping,
                )
        # Keep this mapping so a topK model can simply be loaded without needing the new eid mapping
        self.register_buffer("eid2topkeid", eid2topkeid)
        self.register_buffer("eid2reg", eid2reg)

    def load_regularization_mapping(
        cls,
        data_config,
        qid2topk_eid,
        num_entities_with_pad_and_nocand,
        reg_file,
    ):
        """Reads in a csv file with columns [qid, regularization].

        In the forward pass, the entity id with associated qid will be
        regularized with probability regularization.

        Args:
            data_config: data config
            qid2topk_eid: Dict from QID to eid in the topK entity embedding
            num_entities_with_pad_and_nocand: number of entities including pad and null candidate option
            reg_file: regularization csv file

        Returns: Tensor where each value is the regularization value for EID
        """
        reg_str = reg_file.split(".csv")[0]
        prep_dir = data_utils.get_data_prep_dir(data_config)
        prep_file = os.path.join(
            prep_dir, f"entity_topk_regularization_mapping_{reg_str}.pt"
        )
        utils.ensure_dir(os.path.dirname(prep_file))
        log_rank_0_debug(logger, f"Looking for regularization mapping in {prep_file}")
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(
                logger,
                f"Loading existing entity regularization mapping from {prep_file}",
            )
            start = time.time()
            eid2reg = torch.load(prep_file)
            log_rank_0_debug(
                logger,
                f"Loaded existing entity regularization mapping in {round(time.time() - start, 2)}s",
            )
        else:
            start = time.time()
            log_rank_0_info(
                logger, f"Building entity regularization mapping from {reg_file}"
            )
            qid2reg = pd.read_csv(reg_file)
            assert (
                "qid" in qid2reg.columns and "regularization" in qid2reg.columns
            ), f"Expected qid and regularization as the column names for {reg_file}"
            # default of no mask
            eid2reg_arr = [0.0] * num_entities_with_pad_and_nocand
            for row_idx, row in qid2reg.iterrows():
                eid = qid2topk_eid[row["qid"]]
                eid2reg_arr[eid] = row["regularization"]
            eid2reg = torch.tensor(eid2reg_arr)
            torch.save(eid2reg, prep_file)
            log_rank_0_debug(
                logger,
                f"Finished building and saving entity regularization mapping in {round(time.time() - start, 2)}s.",
            )
        return eid2reg

    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)
            batch_on_the_fly_data: dict of batch on the fly embeddings

        Returns: B x M x K x dim tensor of top K entity embeddings
        """
        entity_ids = self.eid2topkeid[entity_cand_eid]
        tensor = self.learned_entity_embedding(entity_ids)
        if self.eid2reg is not None:
            regularization_tensor = self.eid2reg[entity_ids]
            tensor = model_utils.emb_dropout_by_tensor(
                self.training, regularization_tensor, tensor
            )
        tensor = self.normalize_and_dropout_emb(tensor)
        return tensor

    def get_dim(self):
        return self.learned_embedding_size


class StaticEmb(EntityEmb):
    """Loads a static embedding for each entity.

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
        super(StaticEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        assert "emb_file" in emb_args, f"Must have emb_file in args for StaticEmb"
        self.entity2static = self.prep(
            data_config=main_args.data_config,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
        )
        self.orig_dim = self.entity2static.shape[1]
        # entity2static_embedding = torch.nn.Embedding(
        #     entity_symbols.num_entities_with_pad_and_nocand,
        #     self.orig_dim,
        #     padding_idx=-1,
        #     sparse=True,
        # )

        self.normalize = False
        if self.orig_dim > 1:
            self.normalize = True
        self.proj = None
        self._dim = self.orig_dim
        if "proj" in emb_args:
            log_rank_0_debug(
                logger,
                f"Adding a projection layer to the static emb to go to dim {emb_args.proj}",
            )
            self._dim = emb_args.proj
            self.proj = MLP(
                input_size=self.orig_dim,
                num_hidden_units=None,
                output_size=self._dim,
                num_layers=1,
            )
        # entity2static_embedding.weight.data[:].copy_(entity2static_table)
        # del entity2static_embedding
        # log_rank_0_info(logger, f"Freezing the static emb for {key}")
        # for param in self.entity2static_embedding.parameters():
        #     param.requires_grad = False

    @classmethod
    def prep(cls, data_config, emb_args, entity_symbols):
        """Static embedding prep.

        Args:
            data_config: data config
            emb_args: embedding args
            entity_symbols: entity symbols

        Returns: numpy embedding array where each row is the embedding for an EID
        """
        static_str = os.path.splitext(os.path.basename(emb_args.emb_file))[0]
        prep_dir = data_utils.get_emb_prep_dir(data_config)
        prep_file = os.path.join(prep_dir, f"static_table_{static_str}.npy")
        log_rank_0_debug(logger, f"Looking for static embedding saved at {prep_file}")
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(
                logger, f"Loading existing static embedding table from {prep_file}"
            )
            start = time.time()
            entity2staticemb_table = np.load(prep_file)  # , mmap_mode="r")
            log_rank_0_debug(
                logger,
                f"Loaded existing static embedding table in {round(time.time() - start, 2)}s",
            )
        else:
            start = time.time()
            emb_file = emb_args.emb_file
            log_rank_0_debug(logger, f"Building static table from file {emb_file}")
            entity2staticemb_table = cls.build_static_embeddings(
                emb_file=emb_file, entity_symbols=entity_symbols
            )
            np.save(prep_file, entity2staticemb_table)
            entity2staticemb_table = np.load(prep_file)  # , mmap_mode="r")
            log_rank_0_debug(
                logger,
                f"Finished building and saving static embedding table in {round(time.time() - start, 2)}s.",
            )
        return entity2staticemb_table

    @classmethod
    def build_static_embeddings(cls, emb_file, entity_symbols):
        """Builds the table of the embedding associated with each entity.

        Args:
            emb_file: embedding file to load
            entity_symbols: entity symbols

        Returns: numpy embedding matrix where each row is an emedding
        """
        ending = os.path.splitext(emb_file)[1]
        if ending == ".json":
            dct = utils.load_json_file(emb_file)
            val = next(iter(dct.values()))
            if type(val) is int or type(val) is float:
                embedding_size = 1
                conver_func = lambda x: np.array([x])
            elif type(val) is list:
                embedding_size = len(val)
                conver_func = lambda x: np.array([y for y in x])
            else:
                raise ValueError(
                    f"Unrecognized type for the array value of {type(val)}"
                )
            embeddings = {}
            for k in dct:
                embeddings[k] = conver_func(dct[k])
                assert len(embeddings[k]) == embedding_size
            entity2staticemb_table = np.zeros(
                (entity_symbols.num_entities_with_pad_and_nocand, embedding_size)
            )
            found = 0
            for qid in tqdm(entity_symbols.get_all_qids()):
                if qid in embeddings:
                    found += 1
                    emb = embeddings[qid]
                    eid = entity_symbols.get_eid(qid)
                    entity2staticemb_table[eid, :embedding_size] = emb
            log_rank_0_debug(
                logger,
                f"Found {found/len(entity_symbols.get_all_qids())} percent of all entities have a static embedding",
            )
        elif ending == ".pt":
            log_rank_0_debug(
                logger,
                f"We are readining in the embedding file from a .pt. We assume this is already mapped to eids",
            )
            entity2staticemb_table = torch.load(emb_file).detach().cpu().numpy()
            assert (
                entity2staticemb_table.shape[0]
                == entity_symbols.num_entities_with_pad_and_nocand
            ), (
                f"To load a saved pt file, it must be of shape {entity_symbols.num_entities_with_pad_and_nocand} "
                f"which is in eid space with the number of entities (including PAD and UNK)"
            )
        else:
            raise ValueError(
                f"We do not support static embeddings from {ending}. We only support .json and .pt"
            )
        return entity2staticemb_table

    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)
            batch_on_the_fly_data: dict of batch on the fly embeddings

        Returns: B x M x K x dim tensor of static embeddings
        """
        tensor = torch.from_numpy(self.entity2static[entity_cand_eid.cpu()]).float()
        # tensor = add_attn(tensor) T toekn ids -> 1 "smartly average the titles"
        tensor = model_utils.move_to_device(tensor)
        self.proj = model_utils.move_to_device(self.proj)
        if self.proj is not None:
            tensor = self.proj(tensor)
        tensor = self.normalize_and_dropout_emb(tensor)
        return tensor

    def get_dim(self):
        return self._dim
