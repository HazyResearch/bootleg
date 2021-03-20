"""Type embeddings."""

import logging
import os
import time

import pandas as pd
import torch
import ujson as json

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.embeddings import EntityEmb
from bootleg.layers.helper_modules import PositionAwareAttention
from bootleg.utils import data_utils, model_utils, utils
from bootleg.utils.model_utils import selective_avg

logger = logging.getLogger(__name__)


# Base type embedding
class TypeEmb(EntityEmb):
    """Type embedding base class.

    Forward returns batch x M x K x dim relation embedding.

    Args:
        main_args: main args
        emb_args: specific embedding args
        entity_symbols: entity symbols
        key: unique embedding key
        cpu: bool of if one cpu or not
        normalize: bool if normalize embeddings or not
        dropout1d_perc: 1D dropout percent
        dropout2d_perc: 2D dropout percent

    Attributes:
        merge_func: determines how the types for a single candidate will be merged.
                    Support average, softattn, and addattn. Specified in config.
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
        super(TypeEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        allowable_keys = {
            "max_types",
            "type_dim",
            "type_labels",
            "type_vocab",
            "merge_func",
            "attn_hidden_size",
            "regularize_mapping",
        }
        correct, bad_key = utils.assert_keys_in_dict(allowable_keys, emb_args)
        if not correct:
            raise ValueError(f"The key {bad_key} is not in {allowable_keys}")
        assert (
            "max_types" in emb_args
        ), "Type embedding requires max_types to be set in args"
        assert (
            "type_dim" in emb_args
        ), "Type embedding requires type_dim to be set in args"
        assert (
            "type_labels" in emb_args
        ), "Type embedding requires type_labels to be set in args. A Dict from QID -> TypeId or TypeName"
        assert (
            "type_vocab" in emb_args
        ), "Type embedding requires type_vocab to be set in args. A Dict from TypeName -> TypeId"
        assert (
            self.cpu is False
        ), f"We don't support putting type embeddings on CPU right now"
        self.merge_func = self.average_types
        self.orig_dim = emb_args.type_dim
        self.add_attn = None
        # Function for merging multiple types
        if "merge_func" in emb_args:
            assert emb_args.merge_func in ["average", "addattn",], (
                f"{key}: You have set the type merge_func to be {emb_args.merge_func} but"
                f" that is not in the allowable list of [average, addattn]"
            )
            if emb_args.merge_func == "addattn":
                if "attn_hidden_size" in emb_args:
                    attn_hidden_size = emb_args.attn_hidden_size
                else:
                    attn_hidden_size = 100
                # Softmax of types using the sentence context
                self.add_attn = PositionAwareAttention(
                    input_size=self.orig_dim, attn_size=attn_hidden_size, feature_size=0
                )
                self.merge_func = self.add_attn_merge
        self.max_types = emb_args.max_types
        (
            eid2typeids_table,
            self.type2row_dict,
            num_types_with_unk,
            self.prep_file,
        ) = self.prep(
            data_config=main_args.data_config,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
        )
        self.register_buffer("eid2typeids_table", eid2typeids_table, persistent=False)
        # self.eid2typeids_table.requires_grad = False
        self.num_types_with_pad_and_unk = num_types_with_unk + 1

        # Regularization mapping goes from typeid to 2d dropout percent
        if "regularize_mapping" in emb_args:
            typeid2reg = torch.zeros(self.num_types_with_pad_and_unk)
        else:
            typeid2reg = None

        if not self.from_pretrained:
            if "regularize_mapping" in emb_args:
                if self.dropout1d_perc > 0 or self.dropout2d_perc > 0:
                    logger.warning(
                        f"You have 1D or 2D regularization set with a regularize_mapping. Do you mean to do this?"
                    )
                log_rank_0_info(
                    logger,
                    f"Using regularization mapping in enity embedding from {emb_args.regularize_mapping}",
                )
                typeid2reg = self.load_regularization_mapping(
                    main_args.data_config,
                    self.num_types_with_pad_and_unk,
                    self.type2row_dict,
                    emb_args.regularize_mapping,
                )
        self.register_buffer("typeid2reg", typeid2reg)
        assert self.eid2typeids_table.shape[1] == emb_args.max_types, (
            f"Something went wrong with loading type file."
            f" The given max types {emb_args.max_types} does not match that "
            f"of type table {self.eid2typeids_table.shape[1]}"
        )
        log_rank_0_debug(
            logger,
            f"{key}: Type embedding with {self.max_types} types with dim {self.orig_dim}. "
            f"Setting merge_func to be {self.merge_func.__name__} in type emb.",
        )

    @classmethod
    def prep(cls, data_config, emb_args, entity_symbols):
        """Prep the type id table.

        Args:
            data_config: data config
            emb_args: embedding args
            entity_symbols: entity synbols

        Returns: torch tensor from EID to type IDS, type ID to row in type embedding matrix,
                 and number of types with unk type
        """
        type_str = os.path.splitext(emb_args.type_labels.replace("/", "_"))[0]
        prep_dir = data_utils.get_emb_prep_dir(data_config)
        prep_file = os.path.join(
            prep_dir, f"type_table_{type_str}_{emb_args.max_types}.pt"
        )
        utils.ensure_dir(os.path.dirname(prep_file))
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(logger, f"Loading existing type table from {prep_file}")
            start = time.time()
            eid2typeids_table, type2row_dict, num_types_with_unk = torch.load(prep_file)
            log_rank_0_debug(
                logger,
                f"Loaded existing type table in {round(time.time() - start, 2)}s",
            )
        else:
            start = time.time()
            type_labels = os.path.join(data_config.emb_dir, emb_args.type_labels)
            type_vocab = os.path.join(data_config.emb_dir, emb_args.type_vocab)
            log_rank_0_debug(logger, f"Building type table from {type_labels}")
            eid2typeids_table, type2row_dict, num_types_with_unk = cls.build_type_table(
                type_labels=type_labels,
                type_vocab=type_vocab,
                max_types=emb_args.max_types,
                entity_symbols=entity_symbols,
            )
            torch.save(
                (eid2typeids_table, type2row_dict, num_types_with_unk), prep_file
            )
            log_rank_0_debug(
                logger,
                f"Finished building and saving type table in {round(time.time() - start, 2)}s.",
            )
        return eid2typeids_table, type2row_dict, num_types_with_unk, prep_file

    @classmethod
    def build_type_table(cls, type_labels, type_vocab, max_types, entity_symbols):
        """Builds the EID to type ids table.

        Args:
            type_labels: QID to type ids or type names json mapping
            type_vocab: type name to type ids
            max_types: maximum number of types for an entity
            entity_symbols: entity symbols

        Returns: torch tensor from EID to type IDS, type ID to row in type embedding matrix,
                 and number of types with unk type
        """
        with open(type_vocab) as f:
            vocab = json.load(f)
        all_type_ids = set(list(vocab.values()))
        assert (
            0 not in all_type_ids
        ), f"The type id of 0 is reserved for UNK type. Please offset the typeids by 1"
        # all eids are initially assigned to unk types
        # if they occur in the type file, then they are assigned the types in the file plus padded types
        eid2typeids = torch.zeros(
            entity_symbols.num_entities_with_pad_and_nocand, max_types
        )
        eid2typeids[0] = torch.zeros(1, max_types)

        max_type_id_all = max(all_type_ids)
        type_hit = 0
        type2row_dict = {}
        with open(type_labels) as f:
            qid2typeid = json.load(f)
            for qid, row_types in qid2typeid.items():
                if not entity_symbols.qid_exists(qid):
                    continue
                # assign padded types to the last row
                typeids = torch.ones(max_types) * -1
                if len(row_types) > 0:
                    type_hit += 1
                    # increment by 1 to account for unk row
                    typeids_list = []
                    for type_id_or_name in row_types:
                        # If typename, map to typeid
                        if type(type_id_or_name) is str:
                            type_id = vocab[type_id_or_name]
                        else:
                            type_id = type_id_or_name
                        assert (
                            type_id > 0
                        ), f"Typeid for {qid} is 0. That is reserved. Please offset by 1"
                        assert (
                            type_id in all_type_ids
                        ), f"Typeid for {qid} isn't in vocab"
                        typeids_list.append(type_id)
                        type2row_dict[type_id] = type_id
                    num_types = min(len(typeids_list), max_types)
                    typeids[:num_types] = torch.tensor(typeids_list)[:num_types]
                    eid2typeids[entity_symbols.get_eid(qid)] = typeids
        # + 1 bc we need to account for pad row
        labeled_num_types = max_type_id_all + 1
        # assign padded types to the last row of the type embedding
        # make sure adding type labels doesn't add new types
        assert (max_type_id_all + 1) <= labeled_num_types
        eid2typeids[eid2typeids == -1] = labeled_num_types
        log_rank_0_debug(
            logger,
            f"{round(type_hit/entity_symbols.num_entities, 2)*100}% of entities are assigned types",
        )
        return eid2typeids.long(), type2row_dict, labeled_num_types

    @classmethod
    def load_regularization_mapping(
        cls, data_config, num_types_with_pad_and_unk, type2row_dict, reg_file
    ):
        """Reads in a csv file with columns [qid, regularization].

        In the forward pass, the entity id with associated qid will be
        regularized with probability regularization.

        Args:
            data_config: data config
            num_entities_with_pad_and_nocand: number of types including pad and null option
            type2row_dict: Dict from typeID to row id in the type embedding matrix
            reg_file: regularization csv file

        Returns: Tensor where each value is the regularization value for EID
        """
        reg_str = os.path.splitext(os.path.basename(reg_file.replace("/", "_")))[0]
        prep_dir = data_utils.get_data_prep_dir(data_config)
        prep_file = os.path.join(prep_dir, f"type_regularization_mapping_{reg_str}.pt")
        utils.ensure_dir(os.path.dirname(prep_file))
        log_rank_0_debug(logger, f"Looking for regularization mapping in {prep_file}")
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(
                logger,
                f"Loading existing entity regularization mapping from {prep_file}",
            )
            start = time.time()
            typeid2reg = torch.load(prep_file)
            log_rank_0_debug(
                logger,
                f"Loaded existing entity regularization mapping in {round(time.time() - start, 2)}s",
            )
        else:
            start = time.time()
            log_rank_0_debug(
                logger, f"Building entity regularization mapping from {reg_file}"
            )
            typeid2reg_raw = pd.read_csv(reg_file)
            assert (
                "typeid" in typeid2reg_raw.columns
                and "regularization" in typeid2reg_raw.columns
            ), f"Expected typeid and regularization as the column names for {reg_file}"
            # default of no mask
            typeid2reg_arr = [0.0] * num_types_with_pad_and_unk
            for row_idx, row in typeid2reg_raw.iterrows():
                # Happens when we filter QIDs not in our entity db and the max typeid is smaller than the total number
                if int(row["typeid"]) not in type2row_dict:
                    continue
                typeid = type2row_dict[int(row["typeid"])]
                typeid2reg_arr[typeid] = row["regularization"]
            typeid2reg = torch.Tensor(typeid2reg_arr)
            torch.save(typeid2reg, prep_file)
            log_rank_0_debug(
                logger,
                f"Finished building and saving entity regularization mapping in {round(time.time() - start, 2)}s.",
            )
        return typeid2reg

    def _selective_avg_types(self, type_ids, embeds):
        """Selects the average embedding, ignoring padded types.

        Args:
            type_ids: type ids
            embeds: embeddings

        Returns: average embedding
        """
        # mask of True means keep in the average
        mask = (type_ids < (self.num_types_with_pad_and_unk - 1)) & (type_ids > 0)
        average_val = selective_avg(mask, embeds)
        num_unk_types = (type_ids == 0).sum(3) == type_ids.shape[-1]
        unk_types = torch.where(
            num_unk_types.unsqueeze(3),
            embeds[:, :, :, 0],
            torch.zeros_like(average_val),
        )
        return average_val + unk_types

    def average_types(self, batch_type_ids, batch_type_emb, extras=None):
        """Averages the type embeddings for each candidate.

        Args:
            batch_type_ids: type ids for a batch
            batch_type_emb: type embeddings for a batch
            extras: extras to allow for compatability with other merge_funcs

        Returns: Tensor of averaged type embeddings for a batch
        """
        batch_type_emb = self._selective_avg_types(batch_type_ids, batch_type_emb)
        return batch_type_emb

    def add_attn_merge(self, batch_type_ids, batch_type_emb, add_attn):
        """Using additive attention to average the type embeddings for a
        candidate.

        Important! We must pass in add_attn to avoid bugs with DP not using the add_attn from the right device

        Args:
            batch_type_ids: type ids for a batch
            batch_type_emb: type embeddings for a batch
            add_attn: the addative attention module

        Returns:
        """
        batch, M, K, num_types, dim = batch_type_emb.shape
        # We don't want to compute probabilities over padded types
        mask = (batch_type_ids < (self.num_types_with_pad_and_unk - 1)).reshape(
            batch * M * K, num_types
        )
        # Reshape for add attn
        batch_type_emb = batch_type_emb.contiguous().reshape(
            batch * M * K, num_types, dim
        )
        # Get add attn
        batch_type_emb = add_attn(batch_type_emb, mask=mask)
        # Convert batch back to original shape
        batch_type_emb = batch_type_emb.reshape(batch, M, K, dim)
        return batch_type_emb

    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        raise ValueError("Not implemented")

    def get_dim(self):
        raise ValueError("Not implemented")


class LearnedTypeEmb(TypeEmb):
    """Learned ype embedding class. Forward returns batch x M x K x dim
    relation embedding. Add to your config via.::

        ent_embeddings:
           - key: learned_type
             load_class: LearnedTypeEmb
             freeze: false
             args:
               type_labels: <path to type json mapping>
               max_types: 3
               type_dim: 128
               merge_func: addattn
               attn_hidden_size: 128

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
        super(LearnedTypeEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        self.orig_dim = emb_args.type_dim
        self._dim = main_args.model_config.hidden_size
        self.type_emb = torch.nn.Embedding(
            self.num_types_with_pad_and_unk, self.orig_dim, padding_idx=-1
        )

    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)
            batch_on_the_fly_data: dict of batch on the fly embeddings

        Returns: B x M x K x dim tensor of type embeddings
        """
        batch_type_ids = self.eid2typeids_table[entity_cand_eid].long()
        batch_type_emb = self.type_emb(batch_type_ids)
        if self.typeid2reg is not None:
            assert self.typeid2reg.requires_grad is False
            regularization_tensor = self.typeid2reg[batch_type_ids]
            batch_type_emb = model_utils.emb_dropout_by_tensor(
                self.training, regularization_tensor, batch_type_emb
            )

        # MUST pass add_attn into method - DataParallel gets messed up with this subclassing of functions and objects
        batch_type_emb = self.merge_func(batch_type_ids, batch_type_emb, self.add_attn)
        batch_type_emb = self.normalize_and_dropout_emb(batch_type_emb)
        return batch_type_emb

    def get_dim(self):
        return self.orig_dim
