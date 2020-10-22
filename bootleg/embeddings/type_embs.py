"""Type embeddings"""

import os
import ujson as json
import time
import torch
import pandas as pd
from bootleg.embeddings import EntityEmb
from bootleg.layers.layers import SoftAttn, PositionAwareAttention
from bootleg.utils.model_utils import selective_avg
from bootleg.utils import logging_utils, data_utils, utils, model_utils

# Base type embedding
class TypeEmb(EntityEmb):
    """
    Type embedding base class.

    Attributes:
        merge_func: determines how the types for a single candidate will be merged.
                    Support average, softattn, and addattn. Specified in config.

    Forward returns batch x M x K x dim relation embedding.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(TypeEmb, self).__init__(main_args=main_args, emb_args=emb_args,
            model_device=model_device, entity_symbols=entity_symbols,
            word_symbols=word_symbols, word_emb=word_emb, key=key)
        self.logger = logging_utils.get_logger(main_args)
        self.merge_func = self.average_types
        self.orig_dim = emb_args.type_dim
        # Function for merging multiple types
        if "merge_func" in emb_args:
            if emb_args.merge_func not in ["average", "softattn", "addattn"]:
                self.logger.warning(f"{key}: You have set the type merge_func to be {emb_args.merge_func} but that is not in the allowable list of [average, sofftattn]")
            elif emb_args.merge_func == "softattn":
                if "attn_hidden_size" in emb_args:
                    attn_hidden_size = emb_args.attn_hidden_size
                else:
                    attn_hidden_size = 100
                # Softmax of types using the sentence context
                self.soft_attn = SoftAttn(emb_dim=self.orig_dim, context_dim=main_args.model_config.hidden_size, size=attn_hidden_size)
                self.merge_func = self.soft_attn_merge
            elif emb_args.merge_func == "addattn":
                if "attn_hidden_size" in emb_args:
                    attn_hidden_size = emb_args.attn_hidden_size
                else:
                    attn_hidden_size = 100
                # Softmax of types using the sentence context
                self.add_attn = PositionAwareAttention(input_size=self.orig_dim, attn_size=attn_hidden_size, feature_size=0)
                self.merge_func = self.add_attn_merge
        self.normalize = True
        self.entity_symbols = entity_symbols
        self.max_types = emb_args.max_types
        self.eid2typeids_table, self.type2row_dict, num_types_with_unk, self.prep_file = self.prep(main_args=main_args,
            emb_args=emb_args, entity_symbols=entity_symbols, log_func=self.logger.debug)
        self.num_types_with_pad_and_unk = num_types_with_unk + 1
        self.eid2typeids_table = self.eid2typeids_table.to(model_device)
        # Regularization mapping goes from typeid to 2d dropout percent
        self.typeid2reg = None
        if "regularize_mapping" in emb_args:
            self.logger.debug(f"Using regularization mapping in enity embedding from {emb_args.regularize_mapping}")
            self.typeid2reg = self.load_regularization_mapping(main_args, self.num_types_with_pad_and_unk, self.type2row_dict, emb_args.regularize_mapping, self.logger.debug)
            self.typeid2reg = self.typeid2reg.to(model_device)
        assert self.eid2typeids_table.shape[1] == emb_args.max_types, f"Something went wrong with loading type file." \
                                                             f" The given max types {emb_args.max_types} does not match that of type table {self.eid2typeids_table.shape[1]}"
        self.logger.debug(f"{key}: Type embedding with {self.max_types} types with dim {self.orig_dim}. Setting merge_func to be {self.merge_func.__name__} in type emb.")

    @classmethod
    def prep(cls, main_args, emb_args, entity_symbols, log_func=print, word_symbols=None):
        type_str = os.path.splitext(emb_args.type_labels)[0]
        prep_dir = data_utils.get_emb_prep_dir(main_args)
        prep_file = os.path.join(prep_dir,
            f'type_table_{type_str}_{emb_args.max_types}.pt')
        utils.ensure_dir(os.path.dirname(prep_file))
        if (not main_args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file)):
            log_func(f'Loading existing type table from {prep_file}')
            start = time.time()
            eid2typeids_table, type2row_dict, num_types_with_unk = torch.load(prep_file)
            log_func(f'Loaded existing type table in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            type_labels = os.path.join(main_args.data_config.emb_dir, emb_args.type_labels)
            log_func(f'Building type table from {type_labels}')
            eid2typeids_table, type2row_dict, num_types_with_unk = cls.build_type_table(type_labels=type_labels,
                max_types=emb_args.max_types, entity_symbols=entity_symbols)
            torch.save((eid2typeids_table, type2row_dict, num_types_with_unk), prep_file)
            log_func(f"Finished building and saving type table in {round(time.time() - start, 2)}s.")
        return eid2typeids_table, type2row_dict, num_types_with_unk, prep_file

    @classmethod
    def build_type_table(cls, type_labels, max_types, entity_symbols):
        # all eids are initially assigned to unk types
        # if they occur in the type file, then they are assigned the types in the file plus padded types
        eid2typeids = torch.zeros(entity_symbols.num_entities_with_pad_and_nocand,
        max_types)
        eid2typeids[0] = torch.zeros(1, max_types)
        # currently types are assigned by wikipageid
        # keep track of the max_type_id to set the size of the type embedding
        max_type_id_all = -1
        type_hit = 0
        type2row_dict = {}
        with open(type_labels) as f:
            qid2typeid = json.load(f)
            for qid, row_types in qid2typeid.items():
                if not entity_symbols.qid_exists(qid):
                    continue
                # assign padded types to the last row
                typeids = torch.ones(max_types)*-1
                if len(row_types) > 0:
                    type_hit += 1
                    # increment by 1 to account for unk row
                    typeids_list = []
                    for type_id in row_types:
                        typeids_list.append(type_id+1)
                        type2row_dict[type_id] = type_id+1
                    # we take the max_id over all of the types
                    # not just the ones we filter with max_types
                    max_type_id = max(typeids_list)
                    if max_type_id > max_type_id_all:
                        max_type_id_all = max_type_id
                    num_types = min(len(typeids_list), max_types)
                    typeids[:num_types] = torch.tensor(typeids_list)[:num_types]
                    eid2typeids[entity_symbols.get_eid(qid)] = typeids
        # + 1 bc indices start at 0 (we've already incremented for the unk row)
        labeled_num_types = max_type_id_all + 1
        # assign padded types to the last row of the type embedding
        # make sure adding type labels doesn't add new types
        assert (max_type_id_all+1) <= labeled_num_types
        eid2typeids[eid2typeids == -1] = labeled_num_types
        print(f"{round(type_hit/entity_symbols.num_entities, 2)*100}% of entities are assigned types")
        return eid2typeids.long(), type2row_dict, labeled_num_types

    def load_regularization_mapping(cls, main_args, num_types_with_pad_and_unk, type2row_dict, reg_file, log_func):
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
            typeid2reg = torch.load(prep_file)
            log_func(f'Loaded existing entity regularization mapping in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            reg_file = os.path.join(main_args.data_config.data_dir, reg_file)
            log_func(f'Building entity regularization mapping from {reg_file}')
            typeid2reg_raw = pd.read_csv(reg_file)
            assert "typeid" in typeid2reg_raw.columns and "regularization" in typeid2reg_raw.columns, f"Expected typeid and regularization as the column names for {reg_file}"
            # default of no mask
            typeid2reg_arr = [0.0]*num_types_with_pad_and_unk
            for row_idx, row in typeid2reg_raw.iterrows():
                # Happens when we filter QIDs not in our entity dump and the max typeid is smaller than the total number
                if int(row["typeid"]) not in type2row_dict:
                    continue
                typeid = type2row_dict[int(row["typeid"])]
                typeid2reg_arr[typeid] = row["regularization"]
            typeid2reg = torch.tensor(typeid2reg_arr)
            torch.save(typeid2reg, prep_file)
            log_func(f"Finished building and saving entity regularization mapping in {round(time.time() - start, 2)}s.")
        return typeid2reg

    def get_typeids(self, entity_package):
        batch_type_ids = self.eid2typeids_table[entity_package.tensor].long()
        return batch_type_ids

    def _selective_avg_types(self, type_ids, embeds):
        mask = ((type_ids < (self.num_types_with_pad_and_unk-1)) & (type_ids > 0))
        average_val = selective_avg(type_ids, mask, embeds)
        num_unk_types = ((type_ids == 0).sum(3) == type_ids.shape[-1])
        unk_types = torch.where(num_unk_types.unsqueeze(3), embeds[:,:,:,0], torch.zeros_like(average_val))
        return average_val + unk_types

    def average_types(self, entity_package, batch_type_ids, batch_type_emb, sent_emb):
        """Averages the type embs for a single candidate."""
        batch_type_emb = self._selective_avg_types(batch_type_ids, batch_type_emb)
        return batch_type_emb

    def soft_attn_merge(self, entity_package, batch_type_ids, batch_type_emb, sent_emb):
        """For each candidate, use a weighted average of the type embs based on type emb similarity to the contextualized mention embedding."""
        batch, M, K, num_types, dim = batch_type_emb.shape
        # we don't want to compute probabilities over padded types
        # when there are no types -- we'll just get an average over unk types (i.e. the unk type)
        mask = (batch_type_ids < (self.num_types_with_pad_and_unk-1)).reshape(
            batch*M*K, num_types)
        # Get alias tensor and expand to be for each candidate for soft attn
        alias_word_tensor = model_utils.select_alias_word_sent(entity_package.pos_in_sent, sent_emb, index=0)
        _, _, sent_dim = alias_word_tensor.shape
        alias_word_tensor = alias_word_tensor.unsqueeze(2).expand(batch, M, K, sent_dim)
        # Reshape for soft attn
        batch_type_emb = batch_type_emb.contiguous().reshape(batch*M*K, num_types, dim)
        alias_word_tensor = alias_word_tensor.contiguous().reshape(batch*M*K, sent_dim)
        # Get soft attn
        batch_type_emb = self.soft_attn(batch_type_emb, alias_word_tensor, mask=mask)
        # Convert batch back to original shape
        batch_type_emb = batch_type_emb.reshape(batch, M, K, dim)
        return batch_type_emb

    def add_attn_merge(self, entity_package, batch_type_ids, batch_type_emb, sent_emb):
        """Using additive attention to average the type embeddings for a candidate."""
        batch, M, K, num_types, dim = batch_type_emb.shape
        # We don't want to compute probabilities over padded types
        mask = (batch_type_ids < (self.num_types_with_pad_and_unk-1)).reshape(
            batch*M*K, num_types)
        # Reshape for add attn
        batch_type_emb = batch_type_emb.contiguous().reshape(batch*M*K, num_types, dim)
        # Get add attn
        batch_type_emb = self.add_attn(batch_type_emb, mask=mask)
        # Convert batch back to original shape
        batch_type_emb = batch_type_emb.reshape(batch, M, K, dim)
        return batch_type_emb

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        raise ValueError("Not implemented")

    def get_dim(self):
        raise ValueError("Not implemented")


class LearnedTypeEmb(TypeEmb):
    """
    Learned type embedding class.
    """
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(LearnedTypeEmb, self).__init__(main_args=main_args, emb_args=emb_args,
            model_device=model_device, entity_symbols=entity_symbols,
            word_symbols=word_symbols, word_emb=word_emb, key=key)
        self.orig_dim = emb_args.type_dim
        self._dim = main_args.model_config.hidden_size
        self.type_emb = torch.nn.Embedding(self.num_types_with_pad_and_unk, self.orig_dim, padding_idx=-1).to(model_device)

    @classmethod
    def prep(cls, main_args, emb_args, entity_symbols, log_func=print, word_symbols=None):
        return super(LearnedTypeEmb, cls).prep(main_args=main_args, emb_args=emb_args,
            entity_symbols=entity_symbols, word_symbols=word_symbols, log_func=log_func)

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        batch_type_ids = self.eid2typeids_table[entity_package.tensor].long()
        batch_type_emb = self.type_emb(batch_type_ids)
        if self.typeid2reg is not None:
            regularization_tensor = self.typeid2reg[batch_type_ids]
            batch_type_emb = model_utils.emb_dropout_by_tensor(self.training, regularization_tensor, batch_type_emb)
        batch_type_emb = self.merge_func(entity_package, batch_type_ids, batch_type_emb, sent_emb)
        res = self._package(tensor=batch_type_emb,
                            pos_in_sent=entity_package.pos_in_sent,
                            alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return res

    def get_dim(self):
        return self.orig_dim