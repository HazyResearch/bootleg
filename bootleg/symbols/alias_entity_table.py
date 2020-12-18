"""AliasEntityTable class"""

import numpy as np
import time
import torch
import torch.nn as nn
import os

from tqdm import tqdm

from bootleg.utils import logging_utils, data_utils, utils

class AliasEntityTable(nn.Module):
    """Stores table of the K candidate entity ids for each alias."""
    def __init__(self, args, entity_symbols):
        super(AliasEntityTable, self).__init__()
        self.args = args
        self.logger = logging_utils.get_logger(args)
        self.num_entities_with_pad_and_nocand = entity_symbols.num_entities_with_pad_and_nocand
        self.num_aliases_with_pad = len(entity_symbols.get_all_aliases()) + 1
        self.M = args.data_config.max_aliases
        self.K = entity_symbols.max_candidates + (not args.data_config.train_in_candidates)
        self.alias2entity_table, self.prep_file = self.prep(args, entity_symbols,
            num_aliases_with_pad=self.num_aliases_with_pad, num_cands_K=self.K,
            log_func=self.logger.debug)
        # Small check that loading was done correctly. This isn't a catch all, but will catch is the same or something went wrong.
        assert np.all(np.array(self.alias2entity_table[-1]) == np.ones(self.K)*-1), f"The last row of the alias table isn't -1, something wasn't loaded right."

    @classmethod
    def prep(cls, args, entity_symbols, num_aliases_with_pad, num_cands_K, log_func=print):
        # we pass num_aliases_with_pad and num_cands_K to remove the dependence on entity_symbols
        # when the alias table is already prepped
        data_shape = (num_aliases_with_pad, num_cands_K)
        # dependent on train_in_candidates flag
        prep_dir = data_utils.get_emb_prep_dir(args)
        alias_str = os.path.splitext(args.data_config.alias_cand_map)[0]
        prep_file = os.path.join(prep_dir,
            f'alias2entity_table_{alias_str}_InC{int(args.data_config.train_in_candidates)}.pt')
        if (not args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file)):
            log_func(f'Loading alias table from {prep_file}')
            start = time.time()
            alias2entity_table = np.memmap(prep_file, dtype='int64', mode='r', shape=data_shape)
            log_func(f'Loaded alias table in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            log_func(f'Building alias table')
            utils.ensure_dir(prep_dir)
            mmap_file = np.memmap(prep_file, dtype='int64', mode='w+', shape=data_shape)
            alias2entity_table  = cls.build_alias_table(args, entity_symbols)
            mmap_file[:] = alias2entity_table[:]
            mmap_file.flush()
            log_func(f"Finished building and saving alias table in {round(time.time() - start, 2)}s.")
        return alias2entity_table, prep_file

    @classmethod
    def build_alias_table(cls, args, entity_symbols):
        """Builds the alias to entity ids table"""
        # we need to include a non candidate entity option for each alias and a row for unk alias
        # +1 is for UNK alias (last row)
        num_aliases_with_pad = len(entity_symbols.get_all_aliases()) + 1
        alias2entity_table = torch.ones(num_aliases_with_pad, entity_symbols.max_candidates+(not args.data_config.train_in_candidates)) * -1
        for alias in tqdm(entity_symbols.get_all_aliases(), desc="Building alias table"):
            # first row for unk alias
            alias_id = entity_symbols.get_alias_idx(alias)
            # set all to -1 and fill in with real values for padding and fill in with real values
            entity_list = torch.ones(entity_symbols.max_candidates+(not args.data_config.train_in_candidates)) * -1
            # set first column to zero
            # if we are using noncandidate entity, this will remain a 0
            # if we are not using noncandidate entities, this will get overwritten below.
            entity_list[0] = 0
            eid_cands = entity_symbols.get_eid_cands(alias)
            # we get qids and want entity ids
            # first entry is the non candidate class
            # val[0] because vals is [qid, page_counts]
            entity_list[(not args.data_config.train_in_candidates):len(eid_cands)+(not args.data_config.train_in_candidates)] = torch.tensor(eid_cands)
            alias2entity_table[alias_id, :] = entity_list
        return alias2entity_table.long()

    def forward(self, alias_indices):
        alias_entity_ids = self.alias2entity_table[alias_indices]
        return alias_entity_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        # Not picklable
        del state['alias2entity_table']
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.alias2entity_table = torch.tensor(np.memmap(self.prep_file, dtype='int64', mode='r',
            shape=(self.num_aliases_with_pad, self.K)))
        self.logger = logging_utils.get_logger(self.args)