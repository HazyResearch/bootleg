"""AliasEntityTable class."""
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from bootleg import log_rank_0_debug
from bootleg.utils import data_utils, utils
from bootleg.utils.embedding_utils import get_max_candidates

logger = logging.getLogger(__name__)


class AliasEntityTable(nn.Module):
    """Stores table of the K candidate entity ids for each alias.

    Args:
        data_config: data config
        entity_symbols: entity symbols
    """

    def __init__(self, data_config, entity_symbols):
        super(AliasEntityTable, self).__init__()
        self.num_entities_with_pad_and_nocand = (
            entity_symbols.num_entities_with_pad_and_nocand
        )
        self.num_aliases_with_pad_and_unk = len(entity_symbols.get_all_aliases()) + 2
        self.M = data_config.max_aliases
        self.K = get_max_candidates(entity_symbols, data_config)
        self.alias2entity_table, self.prep_file = self.prep(
            data_config,
            entity_symbols,
            num_aliases_with_pad_and_unk=self.num_aliases_with_pad_and_unk,
            num_cands_K=self.K,
        )
        # self.alias2entity_table = model_utils.move_to_device(self.alias2entity_table)
        # Small check that loading was done correctly. This isn't a catch all,
        # but will catch is the same or something went wrong. -2 is for alias not in our set and -1 is pad.
        assert torch.equal(
            self.alias2entity_table[-2],
            torch.ones_like(self.alias2entity_table[-1]) * -1,
        ), f"The second to last row of the alias table isn't -1, something wasn't loaded right."
        assert torch.equal(
            self.alias2entity_table[-1],
            torch.ones_like(self.alias2entity_table[-1]) * -1,
        ), f"The last row of the alias table isn't -1, something wasn't loaded right."

    @classmethod
    def prep(
        cls,
        data_config,
        entity_symbols,
        num_aliases_with_pad_and_unk,
        num_cands_K,
    ):
        """Preps the alias to entity EID table.

        Args:
            data_config: data config
            entity_symbols: entity symbols
            num_aliases_with_pad_and_unk: number of aliases including pad and unk
            num_cands_K: number of candidates per alias (aka K)

        Returns: torch Tensor of the alias to EID table, save pt file
        """
        # we pass num_aliases_with_pad_and_unk and num_cands_K to remove the dependence on entity_symbols
        # when the alias table is already prepped
        data_shape = (num_aliases_with_pad_and_unk, num_cands_K)
        # dependent on train_in_candidates flag
        prep_dir = data_utils.get_emb_prep_dir(data_config)
        alias_str = os.path.splitext(data_config.alias_cand_map.replace("/", "_"))[0]
        prep_file = os.path.join(
            prep_dir,
            f"alias2entity_table_{alias_str}_InC{int(data_config.train_in_candidates)}.pt",
        )
        log_rank_0_debug(logger, f"Looking for alias table in {prep_file}")
        if not data_config.overwrite_preprocessed_data and os.path.exists(prep_file):
            log_rank_0_debug(logger, f"Loading alias table from {prep_file}")
            start = time.time()
            alias2entity_table = np.memmap(
                prep_file, dtype="int64", mode="r+", shape=data_shape
            )
            log_rank_0_debug(
                logger, f"Loaded alias table in {round(time.time() - start, 2)}s"
            )
        else:
            start = time.time()
            log_rank_0_debug(logger, f"Building alias table")
            utils.ensure_dir(prep_dir)
            mmap_file = np.memmap(prep_file, dtype="int64", mode="w+", shape=data_shape)
            alias2entity_table = cls.build_alias_table(data_config, entity_symbols)
            mmap_file[:] = alias2entity_table[:]
            mmap_file.flush()
            log_rank_0_debug(
                logger,
                f"Finished building and saving alias table in {round(time.time() - start, 2)}s.",
            )
        alias2entity_table = torch.from_numpy(alias2entity_table)
        return alias2entity_table, prep_file

    @classmethod
    def build_alias_table(cls, data_config, entity_symbols):
        """Constructs the alias to EID table.

        Args:
            data_config: data config
            entity_symbols: entity symbols

        Returns: numpy array where row is alias ID and columns are EID
        """
        # we need to include a non candidate entity option for each alias and a row for PAD alias and not in dump alias
        # +2 is for PAD alias (last row) and not in dump alias (second to last row)
        # - same as -2 entity ids being not in cand list
        num_aliases_with_pad_and_unk = len(entity_symbols.get_all_aliases()) + 2
        alias2entity_table = (
            np.ones(
                (
                    num_aliases_with_pad_and_unk,
                    get_max_candidates(entity_symbols, data_config),
                )
            )
            * -1
        )
        for alias in tqdm(
            entity_symbols.get_all_aliases(), desc="Iterating over aliases"
        ):
            alias_id = entity_symbols.get_alias_idx(alias)
            # set all to -1 and fill in with real values for padding and fill in with real values
            entity_list = np.ones(get_max_candidates(entity_symbols, data_config)) * -1
            # set first column to zero
            # if we are using noncandidate entity, this will remain a 0
            # if we are not using noncandidate entities, this will get overwritten below.
            entity_list[0] = 0
            eid_cands = entity_symbols.get_eid_cands(alias)
            # we get qids and want entity ids
            # first entry is the non candidate class
            # val[0] because vals is [qid, page_counts]
            entity_list[
                (not data_config.train_in_candidates) : len(eid_cands)
                + (not data_config.train_in_candidates)
            ] = np.array(eid_cands)
            alias2entity_table[alias_id, :] = entity_list
        return alias2entity_table

    def forward(self, alias_indices):
        """Model forward.

        Args:
            alias_indices: alias indices (B x M)

        Returns: entity candidate EIDs (B x M x K)
        """
        candidate_entity_ids = self.alias2entity_table[alias_indices]
        return candidate_entity_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        # Not picklable
        del state["alias2entity_table"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.alias2entity_table = torch.tensor(
            np.memmap(
                self.prep_file,
                dtype="int64",
                mode="r",
                shape=(self.num_aliases_with_pad_and_unk, self.K),
            )
        )
