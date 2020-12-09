import gc
import os
import time
import ujson as json
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist

from bootleg.symbols.constants import *
from bootleg.utils import logging_utils, utils, data_utils, train_utils
from bootleg.prep import prep_slice

class WikiSlices(Dataset):
    """
    Slice dataset class. The slices are determined by the "slices" field in the jsonl data file. Slices are prepped in prep.py
    """
    def __init__(self, args, use_weak_label, input_src, dataset_name,
                 is_writer, distributed, dataset_is_eval):
        self.logger = logging_utils.get_logger(args)
        self.args = args
        self.dataset_is_eval = dataset_is_eval
        self.storage_type = None
        self.dataset_name = dataset_name
        self.config_dataset_name = data_utils.get_slice_storage_file(dataset_name)
        self.sent_idx_to_idx_file = data_utils.get_sent_idx_file(dataset_name)
        # load memory mapped files
        self.logger.info(f"Loading slices...")
        self.logger.debug("Seeing if " + dataset_name + " exists")
        start = time.time()
        if (not args.data_config.overwrite_preprocessed_data and os.path.exists(self.dataset_name)
        and os.path.exists(self.config_dataset_name) and os.path.exists(self.sent_idx_to_idx_file)):
            self.logger.debug(f"Will load existing dataset {dataset_name}")
        else:
            self.logger.debug(f"Building dataset with {input_src}")
            # only prep data once per node
            if is_writer:
                self.storage_type  = prep_slice(args=args, file=os.path.basename(input_src),
                    use_weak_label=use_weak_label, dataset_is_eval=dataset_is_eval,
                    dataset_name=self.dataset_name,
                    sent_idx_file=self.sent_idx_to_idx_file,
                    storage_config=self.config_dataset_name,
                    logger=self.logger
                    )
                np.save(self.config_dataset_name, self.storage_type, allow_pickle=True)
            if distributed:
                # Make sure all processes wait for data to be created
                dist.barrier()
            self.logger.debug(f"Finished building and saving dataset in {round(time.time() - start, 2)}s.")
        self.storage_type = np.load(self.config_dataset_name, allow_pickle=True).item()
        self.sent_idx_arr = np.memmap(self.sent_idx_to_idx_file, dtype=np.int, mode='r')
        st = time.time()
        # Load and reformat it to be the proper recarray shape of # rows x 1
        self.data = np.expand_dims(np.memmap(self.dataset_name, dtype=self.storage_type, mode='r').view(np.recarray), axis=1)
        assert len(self.data) > 0
        assert len(self.sent_idx_arr) > 0
        self.logger.info(f"Finished loading slices.")
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, key):
        """ Get an example with index. """
        example = self.data[key]
        return example

    def __getstate__(self):
        state = self.__dict__.copy()
        # Not picklable
        del state['data']
        del state['sent_idx_arr']
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data = np.expand_dims(np.memmap(self.dataset_name, dtype=self.storage_type, mode='r').view(np.recarray), axis=1)
        self.sent_idx_arr = np.memmap(self.sent_idx_to_idx_file, dtype=np.int, mode='r')
        self.logger = logging_utils.get_logger(self.args)

    def __repr__(self):
        return f"Slice {self.dataset_name} with {self.data_len} items"

    def get_non_empty_sent_idxs(self, slice_name):
        """ Used when generating samples over the slices. We want to make sure we sample from each slice independently."""
        sent_idx = self.data[slice_name].sent_idx
        aliases_to_predict = self.data[slice_name].alias_to_predict
        # -1 is for dimension for np.all
        return sent_idx[~np.all(aliases_to_predict == 0, -1)]