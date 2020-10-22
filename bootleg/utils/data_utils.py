import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from bootleg.dataloaders.distr_wrapper import DistributedIndicesWrapper
import bootleg.symbols.word_symbols as word_module
from bootleg.utils.utils import import_class
from bootleg.utils import logging_utils, utils

# We must correct all mappings with alias indexes in them
# This is because the indexing will change when we remove False golds (see example in prep.py)
def correct_not_augmented_max_cands(gold, max_cands_dict):
    gold_idx = [i for i in range(len(gold)) if gold[i] is True]
    new_max_cands = {str(gold_idx.index(int(i))):max_cands_dict[i] for i in max_cands_dict if int(i) in gold_idx}
    return new_max_cands

# used for slices
def correct_not_augmented_dict_values(gold, dict_values):
    new_dict_values = {}
    gold_idx = [i for i in range(len(gold)) if gold[i] is True]
    for slice_name in list(dict_values.keys()):
        alias_dict = dict_values[slice_name]
        # i will not be in gold_idx if it wasn't an gold to being with
        new_dict_values[slice_name] = {str(gold_idx.index(int(i))):alias_dict[i] for i in alias_dict if int(i) in gold_idx}
        if len(new_dict_values[slice_name]) <= 0:
            del new_dict_values[slice_name]
    return new_dict_values

def get_base_slice(gold, slices, slice_names, dataset_is_eval):
    if dataset_is_eval:
        return {str(i):1.0 for i in range(len(gold)) if gold[i] is True}
    else:
        return {str(i):1.0 for i in range(len(gold))}

def generate_save_data_name(data_args, use_weak_label, split_name):
    return f"{split_name}_{data_args.word_embedding.word_symbols}_L{data_args.max_word_token_len}" \
           f"_A{data_args.max_aliases}_InC{int(data_args.train_in_candidates)}_Aug{int(use_weak_label)}.pt"

def generate_slice_name(args, data_args, use_weak_label, split_name, dataset_is_eval):
    dataset_name = generate_save_data_name(data_args, use_weak_label, split_name)
    if dataset_is_eval:
        names_for_dataset = args.run_config.eval_slices
    else:
        names_for_dataset = args.train_config.train_heads
    names_for_dataset = [st[:2] for st in names_for_dataset]
    dataset_name = os.path.splitext(dataset_name)[0] + "_" + "_".join(names_for_dataset) + ".pt"
    return dataset_name

def get_emb_prep_dir(args):
    prep_dir = os.path.join(args.data_config.entity_dir, args.data_config.entity_prep_dir)
    utils.ensure_dir(prep_dir)
    return prep_dir

def get_data_prep_dir(args):
    prep_dir = os.path.join(args.data_config.data_dir, args.data_config.data_prep_dir)
    utils.ensure_dir(prep_dir)
    return prep_dir

def get_slice_storage_file(dataset_name):
    return os.path.splitext(dataset_name)[0] + '_config.npy'

def get_storage_file(dataset_name):
    return os.path.splitext(dataset_name)[0] + '_storage_type.pkl'

def get_sent_idx_file(dataset_name):
    return os.path.splitext(dataset_name)[0] + '_sent_idx_to_idx.pt'

def get_batch_prep_config(dataset_name):
    return os.path.splitext(dataset_name)[0] + "_batch_prep_config.json"

def create_dataset(args, data_args, is_writer, word_symbols,
    entity_symbols, slice_dataset=None, dataset_is_eval=False):
    dataset_name = generate_save_data_name(data_args=args.data_config,
        use_weak_label=data_args.use_weak_label,
        split_name=os.path.splitext(data_args.file)[0])
    prep_dir = get_data_prep_dir(args)
    full_dataset_name = os.path.join(prep_dir, dataset_name)
    mod, load_class = import_class("bootleg.dataloaders", data_args.load_class)
    dataset = getattr(mod, load_class)(args=args, use_weak_label=data_args.use_weak_label, input_src=os.path.join(args.data_config.data_dir, data_args.file),
                                       dataset_name=full_dataset_name, is_writer=is_writer,
                                       distributed=args.run_config.distributed, word_symbols=word_symbols,
                                       entity_symbols=entity_symbols, slice_dataset=slice_dataset,
                                       dataset_is_eval=dataset_is_eval)
    return dataset

# For a given slice, finds all the subsentence examples in the dataset corresponding to each index in the slice
# Ex: if the sentences "43 a b c d e f" (sent_idx 43) gets turned into "43 0 a b c" (sent_idx 43 and subsent_idx 0) and "43 1 d e f" in prep (sent_idx 43 and subsent_idx 1)
# then this will add both the indexes in dataset.data corresponding to the two subsent indexes
def get_eval_slice_subset_indices(args, eval_slice_dataset, dataset):
    logger = logging_utils.get_logger(args)
    logger.debug('Starting to sample indices')
    eval_slices = args.run_config.eval_slices
    # Get unique sentence indexes from samples for all slices
    # Will take union of all the data rows that map from these sentence indexes to eval
    sent_indices = set()
    for slice_name in eval_slices:
        # IF THE SEED CHANGES WHAT IS SAMPLED FOR DEV WILL CHANGE
        # randomly sample indices from slice
        slice_indexes = eval_slice_dataset.get_non_empty_sent_idxs(slice_name)
        perc_eval_examples = int(args.run_config.perc_eval * len(slice_indexes))
        data_len = max(perc_eval_examples, args.run_config.min_eval_size, 1)
        if data_len >= len(slice_indexes):
            # if requested sample is larger than the actual data, just use the whole slice
            random_indices = range(len(slice_indexes))
        else:
            random_indices = np.random.choice(len(slice_indexes), data_len, replace=False)
        for idx in random_indices:
            sent_indices.add(slice_indexes[idx])

    logger.debug('Starting to gather indices')
    # Get corresponding indices for key set
    indices = []
    for sent_idx in sent_indices:
        if sent_idx in dataset.sent_idx_to_idx:
            samples = dataset.sent_idx_to_idx[sent_idx]
            for data_idx in samples:
                indices.append(data_idx)
    logger.info(f'Sampled {len(indices)} indices from dataset (dev/test) for evaluation.')
    return indices


# For eval data, to save time, we allow for deterministic sampling. To ensure coverage of the slices, we sample
# each slice independently. Therefore, whether this is a train dataset or eval dataset matters for the samplers.
def create_dataloader(args, dataset, batch_size, eval_slice_dataset=None, world_size=None, rank=None):
    logger = logging_utils.get_logger(args)
    if eval_slice_dataset is not None and not args.run_config.distributed:
        indices = get_eval_slice_subset_indices(args, eval_slice_dataset=eval_slice_dataset, dataset=dataset)
        # Form sampler with for indices from eval_slice_dataset
        sampler = SubsetRandomSampler(indices)
    elif args.run_config.distributed:
        # wrap dataset object to use a subsetsampler with distributed
        if eval_slice_dataset is not None:
            indices = get_eval_slice_subset_indices(args, eval_slice_dataset=eval_slice_dataset, dataset=dataset)
            dataset = DistributedIndicesWrapper(dataset, torch.tensor(indices))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(sampler is None),
                            sampler=sampler,
                            num_workers=args.run_config.dataloader_threads,
                            pin_memory=False)
    return dataloader, sampler


def create_slice_dataset(args, data_args, is_writer, dataset_is_eval):
    # Note that the weak labelling is going to alter our indexing for the slices. Our slices still only score gold==True
    dataset_name = generate_slice_name(args, args.data_config, use_weak_label=data_args.use_weak_label,
                                           split_name="slice_" + os.path.splitext(data_args.file)[0],
                                           dataset_is_eval=dataset_is_eval)
    prep_dir = get_data_prep_dir(args)
    full_dataset_name = os.path.join(prep_dir, dataset_name)
    mod, load_class = import_class("bootleg.dataloaders", data_args.slice_class)
    dataset = getattr(mod, load_class)(args=args, use_weak_label=data_args.use_weak_label, input_src=os.path.join(args.data_config.data_dir, data_args.file),
                      dataset_name=full_dataset_name, is_writer=is_writer,
                      distributed=args.run_config.distributed, dataset_is_eval=dataset_is_eval)
    return dataset


def load_wordsymbols(data_args, is_writer=True, distributed=False):
    embedding_config = data_args.word_embedding
    my_class = getattr(word_module, embedding_config.word_symbols)
    word_symbols = my_class(data_args, is_writer=is_writer,
        distributed=distributed)
    return word_symbols


def load_glove(file_path, log_func):
    log_func('  Loading Glove format file {}'.format(file_path))
    embeddings = {}
    embedding_size = 0

    # collect embeddings size assuming the first line is correct
    num_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        found_line = False
        for line in f:
            num_lines += 1
            if not found_line:
                embedding_size = len(line.split()) - 1
                found_line = True

    # collect embeddings
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in tqdm(enumerate(f), total=num_lines):
            if line:
                try:
                    split = line.split()
                    if len(split) != embedding_size + 1:
                        raise ValueError
                    word = split[0]
                    embedding = np.array(
                        [float(val) for val in split[-embedding_size:]]
                    )
                    embeddings[word] = embedding
                except ValueError:
                    log_func(
                        'Line {} in the GloVe file {} is malformed, '
                        'skipping it'.format(
                            line_number, file_path
                        )
                    )
    log_func('  {0} embeddings loaded'.format(len(embeddings)))
    return embeddings, embedding_size