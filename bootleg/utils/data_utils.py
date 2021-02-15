import os

from bootleg.symbols.constants import FINAL_LOSS
from bootleg.utils import utils


def correct_not_augmented_dict_values(gold, dict_values):
    """Modifies the dict_values to only contain those mentions that are gold
    labels. The new dictionary has the alias indices be corrected to start at 0
    and end at the number of gold mentions.

    Args:
        gold: List of T/F values if mention is gold label or not
        dict_values: Dict of slice_name -> Dict[alias_idx] -> slice probability

    Returns: adjusted dict_values such that only gold = True aliases are kept (dict is reindexed to start at 0)
    """
    new_dict_values = {}
    gold_idx = [i for i in range(len(gold)) if gold[i] is True]
    for slice_name in list(dict_values.keys()):
        alias_dict = dict_values[slice_name]
        # i will not be in gold_idx if it wasn't an gold to being with
        new_dict_values[slice_name] = {
            str(gold_idx.index(int(i))): alias_dict[i]
            for i in alias_dict
            if int(i) in gold_idx
        }
        if len(new_dict_values[slice_name]) <= 0:
            del new_dict_values[slice_name]
    return new_dict_values


# eval_slices must include FINAL_LOSS
def get_eval_slices(eval_slices):
    """Given input eval slices (passed in config), ensure FINAL_LOSS is in the
    eval slices. FINAL_LOSS gives overall metrics.

    Args:
        eval_slices: list of input eval slices

    Returns: list of eval slices to use in the model
    """
    slice_names = eval_slices[:]
    # FINAL LOSS is in ALL MODELS for ALL SLICES
    if FINAL_LOSS not in slice_names:
        slice_names.insert(0, FINAL_LOSS)
    return slice_names


def get_save_data_folder(data_args, use_weak_label, dataset):
    """Give save data folder for the prepped data.

    Args:
        data_args: data config
        use_weak_label: whether to use weak labelling or not
        dataset: dataset name

    Returns: folder string path
    """
    name = os.path.splitext(os.path.basename(dataset))[0]
    direct = os.path.dirname(dataset)
    fold_name = (
        f"{name}_{data_args.word_embedding.bert_model}_L{data_args.max_seq_len}"
        f"_A{data_args.max_aliases}_InC{int(data_args.train_in_candidates)}_Aug{int(use_weak_label)}"
    )
    return os.path.join(direct, data_args.data_prep_dir, fold_name)


def generate_slice_name(data_args, slice_names, use_weak_label, dataset):
    """Generate name for slice datasets, taking into account the eval slices in
    the config.

    Args:
        data_args: data args
        slice_names: slice names
        use_weak_label: if using weak labels or not
        dataset: dataset name

    Returns: dataset name for saving slice data
    """
    dataset_name = os.path.join(
        get_save_data_folder(data_args, use_weak_label, dataset), "slices.pt"
    )
    names_for_dataset = str(hash(slice_names))
    dataset_name = os.path.splitext(dataset_name)[0] + "_" + names_for_dataset + ".pt"
    return dataset_name


def get_emb_prep_dir(data_config):
    """Get embedding prep directory for saving prep files. Lives inside
    entity_dir.

    Args:
        data_config: data config

    Returns: directory path
    """
    prep_dir = os.path.join(data_config.entity_dir, data_config.entity_prep_dir)
    utils.ensure_dir(prep_dir)
    return prep_dir


def get_data_prep_dir(data_config):
    """Get data prep directory for saving prep files. Lives inside data_dir.

    Args:
        data_config: data config

    Returns: directory path
    """
    prep_dir = os.path.join(data_config.data_dir, data_config.data_prep_dir)
    utils.ensure_dir(prep_dir)
    return prep_dir


def get_chunk_dir(prep_dir):
    """Get directory for saving data chunks.

    Args:
        prep_dir: prep directory

    Returns: directory path
    """
    return os.path.join(prep_dir, "chunks")
