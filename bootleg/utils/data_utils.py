"""Bootleg data utils."""
import os

from bootleg.symbols.constants import FINAL_LOSS, SPECIAL_TOKENS
from bootleg.utils import utils


def correct_not_augmented_dict_values(gold, dict_values):
    """
    Correct gold label dict values in data prep.

    Modifies the dict_values to only contain those mentions that are gold
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
    """
    Get eval slices in data prep.

    Given input eval slices (passed in config), ensure FINAL_LOSS is in the
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
    """
    Get save data folder for the prepped data.

    Args:
        data_args: data config
        use_weak_label: whether to use weak labelling or not
        dataset: dataset name

    Returns: folder string path
    """
    name = os.path.splitext(os.path.basename(dataset))[0]
    direct = os.path.dirname(dataset)
    bert_mod = data_args.word_embedding.bert_model.replace("/", "_")
    fold_name = (
        f"{name}_{bert_mod}_L{data_args.max_seq_len}_E{data_args.max_ent_len}"
        f"_W{data_args.max_seq_window_len}"
        f"_T{data_args.entity_type_data.use_entity_types}"
        f"_K{data_args.entity_kg_data.use_entity_kg}"
        f"_D{data_args.use_entity_desc}"
        f"_InC{int(data_args.train_in_candidates)}"
        f"_Aug{int(use_weak_label)}"
    )
    return os.path.join(direct, data_args.data_prep_dir, fold_name)


def get_save_data_folder_candgen(data_args, use_weak_label, dataset):
    """Give save data folder for the prepped data.

    Args:
        data_args: data config
        use_weak_label: whether to use weak labelling or not
        dataset: dataset name

    Returns: folder string path
    """
    name = os.path.splitext(os.path.basename(dataset))[0]
    direct = os.path.dirname(dataset)
    bert_mod = data_args.word_embedding.bert_model.replace("/", "_")
    fold_name = (
        f"{name}_{bert_mod}_L{data_args.max_seq_len}_E{data_args.max_ent_len}"
        f"_W{data_args.max_seq_window_len}"
        f"_A{data_args.use_entity_akas}"
        f"_D{data_args.use_entity_desc}"
        f"_InC{int(data_args.train_in_candidates)}"
        f"_Aug{int(use_weak_label)}"
    )
    return os.path.join(direct, data_args.data_prep_dir, fold_name)


def generate_slice_name(data_args, slice_names, use_weak_label, dataset):
    """
    Generate name for slice datasets, taking into account the config eval slices.

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
    """
    Get embedding prep directory for saving prep files.

    Args:
        data_config: data config

    Returns: directory path
    """
    prep_dir = os.path.join(data_config.entity_dir, data_config.entity_prep_dir)
    utils.ensure_dir(prep_dir)
    return prep_dir


def get_data_prep_dir(data_config):
    """
    Get data prep directory for saving prep files.

    Args:
        data_config: data config

    Returns: directory path
    """
    prep_dir = os.path.join(data_config.data_dir, data_config.data_prep_dir)
    utils.ensure_dir(prep_dir)
    return prep_dir


def get_chunk_dir(prep_dir):
    """
    Get directory for saving data chunks.

    Args:
        prep_dir: prep directory

    Returns: directory path
    """
    return os.path.join(prep_dir, "chunks")


def add_special_tokens(tokenizer):
    """
    Add special tokens.

    Args:
        tokenizer: tokenizer
        data_config: data config
        entitysymbols: entity symbols
    """
    # Add standard tokens
    tokenizer.add_special_tokens(SPECIAL_TOKENS)


def read_in_akas(entitysymbols):
    """Read in alias to QID mappings and generates a QID to list of alternate names.

    Args:
        entitysymbols: entity symbols

    Returns: dictionary of QID to type names
    """
    # take the first type; UNK type is 0
    qid2aliases = {}
    for al in entitysymbols.get_all_aliases():
        for qid in entitysymbols.get_qid_cands(al):
            if qid not in qid2aliases:
                qid2aliases[qid] = set()
            qid2aliases[qid].add(al)
    # Turn into sets for dumping
    for qid in qid2aliases:
        qid2aliases[qid] = list(qid2aliases[qid])
    return qid2aliases
