import copy
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DistributedSampler, RandomSampler

from bootleg import log_rank_0_info
from bootleg.dataset import BootlegDataset, BootlegEntityDataset
from bootleg.slicing.slice_dataset import BootlegSliceDataset
from bootleg.task_config import BATCH_CANDS_LABEL, CANDS_LABEL
from emmental import Meta
from emmental.data import EmmentalDataLoader, emmental_collate_fn
from emmental.utils.utils import list_to_tensor

logger = logging.getLogger(__name__)


def get_slicedatasets(args, splits, entity_symbols):
    """Get the slice datasets.

    Args:
        args: main args
        splits: splits to get datasets for
        entity_symbols: entity symbols

    Returns: Dict of slice datasets
    """
    datasets = {}
    splits = splits
    for split in splits:
        dataset_path = os.path.join(
            args.data_config.data_dir, args.data_config[f"{split}_dataset"].file
        )
        datasets[split] = BootlegSliceDataset(
            main_args=args,
            dataset=dataset_path,
            use_weak_label=args.data_config[f"{split}_dataset"].use_weak_label,
            entity_symbols=entity_symbols,
            dataset_threads=args.run_config.dataset_threads,
            split=split,
        )
    return datasets


def get_dataloaders(
    args,
    tasks,
    use_batch_cands,
    splits,
    entity_symbols,
    tokenizer,
):
    """Gets the dataloaders.

    Args:
        args: main args
        tasks: task names
        use_batch_cands: whether to use candidates across a batch (train and eval_batch_cands)
        splits: data splits to generate dataloaders for
        entity_symbols: entity symbols

    Returns: list of dataloaders
    """
    task_to_label_dict = {
        t: BATCH_CANDS_LABEL if use_batch_cands else CANDS_LABEL for t in tasks
    }
    is_bert = True

    datasets = {}
    for split in splits:
        dataset_path = os.path.join(
            args.data_config.data_dir, args.data_config[f"{split}_dataset"].file
        )
        datasets[split] = BootlegDataset(
            main_args=args,
            name=f"Bootleg",
            dataset=dataset_path,
            use_weak_label=args.data_config[f"{split}_dataset"].use_weak_label,
            tokenizer=tokenizer,
            entity_symbols=entity_symbols,
            dataset_threads=args.run_config.dataset_threads,
            split=split,
            is_bert=is_bert,
        )
    dataloaders = []
    for split, dataset in datasets.items():
        if split in args.learner_config.train_split:
            dataset_sampler = (
                RandomSampler(dataset)
                if Meta.config["learner_config"]["local_rank"] == -1
                else DistributedSampler(
                    dataset, seed=Meta.config["meta_config"]["seed"]
                )
            )
        else:
            dataset_sampler = None
            if Meta.config["learner_config"]["local_rank"] != -1:
                log_rank_0_info(
                    logger,
                    f"You are using distributed computing for eval. We are not using a distributed sampler. "
                    f"Please use DataParallel and not DDP.",
                )
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=dataset,
                sampler=dataset_sampler,
                split=split,
                collate_fn=bootleg_collate_fn
                if use_batch_cands
                else emmental_collate_fn,
                batch_size=args.train_config.batch_size
                if split in args.learner_config.train_split
                or args.run_config.eval_batch_size is None
                else args.run_config.eval_batch_size,
                num_workers=args.run_config.dataloader_threads,
                pin_memory=False,
            )
        )
        log_rank_0_info(
            logger,
            f"Built dataloader for {split} set with {len(dataset)} and {args.run_config.dataloader_threads} threads "
            f"samples (Shuffle={split in args.learner_config.train_split}, "
            f"Batch size={dataloaders[-1].batch_size}).",
        )

    return dataloaders


def get_entity_dataloaders(
    args,
    tasks,
    entity_symbols,
    tokenizer,
):
    """Gets the dataloaders.

    Args:
        args: main args
        tasks: task names
        entity_symbols: entity symbols

    Returns: list of dataloaders
    """
    task_to_label_dict = {t: CANDS_LABEL for t in tasks}
    split = "test"

    dataset_path = os.path.join(
        args.data_config.data_dir, args.data_config[f"{split}_dataset"].file
    )
    dataset = BootlegEntityDataset(
        main_args=args,
        name=f"Bootleg",
        dataset=dataset_path,
        tokenizer=tokenizer,
        entity_symbols=entity_symbols,
        dataset_threads=args.run_config.dataset_threads,
        split=split,
    )
    dataset_sampler = None
    if Meta.config["learner_config"]["local_rank"] != -1:
        log_rank_0_info(
            logger,
            f"You are using distributed computing for eval. We are not using a distributed sampler. "
            f"Please use DataParallel and not DDP.",
        )
    dataloader = EmmentalDataLoader(
        task_to_label_dict=task_to_label_dict,
        dataset=dataset,
        sampler=dataset_sampler,
        split=split,
        collate_fn=emmental_collate_fn,
        batch_size=args.train_config.batch_size
        if split in args.learner_config.train_split
        or args.run_config.eval_batch_size is None
        else args.run_config.eval_batch_size,
        num_workers=args.run_config.dataloader_threads,
        pin_memory=False,
    )
    log_rank_0_info(
        logger,
        f"Built dataloader for {split} set with {len(dataset)} and {args.run_config.dataloader_threads} threads "
        f"samples (Shuffle={split in args.learner_config.train_split}, "
        f"Batch size={dataloader.batch_size}).",
    )

    return dataloader


def bootleg_collate_fn(
    batch: Union[
        List[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]], List[Dict[str, Any]]
    ]
) -> Union[Tuple[Dict[str, Any], Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Collate function (modified from emmental collate fn). The main
    difference is our collate function merges candidates from across the batch for disambiguation.
    Args:
      batch: The batch to collate.
    Returns:
      The collated batch.
    """
    X_batch: defaultdict = defaultdict(list)
    # In Bootleg, we may have a nested dictionary in x_dict; we want to keep this structure but
    # collate the subtensors
    X_sub_batch: defaultdict = defaultdict(lambda: defaultdict(list))
    Y_batch: defaultdict = defaultdict(list)

    # Learnable batch should be a pair of dict, while non learnable batch is a dict
    is_learnable = True if not isinstance(batch[0], dict) else False
    if is_learnable:
        for x_dict, y_dict in batch:
            if isinstance(x_dict, dict) and isinstance(y_dict, dict):
                for field_name, value in x_dict.items():
                    if isinstance(value, list):
                        X_batch[field_name] += value
                    elif isinstance(value, dict):
                        # We reinstantiate the field_name here
                        # This keeps the field_name key intact
                        if field_name not in X_sub_batch:
                            X_sub_batch[field_name] = defaultdict(list)
                        for sub_field_name, sub_value in value.items():
                            if isinstance(sub_value, list):
                                X_sub_batch[field_name][sub_field_name] += sub_value
                            else:
                                X_sub_batch[field_name][sub_field_name].append(
                                    sub_value
                                )
                    else:
                        X_batch[field_name].append(value)
                for label_name, value in y_dict.items():
                    if isinstance(value, list):
                        Y_batch[label_name] += value
                    else:
                        Y_batch[label_name].append(value)
    else:
        for x_dict in batch:  # type: ignore
            for field_name, value in x_dict.items():  # type: ignore
                if isinstance(value, list):
                    X_batch[field_name] += value
                elif isinstance(value, dict):
                    # We reinstantiate the field_name here
                    # This keeps the field_name key intact
                    if field_name not in X_sub_batch:
                        X_sub_batch[field_name] = defaultdict(list)
                    for sub_field_name, sub_value in value.items():
                        if isinstance(sub_value, list):
                            X_sub_batch[field_name][sub_field_name] += sub_value
                        else:
                            X_sub_batch[field_name][sub_field_name].append(sub_value)
                else:
                    X_batch[field_name].append(value)
    field_names = copy.deepcopy(list(X_batch.keys()))
    for field_name in field_names:
        values = X_batch[field_name]
        # Only merge list of tensors
        if isinstance(values[0], torch.Tensor):
            item_tensor, item_mask_tensor = list_to_tensor(
                values,
                min_len=Meta.config["data_config"]["min_data_len"],
                max_len=Meta.config["data_config"]["max_data_len"],
            )
            X_batch[field_name] = item_tensor

    field_names = copy.deepcopy(list(X_sub_batch.keys()))
    for field_name in field_names:
        sub_field_names = copy.deepcopy(list(X_sub_batch[field_name].keys()))
        for sub_field_name in sub_field_names:
            values = X_sub_batch[field_name][sub_field_name]
            # Only merge list of tensors
            if isinstance(values[0], torch.Tensor):
                item_tensor, item_mask_tensor = list_to_tensor(
                    values,
                    min_len=Meta.config["data_config"]["min_data_len"],
                    max_len=Meta.config["data_config"]["max_data_len"],
                )
                X_sub_batch[field_name][sub_field_name] = item_tensor
    # Add sub batch to batch
    for field_name in field_names:
        X_batch[field_name] = dict(X_sub_batch[field_name])
    if is_learnable:
        for label_name, values in Y_batch.items():
            Y_batch[label_name] = list_to_tensor(
                values,
                min_len=Meta.config["data_config"]["min_data_len"],
                max_len=Meta.config["data_config"]["max_data_len"],
            )[0]
    # ACROSS BATCH CANDIDATE MERGING
    # Turns from b x m x k to E where E is the number of unique entities
    all_uniq_eids = []
    all_uniq_eid_idx = []
    label = []
    # print("BATCH", X_batch["entity_cand_eid"])
    for k, batch_eids in enumerate(X_batch["entity_cand_eid"]):
        for j, eid in enumerate(batch_eids):
            # Skip if already in batch or if it's the unk...we don't use masking in the softmax for batch_cands
            # data loading (training and during train eval)
            if (
                eid in all_uniq_eids
                or X_batch["entity_cand_eval_mask"][k][j].item() is True
            ):
                continue
            all_uniq_eids.append(eid)
            all_uniq_eid_idx.append([k, j])

    for eid in X_batch["gold_eid"]:
        men_label = []
        if eid not in all_uniq_eids:
            men_label.append(-2)
        else:
            men_label.append(all_uniq_eids.index(eid))
        label.append(men_label)

    # Super rare edge case if doing eval during training on small batch sizes and have an entire batch
    # where the alias is -2 (i.e., we don't have it in our dump)
    if len(all_uniq_eids) == 0:
        # Give the unq entity in this case -> we want the model to get the wrong answer anyways and it will
        # all_uniq_eids = [X_batch["entity_cand_eid"][0][0]]
        all_uniq_eid_idx = [[0, 0]]

    all_uniq_eid_idx = torch.LongTensor(all_uniq_eid_idx)
    assert len(all_uniq_eid_idx.size()) == 2 and all_uniq_eid_idx.size(1) == 2
    for key in X_batch.keys():
        # Don't transform the mask as that's only used for no batch cands
        if (
            key.startswith("entity_")
            and key != "entity_cand_eval_mask"
            and key != "entity_to_mask"
        ):
            X_batch[key] = X_batch[key][all_uniq_eid_idx[:, 0], all_uniq_eid_idx[:, 1]]
    # print("FINAL", X_batch["entity_cand_eid"])
    Y_batch["gold_unq_eid_idx"] = torch.LongTensor(label)
    # for k in X_batch:
    #     try:
    #         print(k, X_batch[k].shape)
    #     except:
    #         print(k, len(X_batch[k]))
    # for k in Y_batch:
    #     print(k, Y_batch[k].shape, Y_batch[k])
    if is_learnable:
        return dict(X_batch), dict(Y_batch)
    else:
        return dict(X_batch)
