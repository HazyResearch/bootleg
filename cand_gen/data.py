"""Data"""
import logging
import os

from emmental import Meta
from emmental.data import EmmentalDataLoader, emmental_collate_fn
from torch.utils.data import DistributedSampler, RandomSampler

from bootleg import log_rank_0_info
from bootleg.data import bootleg_collate_fn
from cand_gen.dataset import CandGenContextDataset, CandGenDataset, CandGenEntityDataset
from cand_gen.task_config import BATCH_CANDS_LABEL

logger = logging.getLogger(__name__)


def get_dataloaders(
    args,
    tasks,
    splits,
    entity_symbols,
    tokenizer,
):
    """Get the dataloaders.

    Args:
        args: main args
        tasks: task names
        use_batch_cands: whether to use candidates across a batch (train and eval_batch_cands)
        splits: data splits to generate dataloaders for
        entity_symbols: entity symbols

    Returns: list of dataloaders
    """
    task_to_label_dict = {t: BATCH_CANDS_LABEL for t in tasks}
    is_bert = True

    datasets = {}
    for split in splits:
        dataset_path = os.path.join(
            args.data_config.data_dir, args.data_config[f"{split}_dataset"].file
        )
        datasets[split] = CandGenDataset(
            main_args=args,
            name="Bootleg",
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
                    "You are using distributed computing for eval. We are not using a distributed sampler. "
                    "Please use DataParallel and not DDP.",
                )
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=dataset,
                sampler=dataset_sampler,
                split=split,
                collate_fn=bootleg_collate_fn,
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


def get_entity_dataloader(
    args,
    tasks,
    entity_symbols,
    tokenizer,
):
    """Get the dataloaders.

    Args:
        args: main args
        tasks: task names
        entity_symbols: entity symbols

    Returns: list of dataloaders
    """
    task_to_label_dict = {t: None for t in tasks}
    split = "test"

    dataset_path = os.path.join(
        args.data_config.data_dir, args.data_config[f"{split}_dataset"].file
    )
    dataset = CandGenEntityDataset(
        main_args=args,
        name="Bootleg",
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
            "You are using distributed computing for eval. We are not using a distributed sampler. "
            "Please use DataParallel and not DDP.",
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


def get_context_dataloader(
    args,
    tasks,
    entity_symbols,
    tokenizer,
    dataset_range=None,
):
    """Get the dataloaders.

    Args:
        args: main args
        tasks: task names
        entity_symbols: entity symbols
        tokenizer: tokenizer
        dataset_range: the subset of the dataset to wrap

    Returns: list of dataloaders
    """
    task_to_label_dict = {t: None for t in tasks}
    split = "test"
    is_bert = True
    dataset_path = os.path.join(
        args.data_config.data_dir, args.data_config[f"{split}_dataset"].file
    )
    dataset = CandGenContextDataset(
        main_args=args,
        name="Bootleg",
        dataset=dataset_path,
        use_weak_label=args.data_config[f"{split}_dataset"].use_weak_label,
        tokenizer=tokenizer,
        entity_symbols=entity_symbols,
        dataset_threads=args.run_config.dataset_threads,
        split=split,
        is_bert=is_bert,
        dataset_range=dataset_range,
    )
    dataset_sampler = None
    if Meta.config["learner_config"]["local_rank"] != -1:
        log_rank_0_info(
            logger,
            "You are using distributed computing for eval. We are not using a distributed sampler. "
            "Please use DataParallel and not DDP.",
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
