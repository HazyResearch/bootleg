import copy
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from torch import Tensor
from torch.utils.data import DistributedSampler, RandomSampler
from transformers import BertTokenizer

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.datasets.dataset import BootlegDataset
from bootleg.slicing.slice_dataset import BootlegSliceDataset
from bootleg.task_config import NED_TASK_TO_LABEL
from bootleg.utils import embedding_utils
from bootleg.utils.utils import import_class
from emmental import Meta
from emmental.data import EmmentalDataLoader
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
    splits,
    entity_symbols,
    batch_on_the_fly_kg_adj,
):
    """Gets the dataloaders.

    Args:
        args: main args
        tasks: task names
        splits: data splits to generate dataloaders for
        entity_symbols: entity symbols
        batch_on_the_fly_kg_adj: kg embeddings metadata for the __get_item__ method (see get_dataloader_embeddings)

    Returns: list of dataloaders
    """
    task_to_label_dict = {t: NED_TASK_TO_LABEL[t] for t in tasks}
    is_bert = len(args.data_config.word_embedding.bert_model) > 0
    tokenizer = BertTokenizer.from_pretrained(
        args.data_config.word_embedding.bert_model,
        do_lower_case=True
        if "uncased" in args.data_config.word_embedding.bert_model
        else False,
        cache_dir=args.data_config.word_embedding.cache_dir,
    )

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
            batch_on_the_fly_kg_adj=batch_on_the_fly_kg_adj,
        )

    dataloaders = []
    for split, dataset in datasets.items():
        if split in args.learner_config.train_split:
            dataset_sampler = (
                RandomSampler(dataset)
                if Meta.config["learner_config"]["local_rank"] == -1
                else DistributedSampler(dataset)
            )
        else:
            dataset_sampler = None
            if Meta.config["learner_config"]["local_rank"] != -1:
                log_rank_0_info(
                    logger,
                    f"You are using distributed computing for eval. We are not using a distributed sampler. Please use DataParallel and not DDP.",
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


def get_dataloader_embeddings(main_args, entity_symbols):
    """Gets KG embeddings that need to be processed in the __get_item__ method
    of a dataset (e.g., querying a sparce numpy matrix). We save, for each KG
    embedding class that needs this preprocessing, the adjacency matrix (for KG
    connections), the processing function to run in __get_item__, and the file
    to load the adj matrix for dumping/loading.

    Args:
        main_args: main arguments
        entity_symbols: entity symbols

    Returns: Dict of KG metadata for using in the __get_item__ method.
    """
    batch_on_the_fly_kg_adj = {}
    for emb in main_args.data_config.ent_embeddings:
        batch_on_fly = "batch_on_the_fly" in emb and emb["batch_on_the_fly"] is True
        # Find embeddings that have a "batch of the fly" key
        if batch_on_fly:
            log_rank_0_debug(
                logger,
                f"Loading class {emb.load_class} for preprocessing as on the fly or in data prep embeddings",
            )
            (
                cpu,
                dropout1d_perc,
                dropout2d_perc,
                emb_args,
                freeze,
                normalize,
                through_bert,
            ) = embedding_utils.get_embedding_args(emb)
            try:
                # Load the object
                mod, load_class = import_class("bootleg.embeddings", emb.load_class)
                kg_class = getattr(mod, load_class)(
                    main_args=main_args,
                    emb_args=emb_args,
                    entity_symbols=entity_symbols,
                    key=emb.key,
                    cpu=cpu,
                    normalize=normalize,
                    dropout1d_perc=dropout1d_perc,
                    dropout2d_perc=dropout2d_perc,
                )
                # Extract its kg adj, we'll use this later
                # Extract the kg_adj_process_func (how to process the embeddings in __get_item__ or dataset prep)
                # Extract the prep_file. We use this to load the kg_adj back after saving/loading state using scipy.sparse.load_npz(prep_file)
                assert hasattr(
                    kg_class, "kg_adj"
                ), f"The embedding class {emb.key} does not have a kg_adj attribute and it needs to."
                assert hasattr(
                    kg_class, "kg_adj_process_func"
                ), f"The embedding class {emb.key} does not have a kg_adj_process_func attribute and it needs to."
                assert hasattr(kg_class, "prep_file"), (
                    f"The embedding class {emb.key} does not have a prep_file attribute and it needs to. We will call"
                    f" `scipy.sparse.load_npz(prep_file)` to load the kg_adj matrix."
                )
                batch_on_the_fly_kg_adj[emb.key] = {
                    "kg_adj": kg_class.kg_adj,
                    "kg_adj_process_func": kg_class.kg_adj_process_func,
                    "prep_file": kg_class.prep_file,
                }
            except AttributeError as e:
                logger.warning(
                    f"No prep method found for {emb.load_class} with error {e}"
                )
                raise
            except Exception as e:
                print("ERROR", e)
                raise
    return batch_on_the_fly_kg_adj


def bootleg_collate_fn(
    batch: Union[List[Tuple[Dict[str, Any], Dict[str, Tensor]]], List[Dict[str, Any]]]
) -> Union[Tuple[Dict[str, Any], Dict[str, Tensor]], Dict[str, Any]]:
    """Collate function (modified from emmental collate fn). The main
    difference is our collate function handles the kg_adj dictionary items from
    the dataset. We collate each value of each dict key.

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
                        # We reinstantiate the field_name here in case there is not kg adj data - this keeps the field_name key intact
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
                    # We reinstantiate the field_name here in case there is not kg adj data - this keeps the field_name key intact
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
        if isinstance(values[0], Tensor):
            item_tensor, item_mask_tensor = list_to_tensor(
                values,
                min_len=Meta.config["data_config"]["min_data_len"],
                max_len=Meta.config["data_config"]["max_data_len"],
            )
            X_batch[field_name] = item_tensor
            if item_mask_tensor is not None:
                X_batch[f"{field_name}_mask"] = item_mask_tensor

    field_names = copy.deepcopy(list(X_sub_batch.keys()))
    for field_name in field_names:
        sub_field_names = copy.deepcopy(list(X_sub_batch[field_name].keys()))
        for sub_field_name in sub_field_names:
            values = X_sub_batch[field_name][sub_field_name]
            # Only merge list of tensors
            if isinstance(values[0], Tensor):
                item_tensor, item_mask_tensor = list_to_tensor(
                    values,
                    min_len=Meta.config["data_config"]["min_data_len"],
                    max_len=Meta.config["data_config"]["max_data_len"],
                )
                X_sub_batch[field_name][sub_field_name] = item_tensor
                if item_mask_tensor is not None:
                    X_sub_batch[field_name][f"{sub_field_name}_mask"] = item_mask_tensor

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
    if is_learnable:
        return dict(X_batch), dict(Y_batch)
    else:
        return dict(X_batch)
