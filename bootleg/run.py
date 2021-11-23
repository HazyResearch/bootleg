"""Bootleg run command."""

import argparse
import itertools
import logging
import os
import shutil
import subprocess
import sys
import warnings
from copy import copy

import emmental
import numpy as np
import torch
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from rich.logging import RichHandler
from transformers import AutoTokenizer

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.data import get_dataloaders, get_slicedatasets
from bootleg.symbols.constants import DEV_SPLIT, TEST_SPLIT, TRAIN_SPLIT
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.task_config import NED_TASK
from bootleg.tasks import ned_task
from bootleg.utils import data_utils, eval_utils, utils
from bootleg.utils.eval_utils import collect_and_merge_results, dump_model_outputs
from bootleg.utils.model_utils import count_parameters
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args
from bootleg.utils.utils import (
    dump_yaml_file,
    load_yaml_file,
    recurse_redict,
    write_to_file,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def parse_cmdline_args():
    """
    Take an input config file and parse it into the correct subdictionary groups for the model.

    Returns:
        model run mode of train, eval, or dumping
        parsed Dict config
        path to original config path
    """
    # Parse cmdline args to specify config and mode
    cli_parser = argparse.ArgumentParser(
        description="Bootleg CLI Config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    cli_parser.add_argument(
        "--config_script",
        type=str,
        default="",
        help="Should mimic the config_args found in utils/parser/bootleg_args.py with parameters you want to override."
        "You can also override the parameters from config_script by passing them in directly after config_script. "
        "E.g., --train_config.batch_size 5",
    )
    cli_parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "dump_preds"],
    )
    cli_parser.add_argument(
        "--entity_emb_file",
        type=str,
        default=None,
        help="Path to dumped entity embeddings (see ```extract_all_entities.py``` for how). Used in eval and dumping",
    )

    # you can add other args that will override those in the config_script
    # parse_known_args returns 'args' that are the same as what parse_args() returns
    # and 'unknown' which are args that the parser doesn't recognize but you want to keep.
    # 'unknown' are what we pass on to our override any args from the second phase of arg parsing from the json file
    cli_args, unknown = cli_parser.parse_known_args()
    if len(cli_args.config_script) == 0:
        raise ValueError("You must pass a config script via --config.")
    config = parse_boot_and_emm_args(cli_args.config_script, unknown)

    #  Modify the local rank param from the cli args
    config.learner_config.local_rank = int(os.getenv("LOCAL_RANK", -1))
    mode = cli_args.mode
    entity_emb_file = cli_args.entity_emb_file
    return mode, config, cli_args.config_script, entity_emb_file


def setup(config, run_config_path=None):
    """
    Set distributed backend and save configuration files.

    Args:
        config: config
        run_config_path: path for original run config
    """
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # spawn method must be fork to work with Meta.config
    torch.multiprocessing.set_start_method("fork", force=True)
    """
    ulimit -n 500000
    python3 -m torch.distributed.launch --nproc_per_node=2  bootleg/run.py --config_script ...
    """
    log_level = logging.getLevelName(config.run_config.log_level.upper())
    emmental.init(
        log_dir=config["meta_config"]["log_path"],
        config=config,
        use_exact_log_path=config["meta_config"]["use_exact_log_path"],
        local_rank=config.learner_config.local_rank,
        level=log_level,
    )
    log = logging.getLogger()
    # Remove streaming handlers and use rich
    log.handlers = [h for h in log.handlers if not type(h) is logging.StreamHandler]
    log.addHandler(RichHandler())
    # Set up distributed backend
    emmental.Meta.init_distributed_backend()

    cmd_msg = " ".join(sys.argv)
    # Recast to dictionaries for emmental - will remove Dotteddicts
    emmental.Meta.config = recurse_redict(copy(emmental.Meta.config))
    # Log configuration into filess
    if config.learner_config.local_rank in [0, -1]:
        write_to_file(f"{emmental.Meta.log_path}/cmd.txt", cmd_msg)
        dump_yaml_file(
            f"{emmental.Meta.log_path}/parsed_config.yaml", emmental.Meta.config
        )
        # Dump the run config (does not contain defaults)
        if run_config_path is not None:
            dump_yaml_file(
                f"{emmental.Meta.log_path}/run_config.yaml",
                load_yaml_file(run_config_path),
            )

    log_rank_0_info(logger, f"COMMAND: {cmd_msg}")
    log_rank_0_info(
        logger, f"Saving config to {emmental.Meta.log_path}/parsed_config.yaml"
    )

    git_hash = "Not able to retrieve git hash"
    try:
        git_hash = subprocess.check_output(
            ["git", "log", "-n", "1", "--pretty=tformat:%h-%ad", "--date=short"]
        ).strip()
    except subprocess.CalledProcessError:
        pass
    log_rank_0_info(logger, f"Git Hash: {git_hash}")


def configure_optimizer():
    """
    Configure the optimizer for Bootleg.

    Args:
        config: config
    """
    # Specify parameter group for Adam BERT
    def grouped_parameters(model):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        return [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": emmental.Meta.config["learner_config"][
                    "optimizer_config"
                ]["l2"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    emmental.Meta.config["learner_config"]["optimizer_config"][
        "parameters"
    ] = grouped_parameters
    return


# TODO: optimize slices so we split them based on max aliases (save A LOT of memory)
def run_model(mode, config, run_config_path=None, entity_emb_file=None):
    """
    Run Emmental Bootleg models.

    Args:
        mode: run mode (train, eval, dump_preds)
        config: parsed model config
        run_config_path: original config path (for saving)
        entity_emb_file: file for dumped entity embeddings
    """
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # Set up distributed backend and save configuration files
    setup(config, run_config_path)

    # Load entity symbols
    log_rank_0_info(logger, "Loading entity symbols...")
    entity_symbols = EntitySymbols.load_from_cache(
        load_dir=os.path.join(
            config.data_config.entity_dir, config.data_config.entity_map_dir
        ),
        alias_cand_map_dir=config.data_config.alias_cand_map,
        alias_idx_dir=config.data_config.alias_idx_map,
    )
    # Create tasks
    tasks = [NED_TASK]

    # Create splits for data loaders
    data_splits = [TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT]
    # Slices are for eval so we only split on test/dev
    slice_splits = [DEV_SPLIT, TEST_SPLIT]
    # If doing eval, only run on test data
    if mode in ["eval"]:
        data_splits = [TEST_SPLIT]
        slice_splits = [TEST_SPLIT]
    elif mode in ["dump_preds"]:
        data_splits = [TEST_SPLIT]
        slice_splits = []
        # We only do dumping if weak labels is True
        if config.data_config[f"{TEST_SPLIT}_dataset"].use_weak_label is False:
            raise ValueError(
                "When calling dump_preds, we require use_weak_label to be True."
            )

    load_entity_data = True
    if mode == "train":
        assert (
            entity_emb_file is None
        ), "We do not accept entity_emb_file when training."
    else:
        # If we are doing eval with the entity embeddings, do not create/load entity token data
        if entity_emb_file is not None:
            load_entity_data = False

    # Batch cands is for training
    use_batch_cands = mode == "train"

    # Create tokenizer
    context_tokenizer = AutoTokenizer.from_pretrained(
        config.data_config.word_embedding.bert_model
    )
    data_utils.add_special_tokens(context_tokenizer)

    # Gets dataloaders
    dataloaders = get_dataloaders(
        config,
        tasks,
        use_batch_cands,
        load_entity_data,
        data_splits,
        entity_symbols,
        context_tokenizer,
    )
    slice_datasets = get_slicedatasets(config, slice_splits, entity_symbols)

    configure_optimizer()

    # Create models and add tasks
    log_rank_0_info(logger, "Starting Bootleg Model")
    model_name = "Bootleg"
    model = EmmentalModel(name=model_name)

    model.add_task(
        ned_task.create_task(
            config,
            use_batch_cands,
            len(context_tokenizer),
            slice_datasets,
            entity_emb_file,
        )
    )
    # Print param counts
    if mode == "train":
        log_rank_0_debug(logger, "PARAMS WITH GRAD\n" + "=" * 30)
        total_params = count_parameters(model, requires_grad=True, logger=logger)
        log_rank_0_info(logger, f"===> Total Params With Grad: {total_params}")
        log_rank_0_debug(logger, "PARAMS WITHOUT GRAD\n" + "=" * 30)
        total_params = count_parameters(model, requires_grad=False, logger=logger)
        log_rank_0_info(logger, f"===> Total Params Without Grad: {total_params}")

    # Load the best model from the pretrained model
    if config["model_config"]["model_path"] is not None:
        model.load(config["model_config"]["model_path"])

    # Train model
    if mode == "train":
        emmental_learner = EmmentalLearner()
        emmental_learner._set_optimizer(model)
        # Save first checkpoint
        if config.learner_config.local_rank in [0, -1]:
            model.save(f"{emmental.Meta.log_path}/checkpoint_0.0.model.pth")
        emmental_learner.learn(model, dataloaders)
        if config.learner_config.local_rank in [0, -1]:
            model.save(f"{emmental.Meta.log_path}/last_model.pth")

    # Multi-gpu DataParallel eval (NOT distributed)
    if mode in ["eval", "dump_preds"]:
        # This happens inside EmmentalLearner for training
        if (
            config["learner_config"]["local_rank"] == -1
            and config["model_config"]["dataparallel"]
        ):
            model._to_dataparallel()

    # If just finished training a model or in eval mode, run eval
    if mode in ["train", "eval"]:
        if config.learner_config.local_rank in [0, -1]:
            if mode == "train":
                # Skip the TRAIN dataloader
                scores = model.score(dataloaders[1:])
            else:
                scores = model.score(dataloaders)
            # Save metrics and models
            log_rank_0_info(logger, f"Saving metrics to {emmental.Meta.log_path}")
            log_rank_0_info(logger, f"Metrics: {scores}")
            scores["log_path"] = emmental.Meta.log_path
            write_to_file(f"{emmental.Meta.log_path}/{mode}_metrics.txt", scores)
            eval_utils.write_disambig_metrics_to_csv(
                f"{emmental.Meta.log_path}/{mode}_disambig_metrics.csv", scores
            )
        else:
            scores = {}
        return scores
    # If you want detailed dumps, save model outputs
    assert mode in [
        "dump_preds",
    ], 'Mode must be "dump_preds"'
    assert (
        len(dataloaders) == 1
    ), "We should only have length 1 dataloaders for dump_preds!"
    final_result_file = None
    # Get emmental action output for entity embeddings
    num_dump_file_splits = config.run_config.dump_preds_num_data_splits
    if config.learner_config.local_rank in [0, -1]:
        # Setup files/folders
        filename = os.path.basename(dataloaders[0].dataset.raw_filename)
        eval_folder = eval_utils.get_eval_folder(filename)
        temp_eval_folder = os.path.join(eval_folder, "_cache")
        utils.ensure_dir(temp_eval_folder)
        log_rank_0_debug(
            logger,
            f"Will split {os.path.join(config.data_config.data_dir, filename)} int {num_dump_file_splits} splits.",
        )
        # Chunk file into splits if desired
        if num_dump_file_splits > 1:
            chunk_prep_dir = os.path.join(temp_eval_folder, "_data_split_in")
            utils.ensure_dir(chunk_prep_dir)
            total_input = sum(
                1 for _ in open(os.path.join(config.data_config.data_dir, filename))
            )
            chunk_input = int(np.ceil(total_input / num_dump_file_splits))
            log_rank_0_debug(
                logger,
                f"Chunking up {total_input} lines into subfiles of size {chunk_input} lines",
            )
            total_input_from_chunks, input_files_dict = utils.chunk_file(
                os.path.join(config.data_config.data_dir, filename),
                chunk_prep_dir,
                chunk_input,
            )
            input_files = list(input_files_dict.keys())
        else:
            input_files = [os.path.join(config.data_config.data_dir, filename)]
        # Before running dump, we need to collect a mapping from sent_idx to prepped dataset indexes. We don't
        # want to reprep that data and have no guarantees as to the order of the prepped data w.r.t these chunks.
        sent_idx2preppedids = dataloaders[0].dataset.get_sentidx_to_rowids()
        # For each split, run dump preds
        output_files = []
        total_mentions_seen = 0
        for input_id, input_filename in enumerate(input_files):
            sentidx2num_mentions, sent_idx2row = eval_utils.get_sent_idx2num_mens(
                input_filename
            )
            log_rank_0_debug(logger, "Done collecting sentence to mention map")
            dataloader = get_dataloaders(
                config,
                tasks,
                use_batch_cands,
                load_entity_data,
                data_splits,
                entity_symbols,
                context_tokenizer,
                dataset_offsets={
                    data_splits[0]: list(
                        itertools.chain(
                            *[
                                sent_idx2preppedids.get(sent_id, [])
                                for sent_id in sentidx2num_mentions
                            ]
                        )
                    )
                },
            )[0]
            input_file_save_folder = os.path.join(
                temp_eval_folder, f"_data_out_{input_id}"
            )
            saved_dump_memmap, save_dump_memmap_config = dump_model_outputs(
                model,
                dataloader,
                config,
                sentidx2num_mentions,
                input_file_save_folder,
                entity_symbols,
                NED_TASK,
                config.run_config.overwrite_eval_dumps,
            )
            log_rank_0_debug(
                logger,
                f"Saving intermediate files to {saved_dump_memmap} and {save_dump_memmap_config}",
            )
            del dataloader
            result_file, mentions_seen = collect_and_merge_results(
                saved_dump_memmap,
                save_dump_memmap_config,
                config,
                sentidx2num_mentions,
                sent_idx2row,
                input_file_save_folder,
                entity_symbols,
            )
            log_rank_0_info(
                logger,
                f"{mentions_seen} mentions seen. Bootleg labels saved at {result_file}",
            )
            # Collect results
            total_mentions_seen += mentions_seen
            assert (
                result_file not in output_files
            ), f"{result_file} already in output_files"
            output_files.append(result_file)

        # Merge results
        final_result_file = eval_utils.get_result_file(eval_folder)
        with open(final_result_file, "wb") as outfile:
            for filename in output_files:
                with open(filename, "rb") as readfile:
                    shutil.copyfileobj(readfile, outfile)
        log_rank_0_info(
            logger,
            f"Saved final bootleg outputs at {final_result_file}. "
            f"Removing cached folder {temp_eval_folder}",
        )
        eval_utils.try_rmtree(temp_eval_folder)
    return final_result_file


if __name__ == "__main__":
    mode, config, run_config_path, entity_emb_file = parse_cmdline_args()
    run_model(mode, config, run_config_path, entity_emb_file)
