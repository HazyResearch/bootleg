"""Bootleg run command."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
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
from bootleg.utils.model_utils import count_parameters
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args
from bootleg.utils.utils import dump_yaml_file, load_yaml_file, recurse_redict, try_rmtree, write_to_file

logger = logging.getLogger(__name__)


def parse_cmdline_args():
    """Takes an input config file and parses it into the correct subdictionary
    groups for the model.

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
        choices=["train", "eval_batch_cands", "eval", "dump_preds", "dump_embs"],
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
    return mode, config, cli_args.config_script


def setup(config, run_config_path=None):
    """
    Setup distributed backend and save configuration files.
    Args:
        config: config
        run_config_path: path for original run config

    Returns:
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
    """Configures the optimizer for Bootleg. By default, we use
    SparseDenseAdam. We always change the parameter group for layer norms
    following standard BERT finetuning methods.

    Args:
        config: config

    Returns:
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
def run_model(mode, config, run_config_path=None):
    """
    Main run method for Emmental Bootleg models.
    Args:
        mode: run mode (train, eval, dump_preds, dump_embs)
        config: parsed model config
        run_config_path: original config path (for saving)

    Returns:

    """

    # Set up distributed backend and save configuration files
    setup(config, run_config_path)

    # Load entity symbols
    log_rank_0_info(logger, "Loading entity symbols...")
    entity_symbols = EntitySymbols.load_from_cache(
        load_dir=os.path.join(
            config.data_config.entity_dir, config.data_config.entity_map_dir
        ),
        alias_cand_map_file=config.data_config.alias_cand_map,
        alias_idx_file=config.data_config.alias_idx_map,
    )
    # Create tasks
    tasks = [NED_TASK]

    # Create splits for data loaders
    data_splits = [TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT]
    # Slices are for eval so we only split on test/dev
    slice_splits = [DEV_SPLIT, TEST_SPLIT]
    # If doing eval, only run on test data
    if mode in ["eval_batch_cands", "eval", "dump_preds", "dump_embs"]:
        data_splits = [TEST_SPLIT]
        slice_splits = [TEST_SPLIT]
        # We only do dumping if weak labels is True
        if mode in ["dump_preds", "dump_embs"]:
            if config.data_config[f"{TEST_SPLIT}_dataset"].use_weak_label is False:
                raise ValueError(
                    "When calling dump_preds or dump_embs, we require use_weak_label to be True."
                )

    # Batch cands is for training and eval_batch_cands
    use_batch_cands = mode in ["train", "eval_batch_cands"]

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
    if mode in ["eval_batch_cands", "eval", "dump_embs", "dump_preds"]:
        # This happens inside EmmentalLearner for training
        if (
            config["learner_config"]["local_rank"] == -1
            and config["model_config"]["dataparallel"]
        ):
            model._to_dataparallel()

    # If just finished training a model or in eval mode, run eval
    if mode in ["train", "eval_batch_cands", "eval"]:
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
        "dump_embs",
    ], 'Mode must be "dump_preds" or "dump_embs"'
    dump_embs = False if mode != "dump_embs" else True
    assert (
        len(dataloaders) == 1
    ), "We should only have length 1 dataloaders for dump_embs and dump_preds!"
    final_result_file, final_out_emb_file = None, None
    if config.learner_config.local_rank in [0, -1]:
        # Setup files/folders
        filename = os.path.basename(dataloaders[0].dataset.raw_filename)
        log_rank_0_debug(
            logger,
            f"Collecting sentence to mention map {os.path.join(config.data_config.data_dir, filename)}",
        )
        sentidx2num_mentions, sent_idx2row = eval_utils.get_sent_idx2num_mens(
            os.path.join(config.data_config.data_dir, filename)
        )
        log_rank_0_debug(logger, "Done collecting sentence to mention map")
        eval_folder = eval_utils.get_eval_folder(filename)
        subeval_folder = os.path.join(eval_folder, "batch_results")
        utils.ensure_dir(subeval_folder)
        # Will keep track of sentences dumped already. These will only be ones with mentions
        all_dumped_sentences = set()
        number_dumped_batches = 0
        total_mentions_seen = 0
        all_result_files = []
        all_out_emb_files = []
        # Iterating over batches of predictions
        for res_i, res_dict in enumerate(
            eval_utils.batched_pred_iter(
                model,
                dataloaders[0],
                config.run_config.eval_accumulation_steps,
                sentidx2num_mentions,
            )
        ):
            (
                result_file,
                out_emb_file,
                final_sent_idxs,
                mentions_seen,
            ) = eval_utils.disambig_dump_preds(
                res_i,
                total_mentions_seen,
                config,
                res_dict,
                sentidx2num_mentions,
                sent_idx2row,
                subeval_folder,
                entity_symbols,
                dump_embs,
                NED_TASK,
            )
            all_dumped_sentences.update(final_sent_idxs)
            all_result_files.append(result_file)
            all_out_emb_files.append(out_emb_file)
            total_mentions_seen += mentions_seen
            number_dumped_batches += 1

        # Dump the sentences that had no mentions and were not already dumped
        # Assert all remaining sentences have no mentions
        assert all(
            v == 0
            for k, v in sentidx2num_mentions.items()
            if k not in all_dumped_sentences
        ), (
            f"Sentences with mentions were not dumped: "
            f"{[k for k, v in sentidx2num_mentions.items() if k not in all_dumped_sentences]}"
        )
        empty_sentidx2row = {
            k: v for k, v in sent_idx2row.items() if k not in all_dumped_sentences
        }
        empty_resultfile = eval_utils.get_result_file(
            number_dumped_batches, subeval_folder
        )
        all_result_files.append(empty_resultfile)
        # Dump the outputs
        eval_utils.write_data_labels_single(
            sentidx2row=empty_sentidx2row,
            output_file=empty_resultfile,
            filt_emb_data=None,
            sental2embid={},
            alias_cand_map=entity_symbols.get_alias2qids(),
            qid2eid=entity_symbols.get_qid2eid(),
            result_alias_offset=total_mentions_seen,
            train_in_cands=config.data_config.train_in_candidates,
            max_cands=entity_symbols.max_candidates,
            dump_embs=dump_embs,
        )

        log_rank_0_info(
            logger, "Finished dumping. Merging results across accumulation steps."
        )
        # Final result files for labels and embeddings
        final_result_file = os.path.join(
            eval_folder, config.run_config.result_label_file
        )
        # Copy labels
        output = open(final_result_file, "wb")
        for file in all_result_files:
            shutil.copyfileobj(open(file, "rb"), output)
        output.close()
        log_rank_0_info(logger, f"Bootleg labels saved at {final_result_file}")
        # Try to copy embeddings
        if dump_embs:
            final_out_emb_file = os.path.join(
                eval_folder, config.run_config.result_emb_file
            )
            log_rank_0_info(
                logger,
                f"Trying to merge numpy embedding arrays. "
                f"If your machine is limited in memory, this may cause OOM errors. "
                f"Is that happens, result files should be saved in {subeval_folder}.",
            )
            all_arrays = []
            for i, npfile in enumerate(all_out_emb_files):
                all_arrays.append(np.load(npfile))
            np.save(final_out_emb_file, np.concatenate(all_arrays))
            log_rank_0_info(logger, f"Bootleg embeddings saved at {final_out_emb_file}")

        # Cleanup
        try_rmtree(subeval_folder)
    return final_result_file, final_out_emb_file


if __name__ == "__main__":
    mode, config, run_config_path = parse_cmdline_args()
    run_model(mode, config, run_config_path)
