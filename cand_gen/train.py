"""Bootleg run command."""

import argparse
import logging
import os
import subprocess
import sys
from copy import copy

import emmental
import torch
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from rich.logging import RichHandler
from transformers import AutoTokenizer

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.data import get_slicedatasets
from bootleg.symbols.constants import DEV_SPLIT, TEST_SPLIT, TRAIN_SPLIT
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils
from bootleg.utils.model_utils import count_parameters
from bootleg.utils.utils import (
    dump_yaml_file,
    load_yaml_file,
    recurse_redict,
    write_to_file,
)
from cand_gen.data import get_dataloaders
from cand_gen.task_config import CANDGEN_TASK
from cand_gen.tasks import candgen_task
from cand_gen.utils.parser.parser_utils import parse_boot_and_emm_args

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

    cli_parser.add_argument("--local_rank", type=int, default=-1)

    # you can add other args that will override those in the config_script
    # parse_known_args returns 'args' that are the same as what parse_args() returns
    # and 'unknown' which are args that the parser doesn't recognize but you want to keep.
    # 'unknown' are what we pass on to our override any args from the second phase of arg parsing from the json file
    cli_args, unknown = cli_parser.parse_known_args()
    if len(cli_args.config_script) == 0:
        raise ValueError("You must pass a config script via --config.")
    config = parse_boot_and_emm_args(cli_args.config_script, unknown)

    #  Modify the local rank param from the cli args
    config.learner_config.local_rank = int(os.getenv("LOCAL_RANK", cli_args.local_rank))
    return config, cli_args.config_script


def setup(config, run_config_path=None):
    """
    Setup distributed backend and save configuration files.
    Args:
        config: config
        run_config_path: path for original run config

    Returns:
    """
    if "mp_sharing_strategy" in config.data_config:
        torch.multiprocessing.set_sharing_strategy(config["mp_sharing_strategy"])
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
def run_model(config, run_config_path=None):
    """
    Main run method for Emmental Bootleg models.
    Args:
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
        alias_cand_map_dir=config.data_config.alias_cand_map,
        alias_idx_dir=config.data_config.alias_idx_map,
    )
    # Create tasks
    tasks = [CANDGEN_TASK]

    # Create splits for data loaders
    data_splits = [TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT]
    # Slices are for eval so we only split on test/dev
    slice_splits = [DEV_SPLIT, TEST_SPLIT]

    # Create tokenizer
    context_tokenizer = AutoTokenizer.from_pretrained(
        config.data_config.word_embedding.bert_model
    )
    data_utils.add_special_tokens(context_tokenizer)

    # Gets dataloaders
    dataloaders = get_dataloaders(
        config,
        tasks,
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
        candgen_task.create_task(
            config,
            len(context_tokenizer),
            slice_datasets,
        )
    )
    # Print param counts
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
    emmental_learner = EmmentalLearner()
    emmental_learner._set_optimizer(model)
    # Save first checkpoint
    if config.learner_config.local_rank in [0, -1]:
        model.save(f"{emmental.Meta.log_path}/checkpoint_0.0.model.pth")
    emmental_learner.learn(model, dataloaders)
    if config.learner_config.local_rank in [0, -1]:
        model.save(f"{emmental.Meta.log_path}/last_model.pth")

    # If just finished training a model
    if config.learner_config.local_rank in [0, -1]:
        scores = model.score(dataloaders[1:])
        # Save metrics and models
        log_rank_0_info(logger, f"Saving metrics to {emmental.Meta.log_path}")
        log_rank_0_info(logger, f"Metrics: {scores}")
        scores["log_path"] = emmental.Meta.log_path
        write_to_file(f"{emmental.Meta.log_path}/train_metrics.txt", scores)
    else:
        scores = {}
    return scores


if __name__ == "__main__":
    config, run_config_path = parse_cmdline_args()
    run_model(config, run_config_path)
