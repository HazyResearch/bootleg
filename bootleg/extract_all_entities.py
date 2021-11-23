"""Bootleg run command."""

import argparse
import logging
import os
import subprocess
import sys
from copy import copy

import emmental
import numpy as np
import torch
from emmental.model import EmmentalModel
from rich.logging import RichHandler
from transformers import AutoTokenizer

from bootleg import log_rank_0_info
from bootleg.data import get_entity_dataloaders
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.task_config import NED_TASK
from bootleg.tasks import entity_gen_task
from bootleg.utils import data_utils
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args
from bootleg.utils.utils import (
    dump_yaml_file,
    load_yaml_file,
    recurse_redict,
    write_to_file,
)

logger = logging.getLogger(__name__)


def parse_cmdline_args():
    """
    Parse command line.

    Takes an input config file and parses it into the correct subdictionary
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
    return config, cli_args.config_script


def setup(config, run_config_path=None):
    """
    Set distributed backend and save configuration files.

    Args:
        config: config
        run_config_path: path for original run config
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


def run_model(config, run_config_path=None):
    """
    Run Emmental Bootleg model.

    Args:
        config: parsed model config
        run_config_path: original config path (for saving)
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
    qid2eid = entity_symbols.get_qid2eid_dict()
    eid2qid = {v: k for k, v in qid2eid.items()}
    assert len(qid2eid) == len(eid2qid), "Duplicate EIDs detected"

    # Create tasks
    tasks = [NED_TASK]

    # Create tokenizer
    context_tokenizer = AutoTokenizer.from_pretrained(
        config.data_config.word_embedding.bert_model
    )
    data_utils.add_special_tokens(context_tokenizer)

    # Gets dataloader - will set the split to be TEST even though there is no eval file used to generate entities
    dataloader = get_entity_dataloaders(
        config,
        tasks,
        entity_symbols,
        context_tokenizer,
    )

    # Create models and add tasks
    log_rank_0_info(logger, "Starting Bootleg Model")
    model_name = "Bootleg"
    model = EmmentalModel(name=model_name)

    model.add_task(
        entity_gen_task.create_task(
            config,
            len(context_tokenizer),
        )
    )

    # Load the best model from the pretrained model
    if config["model_config"]["model_path"] is not None:
        model.load(config["model_config"]["model_path"])

    # This happens inside EmmentalLearner for training
    if (
        config["learner_config"]["local_rank"] == -1
        and config["model_config"]["dataparallel"]
    ):
        model._to_dataparallel()

    preds = model.predict(dataloader, return_preds=True, return_action_outputs=False)

    final_out_emb_file = os.path.join(emmental.Meta.log_path, "entity_embeddings.npy")
    log_rank_0_info(logger, f"Saving entity embeddings into {final_out_emb_file}")
    log_rank_0_info(
        logger,
        "Use the entity profile's ```get_eid``` command to get the emb ids for QIDs",
    )

    np.save(final_out_emb_file, np.array(preds["probs"][NED_TASK]))

    return final_out_emb_file


if __name__ == "__main__":
    config, run_config_path = parse_cmdline_args()
    run_model(config, run_config_path)
