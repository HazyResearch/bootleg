"""Bootleg run command."""

import argparse
import logging
import os
import subprocess
import sys
from collections import defaultdict
from copy import copy
from pathlib import Path

import emmental
import faiss
import numpy as np
import torch
import ujson
from emmental.model import EmmentalModel
from rich.logging import RichHandler
from rich.progress import track
from transformers import AutoTokenizer

from bootleg import log_rank_0_info
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils
from bootleg.utils.utils import (
    dump_yaml_file,
    load_yaml_file,
    recurse_redict,
    write_to_file,
)
from cand_gen.data import get_context_dataloader, get_entity_dataloader
from cand_gen.task_config import CANDGEN_TASK
from cand_gen.tasks import context_gen_task, entity_gen_task
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
    cli_parser.add_argument(
        "--entity_embs_only",
        action="store_true",
        help="If true will only generate a static embedding file for all entity embeddings. Will not run with context"
        "to generate candidates",
    )
    cli_parser.add_argument(
        "--entity_embs_path",
        type=str,
        default=None,
        help="If already dumped entity embeddings, can provide path here",
    )
    cli_parser.add_argument(
        "--topk",
        default=10,
        type=int,
        help="TopK entities to retrieve. Use spaces to deliminate multiple topks",
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
    return (
        config,
        cli_args.config_script,
        cli_args.entity_embs_only,
        cli_args.entity_embs_path,
        cli_args.topk,
    )


def setup(config, run_config_path=None):
    """
    Setup distributed backend and save configuration files.
    Args:
        config: config
        run_config_path: path for original run config

    Returns:
    """
    if config.data_config["mp_sharing_strategy"]:
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


def gen_entity_embeddings(config, context_tokenizer, entity_symbols):
    # Create tasks
    tasks = [CANDGEN_TASK]
    # Gets dataloader - will set the split to be TEST even though there is no eval file used to generate entities
    dataloader = get_entity_dataloader(
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
    return preds, dataloader, model


def gen_context_embeddings(
    config, context_tokenizer, entity_symbols, dataset_range=None, model=None
):
    # Create tasks
    tasks = [CANDGEN_TASK]
    # Gets dataloader - will set the split to be TEST even though there is no eval file used to generate entities
    dataloader = get_context_dataloader(
        config, tasks, entity_symbols, context_tokenizer, dataset_range
    )
    # Create models and add tasks
    if model is None:
        log_rank_0_info(logger, "Starting Bootleg Model")
        model_name = "Bootleg"
        model = EmmentalModel(name=model_name)
        model.add_task(
            context_gen_task.create_task(
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
    return preds, dataloader, model


def run_model(
    config, run_config_path=None, entity_embs_only=False, entity_embs_path=None, topk=30
):
    """
    Main run method for Emmental Bootleg model.
    Args:
        config: parsed model config
        run_config_path: original config path (for saving)
        entity_embs_only: whether to just generate entity embeddings

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
    qid2eid = entity_symbols.get_qid2eid_dict()
    eid2qid = {v: k for k, v in qid2eid.items()}
    assert len(qid2eid) == len(eid2qid), "Duplicate EIDs detected"

    # GENERATE ENTITY EMBEDDINGS
    # Create tokenizer
    context_tokenizer = AutoTokenizer.from_pretrained(
        config.data_config.word_embedding.bert_model
    )
    data_utils.add_special_tokens(context_tokenizer)

    out_emb_file = entity_embs_path
    if entity_embs_path is None:
        log_rank_0_info(
            logger, "Gathering embeddings for all entities. Will save for reuse."
        )
        preds, _, _ = gen_entity_embeddings(config, context_tokenizer, entity_symbols)

        final_out_emb_file = os.path.join(
            emmental.Meta.log_path, "entity_embeddings.npy"
        )
        log_rank_0_info(logger, f"Saving entity embeddings into {final_out_emb_file}")
        log_rank_0_info(
            logger,
            "Use the entity profile's ```get_eid``` command to get the emb ids for QIDs",
        )
        np.save(final_out_emb_file, np.array(preds["probs"][CANDGEN_TASK]))
        out_emb_file = final_out_emb_file
        del preds
    else:
        assert Path(entity_embs_path).exists(), f"{entity_embs_path} must exist"

    if entity_embs_only:
        return out_emb_file

    log_rank_0_info(logger, "Loading embeddings for cand gen.")
    entity_embs = np.load(out_emb_file)

    log_rank_0_info(logger, "Building index...")
    if torch.cuda.device_count() > 0 and config["model_config"]["device"] >= 0:
        if config["model_config"]["dataparallel"]:
            faiss_cpu_index = faiss.IndexFlatIP(entity_embs.shape[-1])
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_cpu_index)
        else:
            faiss_cpu_index = faiss.IndexFlatIP(entity_embs.shape[-1])
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_cpu_index)
    else:
        faiss_index = faiss.IndexFlatIP(entity_embs.shape[-1])

    faiss_index.add(entity_embs)

    log_rank_0_info(logger, "Searching...")
    recall_k = [1, 2, 5, 10, 20, 30, 40, 50]
    total_cnt = 0
    cnt_k = {i: 0 for i in recall_k}

    # Make sure data is prepped
    context_dataloader = get_context_dataloader(
        config,
        [CANDGEN_TASK],
        entity_symbols,
        context_tokenizer,
    )
    total_samples = len(context_dataloader.dataset)
    topk_candidates = {}
    context_model = None
    nn_chunk = config["run_config"]["dump_preds_accumulation_steps"]
    for i in range(int(np.ceil(total_samples / nn_chunk))):
        st = i * nn_chunk
        ed = min((i + 1) * nn_chunk, total_samples)
        context_preds, context_dataloader, context_model = gen_context_embeddings(
            config,
            context_tokenizer,
            entity_symbols,
            dataset_range=list(range(st, ed)),
            model=context_model,
        )
        res = {
            "context_ids": context_preds["uids"][CANDGEN_TASK],
            "context_features": np.array(context_preds["probs"][CANDGEN_TASK]),
        }
        # import pdb; pdb.set_trace()
        # +1 as index will return
        D, Is = faiss_index.search(res["context_features"], topk)
        for j in range(Is.shape[0]):
            # No need to offset by st+j as the range offset is accounted for in dataset
            example = context_dataloader.dataset[j]
            sent_id = int(example["sent_idx"])
            alias_id = int(example["alias_orig_list_pos"])
            gt_eid = int(example["for_dump_gold_eid"])
            gt = eid2qid.get(gt_eid, "Q-1")
            topk_nn = [eid2qid.get(k, "Q-1") for k in Is[j]]
            assert tuple([sent_id, alias_id]) not in topk_candidates
            topk_candidates[tuple([sent_id, alias_id])] = [
                sent_id,
                alias_id,
                gt,
                topk_nn[:topk],
                D[j].tolist()[:topk],
            ]

            total_cnt += 1
            try:
                idx = topk_nn.index(gt)
                for ll in recall_k:
                    if idx < ll:
                        cnt_k[ll] += 1
            except ValueError:
                pass

    assert len(topk_candidates) == total_samples, "Missing samples"
    for k in recall_k:
        cnt_k[k] /= total_cnt
    print(cnt_k, total_cnt)

    # Get test dataset filename
    file_name = Path(config.data_config.test_dataset.file).stem
    metrics_file = Path(emmental.Meta.log_path) / f"{file_name}_candgen_metrics.txt"
    write_to_file(
        metrics_file,
        cnt_k,
    )

    sent2output = defaultdict(list)
    for (sent_id, alias_id), v in track(
        topk_candidates.items(), description="Grouping by sentence"
    ):
        sent2output[sent_id].append(v)

    sent2output = dict(sent2output)
    for sent_id, v in track(sent2output.items(), description="Sorting sentences"):
        v = sorted(v, key=lambda x: x[1])
        sent2output[sent_id] = v

    candidates_file = (
        Path(emmental.Meta.log_path) / f"{file_name}_{topk}_candidates.jsonl"
    )
    log_rank_0_info(logger, f"Saving to {candidates_file}")
    with open(candidates_file, "w", encoding="utf-8") as f:
        for sent_id, list_of_values in sent2output.items():
            sent_ids, alias_ids, gts, cands, probs = list(zip(*list_of_values))
            json_obj = {
                "sent_idx_unq": sent_id,
                "alias_idxs": list(alias_ids),
                "qids": list(gts),
                "cands": list(cands),
                "probs": list(probs),
            }
            f.write(ujson.dumps(json_obj) + "\n")

    return candidates_file, metrics_file


if __name__ == "__main__":
    (
        config,
        run_config_path,
        entity_embs_only,
        entity_embs_path,
        topk,
    ) = parse_cmdline_args()
    run_model(config, run_config_path, entity_embs_only, entity_embs_path, topk)
