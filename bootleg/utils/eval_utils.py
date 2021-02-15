import glob
import logging
import multiprocessing
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ujson
from tqdm import tqdm

import emmental
from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.symbols.constants import PRED_LAYER, UNK_AL
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.task_config import NED_TASK
from bootleg.utils import utils

logger = logging.getLogger(__name__)


def masked_softmax(pred, mask, dim=2, temp=1.0):
    """Masked softmax. Mask of 0/False means mask value (ignore it)

    Args:
        pred: input tensor
        mask: mask
        dim: softmax dimension
        temp: softmax temperature

    Returns: masked softmax tensor
    """
    assert temp > 0, f"You can't have a temperature of 0"
    # https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
    pred = pred / temp
    mask = mask.float()
    masked_pred = pred.masked_fill((1 - mask).byte().bool(), -2e15)
    result = F.softmax(masked_pred, dim=dim)
    return result


def masked_class_logsoftmax(pred, mask, dim=2, temp=1.0):
    """Masked logsoftmax. Mask of 0/False means mask value (ignore it)

    Args:
        pred: input tensor
        mask: mask
        dim: softmax dimension
        temp: softmax temperature

    Returns: masked softmax tensor
    """
    assert temp > 0, f"You can't have a temperature of 0"
    # pred is batch x M x K
    # https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
    pred = pred / temp
    pred = (
        pred + (mask + 5e-16).log()
    )  # we could also do 1e-46 but I feel safer with 5e-16 especially if doing FP16
    # compute softmax over the k dimension
    return F.log_softmax(input=pred, dim=dim)


def map_aliases_to_candidates(train_in_candidates, entity_symbols, aliases):
    """Get list of QID candidates for each alias.

    Args:
        train_in_candidates: whether the model has a NC entity or not (does it assume all gold QIDs are in the candidate lists)
        entity_symbols: entity symbols
        aliases: list of aliases

    Returns: List of lists QIDs
    """
    not_tic = 1 - train_in_candidates
    return [
        not_tic * ["NC"] + entity_symbols.get_qid_cands(al, max_cand_pad=True)
        if al != UNK_AL
        else ["-1"] * (entity_symbols.max_candidates + not_tic)
        for al in aliases
    ]


def map_aliases_to_candidates_eid(train_in_candidates, entity_symbols, aliases):
    """Get list of EID candidates for each alias.

    Args:
        train_in_candidates: whether the model has a NC entity or not (does it assume all gold EIDs are in the candidate lists)
        entity_symbols: entity symbols
        aliases: list of aliases

    Returns: List of lists EIDs
    """
    not_tic = 1 - train_in_candidates
    return [
        not_tic * ["NC"] + entity_symbols.get_eid_cands(al, max_cand_pad=True)
        if al != UNK_AL
        else ["-1"] * (entity_symbols.max_candidates + not_tic)
        for al in aliases
    ]


def select_embs(embs, pred_cands, M):
    """Select the embeddings for the predicted indices.

    Args:
        embs: embedding tensor (M x K x H)
        pred_cands: indices into the K candidates to select (M)
        M: M

    Returns: selected embedding (M x H)
    """
    return embs[torch.arange(M).unsqueeze(0), pred_cands]


def get_eval_folder(file):
    """Return eval folder for the given evaluation file. Stored in
    log_path/filename/model_name.

    Args:
        file: eval file

    Returns: eval folder
    """
    return os.path.join(
        emmental.Meta.log_path,
        os.path.splitext(file)[0],
        os.path.splitext(
            os.path.basename(emmental.Meta.config["model_config"]["model_path"])
        )[0],
    )


def write_disambig_metrics_to_csv(file_path, dictionary):
    """Saves disambiguation metrics in the dictionary to file_path.

    Args:
        file_path: file path
        dictionary: dictionary of scores (output of Emmental score)

    Returns:
    """
    # Only saving NED, ignore Type. dictionary has keys such as "NED/Bootleg/dev/unif_HD/total_men" which corresponds to
    # task/dataset/split/slice/metric, and the value is the associated value for that metric as calculated on the dataset.
    # Sort keys to ensure that the rest of the code below remains in the correct order across slices
    all_keys = [x for x in sorted(dictionary.keys()) if x.startswith(NED_TASK)]

    # This line uses endswith("total_men") because we are just trying to get 1 copy of each task/dataset/split/slice combo. We are not
    # actually using the total_men information in this line below (could've used acc_boot instead, etc.)
    task, dataset, split, slices = list(
        zip(*[x.split("/")[:4] for x in all_keys if x.endswith("total_men")])
    )

    acc_boot = [dictionary[x] for x in all_keys if x.endswith("acc_boot")]
    acc_boot_notNC = [dictionary[x] for x in all_keys if x.endswith("acc_notNC_boot")]
    mentions = [dictionary[x] for x in all_keys if x.endswith("total_men")]
    mentions_notNC = [dictionary[x] for x in all_keys if x.endswith("total_notNC_men")]
    acc_pop = [dictionary[x] for x in all_keys if x.endswith("acc_pop")]
    acc_pop_notNC = [dictionary[x] for x in all_keys if x.endswith("acc_notNC_pop")]

    df_info = {
        "task": task,
        "dataset": dataset,
        "split": split,
        "slice": slices,
        "mentions": mentions,
        "mentions_notNC": mentions_notNC,
        "acc_boot": acc_boot,
        "acc_boot_notNC": acc_boot_notNC,
        "acc_pop": acc_pop,
        "acc_pop_notNC": acc_pop_notNC,
    }
    df = pd.DataFrame(data=df_info)

    df.to_csv(file_path, index=False)


def disambig_dump_preds(
    config, res_dict, file_name, entity_symbols, dump_embs, task_name
):
    """Dumps the predictions of a disambiguation task.

    Args:
        config: model config
        res_dict: result dictionary from Emmental predict
        file_name: eval file name
        entity_symbols: entity symbols
        dump_embs: whether to dump the contextualized embeddings or not
        task_name: task name

    Returns: saved prediction file, saved embedding file (will be None if dump_embs is False)
    """
    num_processes = min(
        config.run_config.dataset_threads, int(multiprocessing.cpu_count() * 0.9)
    )

    # This is dumping
    disambig_res_dict = {}
    for k in res_dict:
        assert task_name in res_dict[k], f"{task_name} not in res_dict for key {k}"
        disambig_res_dict[k] = res_dict[k][task_name]

    file_name = os.path.basename(file_name)
    eval_folder = get_eval_folder(file_name)
    utils.ensure_dir(eval_folder)

    # write to file (M x hidden x size for each data point -- next step will deal with recovering original sentence indices for overflowing sentences)
    unmerged_entity_emb_file = os.path.join(eval_folder, f"entity_embs.pt")
    merged_entity_emb_file = os.path.join(eval_folder, f"entity_embs_unmerged.pt")
    emb_file_config = os.path.splitext(unmerged_entity_emb_file)[0] + "_config.npy"
    M = config.data_config.max_aliases
    K = entity_symbols.max_candidates + (not config.data_config.train_in_candidates)
    if dump_embs:
        unmerged_storage_type = np.dtype(
            [
                ("M", int),
                ("K", int),
                ("hidden_size", int),
                ("sent_idx", int),
                ("subsent_idx", int),
                ("alias_list_pos", int, (M,)),
                ("entity_emb", float, M * config.model_config.hidden_size),
                ("final_loss_true", int, (M,)),
                ("final_loss_pred", int, (M,)),
                ("final_loss_prob", float, (M,)),
                ("final_loss_cand_probs", float, M * K),
            ]
        )
        merged_storage_type = np.dtype(
            [
                ("hidden_size", int),
                ("sent_idx", int),
                ("alias_list_pos", int),
                ("entity_emb", float, config.model_config.hidden_size),
                ("final_loss_pred", int),
                ("final_loss_prob", float),
                ("final_loss_cand_probs", float, K),
            ]
        )
    else:
        # don't need to extract contextualized entity embedding
        unmerged_storage_type = np.dtype(
            [
                ("M", int),
                ("K", int),
                ("hidden_size", int),
                ("sent_idx", int),
                ("subsent_idx", int),
                ("alias_list_pos", int, (M,)),
                ("final_loss_true", int, (M,)),
                ("final_loss_pred", int, (M,)),
                ("final_loss_prob", float, (M,)),
                ("final_loss_cand_probs", float, M * K),
            ]
        )
        merged_storage_type = np.dtype(
            [
                ("hidden_size", int),
                ("sent_idx", int),
                ("alias_list_pos", int),
                ("final_loss_pred", int),
                ("final_loss_prob", float),
                ("final_loss_cand_probs", float, K),
            ]
        )
    mmap_file = np.memmap(
        unmerged_entity_emb_file,
        dtype=unmerged_storage_type,
        mode="w+",
        shape=(len(disambig_res_dict["uids"]),),
    )
    # print("MEMMAP FILE SHAPE", len(disambig_res_dict["uids"]))
    # Init sent_idx to -1 for debugging
    mmap_file[:]["sent_idx"] = -1
    np.save(emb_file_config, unmerged_storage_type, allow_pickle=True)
    log_rank_0_debug(
        logger, f"Created file {unmerged_entity_emb_file} to save predictions."
    )

    log_rank_0_debug(logger, f'{len(disambig_res_dict["uids"])} samples')
    for_iteration = [
        disambig_res_dict["uids"],
        disambig_res_dict["golds"],
        disambig_res_dict["probs"],
        disambig_res_dict["preds"],
    ]

    for i, (uid, gold, probs, model_pred) in tqdm(
        enumerate(zip(*for_iteration)), total=len(disambig_res_dict["uids"])
    ):
        # disambig_res_dict["output"] is dict with keys ['_input__alias_orig_list_pos', 'bootleg_pred_1', '_input__sent_idx',
        # '_input__for_dump_gold_cand_K_idx_train', '_input__subsent_idx', 0, 1]
        sent_idx = disambig_res_dict["outputs"]["_input__sent_idx"][i]
        # print("INSIDE LOOP", sent_idx, "AT", i)
        subsent_idx = disambig_res_dict["outputs"]["_input__subsent_idx"][i]
        alias_orig_list_pos = disambig_res_dict["outputs"][
            "_input__alias_orig_list_pos"
        ][i]
        gold_cand_K_idx_train = disambig_res_dict["outputs"][
            "_input__for_dump_gold_cand_K_idx_train"
        ][i]
        output_embeddings = disambig_res_dict["outputs"][f"{PRED_LAYER}_ent_embs"][i]

        mmap_file[i]["M"] = M
        mmap_file[i]["K"] = K
        mmap_file[i]["hidden_size"] = config.model_config.hidden_size
        mmap_file[i]["sent_idx"] = sent_idx
        mmap_file[i]["subsent_idx"] = subsent_idx
        mmap_file[i]["alias_list_pos"] = alias_orig_list_pos
        # This will give all aliases seen by the model during training, independent of if it's gold or not
        mmap_file[i][f"final_loss_true"] = gold_cand_K_idx_train.reshape(M)

        # get max for each alias, probs is M x K
        max_probs = probs.max(axis=1)
        pred_cands = probs.argmax(axis=1)

        mmap_file[i]["final_loss_pred"] = pred_cands
        mmap_file[i]["final_loss_prob"] = max_probs
        mmap_file[i]["final_loss_cand_probs"] = probs.reshape(1, -1)
        # final_entity_embs is M x K x hidden_size, pred_cands is M
        if dump_embs:
            chosen_entity_embs = select_embs(
                embs=output_embeddings, pred_cands=pred_cands, M=M
            )

            # write chosen entity embs to file for contextualized entity embeddings
            mmap_file[i]["entity_emb"] = chosen_entity_embs.reshape(1, -1)

    # for i in range(len(mmap_file)):
    #     si = mmap_file[i]["sent_idx"]
    #     if -1 == si:
    #         import ipdb
    #         ipdb.set_trace()
    #     assert si != -1, f"{i} {mmap_file[i]}"

    result_file = os.path.join(eval_folder, config.run_config.result_label_file)
    log_rank_0_debug(logger, f"Writing predictions to {result_file}...")
    merge_subsentences(
        num_processes,
        os.path.join(config.data_config.data_dir, file_name),
        merged_entity_emb_file,
        merged_storage_type,
        unmerged_entity_emb_file,
        unmerged_storage_type,
        dump_embs=dump_embs,
    )

    write_data_labels(
        num_processes=num_processes,
        merged_entity_emb_file=merged_entity_emb_file,
        merged_storage_type=merged_storage_type,
        data_file=os.path.join(config.data_config.data_dir, file_name),
        out_file=result_file,
        data_config=config.data_config,
        train_in_candidates=config.data_config.train_in_candidates,
        dump_embs=dump_embs,
    )

    out_emb_file = None
    # save easier-to-use embedding file
    if dump_embs:
        filt_emb_data = np.memmap(
            merged_entity_emb_file, dtype=merged_storage_type, mode="r+"
        )
        hidden_size = filt_emb_data[0]["hidden_size"]
        out_emb_file = os.path.join(eval_folder, config.run_config.result_emb_file)
        np.save(out_emb_file, filt_emb_data["entity_emb"].reshape(-1, hidden_size))
        log_rank_0_info(
            logger, f"Saving contextual entity embeddings to {out_emb_file}"
        )
    log_rank_0_info(logger, f"Wrote predictions to {result_file}")
    return result_file, out_emb_file


def get_sent_start_map(data_file):
    """Gets the map from sentence index to number of mentions so we know the
    offset of each sentence.

    Args:
        data_file: eval file

    Returns: Dict of sentence index -> number of mention per sentence
    """
    sent_start_map = {}
    total_num_mentions = 0
    with open(data_file) as f:
        for line in f:
            line = ujson.loads(line)
            # keep track of the start idx in the condensed memory mapped file for each sentence (varying number of aliases)
            assert (
                line["sent_idx_unq"] not in sent_start_map
            ), f'Sentence indices must be unique. {line["sent_idx_unq"]} already seen.'
            # Save as string for Marisa Tri later
            sent_start_map[str(line["sent_idx_unq"])] = total_num_mentions
            # We include false aliases for debugging (and alias_pos includes them)
            total_num_mentions += len(line["aliases"])
            # print("INSIDE SENT MAP", str(line["sent_idx_unq"]), total_num_mentions)

    log_rank_0_debug(
        logger, f"Total number of mentions across all sentences: {total_num_mentions}"
    )
    return sent_start_map, total_num_mentions


#
def merge_subsentences(
    num_processes,
    data_file,
    to_save_file,
    to_save_storage,
    to_read_file,
    to_read_storage,
    dump_embs=False,
):
    """Flatten all sentences back together over sub-sentences; removing the PAD
    aliases from the data I.e., converts from sent_idx -> array of values to
    (sent_idx, alias_idx) -> value with varying numbers of aliases per
    sentence.

    Args:
        num_processes: number of processes
        data_file: eval file
        to_save_file: memmap file to save results to
        to_save_storage: save file storage type
        to_read_file: memmap file to read predictions from
        to_read_storage: read file storage type
        dump_embs: whether to dump embeddings or not

    Returns:
    """
    log_rank_0_debug(logger, f"Getting sentence mapping")
    sent_start_map, total_num_mentions = get_sent_start_map(data_file)
    sent_start_map_file = tempfile.NamedTemporaryFile(suffix="bootleg_sent_start_map")
    utils.create_single_item_trie(sent_start_map, out_file=sent_start_map_file.name)
    log_rank_0_debug(logger, f"Done with sentence mapping")

    full_pred_data = np.memmap(to_read_file, dtype=to_read_storage, mode="r")
    M = int(full_pred_data[0]["M"])
    K = int(full_pred_data[0]["K"])
    hidden_size = int(full_pred_data[0]["hidden_size"])
    # print("TOTAL MENS", total_num_mentions)
    filt_emb_data = np.memmap(
        to_save_file, dtype=to_save_storage, mode="w+", shape=(total_num_mentions,)
    )
    filt_emb_data["hidden_size"] = hidden_size
    filt_emb_data["sent_idx"][:] = -1
    filt_emb_data["alias_list_pos"][:] = -1

    chunk_size = int(np.ceil(len(full_pred_data) / num_processes))
    all_ids = list(range(0, len(full_pred_data)))
    row_idx_set_chunks = [
        all_ids[ids : ids + chunk_size]
        for ids in range(0, len(full_pred_data), chunk_size)
    ]
    input_args = [[M, K, hidden_size, dump_embs, chunk] for chunk in row_idx_set_chunks]
    log_rank_0_info(
        logger, f"Merging sentences together with {num_processes} processes"
    )
    pool = multiprocessing.Pool(
        processes=num_processes,
        initializer=merge_subsentences_initializer,
        initargs=[
            to_save_file,
            to_save_storage,
            to_read_file,
            to_read_storage,
            sent_start_map_file.name,
        ],
    )

    start = time.time()
    seen_ids = set()
    for sent_ids_seen in pool.imap_unordered(
        merge_subsentences_hlp, input_args, chunksize=1
    ):
        for emb_id in sent_ids_seen:
            assert (
                emb_id not in seen_ids
            ), f"{emb_id} already seen, something went wrong with sub-sentences"
            seen_ids.add(emb_id)

    filt_emb_data = np.memmap(to_save_file, dtype=to_save_storage, mode="r")
    # for i in range(len(filt_emb_data)):
    #     si = filt_emb_data[i]["sent_idx"]
    #     al_test = filt_emb_data[i]["alias_list_pos"]
    #     if si == -1 or al_test == -1:
    #         print("BAD", i, filt_emb_data[i])

    # Clean up
    sent_start_map_file.close()
    logging.debug(f"Time to merge sub-sentences {time.time() - start}s")
    return


def merge_subsentences_initializer(
    to_write_file, to_write_storage, to_read_file, to_read_storage, sent_start_map_file
):
    global filt_emb_data_global
    filt_emb_data_global = np.memmap(to_write_file, dtype=to_write_storage, mode="r+")
    global full_pred_data_global
    full_pred_data_global = np.memmap(to_read_file, dtype=to_read_storage, mode="r+")
    global sent_start_map_marisa_global
    sent_start_map_marisa_global = utils.load_single_item_trie(sent_start_map_file)


def merge_subsentences_hlp(args):
    """Helper for merge_sentences."""
    M, K, hidden_size, dump_embs, r_idx_set = args
    seen_ids = set()
    for r_idx in r_idx_set:
        row = full_pred_data_global[r_idx]
        # get corresponding row to start writing into condensed memory mapped file
        sent_idx = str(row["sent_idx"])
        sent_start_idx = sent_start_map_marisa_global[sent_idx][0][0]
        # print("R IDS", r_idx, row["sent_idx"], "START", sent_start_idx)
        # for each VALID mention, need to write into original alias list pos in list
        for i, (true_val, alias_orig_pos) in enumerate(
            zip(row["final_loss_true"], row["alias_list_pos"])
        ):
            # print(
            #     "INSIDE MERGE", i, true_val, alias_orig_pos, true_val + alias_orig_pos
            # )
            # bc we are are using the mentions which includes both true and false golds, true_val == -1 only for padded mentions or sub-sentence mentions
            if true_val != -1:
                # id in condensed embedding
                emb_id = sent_start_idx + alias_orig_pos
                assert (
                    emb_id not in seen_ids
                ), f"{emb_id} already seen, something went wrong with sub-sentences"
                if dump_embs:
                    filt_emb_data_global["entity_emb"][emb_id] = row[
                        "entity_emb"
                    ].reshape(M, hidden_size)[i]
                filt_emb_data_global["sent_idx"][emb_id] = sent_idx
                filt_emb_data_global["alias_list_pos"][emb_id] = alias_orig_pos
                filt_emb_data_global["final_loss_pred"][emb_id] = row[
                    "final_loss_pred"
                ].reshape(M)[i]
                filt_emb_data_global["final_loss_prob"][emb_id] = row[
                    "final_loss_prob"
                ].reshape(M)[i]
                filt_emb_data_global["final_loss_cand_probs"][emb_id] = row[
                    "final_loss_cand_probs"
                ].reshape(M, K)[i]
    return seen_ids


def get_sent_idx_map(merged_entity_emb_file, merged_storage_type):
    """Get sent_idx, alias_idx mapping to emb idx for quick lookup.

    Args:
        merged_entity_emb_file: memmap file after merge sentences
        merged_storage_type: file storage type

    Returns: Dict of f"{sent_idx}_{alias_idx}" -> index in merged_entity_emb_file
    """
    filt_emb_data = np.memmap(
        merged_entity_emb_file, dtype=merged_storage_type, mode="r+"
    )
    sent_idx_map = {}
    for i, row in tqdm(
        enumerate(filt_emb_data),
        desc="Building sent_idx, alias_list_pos mapping",
        total=len(filt_emb_data),
    ):
        sent_idx = row["sent_idx"]
        alias_idx = row["alias_list_pos"]
        assert sent_idx != -1 and alias_idx != -1, f"Sent {sent_idx}, Al {alias_idx}"
        # Keep as string for Marisa Tri later
        sent_idx_map[f"{sent_idx}_{alias_idx}"] = i
    return sent_idx_map


def write_data_labels(
    num_processes,
    merged_entity_emb_file,
    merged_storage_type,
    data_file,
    out_file,
    train_in_candidates,
    dump_embs,
    data_config,
):
    """Takes the flattened data from merge_sentences and writes out predictions
    to a file, one line per sentence.

    The embedding ids are added to the file if dump_embs is True.

    Args:
        num_processes: number of processes
        merged_entity_emb_file: input memmap file after merge sentences
        merged_storage_type: input file storage type
        data_file: eval file
        out_file: final output file for predictions
        train_in_candidates: whether NC entities are not in candidate lists
        dump_embs: whether to dump embeddings or not
        data_config: data config

    Returns:
    """
    st = time.time()
    sent_idx_map = get_sent_idx_map(merged_entity_emb_file, merged_storage_type)
    sent_idx_map_file = tempfile.NamedTemporaryFile(suffix="bootleg_sent_idx_map")
    utils.create_single_item_trie(sent_idx_map, out_file=sent_idx_map_file.name)
    log_rank_0_debug(logger, f"Finished getting sentence map {time.time() - st}s")

    # Chunk file for parallel writing
    # We do not use TemporaryFolders as the temp dir may not have enough space for large files
    create_ex_indir = os.path.join(
        os.path.dirname(data_file), "_bootleg_eval_temp_indir"
    )
    utils.ensure_dir(create_ex_indir)
    create_ex_outdir = os.path.join(
        os.path.dirname(data_file), "_bootleg_eval_temp_outdir"
    )
    utils.ensure_dir(create_ex_outdir)
    logger.debug(f"Counting lines")
    total_input = sum(1 for _ in open(data_file))
    chunk_input = int(np.ceil(total_input / num_processes))
    logger.debug(
        f"Chunking up {total_input} lines into subfiles of size {chunk_input} lines"
    )
    total_input_from_chunks, input_files_dict = utils.chunk_file(
        data_file, create_ex_indir, chunk_input
    )

    # Generation input/output pairs
    input_files = list(input_files_dict.keys())
    input_file_lines = [input_files_dict[k] for k in input_files]
    output_files = [
        in_file_name.replace(create_ex_indir, create_ex_outdir)
        for in_file_name in input_files
    ]
    assert (
        total_input == total_input_from_chunks
    ), f"Lengths of files {total_input} doesn't mathc {total_input_from_chunks}"
    log_rank_0_debug(logger, f"Done chunking files. Starting pool")

    pool = multiprocessing.Pool(
        processes=num_processes,
        initializer=write_data_labels_initializer,
        initargs=[
            merged_entity_emb_file,
            merged_storage_type,
            sent_idx_map_file.name,
            train_in_candidates,
            dump_embs,
            data_config,
        ],
    )

    input_args = list(zip(input_files, input_file_lines, output_files))

    total = 0
    for res in pool.imap(write_data_labels_hlp, input_args, chunksize=1):
        total += 1

    # Merge output files to final file
    log_rank_0_info(logger, f"Merging output files")
    with open(out_file, "wb") as outfile:
        for filename in glob.glob(os.path.join(create_ex_outdir, "*")):
            if filename == out_file:
                # don't want to copy the output into the output
                continue
            with open(filename, "rb") as readfile:
                shutil.copyfileobj(readfile, outfile)
    # Remove temporary files/folders
    sent_idx_map_file.close()
    shutil.rmtree(create_ex_indir)
    shutil.rmtree(create_ex_outdir)


def write_data_labels_initializer(
    merged_entity_emb_file,
    merged_storage_type,
    sent_idx_map_file,
    train_in_candidates,
    dump_embs,
    data_config,
):
    global filt_emb_data_global
    filt_emb_data_global = np.memmap(
        merged_entity_emb_file, dtype=merged_storage_type, mode="r+"
    )
    global sent_idx_map_global
    sent_idx_map_global = utils.load_single_item_trie(sent_idx_map_file)
    global train_in_candidates_global
    train_in_candidates_global = train_in_candidates
    global dump_embs_global
    dump_embs_global = dump_embs
    global entity_dump_global
    entity_dump_global = EntitySymbols(
        load_dir=os.path.join(data_config.entity_dir, data_config.entity_map_dir),
        alias_cand_map_file=data_config.alias_cand_map,
    )


def write_data_labels_hlp(args):
    """Write data labels helper."""
    input_file, input_lines, output_file = args
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for line in tqdm(f_in, total=input_lines, desc="Writing data"):
            line = ujson.loads(line)
            aliases = line["aliases"]
            sent_idx = line["sent_idx_unq"]
            qids = []
            ctx_emb_ids = []
            entity_ids = []
            probs = []
            cands = []
            cand_probs = []
            entity_cands_qid = map_aliases_to_candidates(
                train_in_candidates_global, entity_dump_global, aliases
            )
            # eid is entity id
            entity_cands_eid = map_aliases_to_candidates_eid(
                train_in_candidates_global, entity_dump_global, aliases
            )
            for al_idx, alias in enumerate(aliases):
                sent_idx_key = f"{sent_idx}_{al_idx}"
                assert (
                    sent_idx_key in sent_idx_map_global
                ), f"Dumped prediction data does not match data file. Can not find {sent_idx} - {al_idx}"
                emb_idx = sent_idx_map_global[sent_idx_key][0][0]
                ctx_emb_ids.append(emb_idx)
                prob = filt_emb_data_global[emb_idx]["final_loss_prob"]
                cand_prob = filt_emb_data_global[emb_idx]["final_loss_cand_probs"]
                pred_cand = filt_emb_data_global[emb_idx]["final_loss_pred"]
                eid = entity_cands_eid[al_idx][pred_cand]
                qid = entity_cands_qid[al_idx][pred_cand]
                qids.append(qid)
                probs.append(prob)
                cands.append(list(entity_cands_qid[al_idx]))
                cand_probs.append(list(cand_prob))
                entity_ids.append(eid)
            line["qids"] = qids
            line["probs"] = probs
            line["cands"] = cands
            line["cand_probs"] = cand_probs
            line["entity_ids"] = entity_ids
            if dump_embs_global:
                line["ctx_emb_ids"] = ctx_emb_ids
            f_out.write(ujson.dumps(line) + "\n")
