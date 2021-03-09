import glob
import logging
import multiprocessing
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import ujson
from tqdm import tqdm

import emmental
from bootleg import log_rank_0_debug
from bootleg.symbols.constants import PRED_LAYER, UNK_AL
from bootleg.task_config import NED_TASK
from bootleg.utils import data_utils, utils
from bootleg.utils.classes.aliasmention_trie import AliasCandRecordTrie
from emmental.utils.utils import array_to_numpy, prob_to_pred

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


def map_aliases_to_candidates(
    train_in_candidates, max_candidates, alias_cand_map, aliases
):
    """Get list of QID candidates for each alias.

    Args:
        train_in_candidates: whether the model has a NC entity or not (does it assume all gold QIDs are in the candidate lists)
        alias_cand_map: alias -> candidate qids in dict or AliasCandRecordTrie format
        aliases: list of aliases

    Returns: List of lists QIDs
    """
    not_tic = 1 - train_in_candidates
    res = []
    for al in aliases:
        if al == UNK_AL:
            res.append(["-1"] * (max_candidates + not_tic))
        else:
            if isinstance(alias_cand_map, dict):
                cands = [qid_pair[0] for qid_pair in alias_cand_map[al]]
            else:
                cands = alias_cand_map.get_value(al, getter=lambda x: x[0])
            cands = cands + ["-1"] * (max_candidates - len(cands))
            res.append(not_tic * ["NC"] + cands)
    return res


def map_candidate_qids_to_eid(candidate_qids, qid2eid):
    """Get list of EID candidates for each alias.

    Args:
        candidate_qids: list of list of candidate QIDs
        qid2eid: mapping of qid to entity id

    Returns: List of lists EIDs
    """
    if isinstance(qid2eid, dict):
        return [
            [qid2eid[q] if q not in {"-1", "NC"} else q for q in cand_list]
            for cand_list in candidate_qids
        ]
    else:
        # qid2eid is trie so must use trie getters
        return [
            [qid2eid[q][0][0] if q not in {"-1", "NC"} else q for q in cand_list]
            for cand_list in candidate_qids
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


def get_sent_idx2num_mentions(data_file):
    """Gets the map from sentence index to number of mentions and to data. Used
    for calculating offsets and chunking file.

    Args:
        data_file: eval file

    Returns: Dict of sentence index -> number of mention per sentence, Dict of sentence index -> input line
    """
    sent_idx2num_mentions = {}
    sent_idx2row = {}
    total_num_mentions = 0
    with open(data_file) as f:
        for line in f:
            line = ujson.loads(line)
            # keep track of the start idx in the condensed memory mapped file for each sentence (varying number of aliases)
            assert (
                line["sent_idx_unq"] not in sent_idx2num_mentions
            ), f'Sentence indices must be unique. {line["sent_idx_unq"]} already seen.'
            sent_idx2row[str(line["sent_idx_unq"])] = line
            # Save as string for Marisa Tri later
            sent_idx2num_mentions[str(line["sent_idx_unq"])] = len(line["aliases"])
            # We include false aliases for debugging (and alias_pos includes them)
            total_num_mentions += len(line["aliases"])
            # print("INSIDE SENT MAP", str(line["sent_idx_unq"]), total_num_mentions)

    log_rank_0_debug(
        logger, f"Total number of mentions across all sentences: {total_num_mentions}"
    )
    return sent_idx2num_mentions, sent_idx2row


# Modified from
# https://github.com/SenWu/emmental/blob/master/src/emmental/model.py#L455
# to support eval_accumulation_steps
@torch.no_grad()
def batched_pred_iter(
    model,
    dataloader,
    eval_accumulation_steps,
    sent_idx2num_mentions,
):
    """Predict from dataloader taking into account eval accumulation steps.
    Will yield a new prediction set after each set accumulation steps for
    writing out.

    If a sentence or batch doesn't have any mentions, it will not be returned by this method.

    Recall that we split up sentences that are too long to feed to the model.
    We use the sent_idx2num_mentions dict to ensure we have full sentences evaluated before
    returning, otherwise we'll have incomplete sentences to merge together when dumping.

    Args:
      model: model
      dataloader: The dataloader to predict
      eval_accumulation_steps: Number of eval steps to run before returning
      sent_idx2num_mentions: list of sent index to number of mentions

    Returns:
      Iterator over result dict.
    """

    def collect_result(uid_d, gold_d, pred_d, prob_d, out_d, cur_sentidx_nummen):
        """Merges results for the sentences where all mentions have been
        evaluated."""
        final_uid_d = defaultdict(list)
        final_prob_d = defaultdict(list)
        final_pred_d = defaultdict(list)
        final_gold_d = defaultdict(list)
        final_out_d = defaultdict(lambda: defaultdict(list))
        sentidxs_finalized = []
        # print("FINALIZE", cur_sentidx_nummen, sent_idx2num_mentions)
        for sent_idx, cur_mention_set in cur_sentidx_nummen.items():
            assert (
                len(cur_mention_set) <= sent_idx2num_mentions[str(sent_idx)]
            ), f"Too many mentions for {sent_idx}: {cur_mention_set} VS {sent_idx2num_mentions[str(sent_idx)]}"
            if len(cur_mention_set) == sent_idx2num_mentions[str(sent_idx)]:
                sentidxs_finalized.append(sent_idx)
                for task_name in uid_d:
                    final_uid_d[task_name].extend(uid_d[task_name][sent_idx])
                    final_prob_d[task_name].extend(prob_d[task_name][sent_idx])
                    final_pred_d[task_name].extend(pred_d[task_name][sent_idx])
                    final_gold_d[task_name].extend(gold_d[task_name][sent_idx])
                    if task_name in out_d.keys():
                        for action_name in out_d[task_name].keys():
                            final_out_d[task_name][action_name].extend(
                                out_d[task_name][action_name][sent_idx]
                            )
        # If batch size is close to 1 and accumulation step was close to 1, we may get to where there are no complete sentences
        if len(sentidxs_finalized) == 0:
            return {}, sentidxs_finalized
        res = {
            "uids": final_uid_d,
            "golds": final_gold_d,
        }
        for task_name in final_prob_d.keys():
            final_prob_d[task_name] = array_to_numpy(final_prob_d[task_name])
        res["probs"] = final_prob_d
        for task_name in final_pred_d.keys():
            final_pred_d[task_name] = array_to_numpy(final_pred_d[task_name])
        res["preds"] = final_pred_d
        res["outputs"] = final_out_d
        return res, sentidxs_finalized

    model.eval()

    # Will store sent_idx -> task_name -> list output
    uid_dict = defaultdict(lambda: defaultdict(list))
    prob_dict = defaultdict(lambda: defaultdict(list))
    pred_dict = defaultdict(lambda: defaultdict(list))
    gold_dict = defaultdict(lambda: defaultdict(list))
    # Will store sent_idx -> task_name -> output key -> list output
    out_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # list of all finalized and yielded sentences
    all_finalized_sentences = []
    # Storing currently stored sent idx -> unique mentions seed (for sentences that aren't complete, we'll hold until they are)
    cur_sentidx2_nummentions = dict()
    num_eval_steps = 0

    # Collect dataloader information
    task_to_label_dict = dataloader.task_to_label_dict
    uid = dataloader.uid

    with torch.no_grad():
        for batch_num, bdict in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Evaluating {dataloader.data_name} ({dataloader.split})",
        ):
            num_eval_steps += 1
            X_bdict, Y_bdict = bdict

            (
                uid_bdict,
                loss_bdict,
                prob_bdict,
                gold_bdict,
                out_bdict,
            ) = model.forward(  # type: ignore
                X_bdict[uid],
                X_bdict,
                Y_bdict,
                task_to_label_dict,
                return_action_outputs=True,
                return_probs=True,
            )
            assert (
                NED_TASK in uid_bdict
            ), f"{NED_TASK} task needs to be in returned in uid to get number of mentions"
            for task_name in uid_bdict.keys():
                for ex_idx in range(len(uid_bdict[task_name])):
                    # Recall that our uid is
                    # ============================
                    # guid_dtype = np.dtype(
                    #     [
                    #         ("sent_idx", "i8", 1),
                    #         ("subsent_idx", "i8", 1),
                    #         ("alias_orig_list_pos", "i8", (data_config.max_aliases,)),
                    #     ]
                    # )
                    # ============================
                    # Index 0 -> sent_idx, Index 1 -> subsent_idx, Index 2 -> List of aliases positions (-1 means no mention)
                    sent_idx = uid_bdict[task_name][ex_idx][0]
                    # Only incredment for NED TASK
                    if task_name == NED_TASK:
                        if sent_idx not in cur_sentidx2_nummentions:
                            cur_sentidx2_nummentions[sent_idx] = set()
                        # Index 2 is index of alias positions in original list (-1 means no mention)
                        cur_sentidx2_nummentions[sent_idx].update(
                            set([i for i in uid_bdict[task_name][ex_idx][2] if i != -1])
                        )
                    uid_dict[task_name][sent_idx].extend(
                        uid_bdict[task_name][ex_idx : ex_idx + 1]
                    )
                    prob_dict[task_name][sent_idx].extend(prob_bdict[task_name][ex_idx : ex_idx + 1])  # type: ignore
                    pred_dict[task_name][sent_idx].extend(  # type: ignore
                        prob_to_pred(prob_bdict[task_name][ex_idx : ex_idx + 1])
                    )
                    gold_dict[task_name][sent_idx].extend(
                        gold_bdict[task_name][ex_idx : ex_idx + 1]
                    )
                    if task_name in out_bdict.keys():
                        for action_name in out_bdict[task_name].keys():
                            out_dict[task_name][action_name][sent_idx].extend(
                                out_bdict[task_name][action_name][ex_idx : ex_idx + 1]
                            )
            if num_eval_steps >= eval_accumulation_steps:
                # Collect the sentences that have all mentions collected
                res, finalized_sent_idxs = collect_result(
                    uid_dict,
                    gold_dict,
                    pred_dict,
                    prob_dict,
                    out_dict,
                    cur_sentidx2_nummentions,
                )
                all_finalized_sentences.extend([str(s) for s in finalized_sent_idxs])
                num_eval_steps = 0
                for final_sent_i in finalized_sent_idxs:
                    assert final_sent_i in cur_sentidx2_nummentions
                    del cur_sentidx2_nummentions[final_sent_i]
                    for task_name in uid_dict.keys():
                        del uid_dict[task_name][final_sent_i]
                        del prob_dict[task_name][final_sent_i]
                        del pred_dict[task_name][final_sent_i]
                        del gold_dict[task_name][final_sent_i]
                        if task_name in out_dict.keys():
                            for action_name in out_dict[task_name].keys():
                                del out_dict[task_name][action_name][final_sent_i]
                if len(res) > 0:
                    # print("FINALIZED", finalized_sent_idxs)
                    yield res
    res, finalized_sent_idxs = collect_result(
        uid_dict, gold_dict, pred_dict, prob_dict, out_dict, cur_sentidx2_nummentions
    )
    all_finalized_sentences.extend([str(s) for s in finalized_sent_idxs])
    for final_sent_i in finalized_sent_idxs:
        del cur_sentidx2_nummentions[final_sent_i]
    if len(res) > 0:
        # print("FINALIZED", finalized_sent_idxs)
        yield res
    assert (
        len(cur_sentidx2_nummentions) == 0
    ), f"After eval, some sentences had left over mentions {cur_sentidx2_nummentions}"
    assert set(all_finalized_sentences).intersection(
        sent_idx2num_mentions.keys()
    ) == set([k for k, v in sent_idx2num_mentions.items() if v > 0]), (
        f"Some sentences are left over "
        f"{[s for s in sent_idx2num_mentions if s not in set(all_finalized_sentences) and sent_idx2num_mentions[s] > 0]}"
    )
    return None


def check_and_create_alias_cand_trie(save_folder, entity_symbols):
    try:
        AliasCandRecordTrie(load_dir=save_folder)
    except FileNotFoundError as e:
        log_rank_0_debug(
            logger,
            "Creating the alias candidate trie for faster parallel processing. "
            "This is a one time cost",
        )
        alias_trie = AliasCandRecordTrie(
            input_dict=entity_symbols.get_alias2qids(),
            vocabulary=entity_symbols.get_qid2title(),
            max_value=entity_symbols.max_candidates,
        )
        alias_trie.dump(save_folder)
    return


def get_emb_file(result_idx, save_folder):
    return os.path.join(save_folder, f"out_emb_file_{result_idx}.npy")


def get_result_file(result_idx, save_folder):
    return os.path.join(save_folder, f"result_label_file_{result_idx}.jsonl")


def disambig_dump_preds(
    result_idx,
    config,
    res_dict,
    sent_idx2num_mentions,
    sent_idx2row,
    save_folder,
    entity_symbols,
    dump_embs,
    task_name,
):
    """Dumps the predictions of a disambiguation task.

    Args:
        result_idx: batch index of the result arrays
        config: model config
        res_dict: result dictionary from Emmental predict
        sent_idx2num_mentions: Dict sentence idx to number of mentions
        sent_idx2row: Dict sentence idx to row of eval data
        save_folder: folder to save results
        entity_symbols: entity symbols
        dump_embs: whether to save the contextualized embeddings or not
        task_name: task name

    Returns: saved prediction file, saved embedding file (will be None if dump_embs is False)
    """
    num_processes = min(
        config.run_config.dataset_threads, int(multiprocessing.cpu_count() * 0.9)
    )
    cache_dir = os.path.join(save_folder, f"cache_{result_idx}")
    utils.ensure_dir(cache_dir)
    trie_candidate_map_folder = None
    trie_qid2eid_file = None
    # Save the alias->QID candidate map and the QID->EID mapping in memory efficient structures for faster
    # prediction dumping
    if num_processes > 1:
        entity_prep_dir = data_utils.get_emb_prep_dir(config.data_config)
        trie_candidate_map_folder = os.path.join(
            entity_prep_dir, "for_dumping_preds", "alias_cand_trie"
        )
        utils.ensure_dir(trie_candidate_map_folder)
        check_and_create_alias_cand_trie(trie_candidate_map_folder, entity_symbols)
        trie_qid2eid_file = os.path.join(
            entity_prep_dir, "for_dumping_preds", "qid2eid_trie.marisa"
        )
        if not os.path.exists(trie_qid2eid_file):
            utils.create_single_item_trie(
                entity_symbols.get_qid2eid(), out_file=trie_qid2eid_file
            )

    # This is dumping
    disambig_res_dict = {}
    for k in res_dict:
        assert task_name in res_dict[k], f"{task_name} not in res_dict for key {k}"
        disambig_res_dict[k] = res_dict[k][task_name]

    # write to file (M x hidden x size for each data point -- next step will deal with recovering original sentence indices for overflowing sentences)
    unmerged_entity_emb_file = os.path.join(save_folder, f"entity_embs.pt")
    merged_entity_emb_file = os.path.join(save_folder, f"entity_embs_unmerged.pt")
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
    all_sent_idx = set()
    for i, (uid, gold, probs, model_pred) in enumerate(zip(*for_iteration)):
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

        all_sent_idx.add(str(sent_idx))
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
    #         import pdb
    #         pdb.set_trace()
    #     assert si != -1, f"{i} {mmap_file[i]}"
    # Store all predicted sentences to filter the sentence mapping by
    subset_sent_idx2num_mentions = {
        k: v for k, v in sent_idx2num_mentions.items() if k in all_sent_idx
    }
    # print("ALL SEEN", all_sent_idx)
    subsent_sent_idx2row = {k: v for k, v in sent_idx2row.items() if k in all_sent_idx}
    result_file = get_result_file(result_idx, save_folder)
    log_rank_0_debug(logger, f"Writing predictions to {result_file}...")
    merge_subsentences(
        num_processes=num_processes,
        subset_sent_idx2num_mentions=subset_sent_idx2num_mentions,
        cache_folder=cache_dir,
        to_save_file=merged_entity_emb_file,
        to_save_storage=merged_storage_type,
        to_read_file=unmerged_entity_emb_file,
        to_read_storage=unmerged_storage_type,
        dump_embs=dump_embs,
    )
    write_data_labels(
        num_processes=num_processes,
        merged_entity_emb_file=merged_entity_emb_file,
        merged_storage_type=merged_storage_type,
        sent_idx2row=subsent_sent_idx2row,
        cache_folder=cache_dir,
        out_file=result_file,
        entity_dump=entity_symbols,
        train_in_candidates=config.data_config.train_in_candidates,
        max_candidates=entity_symbols.max_candidates,
        dump_embs=dump_embs,
        trie_candidate_map_folder=trie_candidate_map_folder,
        trie_qid2eid_file=trie_qid2eid_file,
    )

    out_emb_file = None
    # save easier-to-use embedding file
    if dump_embs:
        filt_emb_data = np.memmap(
            merged_entity_emb_file, dtype=merged_storage_type, mode="r+"
        )
        hidden_size = filt_emb_data[0]["hidden_size"]
        out_emb_file = get_emb_file(result_idx, save_folder)
        np.save(out_emb_file, filt_emb_data["entity_emb"].reshape(-1, hidden_size))
        log_rank_0_debug(
            logger,
            f"Saving contextual entity embeddings for {result_idx} to {out_emb_file}",
        )
    # Cleanup cache
    shutil.rmtree(cache_dir)
    log_rank_0_debug(logger, f"Wrote predictions for {result_idx} to {result_file}")
    return result_file, out_emb_file, all_sent_idx


#
def merge_subsentences(
    num_processes,
    subset_sent_idx2num_mentions,
    cache_folder,
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
        subset_sent_idx2num_mentions: Dict of sentence index to number of mentions for this batch
        cache_folder: cache directory
        to_save_file: memmap file to save results to
        to_save_storage: save file storage type
        to_read_file: memmap file to read predictions from
        to_read_storage: read file storage type
        dump_embs: whether to save embeddings or not

    Returns:
    """
    # Compute sent idx to offset so we know where to fill in mentions
    cur_offset = 0
    sentidx2offset = {}
    for k, v in subset_sent_idx2num_mentions.items():
        sentidx2offset[k] = cur_offset
        cur_offset += v
    total_num_mentions = cur_offset
    # print("TOTAL", total_num_mentions)
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

    all_ids = list(range(0, len(full_pred_data)))
    start = time.time()
    if num_processes == 1:
        seen_ids = merge_subsentences_single(
            M,
            K,
            hidden_size,
            dump_embs,
            all_ids,
            filt_emb_data,
            full_pred_data,
            sentidx2offset,
        )

    else:
        # Get trie for sentence start map
        trie_folder = os.path.join(cache_folder, "bootleg_sent_idx2num_mentions")
        utils.ensure_dir(trie_folder)
        trie_file = os.path.join(trie_folder, "sentidx.marisa")
        utils.create_single_item_trie(sentidx2offset, out_file=trie_file)
        # Chunk up date
        chunk_size = int(np.ceil(len(full_pred_data) / num_processes))
        row_idx_set_chunks = [
            all_ids[ids : ids + chunk_size]
            for ids in range(0, len(full_pred_data), chunk_size)
        ]
        # Start pool
        input_args = [
            [M, K, hidden_size, dump_embs, chunk] for chunk in row_idx_set_chunks
        ]
        log_rank_0_debug(
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
                trie_file,
            ],
        )

        seen_ids = set()
        for sent_ids_seen in pool.imap_unordered(
            merge_subsentences_hlp, input_args, chunksize=1
        ):
            for emb_id in sent_ids_seen:
                assert (
                    emb_id not in seen_ids
                ), f"{emb_id} already seen, something went wrong with sub-sentences"
                seen_ids.add(emb_id)

    # filt_emb_data = np.memmap(to_save_file, dtype=to_save_storage, mode="r")
    # for i in range(len(filt_emb_data)):
    #     si = filt_emb_data[i]["sent_idx"]
    #     al_test = filt_emb_data[i]["alias_list_pos"]
    #     if si == -1 or al_test == -1:
    #         print("BAD", i, filt_emb_data[i])
    logging.debug(f"Saw {len(seen_ids)} sentences")
    logging.debug(f"Time to merge sub-sentences {time.time() - start}s")
    return


def merge_subsentences_initializer(
    to_write_file, to_write_storage, to_read_file, to_read_storage, sentidx2offset_file
):
    global filt_emb_data_global
    filt_emb_data_global = np.memmap(to_write_file, dtype=to_write_storage, mode="r+")
    global full_pred_data_global
    full_pred_data_global = np.memmap(to_read_file, dtype=to_read_storage, mode="r+")
    global sentidx2offset_marisa_global
    sentidx2offset_marisa_global = utils.load_single_item_trie(sentidx2offset_file)


def merge_subsentences_hlp(args):
    M, K, hidden_size, dump_embs, r_idx_set = args
    return merge_subsentences_single(
        M,
        K,
        hidden_size,
        dump_embs,
        r_idx_set,
        filt_emb_data_global,
        full_pred_data_global,
        sentidx2offset_marisa_global,
    )


def merge_subsentences_single(
    M,
    K,
    hidden_size,
    dump_embs,
    r_idx_set,
    filt_emb_data,
    full_pred_data,
    sentidx2offset,
):
    """Helper for merge_sentences."""
    seen_ids = set()
    for r_idx in r_idx_set:
        row = full_pred_data[r_idx]
        # get corresponding row to start writing into condensed memory mapped file
        sent_idx = str(row["sent_idx"])
        if isinstance(sentidx2offset, dict):
            sent_start_idx = sentidx2offset[sent_idx]
        else:
            # Get from Trie
            sent_start_idx = sentidx2offset[sent_idx][0][0]
        # print("R IDS", r_idx, row["sent_idx"], "START", sent_start_idx)
        # for each VALID mention, need to write into original alias list pos in list
        for i, (true_val, alias_orig_pos) in enumerate(
            zip(row["final_loss_true"], row["alias_list_pos"])
        ):
            # bc we are are using the mentions which includes both true and false golds, true_val == -1 only for padded mentions or sub-sentence mentions
            if true_val != -1:
                # print(
                #     "INSIDE MERGE", i, sent_idx, true_val, alias_orig_pos, sent_start_idx, sent_start_idx + alias_orig_pos
                # )
                # id in condensed embedding
                emb_id = sent_start_idx + alias_orig_pos
                assert (
                    emb_id not in seen_ids
                ), f"{emb_id} already seen, something went wrong with sub-sentences"
                seen_ids.add(emb_id)
                if dump_embs:
                    filt_emb_data["entity_emb"][emb_id] = row["entity_emb"].reshape(
                        M, hidden_size
                    )[i]
                filt_emb_data["sent_idx"][emb_id] = sent_idx
                filt_emb_data["alias_list_pos"][emb_id] = alias_orig_pos
                filt_emb_data["final_loss_pred"][emb_id] = row[
                    "final_loss_pred"
                ].reshape(M)[i]
                filt_emb_data["final_loss_prob"][emb_id] = row[
                    "final_loss_prob"
                ].reshape(M)[i]
                filt_emb_data["final_loss_cand_probs"][emb_id] = row[
                    "final_loss_cand_probs"
                ].reshape(M, K)[i]
    return seen_ids


def get_sental2embid(merged_entity_emb_file, merged_storage_type):
    """Get sent_idx, alias_idx mapping to emb idx for quick lookup.

    Args:
        merged_entity_emb_file: memmap file after merge sentences
        merged_storage_type: file storage type

    Returns: Dict of f"{sent_idx}_{alias_idx}" -> index in merged_entity_emb_file
    """
    filt_emb_data = np.memmap(
        merged_entity_emb_file, dtype=merged_storage_type, mode="r+"
    )
    sental2embid = {}
    for i, row in enumerate(filt_emb_data):
        sent_idx = row["sent_idx"]
        alias_idx = row["alias_list_pos"]
        assert (
            sent_idx != -1 and alias_idx != -1
        ), f"{i} {row} Has Sent {sent_idx}, Al {alias_idx}"
        # Keep as string for Marisa Tri later
        sental2embid[f"{sent_idx}_{alias_idx}"] = i
    return sental2embid


def write_data_labels(
    num_processes,
    merged_entity_emb_file,
    merged_storage_type,
    sent_idx2row,
    cache_folder,
    out_file,
    entity_dump,
    train_in_candidates,
    max_candidates,
    dump_embs,
    trie_candidate_map_folder=None,
    trie_qid2eid_file=None,
):
    """Takes the flattened data from merge_sentences and writes out predictions
    to a file, one line per sentence.

    The embedding ids are added to the file if dump_embs is True.

    Args:
        num_processes: number of processes
        merged_entity_emb_file: input memmap file after merge sentences
        merged_storage_type: input file storage type
        sent_idx2row: Dict of sentence idx to row relevant to this subbatch
        cache_folder: folder to save temporary outputs
        out_file: final output file for predictions
        entity_dump: entity dump
        train_in_candidates: whether NC entities are not in candidate lists
        max_candidates: maximum number of candidates
        dump_embs: whether to dump embeddings or not
        trie_candidate_map_folder: folder where trie of alias->candidate map is stored for parallel proccessing
        trie_qid2eid_file: file where trie of qid->eid map is stored for parallel proccessing

    Returns:
    """
    st = time.time()
    sental2embid = get_sental2embid(merged_entity_emb_file, merged_storage_type)
    log_rank_0_debug(logger, f"Finished getting sentence map {time.time() - st}s")

    total_input = len(sent_idx2row)
    if num_processes == 1:
        filt_emb_data = np.memmap(
            merged_entity_emb_file, dtype=merged_storage_type, mode="r+"
        )
        write_data_labels_single(
            sentidx2row=sent_idx2row,
            output_file=out_file,
            filt_emb_data=filt_emb_data,
            sental2embid=sental2embid,
            alias_cand_map=entity_dump.get_alias2qids(),
            qid2eid=entity_dump.get_qid2eid(),
            train_in_cands=train_in_candidates,
            max_cands=max_candidates,
            dump_embs=dump_embs,
        )
    else:
        assert (
            trie_candidate_map_folder is not None
        ), "trie_candidate_map_folder is None and you have parallel turned on"
        assert (
            trie_qid2eid_file is not None
        ), "trie_qid2eid_file is None and you have parallel turned on"

        # Get trie of sentence map
        trie_folder = os.path.join(cache_folder, "bootleg_sental2embid")
        utils.ensure_dir(trie_folder)
        trie_file = os.path.join(trie_folder, "sentidx.marisa")
        utils.create_single_item_trie(sental2embid, out_file=trie_file)

        # Chunk file for parallel writing
        # We do not use TemporaryFolders as the temp dir may not have enough space for large files
        create_ex_indir = os.path.join(cache_folder, "_bootleg_eval_temp_indir")
        utils.ensure_dir(create_ex_indir)
        create_ex_outdir = os.path.join(cache_folder, "_bootleg_eval_temp_outdir")
        utils.ensure_dir(create_ex_outdir)
        chunk_input = int(np.ceil(total_input / num_processes))
        logger.debug(
            f"Chunking up {total_input} lines into subfiles of size {chunk_input} lines"
        )
        # Chunk up dictionary of data for parallel processing
        input_files = []
        i = 0
        cur_lines = 0
        file_split = os.path.join(create_ex_indir, f"out{i}.jsonl")
        open_file = open(file_split, "w")
        for s_idx in sent_idx2row:
            if cur_lines >= chunk_input:
                open_file.close()
                input_files.append(file_split)
                cur_lines = 0
                i += 1
                file_split = os.path.join(create_ex_indir, f"out{i}.jsonl")
                open_file = open(file_split, "w")
            line = sent_idx2row[s_idx]
            open_file.write(ujson.dumps(line) + "\n")
            cur_lines += 1
        open_file.close()
        input_files.append(file_split)
        # Generation input/output pairs
        output_files = [
            in_file_name.replace(create_ex_indir, create_ex_outdir)
            for in_file_name in input_files
        ]
        log_rank_0_debug(logger, f"Done chunking files. Starting pool")

        pool = multiprocessing.Pool(
            processes=num_processes,
            initializer=write_data_labels_initializer,
            initargs=[
                merged_entity_emb_file,
                merged_storage_type,
                trie_file,
                train_in_candidates,
                max_candidates,
                dump_embs,
                trie_candidate_map_folder,
                trie_qid2eid_file,
            ],
        )

        input_args = list(zip(input_files, output_files))

        total = 0
        for res in pool.imap(write_data_labels_hlp, input_args, chunksize=1):
            total += 1

        # Merge output files to final file
        log_rank_0_debug(logger, f"Merging output files")
        with open(out_file, "wb") as outfile:
            for filename in glob.glob(os.path.join(create_ex_outdir, "*")):
                if filename == out_file:
                    # don't want to copy the output into the output
                    continue
                with open(filename, "rb") as readfile:
                    shutil.copyfileobj(readfile, outfile)


def write_data_labels_initializer(
    merged_entity_emb_file,
    merged_storage_type,
    sental2embid_file,
    train_in_candidates,
    max_cands,
    dump_embs,
    trie_candidate_map_folder,
    trie_qid2eid_file,
):
    global filt_emb_data_global
    filt_emb_data_global = np.memmap(
        merged_entity_emb_file, dtype=merged_storage_type, mode="r+"
    )
    global sental2embid_global
    sental2embid_global = utils.load_single_item_trie(sental2embid_file)
    global alias_cand_trie_global
    alias_cand_trie_global = AliasCandRecordTrie(load_dir=trie_candidate_map_folder)
    global qid2eid_global
    qid2eid_global = utils.load_single_item_trie(trie_qid2eid_file)
    global train_in_candidates_global
    train_in_candidates_global = train_in_candidates
    global max_cands_global
    max_cands_global = max_cands
    global dump_embs_global
    dump_embs_global = dump_embs


def write_data_labels_hlp(args):
    input_file, output_file = args
    s_idx2row = {}
    with open(input_file) as in_f:
        for line in in_f:
            line = ujson.loads(line)
            s_idx2row[str(line["sent_idx_unq"])] = line
    return write_data_labels_single(
        s_idx2row,
        output_file,
        filt_emb_data_global,
        sental2embid_global,
        alias_cand_trie_global,
        qid2eid_global,
        train_in_candidates_global,
        max_cands_global,
        dump_embs_global,
    )


def write_data_labels_single(
    sentidx2row,
    output_file,
    filt_emb_data,
    sental2embid,
    alias_cand_map,
    qid2eid,
    train_in_cands,
    max_cands,
    dump_embs,
):
    """Write data labels helper."""
    with open(output_file, "w") as f_out:
        for sent_idx in sentidx2row:
            line = sentidx2row[sent_idx]
            aliases = line["aliases"]
            assert sent_idx == str(line["sent_idx_unq"])
            qids = []
            ctx_emb_ids = []
            entity_ids = []
            probs = []
            cands = []
            cand_probs = []
            entity_cands_qid = map_aliases_to_candidates(
                train_in_cands, max_cands, alias_cand_map, aliases
            )
            # eid is entity id
            entity_cands_eid = map_candidate_qids_to_eid(entity_cands_qid, qid2eid)
            for al_idx, alias in enumerate(aliases):
                sent_idx_key = f"{sent_idx}_{al_idx}"
                assert (
                    sent_idx_key in sental2embid
                ), f"Dumped prediction data does not match data file. Can not find {sent_idx} - {al_idx}"
                if isinstance(sental2embid, dict):
                    emb_idx = sental2embid[sent_idx_key]
                else:
                    # Get from Trie
                    emb_idx = sental2embid[sent_idx_key][0][0]
                ctx_emb_ids.append(emb_idx)
                prob = filt_emb_data[emb_idx]["final_loss_prob"]
                cand_prob = filt_emb_data[emb_idx]["final_loss_cand_probs"]
                pred_cand = filt_emb_data[emb_idx]["final_loss_pred"]
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
            if dump_embs:
                line["ctx_emb_ids"] = ctx_emb_ids
            f_out.write(ujson.dumps(line) + "\n")
