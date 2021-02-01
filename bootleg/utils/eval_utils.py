import glob
import multiprocessing
import shutil
import tempfile
from collections import  OrderedDict

import logging
import jsonlines
import numpy as np
import os

import ujson
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import time
import torch
from tqdm import tqdm

from bootleg.symbols.constants import *
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import train_utils, utils

import torch.nn.functional as F


def masked_softmax(pred, mask, dim=2, temp=1.0):
    assert temp > 0, f"You can't have a temperature of 0"
    # https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
    pred = pred / temp
    mask = mask.float()
    masked_pred = pred.masked_fill((1 - mask).byte().bool(), -1e32)
    result = F.softmax(masked_pred, dim=dim)
    return result


def masked_class_logsoftmax(pred, mask, dim=2, temp=1.0):
    assert temp > 0, f"You can't have a temperature of 0"
    # pred is batch x M x K
    # https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
    pred = pred / temp
    pred = pred + (mask + 1e-45).log()  # we could also do 1e-46 but I feel safer with 1e-45
    # compute softmax over the k dimension
    return F.log_softmax(input=pred, dim=dim)


def map_aliases_to_candidates(train_in_candidates, entity_symbols, aliases):
    not_tic = (1 - train_in_candidates)
    return [not_tic * ["NC"] + entity_symbols.get_qid_cands(al, max_cand_pad=True)
            if al != UNK_AL else ["-1"] * (entity_symbols.max_candidates + not_tic)
            for al in aliases]

def map_aliases_to_candidates_eid(train_in_candidates, entity_symbols, aliases):
    not_tic = (1 - train_in_candidates)
    return [not_tic * ["NC"] + entity_symbols.get_eid_cands(al, max_cand_pad=True)
            if al != UNK_AL else ["-1"] * (entity_symbols.max_candidates + not_tic)
            for al in aliases]

def get_eval_columns(topk, is_sbl_model, is_test):
    columns = [
        ("train_head", "head"),
        ("eval_slice", "slice"),
        ("num_mentions", "men"),
        ("correct", "crct"),
        (f"correct_top{topk}", f"crct_top{topk}"),
        ("correct_popular", "crct_pop"),
        ("f1_micro", "f1"),
        (f"f1_micro_top{topk}", f"f1_top{topk}"),
        ("f1_micro_ent_popular", "f1_pop"),
    ]
    if not is_test:
        columns.append(("global_step", "stp"))
    if is_sbl_model:
        columns.extend([
            ("correct_pred", "crct_pred"),
            ("pred_num_mentions", "men_pred"),
            ("ind_precision", "i_prec"),
            ("ind_recall", "i_rec"),
            ("f1_micro_ent_pred", "f1_pred")]
        )
    return OrderedDict(columns)


def run_eval_all_dev_sets(args, global_step, dev_dataset_collection, logger, status_reporter, trainer):
    for dev_id, dev_data_file in enumerate(dev_dataset_collection):
        dev_dataloader = dev_dataset_collection[dev_data_file].data_loader
        logger.info(f"************************RUNNING DEV EVAL {dev_data_file} AT STEP {global_step}")
        # False is for if it's test or dev file so we know what loss dump to use
        run_batched_eval(args, False, global_step, logger, trainer, dev_dataloader, status_reporter, dev_data_file)

def batch_eval_pretty_json(all_dev_results, train_head, eval_slice, topk_val):
    pretty_dict_for_printing = {}
    slice_res = all_dev_results[train_head][eval_slice]
    total_pred_in_slice = all_dev_results[train_head][FINAL_LOSS]['pred_in_slice']
    pred_correct_in_slice = slice_res['pred_in_slice']
    true_total_in_slice = slice_res['true_total_in_slice']
    prec, recall = calc_ind_prec_recall(pred_correct_in_slice, total_pred_in_slice, true_total_in_slice)
    pretty_dict_for_printing["global_step"] = slice_res['step']
    pretty_dict_for_printing["train_head"] = train_head
    pretty_dict_for_printing["eval_slice"] = eval_slice
    pretty_dict_for_printing["num_mentions"] = true_total_in_slice
    pretty_dict_for_printing["correct"] = slice_res["correct_ent"]
    pretty_dict_for_printing[f"correct_top{topk_val}"] = slice_res[f'correct_ent_top{topk_val}']
    pretty_dict_for_printing["correct_popular"] = slice_res["correct_ent_head"]
    pretty_dict_for_printing["correct_pred"] = slice_res["correct_ent_pred"]
    pretty_dict_for_printing["f1_micro"] = slice_res['f1_micro_ent']
    pretty_dict_for_printing[f"f1_micro_top{topk_val}"] = slice_res[f'f1_micro_ent_top{topk_val}']
    pretty_dict_for_printing["f1_micro_ent_popular"] = slice_res['f1_micro_ent_head']
    pretty_dict_for_printing["f1_micro_ent_pred"] = slice_res["f1_micro_ent_pred"]
    pretty_dict_for_printing["pred_num_mentions"] = slice_res["pred_in_slice"]
    pretty_dict_for_printing["ind_precision"] = prec
    pretty_dict_for_printing["ind_recall"] = recall
    return pretty_dict_for_printing


def calc_micros(gold, preds, head, ending):
    prec_macro_head, recall_macro_head, f1_macro_head, _ = precision_recall_fscore_support(gold, head, average='macro')
    prec_micro_head, recall_micro_head, f1_micro_head, _ = precision_recall_fscore_support(gold, head, average='micro')
    prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold, preds, average='macro')
    prec_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(gold, preds, average='micro')
    metrics = {}
    metrics["prec_macro" + ending] = round(prec_macro, 4)
    metrics["recall_macro" + ending] = round(recall_macro, 4)
    metrics["f1_macro" + ending] = round(f1_macro, 4)
    metrics["prec_micro" + ending] = round(prec_micro, 4)
    metrics["recall_micro" + ending] = round(recall_micro, 4)
    metrics["f1_micro" + ending] = round(f1_micro, 4)
    metrics["prec_macro" + ending + "_head"] = round(prec_macro_head, 4)
    metrics["recall_macro" + ending + "_head"] = round(recall_macro_head, 4)
    metrics["f1_macro" + ending + "_head"] = round(f1_macro_head, 4)
    metrics["prec_micro" + ending + "_head"] = round(prec_micro_head, 4)
    metrics["recall_micro" + ending + "_head"] = round(recall_micro_head, 4)
    metrics["f1_micro" + ending + "_head"] = round(f1_micro_head, 4)
    metrics["num_mentions"] = len(gold)
    return metrics


def calc_metrics(gold_ents, preds_ents, head_ents):
    ent_land_metrics = calc_micros(gold_ents, preds_ents, head_ents, "_ent")
    true_false_arr = np.array(gold_ents) == np.array(preds_ents)
    correct = np.sum(true_false_arr)
    ent_land_metrics["correct"] = correct
    true_false_arr = np.array(gold_ents) == np.array(head_ents)
    correct = np.sum(true_false_arr)
    ent_land_metrics["correct_head"] = correct
    return ent_land_metrics


# Dump cumulative scores per slice
def gen_slice_json_results(gold_ents, preds_ents, head_ents, num_aliases, num_test_examples, slice, step=-1, batch_size=-1):
    ent_land_metrics = calc_metrics(gold_ents, preds_ents, head_ents)
    test_results = {}
    if step != -1:
        test_results["step"] = step
    if batch_size != -1:
        test_results["total_examples_seen"] = step * batch_size
    test_results["num_test_examples"] = num_test_examples
    test_results["num_aliases"] = int(num_aliases)
    test_results["num_mentions"] = int(ent_land_metrics["num_mentions"])
    test_results["f1_micro_ent"] = float(ent_land_metrics["f1_micro_ent"])
    test_results["f1_micro_ent_head"] = float(ent_land_metrics["f1_micro_ent_head"])
    test_results["correct"] = int(ent_land_metrics["correct"])
    test_results["correct_head"] = int(ent_land_metrics["correct_head"])
    return {slice: test_results}


def run_batched_eval(args, is_test, global_step, logger, trainer, dataloader, status_reporter, file):
    # reset counts to zero for next eval
    trainer.eval_wrapper.reset_buffers()
    logger.info(f'Evaluating {len(dataloader)} batches')
    for i, batch in tqdm(enumerate(dataloader), desc="Running eval", total=len(dataloader)):
        outs, loss_pack, entity_pack, _ = trainer.update(batch, eval=True)
    is_sbl_model = train_utils.is_slicing_model(args)
    all_dev_results = trainer.eval_wrapper.compute_scores()
    assert FINAL_LOSS in all_dev_results
    # This will include stage heads, too
    train_head_keys = sorted(list(all_dev_results.keys()))
    # Moving final loss so we see it last (purely for aesthetics)
    train_head_keys.remove(FINAL_LOSS)
    train_head_keys.append(FINAL_LOSS)
    eval_slice_keys = sorted(list(all_dev_results[FINAL_LOSS].keys()))
    pretty_dict_for_printing = []
    dict_for_dumping = []
    for train_head in train_head_keys:
        eval_slice_name_for_head = train_utils.get_inv_head_name_eval(train_head)
        for eval_slice in eval_slice_keys:
            # base head and overall slice are the same
            all_dev_results[train_head][eval_slice]['step'] = global_step
            # if eval_slice == BASE_SLICE:
            #     continue
            dict_for_printing = batch_eval_pretty_json(all_dev_results, train_head, eval_slice, topk_val=args.run_config.topk)
            dict_for_dumping.append(dict_for_printing)
            # do not include stage outputs in pretty dump
            if 'stage' in train_head or (not is_sbl_model and eval_slice == BASE_SLICE):
                continue
            if eval_slice == eval_slice_name_for_head or train_head == train_utils.get_slice_head_eval_name(BASE_SLICE) \
                    or FINAL_LOSS in train_head or eval_slice == FINAL_LOSS or args.run_config.verbose:
                # Order the columns
                eval_columns = get_eval_columns(topk=args.run_config.topk, is_sbl_model=is_sbl_model, is_test=is_test)
                ordered_dict = OrderedDict((eval_columns[k], dict_for_printing[k]) for k in eval_columns)
                pretty_dict_for_printing.append(ordered_dict)
    logger.info(f'\n{tabulate(pretty_dict_for_printing, headers="keys", tablefmt="grid")}')
    if status_reporter is not None:
        status_reporter.dump_results(dict_for_dumping, file, is_test)
    return all_dev_results

def select_embs(embs, pred_cands, batch_size, M):
    return embs[torch.arange(batch_size).unsqueeze(-1), torch.arange(M).unsqueeze(0), pred_cands]

def run_dump_preds(args, test_data_file, trainer, dataloader, logger, entity_symbols, dump_embs=False):
    """
    Dumpes Preds:
    Remember that if a sentence has all gold=False anchors, it's dropped and will not be seen
    If a subsplit of a sentence has all gold=False anchors, it will also be dropped and not seen
    """
    num_processes = min(args.run_config.dataset_threads, int(multiprocessing.cpu_count()*0.9))
    # we only care about the entity embeddings for the final slice head
    eval_folder = train_utils.get_eval_folder(args, test_data_file)
    utils.ensure_dir(eval_folder)

    # write to file (M x hidden x size for each data point -- next step will deal with recovering original sentence indices for overflowing sentences)
    test_file_tag = test_data_file.split('.jsonl')[0]
    unmerged_entity_emb_file = os.path.join(eval_folder, f'{test_file_tag}_entity_embs.pt')
    merged_entity_emb_file = os.path.join(eval_folder, f'{test_file_tag}_entity_embs_merged.pt')
    emb_file_config = unmerged_entity_emb_file.split('.pt')[0] + '_config'
    M = args.data_config.max_aliases
    K = entity_symbols.max_candidates + (not args.data_config.train_in_candidates)
    if dump_embs:
        unmerged_storage_type = np.dtype([('M', int),
                                          ('K', int),
                                          ('hidden_size', int),
                                          ('sent_idx', int),
                                          ('subsent_idx', int),
                                          ('alias_list_pos', int, M),
                                          ('entity_emb', float, M*args.model_config.hidden_size),
                                          ('final_loss_true', int, M),
                                          ('final_loss_pred', int, M),
                                          ('final_loss_prob', float, M),
                                          ('final_loss_cand_probs', float, M*K)])
        merged_storage_type = np.dtype([('hidden_size', int),
                             ('sent_idx', int),
                             ('alias_list_pos', int),
                             ('entity_emb', float, args.model_config.hidden_size),
                             ('final_loss_pred', int),
                             ('final_loss_prob', float),
                             ('final_loss_cand_probs', float, K)])
    else:
        # don't need to extract contextualized entity embedding
        unmerged_storage_type = np.dtype([('M', int),
                                          ('K', int),
                                          ('hidden_size', int),
                                          ('sent_idx', int),
                                          ('subsent_idx', int),
                                          ('alias_list_pos', int, M),
                                          ('final_loss_true', int, M),
                                          ('final_loss_pred', int, M),
                                          ('final_loss_prob', float, M),
                                          ('final_loss_cand_probs', float, M*K)])
        merged_storage_type = np.dtype([('hidden_size', int),
                             ('sent_idx', int),
                             ('alias_list_pos', int),
                             ('final_loss_pred', int),
                             ('final_loss_prob', float),
                             ('final_loss_cand_probs', float, K)])

    mmap_file = np.memmap(unmerged_entity_emb_file, dtype=unmerged_storage_type, mode='w+', shape=(len(dataloader.dataset),))
    # Init sent_idx to -1 for debugging
    mmap_file[:]['sent_idx'] = -1
    np.save(emb_file_config, unmerged_storage_type, allow_pickle=True)
    logger.debug(f'Created file {unmerged_entity_emb_file} to save predictions.')

    start_idx = 0
    logger.info(f'{len(dataloader)*args.run_config.eval_batch_size} samples, {len(dataloader)} batches, {len(dataloader.dataset)} len dataset')
    st = time.time()
    for i, batch in enumerate(dataloader):
        curr_batch_size = batch["sent_idx"].shape[0]
        end_idx = start_idx + curr_batch_size
        preds, _, entity_pack, final_entity_embs = trainer.update(batch, eval=True)
        model_preds = preds[DISAMBIG][FINAL_LOSS]
        # don't want to choose padded entity indices
        probs = torch.exp(masked_class_logsoftmax(pred=model_preds, mask=~entity_pack.mask, dim=2))

        mmap_file[start_idx:end_idx]['M'] = M
        mmap_file[start_idx:end_idx]['K'] = K
        mmap_file[start_idx:end_idx]['hidden_size'] = args.model_config.hidden_size
        mmap_file[start_idx:end_idx]['sent_idx'] = batch["sent_idx"]
        mmap_file[start_idx:end_idx]['subsent_idx'] = batch["subsent_idx"]
        mmap_file[start_idx:end_idx]['alias_list_pos'] = batch['alias_list_pos']
        # This will give all aliases seen by the model during training, independent of if it's gold or not
        mmap_file[start_idx:end_idx][f'final_loss_true'] = batch['true_entity_idx_for_train'].reshape(curr_batch_size, M).cpu().numpy()

        # get max for each alias, probs is batch x M x K
        max_probs, pred_cands = probs.max(dim=2)

        mmap_file[start_idx:end_idx]['final_loss_pred'] = pred_cands.cpu().numpy()
        mmap_file[start_idx:end_idx]['final_loss_prob'] = max_probs.cpu().numpy()
        mmap_file[start_idx:end_idx]['final_loss_cand_probs'] = probs.cpu().numpy().reshape(curr_batch_size,-1)

        # final_entity_embs is batch x M x K x hidden_size, pred_cands in batch x M
        if dump_embs:
            chosen_entity_embs = select_embs(embs=final_entity_embs, pred_cands=pred_cands,
                batch_size=curr_batch_size, M=M)

            # write chosen entity embs to file for contextualized entity embeddings
            mmap_file[start_idx:end_idx]['entity_emb'] = chosen_entity_embs.reshape(curr_batch_size,-1).cpu().numpy()

        start_idx += curr_batch_size
        if i % 100 == 0 and i != 0:
            logger.info(f'Saved {i} batches of predictions out of {len(dataloader)}')

    # restitch together and write data file
    result_file = os.path.join(eval_folder, args.run_config.result_label_file)
    logger.info(f'Writing predictions to {result_file}...')
    merge_subsentences(
        num_processes,
        os.path.join(args.data_config.data_dir, test_data_file),
        merged_entity_emb_file,
        merged_storage_type,
        unmerged_entity_emb_file,
        unmerged_storage_type,
        dump_embs=dump_embs)
    st = time.time()
    write_data_labels(num_processes=num_processes,
        merged_entity_emb_file=merged_entity_emb_file, merged_storage_type=merged_storage_type,
        data_file=os.path.join(args.data_config.data_dir, test_data_file), out_file=result_file,
        train_in_candidates=args.data_config.train_in_candidates,
        dump_embs=dump_embs, data_config=args.data_config)
    out_emb_file = None
    # save easier-to-use embedding file
    if dump_embs:
        filt_emb_data = np.memmap(merged_entity_emb_file, dtype=merged_storage_type, mode="r+")
        hidden_size = filt_emb_data[0]['hidden_size']
        out_emb_file = os.path.join(eval_folder, args.run_config.result_emb_file)
        np.save(out_emb_file, filt_emb_data['entity_emb'].reshape(-1,hidden_size))
        logger.info(f'Saving contextual entity embeddings to {out_emb_file}')
    logger.info(f'Wrote predictions to {result_file}')
    return result_file, out_emb_file

def calc_ind_prec_recall(pred_correct_in_slice, total_pred_in_slice, true_total_in_slice):
    if true_total_in_slice == 0:
        if pred_correct_in_slice > 0:
            recall = '{0:.3f}'.format(0.0)
        else:
            recall = '{0:.3f}'.format(0.0)
    else:
        recall = '{0:.3f}'.format(pred_correct_in_slice / true_total_in_slice)
    if total_pred_in_slice == 0:
        if pred_correct_in_slice > 0:
            prec = '{0:.3f}'.format(0.0)
        else:
            prec = '{0:.3f}'.format(1.0)
    else:
        prec = '{0:.3f}'.format(pred_correct_in_slice / total_pred_in_slice)
    return prec, recall

# load in original file to get number of mentions per sentence for length of "compressed" memory mapped file
def get_sent_start_map(data_file):
    sent_start_map = {}
    total_num_mentions = 0
    with jsonlines.open(data_file) as f:
        for line in f:
            # keep track of the start idx in the condensed memory mapped file for each sentence (varying number of aliases)
            assert line['sent_idx_unq'] not in sent_start_map, f'Sentence indices must be unique. {line["sent_idx_unq"]} already seen.'
            # Save as string for Marisa Tri later
            sent_start_map[str(line['sent_idx_unq'])] = total_num_mentions
            # We include false aliases for debugging (and alias_pos includes them)
            total_num_mentions += len(line['aliases'])
    # for k in sent_start_map:
    #     print("K", k, sent_start_map[k])
    logging.info(f'Total number of mentions across all sentences: {total_num_mentions}')
    return sent_start_map, total_num_mentions

# stich embs back together over sub-sentences, convert from sent_idx array to (sent_idx, alias_idx) array with varying numbers of aliases per sentence
def merge_subsentences(num_processes, data_file, to_save_file, to_save_storage, to_read_file, to_read_storage, dump_embs=False):
    logger = logging.getLogger(__name__)
    logger.debug(f"Getting sentence mapping")
    sent_start_map, total_num_mentions = get_sent_start_map(data_file)
    sent_start_map_file = tempfile.NamedTemporaryFile(suffix="bootleg_sent_start_map")
    utils.create_single_item_trie(sent_start_map, out_file=sent_start_map_file.name)
    logger.debug(f"Done with sentence mapping")

    full_pred_data = np.memmap(to_read_file, dtype=to_read_storage, mode='r')
    M = int(full_pred_data[0]['M'])
    K = int(full_pred_data[0]['K'])
    hidden_size = int(full_pred_data[0]['hidden_size'])

    filt_emb_data = np.memmap(to_save_file, dtype=to_save_storage, mode='w+', shape=(total_num_mentions,))
    filt_emb_data['hidden_size'] = hidden_size
    filt_emb_data['sent_idx'][:] = -1
    filt_emb_data['alias_list_pos'][:] = -1

    chunk_size = int(np.ceil(len(full_pred_data)/num_processes))
    all_ids = list(range(0, len(full_pred_data)))
    row_idx_set_chunks = [all_ids[ids:ids+chunk_size] for ids in range(0, len(full_pred_data), chunk_size)]
    input_args = [
        [M, K, hidden_size, dump_embs, chunk]
        for chunk in row_idx_set_chunks
    ]

    logger.info(f"Merging sentences together with {num_processes} processes. Starting pool")

    pool = multiprocessing.Pool(processes=num_processes,
                                initializer=merge_subsentences_initializer,
                                initargs=[
                                    to_save_file,
                                    to_save_storage,
                                    to_read_file,
                                    to_read_storage,
                                    sent_start_map_file.name
                                ])
    logger.debug(f"Finished pool")
    start = time.time()
    seen_ids = set()
    for sent_ids_seen in pool.imap_unordered(merge_subsentences_hlp, input_args, chunksize=1):
        for emb_id in sent_ids_seen:
            assert emb_id not in seen_ids, f'{emb_id} already seen, something went wrong with sub-sentences'
            seen_ids.add(emb_id)
    sent_start_map_file.close()
    logger.info(f'Time to merge sub-sentences {time.time()-start}s')
    return

def merge_subsentences_initializer(to_write_file, to_write_storage, to_read_file, to_read_storage, sent_start_map_file):
    global filt_emb_data_global
    filt_emb_data_global = np.memmap(to_write_file, dtype=to_write_storage, mode='r+')
    global full_pred_data_global
    full_pred_data_global = np.memmap(to_read_file, dtype=to_read_storage, mode='r+')
    global sent_start_map_marisa_global
    sent_start_map_marisa_global = utils.load_single_item_trie(sent_start_map_file)

def merge_subsentences_hlp(args):
    M, K, hidden_size, dump_embs, r_idx_set = args
    seen_ids = set()
    for r_idx in tqdm(r_idx_set):
        row = full_pred_data_global[r_idx]
        # get corresponding row to start writing into condensed memory mapped file
        sent_idx = str(row['sent_idx'])
        sent_start_idx = sent_start_map_marisa_global[sent_idx][0][0]
        # for each VALID mention, need to write into original alias list pos in list
        for i, (true_val, alias_orig_pos) in enumerate(zip(row['final_loss_true'], row['alias_list_pos'])):
            # bc we are are using the mentions which includes both true and false golds, true_val == -1 only for padded mentions or sub-sentence mentions
            if true_val != -1:
                # id in condensed embedding
                emb_id = sent_start_idx + alias_orig_pos
                assert emb_id not in seen_ids, f'{emb_id} already seen, something went wrong with sub-sentences'
                if dump_embs:
                    filt_emb_data_global['entity_emb'][emb_id] = row['entity_emb'].reshape(M, hidden_size)[i]
                filt_emb_data_global['sent_idx'][emb_id] = sent_idx
                filt_emb_data_global['alias_list_pos'][emb_id] = alias_orig_pos
                filt_emb_data_global['final_loss_pred'][emb_id] = row['final_loss_pred'].reshape(M)[i]
                filt_emb_data_global['final_loss_prob'][emb_id] = row['final_loss_prob'].reshape(M)[i]
                filt_emb_data_global['final_loss_cand_probs'][emb_id] = row['final_loss_cand_probs'].reshape(M, K)[i]
    return seen_ids

# get sent_idx, alias_idx mapping to emb idx for quick lookup
def get_sent_idx_map(merged_entity_emb_file, merged_storage_type):
    """ Get sent_idx, alias_idx mapping to emb idx for quick lookup """
    filt_emb_data = np.memmap(merged_entity_emb_file, dtype=merged_storage_type, mode="r+")
    sent_idx_map = {}
    for i, row in enumerate(filt_emb_data):
        sent_idx = row['sent_idx']
        alias_idx = row['alias_list_pos']
        assert sent_idx != -1 and alias_idx != -1, f"Sent {sent_idx}, Al {alias_idx}"
        # string for Marisa Trie later
        sent_idx_map[f"{sent_idx}_{alias_idx}"] = i
    return sent_idx_map

# write new data with qids and entity emb ids
def write_data_labels(num_processes, merged_entity_emb_file,
                      merged_storage_type, data_file,
                      out_file, train_in_candidates, dump_embs, data_config):
    logger = logging.getLogger(__name__)

    # Get sent mapping
    start = time.time()
    sent_idx_map = get_sent_idx_map(merged_entity_emb_file, merged_storage_type)
    sent_idx_map_file = tempfile.NamedTemporaryFile(suffix="bootleg_sent_idx_map")
    utils.create_single_item_trie(sent_idx_map, out_file=sent_idx_map_file.name)

    # Chunk file for parallel writing
    create_ex_indir = tempfile.TemporaryDirectory()
    create_ex_outdir = tempfile.TemporaryDirectory()
    logger.debug(f"Counting lines")
    total_input = sum(1 for _ in open(data_file))
    chunk_input = int(np.ceil(total_input/num_processes))
    logger.debug(f"Chunking up {total_input} lines into subfiles of size {chunk_input} lines")
    total_input_from_chunks, input_files_dict = utils.chunk_file(data_file, create_ex_indir.name, chunk_input)

    input_files = list(input_files_dict.keys())
    input_file_lines = [input_files_dict[k] for k in input_files]
    output_files = [in_file_name.replace(create_ex_indir.name, create_ex_outdir.name) for in_file_name in input_files]
    assert total_input == total_input_from_chunks, f"Lengths of files {total_input} doesn't mathc {total_input_from_chunks}"
    logger.debug(f"Done chunking files")

    logger.info(f'Starting to write files with {num_processes} processes')
    pool = multiprocessing.Pool(processes=num_processes,
                                initializer=write_data_labels_initializer,
                                initargs=[
                                    merged_entity_emb_file,
                                    merged_storage_type,
                                    sent_idx_map_file.name,
                                    train_in_candidates,
                                    dump_embs,
                                    data_config
                                ])

    input_args = list(zip(input_files, input_file_lines, output_files))
    # Store output files and counts for saving in next step
    total = 0
    for res in pool.imap(
                    write_data_labels_hlp,
                    input_args,
                    chunksize=1
                ):
        total += 1

    # Merge output files to final file
    logger.debug(f"Merging output files")
    with open(out_file, 'wb') as outfile:
        for filename in glob.glob(os.path.join(create_ex_outdir.name, "*")):
            if filename == out_file:
                # don't want to copy the output into the output
                continue
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)
    sent_idx_map_file.close()
    create_ex_indir.cleanup()
    create_ex_outdir.cleanup()
    logger.info(f'Time to write files {time.time()-start}s')

def write_data_labels_initializer(merged_entity_emb_file, merged_storage_type, sent_idx_map_file, train_in_candidates, dump_embs, data_config):
    global filt_emb_data_global
    filt_emb_data_global = np.memmap(merged_entity_emb_file, dtype=merged_storage_type, mode="r+")
    global sent_idx_map_global
    sent_idx_map_global = utils.load_single_item_trie(sent_idx_map_file)
    global train_in_candidates_global
    train_in_candidates_global = train_in_candidates
    global dump_embs_global
    dump_embs_global = dump_embs
    global entity_dump_global
    entity_dump_global = EntitySymbols(load_dir=os.path.join(data_config.entity_dir, data_config.entity_map_dir),
                                       alias_cand_map_file=data_config.alias_cand_map)


def write_data_labels_hlp(args):
    input_file, input_lines, output_file = args
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in tqdm(f_in, total=input_lines, desc="Writing data"):
            line = ujson.loads(line)
            aliases = line['aliases']
            sent_idx = line['sent_idx_unq']
            qids = []
            cand_qids = []
            ctx_emb_ids = []
            entity_ids = []
            cand_entity_ids = []
            probs = []
            cand_probs = []
            entity_cands_qid = map_aliases_to_candidates(train_in_candidates_global, entity_dump_global, aliases)
            # eid is entity id
            entity_cands_eid = map_aliases_to_candidates_eid(train_in_candidates_global, entity_dump_global, aliases)
            for al_idx, alias in enumerate(aliases):
                sent_idx_key = f"{sent_idx}_{al_idx}"
                assert sent_idx_key in sent_idx_map_global, f'Dumped prediction data does not match data file. Can not find {sent_idx} - {al_idx}'
                emb_idx = sent_idx_map_global[sent_idx_key][0][0]
                ctx_emb_ids.append(emb_idx)

                prob = filt_emb_data_global[emb_idx]['final_loss_prob']
                cand_prob = filt_emb_data_global[emb_idx]['final_loss_cand_probs']
                pred_cand = filt_emb_data_global[emb_idx]['final_loss_pred']
                
                # sort predicted cands based on cand_probs
                packed_list = zip(cand_prob, entity_cands_eid[al_idx], entity_cands_qid[al_idx])
                packed_list_sorted = sorted(packed_list, key=lambda tup: tup[0], reverse=True)
                cand_prob, cand_entity_id, cand_qid = list(zip(*packed_list_sorted))
                
                eid = entity_cands_eid[al_idx][pred_cand]
                entity_ids.append(eid)
                cand_entity_ids.append(cand_entity_id)
                qid = entity_cands_qid[al_idx][pred_cand]
                qids.append(qid)
                cand_qids.append(cand_qid)
                probs.append(prob)
                cand_probs.append(list(cand_prob))
                
            line['qids'] = qids
            line['probs'] = probs
            line['entity_ids'] = entity_ids
            line['cands'] = cand_qids
            line['cand_probs'] = cand_probs
            line['cand_entity_ids'] = cand_entity_ids
            
            if dump_embs_global:
                line['ctx_emb_ids'] = ctx_emb_ids
            f_out.write(ujson.dumps(line) + "\n")
