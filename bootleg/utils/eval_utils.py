from collections import  OrderedDict

import logging
import jsonlines
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
import time
import torch

from bootleg.symbols.constants import *
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
        # False is for if it's test or not
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
    for i, batch in enumerate(dataloader):
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
    for train_head in train_head_keys:
        # do not include stage outputs in dump
        if 'stage' in train_head:
            continue
        eval_slice_name_for_head = train_utils.get_inv_head_name_eval(train_head)
        for eval_slice in eval_slice_keys:
            # base head and overall slice are the same
            if eval_slice == BASE_SLICE:
                continue
            all_dev_results[train_head][eval_slice]['step'] = global_step
            dict_for_printing = batch_eval_pretty_json(all_dev_results, train_head, eval_slice, topk_val=args.run_config.topk)
            if eval_slice == eval_slice_name_for_head or train_head == train_utils.get_slice_head_eval_name(BASE_SLICE) \
                    or FINAL_LOSS in train_head or eval_slice == FINAL_LOSS or args.run_config.verbose:
                # Order the columns
                eval_columns = get_eval_columns(topk=args.run_config.topk, is_sbl_model=is_sbl_model, is_test=is_test)
                ordered_dict = OrderedDict((eval_columns[k], dict_for_printing[k]) for k in eval_columns)
                pretty_dict_for_printing.append(ordered_dict)
    logger.info(f'\n{tabulate(pretty_dict_for_printing, headers="keys", tablefmt="grid")}')
    if status_reporter is not None:
        status_reporter.dump_results(all_dev_results, pretty_dict_for_printing, file, is_test)
    return all_dev_results

def select_embs(embs, pred_cands, batch_size, M):
    return embs[torch.arange(batch_size).unsqueeze(-1), torch.arange(M).unsqueeze(0), pred_cands]

def run_dump_preds(args, test_data_file, trainer, dataloader, logger, entity_symbols, dump_embs=False):
    # we only care about the entity embeddings for the final slice head
    eval_folder = train_utils.get_eval_folder(args, test_data_file)
    utils.ensure_dir(eval_folder)

    # write to file (M x hidden x size for each data point -- next step will deal with recovering original sentence indices for overflowing sentences)
    test_file_tag = test_data_file.split('.jsonl')[0]
    entity_emb_file = os.path.join(eval_folder, f'{test_file_tag}_entity_embs.pt')
    emb_file_config = entity_emb_file.split('.pt')[0] + '_config'
    M = args.data_config.max_aliases
    K = entity_symbols.max_candidates + (not args.data_config.train_in_candidates)
    # TODO: fix extra dimension issue
    if dump_embs:
        storage_type = np.dtype([('M', int), ('K', int), ('hidden_size', int), ('sent_idx', int), ('subsent_idx', int), ('alias_list_pos', int, M), ('entity_emb', float, M*args.model_config.hidden_size),
        ('final_loss_true', int, M), ('final_loss_pred', int, M), ('final_loss_prob', float, M)])
    else:
        # don't need to extract contextualized entity embedding
        storage_type = np.dtype([('M', int), ('K', int), ('hidden_size', int), ('sent_idx', int), ('subsent_idx', int), ('alias_list_pos', int, M), ('final_loss_true', int, M), ('final_loss_pred', int, M), ('final_loss_prob', float, M)])
    mmap_file = np.memmap(entity_emb_file, dtype=storage_type, mode='w+', shape=(len(dataloader.dataset),))
    np.save(emb_file_config, storage_type, allow_pickle=True)
    logger.debug(f'Created file {entity_emb_file} to save predictions.')

    start_idx = 0
    logger.info(f'{len(dataloader)*args.run_config.eval_batch_size} samples, {len(dataloader)} batches')
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

        # TODO: rename this
        mmap_file[start_idx:end_idx][f'final_loss_true'] = batch['true_entity_idx_for_train'].reshape(curr_batch_size, M).cpu().numpy()
        # mmap_file[start_idx:end_idx][f'final_loss_true'] = batch[train_utils.get_slice_head_pred_name(FINAL_LOSS)][j].flatten().cpu().numpy()

        # get max for each alias, probs is batch x M x K
        max_probs, pred_cands = probs.max(dim=2)

        mmap_file[start_idx:end_idx]['final_loss_pred'] = pred_cands.cpu().numpy()
        mmap_file[start_idx:end_idx]['final_loss_prob'] = max_probs.cpu().numpy()

        # final_entity_embs is batch x M x K x hidden_size, pred_cands in batch x M
        if dump_embs:
            chosen_entity_embs = select_embs(embs=final_entity_embs, pred_cands=pred_cands,
                batch_size=curr_batch_size, M=M)

            # write chosen entity embs to file for contextualized entity embeddings
            mmap_file[start_idx:end_idx]['entity_emb'] = chosen_entity_embs.reshape(curr_batch_size,-1).cpu().numpy()

        start_idx += curr_batch_size
        if i % 100 == 0 and i != 0:
            logger.info(f'Saved {i} batches of predictions')

    # restitch together and write data file
    result_file = os.path.join(eval_folder, args.run_config.result_label_file)
    logger.info(f'Writing predictions...')
    filt_pred_data = merge_subsentences(os.path.join(args.data_config.data_dir, test_data_file),
        mmap_file,
        dump_embs=dump_embs)
    sent_idx_map = get_sent_idx_map(filt_pred_data)

    write_data_labels(filt_pred_data=filt_pred_data, data_file=os.path.join(args.data_config.data_dir, test_data_file), out_file=result_file,
        sent_idx_map=sent_idx_map, entity_dump=entity_symbols, train_in_candidates=args.data_config.train_in_candidates, dump_embs=dump_embs)

    out_emb_file = None
    # save easier-to-use embedding file
    if dump_embs:
        hidden_size = filt_pred_data[0]['hidden_size']
        out_emb_file = os.path.join(eval_folder, args.run_config.result_emb_file)
        np.save(out_emb_file, filt_pred_data['entity_emb'].reshape(-1,hidden_size))
        logger.info(f'Saving contextual entity embeddings to {out_emb_file}')

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
            sent_start_map[line['sent_idx_unq']] = total_num_mentions
            # include false aliases for debugging (and alias_pos includes them)
            total_num_mentions += len(line['aliases'])
    logging.info(f'Total number of mentions across all sentences: {total_num_mentions}')
    return sent_start_map, total_num_mentions

# stich embs back together over sub-sentences, convert from sent_idx array to (sent_idx, alias_idx) array with varying numbers of aliases per sentence
# TODO: parallelize this
def merge_subsentences(data_file, full_pred_data, dump_embs=False):
    sent_start_map, total_num_mentions = get_sent_start_map(data_file)
    M = int(full_pred_data[0]['M'])
    hidden_size = int(full_pred_data[0]['hidden_size'])
    if dump_embs:
        storage_type = np.dtype([('hidden_size', int),
                             ('sent_idx', int),
                             ('alias_list_pos', int),
                             ('entity_emb', float, hidden_size),
                             ('final_loss_pred', int),
                             ('final_loss_prob', float)])
    else:
        storage_type = np.dtype([('hidden_size', int),
                             ('sent_idx', int),
                             ('alias_list_pos', int),
                             ('final_loss_pred', int),
                             ('final_loss_prob', float)])
    filt_emb_data = np.zeros(total_num_mentions, dtype=storage_type)
    filt_emb_data['hidden_size'] = hidden_size
    start = time.time()
    seen_ids = set()
    seen_data = {}
    for row in full_pred_data:
        # get corresponding row to start writing into condensed memory mapped file
        sent_idx = row['sent_idx']
        sent_start_idx = sent_start_map[sent_idx]
        # for each VALID mention, need to write into original alias list pos in list
        for i, (true_val, alias_orig_pos) in enumerate(zip(row['final_loss_true'], row['alias_list_pos'])):
            # bc we are are using the gold mentions which include false anchors, true_val == -1 only for padded mentions or sub-sentence mentions
            if true_val != -1:
                # id in condensed embedding
                emb_id = sent_start_idx + alias_orig_pos
                assert emb_id not in seen_ids, f'{emb_id} already seen, something went wrong with sub-sentences'
                if dump_embs:
                    filt_emb_data['entity_emb'][emb_id] = row['entity_emb'].reshape(M, hidden_size)[i]
                filt_emb_data['sent_idx'][emb_id] = sent_idx
                filt_emb_data['alias_list_pos'][emb_id] = alias_orig_pos
                filt_emb_data['final_loss_pred'][emb_id] = row['final_loss_pred'].reshape(M)[i]
                filt_emb_data['final_loss_prob'][emb_id] = row['final_loss_prob'].reshape(M)[i]
                seen_ids.add(emb_id)
                seen_data[emb_id] = row
    logging.debug(f'Time to merge sub-sentences {time.time()-start}s')
    return filt_emb_data

# get sent_idx, alias_idx mapping to emb idx for quick lookup
def get_sent_idx_map(filt_emb_data):
    sent_idx_map = {}
    for i, row in enumerate(filt_emb_data):
        sent_idx = row['sent_idx']
        alias_idx = row['alias_list_pos']
        sent_idx_map[(sent_idx, alias_idx)] = i
    return sent_idx_map

# write new data with qids and entity emb ids
# TODO: parallelize this
def write_data_labels(filt_pred_data, data_file, out_file, sent_idx_map,
    entity_dump, train_in_candidates, dump_embs):
    with jsonlines.open(data_file) as f_in, jsonlines.open(out_file, 'w') as f_out:
        for line in f_in:
            aliases = line['aliases']
            sent_idx = line['sent_idx_unq']
            qids = []
            ctx_emb_ids = []
            entity_ids = []
            probs = []
            entity_cands_qid = map_aliases_to_candidates(train_in_candidates, entity_dump, aliases)
            # eid is entity id
            entity_cands_eid = map_aliases_to_candidates_eid(train_in_candidates, entity_dump, aliases)
            for al_idx, alias in enumerate(aliases):
                assert (sent_idx, al_idx) in sent_idx_map, 'Dumped prediction data does not match data file'
                emb_idx = sent_idx_map[(sent_idx, al_idx)]
                ctx_emb_ids.append(emb_idx)
                prob = filt_pred_data[emb_idx]['final_loss_prob']
                pred_cand = filt_pred_data[emb_idx]['final_loss_pred']
                eid = entity_cands_eid[al_idx][pred_cand]
                qid = entity_cands_qid[al_idx][pred_cand]
                qids.append(qid)
                probs.append(prob)
                entity_ids.append(eid)
            line['qids'] = qids
            line['probs'] = probs
            line['entity_ids'] = entity_ids
            if dump_embs:
                line['ctx_emb_ids'] = ctx_emb_ids
            f_out.write(line)
    logging.info(f'Finished writing predictions to {out_file}')