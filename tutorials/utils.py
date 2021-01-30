from collections import defaultdict
import jsonlines
import numpy as np
import os
import tagme
import ujson

import pandas as pd
from tqdm import tqdm

pd.options.display.max_colwidth = 500

from bootleg.symbols.constants import *

def copy_candidates(from_alias, to_alias, alias2qids, max_candidates=30, qids_to_add=None):
    """This will copy the candidates from from_alias to to_alias. We assume to_alias does not exist and from_alias does exists.
    qids_to_add will be added to the beginning of the candidate list to ensure they are among the top 30"""
    if qids_to_add is None:
        qids_to_add = []
    assert from_alias in alias2qids, f"The from_alias {from_alias} must be in the alias2qids mapping. Use add_new_alias command from a new alias"
    assert to_alias not in alias2qids, f"The to_alias {to_alias} must not be in alias2qids."
    candidates = alias2qids[from_alias]
    # Add the qids to add to candidates. As the user wants these qids, give them the highest score
    if len(qids_to_add) > 0:
        top_score = candidates[0][1]
        new_candidates = [[q, top_score] for q in qids_to_add]
        candidates = new_candidates + candidates
        if len(candidates) > max_candidates:
            print(f"Filtering candidates down to top {max_candidates}")
            candidates = candidates[:max_candidates]
    alias2qids[to_alias] = candidates
    return alias2qids


def add_new_alias(new_alias, alias2qids, qids_to_add, max_candidates=30):
    assert new_alias not in alias2qids, f"The new_alias {new_alias} must not be in alias2qids."
    # Assign each qid a score of 1.0
    candidates = [[q, 1.0] for q in qids_to_add]
    if len(candidates) > max_candidates:
        print(f"Filtering candidates down to top {max_candidates}")
        candidates = candidates[:max_candidates]
    alias2qids[new_alias] = candidates
    return alias2qids

def load_title_map(entity_mapping_dir):
    return ujson.load(open(os.path.join(entity_mapping_dir, 'qid2title.json')))

def load_cand_map(entity_mapping_dir, alias_map_file):
    return ujson.load(open(os.path.join(entity_mapping_dir, alias_map_file)))

def load_predictions(file):
    lines = {}
    with jsonlines.open(file) as f:
        for line in f:
            lines[line['sent_idx_unq']] = line
    return lines

def score_predictions(orig_file, pred_file, title_map, cands_map=None, type_symbols=None, kg_symbols=None):
    """Loads a jsonl file and joins with the results from dump_preds"""
    if cands_map is None:
        cands_map = {}
    if type_symbols is None:
        type_symbols = []
    if kg_symbols is None:
        kg_symbols = []
    num_lines = sum(1 for line in open(orig_file))
    preds = load_predictions(pred_file)
    correct = 0
    total = 0
    rows = []
    with jsonlines.open(orig_file) as f:
        for line in tqdm(f, total=num_lines):
            sent_idx = line['sent_idx_unq']
            gold_qids = line['qids']
            pred_qids = preds[sent_idx]['qids']
            assert len(gold_qids) == len(pred_qids), 'Gold and pred QIDs have different lengths'
            correct += np.sum([gold_qid == pred_qid for gold_qid, pred_qid in zip(gold_qids, pred_qids)])
            total += len(gold_qids)
            # for each alias, append a row in the merged result table
            for alias_idx in range(len(gold_qids)):
                res = {
                    'sentence': line['sentence'],
                    'sent_idx': line['sent_idx_unq'],
                    'aliases': line['aliases'],
                    'span': line['spans'][alias_idx],
                    'slices': line.get('slices', {}),
                    'alias': line['aliases'][alias_idx],
                    'alias_idx': alias_idx,
                    'is_gold_label': line['gold'][alias_idx],
                    'gold_qid': gold_qids[alias_idx],
                    'pred_qid': pred_qids[alias_idx],
                    'gold_title': title_map[gold_qids[alias_idx]],
                    'pred_title': title_map[pred_qids[alias_idx]],
                    'all_gold_qids': gold_qids,
                    'all_pred_qids': pred_qids,
                    'gold_label_aliases': [al for i, al in enumerate(line['aliases']) if line['gold'][i] is True],
                    'all_is_gold_labels': line['gold'],
                    'all_spans': line['spans']
                }
                slices = []
                if 'slices' in line:
                    for sl_name in line['slices']:
                        if str(alias_idx) in line['slices'][sl_name] and line['slices'][sl_name][str(alias_idx)] > 0.5:
                            slices.append(sl_name)
                res['slices'] = slices
                if len(cands_map) > 0:
                    res["cands"] = [tuple([title_map[q[0]], preds[sent_idx]["cand_probs"][alias_idx][i]]) for i, q in enumerate(cands_map[line['aliases'][alias_idx]])]
                for type_sym in type_symbols:
                    type_nm = os.path.basename(os.path.splitext(type_sym.type_file)[0])
                    gold_types = type_sym.get_types(gold_qids[alias_idx])
                    pred_types = type_sym.get_types(pred_qids[alias_idx])
                    res[f"{type_nm}_gld"] = gold_types
                    res[f"{type_nm}_pred"] = pred_types
                for kg_sym in kg_symbols:
                    kg_nm = os.path.basename(os.path.splitext(kg_sym.kg_adj_file)[0])
                    connected_pairs_gld = []
                    connected_pairs_pred = []
                    for alias_idx2 in range(len(gold_qids)):
                        if kg_sym.is_connected(gold_qids[alias_idx], gold_qids[alias_idx2]):
                            connected_pairs_gld.append(gold_qids[alias_idx2])
                        if kg_sym.is_connected(pred_qids[alias_idx], pred_qids[alias_idx2]):
                            connected_pairs_pred.append(pred_qids[alias_idx2])
                    res[f"{kg_nm}_gld"] = connected_pairs_gld
                    res[f"{kg_nm}_pred"] = connected_pairs_gld
                rows.append(res)
    return pd.DataFrame(rows)

def load_mentions(file):
    lines = []
    with jsonlines.open(file) as f:
        for line in f:
            new_line = {
                'sentence': line['sentence'],
                'aliases': line.get('aliases', []),
                'spans': line.get('spans', [])
            }
            lines.append(new_line)
    return pd.DataFrame(lines)

def create_error(sent_obj, gold_aliases, gold_qids, gold_spans, found_aliases, found_spans, pred_qids, pred_probs, error):
    return {
        "sent_idx": sent_obj["sent_idx_unq"],
        "sentence": sent_obj["sentence"],
        "gold_aliases": gold_aliases,
        "gold_qids": gold_qids,
        "gold_spans": gold_spans,
        "pred_aliases": found_aliases,
        "pred_spans": found_spans,
        "pred_qids": pred_qids,
        "pred_probs": pred_probs,
        "error": error
    }

def compute_precision_and_recall(orig_label_file, new_label_file, threshold=None):
    # read in first file and map by index for fast retrieval
    total_mentions = 0
    correct_mentions = 0
    pred_mentions = 0

    new_labels = {}
    with jsonlines.open(new_label_file) as f:
        for line in f:
            new_labels[line['sent_idx_unq']] = line

    errors = defaultdict(list)
    with jsonlines.open(orig_label_file) as f:
        for line in f:
            gold_aliases = line['aliases']
            gold_spans = line['spans']
            gold_qids = line['qids']

            pred_vals = new_labels[line['sent_idx_unq']]
            pred_aliases = pred_vals['aliases']
            pred_spans = pred_vals['spans']
            pred_qids = pred_vals['qids']
            pred_probs = [round(p, 3) for p in pred_vals['probs']]
            if threshold is not None:
                new_pred_qids = []
                for pred_qid, pred_prob in zip(pred_qids, pred_probs):
                    if pred_prob < threshold:
                        new_pred_qids.append('NC')
                    else:
                        new_pred_qids.append(pred_qid)
                pred_qids = new_pred_qids

            total_mentions += len(gold_aliases)

            # predicted mentions are only those that aren't nil ('NC')
            pred_mentions += sum([pred_qid != 'NC' for pred_qid in pred_qids])

            for gold_alias, gold_qid, gold_span in zip(gold_aliases, gold_qids, gold_spans):
                gold_span_start, gold_span_end = gold_span
                fuzzy_gold_left = [gold_span_start-1,gold_span_end]
                fuzzy_gold_right = [gold_span_start+1,gold_span_end]
                if gold_span in pred_spans or fuzzy_gold_left in pred_spans or fuzzy_gold_right in pred_spans:
                    if gold_span in pred_spans:
                        pred_idx = pred_spans.index(gold_span)
                    elif fuzzy_gold_left in pred_spans:
                        pred_idx = pred_spans.index(fuzzy_gold_left)
                    elif fuzzy_gold_right in pred_spans:
                        pred_idx = pred_spans.index(fuzzy_gold_right)

                    if gold_qid == pred_qids[pred_idx]:
                        correct_mentions += 1
                    # could not find a label for the mention
                    elif pred_qids[pred_idx] == 'NC':
                        errors['missing_mention'].append(create_error(line, gold_aliases, gold_qids,
                                                                gold_spans, pred_aliases, pred_spans,
                                                                 pred_qids, pred_probs, error=gold_alias))
                    else:
                        errors['wrong_entity'].append(create_error(line, gold_aliases,
                                                                gold_qids, gold_spans, pred_aliases, pred_spans,
                                                                pred_qids, pred_probs, error=gold_alias))
                else:
                    errors['missing_mention'].append(create_error(line, gold_aliases, gold_qids,
                                                                gold_spans, pred_aliases, pred_spans,
                                                                 pred_qids, pred_probs, error=gold_alias))

            for pred_alias, pred_span, pred_qid in zip(pred_aliases, pred_spans, pred_qids):
                if pred_qid == 'NC':
                    errors['NC'].append(create_error(line, gold_aliases, gold_qids, gold_spans, pred_aliases, pred_spans,
                                                              pred_qids, pred_probs, error=''))
                pred_span_start, pred_span_end = pred_span
                fuzzy_gold_left = [pred_span_start-1,pred_span_end]
                fuzzy_gold_right = [pred_span_start+1,pred_span_end]
                if pred_span not in gold_spans and fuzzy_gold_left not in gold_spans and fuzzy_gold_right not in gold_spans and pred_qid != 'NC':
                    errors['extra_mention'].append(create_error(line, gold_aliases, gold_qids, gold_spans, pred_aliases, pred_spans,
                                                              pred_qids, pred_probs, error=pred_alias))

    rec = correct_mentions/total_mentions
    prec = correct_mentions/pred_mentions
    print(f'Recall: {round(rec, 2)} ({correct_mentions}/{total_mentions})')
    print(f'Precision: {round(prec, 2)} ({correct_mentions}/{pred_mentions})')
    print(f'F1: {round(2*((prec*rec)/(prec+rec)), 2)}')
    return errors


def tagme_annotate(in_file, out_file, threshold=0.1, wpid2qid=None):
    with jsonlines.open(in_file) as f_in, jsonlines.open(out_file, 'w') as f_out:
        for line in f_in:
            aliases = []
            spans = []
            qids = []
            probs = []
            text = line['sentence']
            text_spans = text.split()
            text_span_indices = []
            total_len = 0
            for i,t in enumerate(text_spans):
                text_span_indices.append(total_len)
                total_len += len(t)+1
            lunch_annotations = tagme.annotate(text)

            # as the threshold increases, the precision increases, but the recall decreases
            for ann in lunch_annotations.get_annotations(threshold):
                mention = ann.mention
                qid = wpid2qid[str(ann.entity_id)]
                span_start = text_span_indices.index(ann.begin)
                try:
                    span_end = text_span_indices.index(ann.end+1)
                except:
                    span_end = len(text_spans)
                aliases.append(mention)
                spans.append([span_start, span_end])
                qids.append(qid)
                probs.append(ann.score)

            line['aliases'] = aliases
            line['qids'] = qids
            line['spans'] = spans
            line['probs'] = probs
            line[ANCHOR_KEY] = [True for _ in aliases]
            f_out.write(line)
