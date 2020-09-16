from collections import defaultdict, OrderedDict
import jsonlines
import numpy as np
import os
import tagme
import torch
import ujson

import pandas as pd
pd.options.display.max_colwidth = 500

from bootleg.utils import data_utils, sentence_utils, eval_utils
from bootleg.utils.parser_utils import get_full_config
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.alias_entity_table import AliasEntityTable
from bootleg.symbols.constants import *
from bootleg.model import Model
from bootleg.extract_mentions import find_aliases_in_sentence_tag, get_all_aliases
from bootleg.utils.utils import import_class

def load_predictions(file):
    lines = {}
    with jsonlines.open(file) as f:
        for line in f:
            lines[line['sent_idx_unq']] = line
    return lines

def score_predictions(orig_file, pred_file, entity_mapping_dir, type_symbols=[]):
    title_mapping = ujson.load(open(os.path.join(entity_mapping_dir, 'qid2title.json')))
    preds = load_predictions(pred_file)
    correct = 0
    total = 0
    rows = []
    with jsonlines.open(orig_file) as f:
        for line in f:
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
                    'aliases': line['aliases'],
                    'alias': line['aliases'][alias_idx],
                    'alias_idx': alias_idx,
                    'gold_qid': gold_qids[alias_idx],
                    'pred_qid': pred_qids[alias_idx],
                    'gold_title': title_mapping[gold_qids[alias_idx]],
                    'pred_title': title_mapping[pred_qids[alias_idx]]
                }
                for type_sym in type_symbols:
                    type_nm = os.path.basename(os.path.splitext(type_sym.type_file)[0])
                    gold_types = type_sym.get_types(gold_qids[alias_idx])
                    pred_types = type_sym.get_types(pred_qids[alias_idx])
                    res[f"{type_nm}_gld"] = gold_types
                    res[f"{type_nm}_pred"] = pred_types
                rows.append(res)
    return pd.DataFrame(rows)

def load_mentions(file):
    lines = []
    with jsonlines.open(file) as f:
        for line in f:
            new_line = {
                'sentence': line['sentence'],
                'aliases': line['aliases'],
                'spans': line['spans']
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

    print(f'Recall: {round(correct_mentions/total_mentions, 2)} ({correct_mentions}/{total_mentions})')
    print(f'Precision: {round(correct_mentions/pred_mentions, 2)} ({correct_mentions}/{pred_mentions})')
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
