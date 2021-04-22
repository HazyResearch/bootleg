import os

import jsonlines
import numpy as np
import pandas as pd
import requests
import tagme
import ujson
from tqdm import tqdm

from bootleg.symbols.entity_profile import EntityProfile

pd.options.display.max_colwidth = 500


def load_train_data(train_file, title_map, entity_profile=None):
    """Loads a jsonl file and creates a pandas DataFrame. Adds candidates, types, and KGs if available."""
    num_lines = sum(1 for _ in open(train_file))
    rows = []
    with jsonlines.open(train_file) as f:
        for line in tqdm(f, total=num_lines):
            gold_qids = line["qids"]
            # for each alias, append a row in the merged result table
            for alias_idx in range(len(gold_qids)):
                res = {
                    "sentence": line["sentence"],
                    "sent_idx": line["sent_idx_unq"],
                    "aliases": line["aliases"],
                    "span": line["spans"][alias_idx],
                    "slices": line.get("slices", {}),
                    "alias": line["aliases"][alias_idx],
                    "alias_idx": alias_idx,
                    "is_gold_label": line["gold"][alias_idx],
                    "gold_qid": gold_qids[alias_idx],
                    "gold_title": title_map[gold_qids[alias_idx]]
                    if gold_qids[alias_idx] != "Q-1"
                    else "Q-1",
                    "all_gold_qids": gold_qids,
                    "gold_label_aliases": [
                        al
                        for i, al in enumerate(line["aliases"])
                        if line["gold"][i] is True
                    ],
                    "all_is_gold_labels": line["gold"],
                    "all_spans": line["spans"],
                }
                slices = []
                if "slices" in line:
                    for sl_name in line["slices"]:
                        if (
                            str(alias_idx) in line["slices"][sl_name]
                            and line["slices"][sl_name][str(alias_idx)] > 0.5
                        ):
                            slices.append(sl_name)
                res["slices"] = slices
                if entity_profile is not None:
                    res["cand_names"] = [
                        title_map[q[0]]
                        for i, q in enumerate(
                            entity_profile.get_qid_count_cands(
                                line["aliases"][alias_idx]
                            )
                        )
                    ]
                    res["cand_qids"] = [
                        q[0]
                        for i, q in enumerate(
                            entity_profile.get_qid_count_cands(
                                line["aliases"][alias_idx]
                            )
                        )
                    ]
                    for type_sym in entity_profile.get_all_typesystems():
                        gold_types = entity_profile.get_types(
                            gold_qids[alias_idx], type_sym
                        )
                        res[f"{type_sym}_gld"] = gold_types

                    connected_pairs_gld = []
                    for alias_idx2 in range(len(gold_qids)):
                        if entity_profile.is_connected(
                            gold_qids[alias_idx], gold_qids[alias_idx2]
                        ):
                            connected_pairs_gld.append(gold_qids[alias_idx2])
                    res[f"kg_gld"] = connected_pairs_gld
                rows.append(res)
    return pd.DataFrame(rows)


def load_title_map(entity_dir, entity_mapping_dir="entity_mappings"):
    return ujson.load(
        open(os.path.join(entity_dir, entity_mapping_dir, "qid2title.json"))
    )


def load_cand_map(entity_dir, alias_map_file, entity_mapping_dir="entity_mappings"):
    return ujson.load(
        open(os.path.join(entity_dir, entity_mapping_dir, alias_map_file))
    )


def load_predictions(file):
    lines = {}
    with jsonlines.open(file) as f:
        for line in f:
            lines[line["sent_idx_unq"]] = line
    return lines


def score_predictions(
    orig_file, pred_file, title_map, entity_profile: EntityProfile = None
):
    """Loads a jsonl file and joins with the results from dump_preds"""
    num_lines = sum(1 for line in open(orig_file))
    preds = load_predictions(pred_file)
    correct = 0
    total = 0
    rows = []
    with jsonlines.open(orig_file) as f:
        for line in tqdm(f, total=num_lines):
            sent_idx = line["sent_idx_unq"]
            gold_qids = line["qids"]
            pred_qids = preds[sent_idx]["qids"]
            assert len(gold_qids) == len(
                pred_qids
            ), "Gold and pred QIDs have different lengths"
            correct += np.sum(
                [
                    gold_qid == pred_qid
                    for gold_qid, pred_qid in zip(gold_qids, pred_qids)
                ]
            )
            total += len(gold_qids)
            # for each alias, append a row in the merged result table
            for alias_idx in range(len(gold_qids)):
                res = {
                    "sentence": line["sentence"],
                    "sent_idx": line["sent_idx_unq"],
                    "aliases": line["aliases"],
                    "span": line["spans"][alias_idx],
                    "slices": line.get("slices", {}),
                    "alias": line["aliases"][alias_idx],
                    "alias_idx": alias_idx,
                    "is_gold_label": line["gold"][alias_idx],
                    "gold_qid": gold_qids[alias_idx],
                    "pred_qid": pred_qids[alias_idx],
                    "gold_title": title_map[gold_qids[alias_idx]]
                    if gold_qids[alias_idx] != "Q-1"
                    else "Q-1",
                    "pred_title": title_map.get(pred_qids[alias_idx], "CouldnotFind")
                    if pred_qids[alias_idx] != "NC"
                    else "NC",
                    "all_gold_qids": gold_qids,
                    "all_pred_qids": pred_qids,
                    "gold_label_aliases": [
                        al
                        for i, al in enumerate(line["aliases"])
                        if line["gold"][i] is True
                    ],
                    "all_is_gold_labels": line["gold"],
                    "all_spans": line["spans"],
                }
                slices = []
                if "slices" in line:
                    for sl_name in line["slices"]:
                        if (
                            str(alias_idx) in line["slices"][sl_name]
                            and line["slices"][sl_name][str(alias_idx)] > 0.5
                        ):
                            slices.append(sl_name)
                res["slices"] = slices
                if entity_profile is not None:
                    res["cands"] = [
                        tuple(
                            [
                                title_map[q[0]],
                                preds[sent_idx]["cand_probs"][alias_idx][i],
                            ]
                        )
                        for i, q in enumerate(
                            entity_profile.get_qid_count_cands(
                                line["aliases"][alias_idx]
                            )
                        )
                    ]
                for type_sym in entity_profile.get_all_typesystems():
                    gold_types = entity_profile.get_types(
                        gold_qids[alias_idx], type_sym
                    )
                    pred_types = entity_profile.get_types(
                        pred_qids[alias_idx], type_sym
                    )
                    res[f"{type_sym}_gld"] = gold_types
                    res[f"{type_sym}_pred"] = pred_types

                connected_pairs_gld = []
                connected_pairs_pred = []
                for alias_idx2 in range(len(gold_qids)):
                    if entity_profile.is_connected(
                        gold_qids[alias_idx], gold_qids[alias_idx2]
                    ):
                        connected_pairs_gld.append(gold_qids[alias_idx2])
                    if entity_profile.is_connected(
                        pred_qids[alias_idx], pred_qids[alias_idx2]
                    ):
                        connected_pairs_pred.append(pred_qids[alias_idx2])
                res[f"kg_gld"] = connected_pairs_gld
                res[f"kg_pred"] = connected_pairs_pred
                rows.append(res)
    return pd.DataFrame(rows)


def load_mentions(file):
    lines = []
    with jsonlines.open(file) as f:
        for line in f:
            new_line = {
                "sentence": line["sentence"],
                "aliases": line.get("aliases", []),
                "spans": line.get("spans", []),
            }
            lines.append(new_line)
    return pd.DataFrame(lines)


def enwiki_title_to_wikidata_id(title: str) -> str:
    protocol = "https"
    base_url = "en.wikipedia.org/w/api.php"
    params = f"action=query&prop=pageprops&format=json&titles={title}"
    url = f"{protocol}://{base_url}?{params}"
    response = requests.get(url)
    json = response.json()
    for pages in json["query"]["pages"].values():
        wikidata_id = pages["pageprops"]["wikibase_item"]
    return wikidata_id


def tagme_annotate(in_file, out_file, threshold=0.1):
    with jsonlines.open(in_file) as f_in, jsonlines.open(out_file, "w") as f_out:
        for line in f_in:
            aliases = []
            spans = []
            qids = []
            probs = []
            text = line["sentence"]
            text_spans = text.split()
            text_span_indices = []
            total_len = 0

            # get word boundaries for converting char spans to word spans
            for i, t in enumerate(text_spans):
                text_span_indices.append(total_len)
                total_len += len(t) + 1
            lunch_annotations = tagme.annotate(text)

            # as the threshold increases, the precision increases, but the recall decreases
            for ann in lunch_annotations.get_annotations(threshold):
                mention = ann.mention
                try:
                    qid = enwiki_title_to_wikidata_id(ann.entity_title)
                except:
                    print(f"No wikidata id found for {ann.entity_title}")
                    continue
                span_start = text_span_indices.index(ann.begin)
                try:
                    span_end = text_span_indices.index(ann.end + 1)
                except:
                    span_end = len(text_spans)
                aliases.append(mention)
                spans.append([span_start, span_end])
                qids.append(qid)
                probs.append(ann.score)

            line["aliases"] = aliases
            line["qids"] = qids
            line["spans"] = spans
            line["probs"] = probs
            line["gold"] = [True for _ in aliases]
            f_out.write(line)


# modified from https://github.com/facebookresearch/BLINK/blob/master/elq/vcg_utils/measures.py
def entity_linking_tp_with_overlap(gold, predicted, ignore_entity=False):
    """
    Partially adopted from: https://github.com/UKPLab/starsem2018-entity-linking
    Counts weak and strong matches
    :param gold:
    :param predicted:
    :return:
    >>> entity_linking_tp_with_overlap([('Q7366', 14, 18),('Q780394', 19, 35)],[('Q7366', 14, 16),('Q780394', 19, 35)])
    2, 1
    >>> entity_linking_tp_with_overlap([('Q7366', 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16)])
    1, 0
    >>> entity_linking_tp_with_overlap([(None, 14, 18), ('Q780394', 19, 35)], [('Q7366', 14, 16)])
    0, 0
    >>> entity_linking_tp_with_overlap([(None, 14, 18), (None, )], [(None,)])
    1, 0
    >>> entity_linking_tp_with_overlap([('Q7366', ), ('Q780394', )], [('Q7366', 14, 16)])
    1, 0
    >>> entity_linking_tp_with_overlap([], [('Q7366', 14, 16)])
    0, 0
    """
    if not gold or not predicted:
        return 0, 0
    # Add dummy spans, if no spans are given, everything is overlapping per default
    if any(len(e) != 3 for e in gold):
        gold = [(e[0], 0, 1) for e in gold]
        predicted = [(e[0], 0, 1) for e in predicted]
    # Replace None KB ids with empty strings
    gold = [("",) + e[1:] if e[0] is None else e for e in gold]
    predicted = [("",) + e[1:] if e[0] is None else e for e in predicted]

    # ignore_entity for computing mention precision and recall without the entity prediction
    if ignore_entity:
        gold = [("",) + e[1:] for e in gold]
        predicted = [("",) + e[1:] for e in predicted]

    gold = sorted(gold, key=lambda x: x[2])
    predicted = sorted(predicted, key=lambda x: x[2])

    # tracks weak matches
    lcs_matrix_weak = np.zeros((len(gold), len(predicted)), dtype=np.int16)
    # tracks strong matches
    lcs_matrix_strong = np.zeros((len(gold), len(predicted)), dtype=np.int16)
    for g_i in range(len(gold)):
        for p_i in range(len(predicted)):
            gm = gold[g_i]
            pm = predicted[p_i]

            # increment lcs_matrix_weak
            if not (gm[1] >= pm[2] or pm[1] >= gm[2]) and (
                gm[0].lower() == pm[0].lower()
            ):
                if g_i == 0 or p_i == 0:
                    lcs_matrix_weak[g_i, p_i] = 1
                else:
                    lcs_matrix_weak[g_i, p_i] = 1 + lcs_matrix_weak[g_i - 1, p_i - 1]
            else:
                if g_i == 0 and p_i == 0:
                    lcs_matrix_weak[g_i, p_i] = 0
                elif g_i == 0 and p_i != 0:
                    lcs_matrix_weak[g_i, p_i] = max(0, lcs_matrix_weak[g_i, p_i - 1])
                elif g_i != 0 and p_i == 0:
                    lcs_matrix_weak[g_i, p_i] = max(lcs_matrix_weak[g_i - 1, p_i], 0)
                elif g_i != 0 and p_i != 0:
                    lcs_matrix_weak[g_i, p_i] = max(
                        lcs_matrix_weak[g_i - 1, p_i], lcs_matrix_weak[g_i, p_i - 1]
                    )

            # increment lcs_matrix_strong
            if (gm[1] == pm[1] and pm[2] == gm[2]) and (gm[0].lower() == pm[0].lower()):
                if g_i == 0 or p_i == 0:
                    lcs_matrix_strong[g_i, p_i] = 1
                else:
                    lcs_matrix_strong[g_i, p_i] = (
                        1 + lcs_matrix_strong[g_i - 1, p_i - 1]
                    )
            else:
                if g_i == 0 and p_i == 0:
                    lcs_matrix_strong[g_i, p_i] = 0
                elif g_i == 0 and p_i != 0:
                    lcs_matrix_strong[g_i, p_i] = max(
                        0, lcs_matrix_strong[g_i, p_i - 1]
                    )
                elif g_i != 0 and p_i == 0:
                    lcs_matrix_strong[g_i, p_i] = max(
                        lcs_matrix_strong[g_i - 1, p_i], 0
                    )
                elif g_i != 0 and p_i != 0:
                    lcs_matrix_strong[g_i, p_i] = max(
                        lcs_matrix_strong[g_i - 1, p_i], lcs_matrix_strong[g_i, p_i - 1]
                    )

    weak_match_count = lcs_matrix_weak[len(gold) - 1, len(predicted) - 1]
    strong_match_count = lcs_matrix_strong[len(gold) - 1, len(predicted) - 1]
    return weak_match_count, strong_match_count


def convert_line_tuple(line):
    qids = line["qids"]
    spans = line["spans"]
    pairs = zip(qids, spans)
    pairs = [(pair[0], pair[1][0], pair[1][1]) for pair in pairs]
    return pairs


# modified from https://github.com/facebookresearch/BLINK
def compute_metrics(pred_file, gold_file, md_step_only=False, threshold=0.0):
    # align by sentence index
    pred_results = {}
    with jsonlines.open(pred_file) as f:
        for line in f:
            pred_results[line["sent_idx_unq"]] = line

    gold_results = {}
    with jsonlines.open(gold_file) as f:
        for line in f:
            gold_results[line["sent_idx_unq"]] = line

    assert len(pred_results) == len(
        gold_results
    ), f"{len(pred_results)} {len(gold_results)}"

    num_mentions_actual = 0
    num_mentions_pred = 0
    weak_match_total = 0
    strong_match_total = 0
    errors = []
    for sent_idx in pred_results:
        gold_line = gold_results[sent_idx]
        pred_line = pred_results[sent_idx]
        pred_triples = convert_line_tuple(pred_results[sent_idx])
        gold_triples = convert_line_tuple(gold_results[sent_idx])

        # filter out triples below the threshold
        if len(pred_triples) > 0 and "probs" in pred_line:
            assert len(pred_triples) == len(pred_line["probs"])
            pred_triples = [
                pt
                for (pt, prob) in zip(pred_triples, pred_line["probs"])
                if prob > threshold
            ]

        weak_match_count, strong_match_count = entity_linking_tp_with_overlap(
            pred_triples, gold_triples, ignore_entity=md_step_only
        )

        num_mentions_actual += len(gold_triples)
        num_mentions_pred += len(pred_triples)
        weak_match_total += weak_match_count
        strong_match_total += strong_match_count
        if weak_match_count != len(gold_triples) or weak_match_count != len(
            pred_triples
        ):
            pred_qids = [p[0] for p in pred_triples]
            pred_spans = [[p[1], p[2]] for p in pred_triples]
            pred_probs = []
            if "probs" in pred_line:
                pred_probs = [p for p in pred_line["probs"] if p > threshold]
                assert len(pred_qids) == len(pred_probs)
            errors.append(
                {
                    "sent_idx": sent_idx,
                    "text": gold_line["sentence"],
                    "gold_aliases": gold_line["aliases"],
                    "gold_qids": gold_line["qids"],
                    "gold_spans": gold_line["spans"],
                    "pred_aliases": pred_line["aliases"],
                    "pred_qids": pred_qids,
                    "pred_spans": pred_spans,
                    "pred_probs": pred_probs,
                }
            )

    print("WEAK MATCHING")
    precision = weak_match_total / num_mentions_pred
    recall = weak_match_total / num_mentions_actual
    print(f"precision = {weak_match_total} / {num_mentions_pred} = {precision}")
    print(f"recall = {weak_match_total} / {num_mentions_actual} = {recall}")
    print(f"f1 = {precision*recall*2/(precision+recall)}")

    print("\nEXACT MATCHING")
    precision = strong_match_total / num_mentions_pred
    recall = strong_match_total / num_mentions_actual
    print(f"precision = {strong_match_total} / {num_mentions_pred} = {precision}")
    print(f"recall = {strong_match_total} / {num_mentions_actual} = {recall}")
    print(f"f1 = {precision*recall*2/(precision+recall)}")

    return pd.DataFrame(errors)
