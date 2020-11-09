from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from task_config import ID_TO_LABEL


def tacred_scorer(
    golds: ndarray,
    probs: ndarray,
    preds: Optional[ndarray],
    uids: Optional[List[str]] = None,
) -> Dict[str, float]:

    NO_RELATION = 0

    res = {}
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(golds)):
        gold = golds[row]
        guess = preds[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    relations = gold_by_relation.keys()

    for relation in sorted(relations):
        # (compute the score)
        correct = correct_by_relation[relation]
        guessed = guessed_by_relation[relation]
        gold = gold_by_relation[relation]
        prec = 1.0
        if guessed > 0:
            prec = float(correct) / float(guessed)
        recall = 0.0
        if gold > 0:
            recall = float(correct) / float(gold)
        f1 = 0.0
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)

        res[f"{ID_TO_LABEL[relation]}_prec"] = prec
        res[f"{ID_TO_LABEL[relation]}_rec"] = recall
        res[f"{ID_TO_LABEL[relation]}_f1"] = f1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values())
        )
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values())
        )
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    res["Precision"] = prec_micro
    res["Recall"] = recall_micro
    res["F1"] = f1_micro

    n_matches = np.where(golds == preds)[0].shape[0]

    res["Accuracy"] = n_matches / golds.shape[0]

    return res
