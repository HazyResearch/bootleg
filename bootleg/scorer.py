"""Bootleg scorer."""
import logging
from collections import Counter
from typing import Dict, List, Optional

from numpy import ndarray

logger = logging.getLogger(__name__)


class BootlegSlicedScorer:
    """Sliced NED scorer init.

    Args:
        train_in_candidates: are we training assuming that all gold qids are in the candidates or not
        slices_datasets: slice dataset (see slicing/slice_dataset.py)
    """

    def __init__(self, train_in_candidates, slices_datasets=None):
        """Bootleg scorer initializer."""
        self.train_in_candidates = train_in_candidates
        self.slices_datasets = slices_datasets

    def get_slices(self, uid):
        """
        Get slices incidence matrices.

        Get slice incidence matrices for the uid Uid is dtype
        (np.dtype([('sent_idx', 'i8', 1), ('subsent_idx', 'i8', 1),
        ("alias_orig_list_pos", 'i8', max_aliases)]) where alias_orig_list_pos
        gives the mentions original positions in the sentence.

        Args:
            uid: unique identifier of sentence

        Returns: dictionary of slice_name -> matrix of 0/1 for if alias is in slice or not (-1 for no alias)
        """
        if self.slices_datasets is None:
            return {}
        for split, dataset in self.slices_datasets.items():
            sent_idx = uid["sent_idx"]
            alias_orig_list_pos = uid["alias_orig_list_pos"]
            if dataset.contains_sentidx(sent_idx):
                return dataset.get_slice_incidence_arr(sent_idx, alias_orig_list_pos)
        return {}

    def bootleg_score(
        self,
        golds: ndarray,
        probs: ndarray,
        preds: Optional[ndarray],
        uids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Scores the predictions using the gold labels and slices.

        Args:
            golds: gold labels
            probs: probabilities
            preds: predictions (max prob candidate)
            uids: unique identifiers

        Returns: dictionary of tensorboard compatible keys and metrics
        """
        batch = golds.shape[0]
        NO_MENTION = -1
        NOT_IN_CANDIDATES = -2 if self.train_in_candidates else 0
        res = {}
        total = Counter()
        total_in_cand = Counter()
        correct_boot = Counter()
        correct_pop_cand = Counter()
        correct_boot_in_cand = Counter()
        correct_pop_cand_in_cand = Counter()
        assert (
            len(uids) == batch
        ), f"Length of uids {len(uids)} does not match batch {batch} in scorer"
        for row in range(batch):
            gold = golds[row]
            pred = preds[row]
            uid = uids[row]
            pop_cand = 0 + int(not self.train_in_candidates)
            if gold == NO_MENTION:
                continue
            # Slices is dictionary of slice_name -> incidence array. Each array value is 1/0 for if in slice or not
            slices = self.get_slices(uid)
            for slice_name in slices:
                assert (
                    slices[slice_name][0] != -1
                ), f"Something went wrong with slices {slices} and uid {uid}"
                # Check if alias is in slice
                if slices[slice_name][0] == 1:
                    total[slice_name] += 1
                    if gold != NOT_IN_CANDIDATES:
                        total_in_cand[slice_name] += 1
                    if gold == pred:
                        correct_boot[slice_name] += 1
                        if gold != NOT_IN_CANDIDATES:
                            correct_boot_in_cand[slice_name] += 1
                    if gold == pop_cand:
                        correct_pop_cand[slice_name] += 1
                        if gold != NOT_IN_CANDIDATES:
                            correct_pop_cand_in_cand[slice_name] += 1
        for slice_name in total:
            res[f"{slice_name}/total_men"] = total[slice_name]
            res[f"{slice_name}/total_notNC_men"] = total_in_cand[slice_name]
            res[f"{slice_name}/acc_boot"] = (
                0
                if total[slice_name] == 0
                else correct_boot[slice_name] / total[slice_name]
            )
            res[f"{slice_name}/acc_notNC_boot"] = (
                0
                if total_in_cand[slice_name] == 0
                else correct_boot_in_cand[slice_name] / total_in_cand[slice_name]
            )
            res[f"{slice_name}/acc_pop"] = (
                0
                if total[slice_name] == 0
                else correct_pop_cand[slice_name] / total[slice_name]
            )
            res[f"{slice_name}/acc_notNC_pop"] = (
                0
                if total_in_cand[slice_name] == 0
                else correct_pop_cand_in_cand[slice_name] / total_in_cand[slice_name]
            )
        return res
