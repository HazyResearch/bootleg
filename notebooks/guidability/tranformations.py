from typing import List, Sequence, Tuple

import numpy as np
from robustnessgym.core.dataset import Batch
from robustnessgym.slicebuilders.transformation import Transformation


class TransformGoldEntities(Transformation):
    def __init__(self):
        super(TransformGoldEntities, self).__init__(
            num_transformed=2,
        )

    def apply(
        self,
        skeleton_batches: List[Batch],
        slice_membership: np.ndarray,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs
    ) -> Tuple[List[Batch], np.ndarray]:

        for i, (gold_qids, spans, sentence) in enumerate(
            zip(*[batch[c] for c in columns])
        ):

            # Split the sentence using spaces
            capitalized_sentence = sentence.split(" ")
            uncapitalized_sentence = sentence.split(" ")

            for span in spans[-len(gold_qids) :]:
                # Capitalize the entity
                capitalized_sentence[span[0] - 1] = capitalized_sentence[
                    span[0] - 1
                ].capitalize()

                # Lower-case the entity
                uncapitalized_sentence[span[0] - 1] = uncapitalized_sentence[
                    span[0] - 1
                ].lower()

            # Fill in the changed sentence: all other columns are exactly the same
            skeleton_batches[0][columns[2]][i] = " ".join(capitalized_sentence)
            skeleton_batches[1][columns[2]][i] = " ".join(uncapitalized_sentence)

        return skeleton_batches, slice_membership


class TransformSportsCandidates(Transformation):
    def __init__(self, title2q):
        super(TransformSportsCandidates, self).__init__(
            num_transformed=1,
        )
        self.quantifiers = {"women", "under-", "junior", "men"}
        self.quantifiers_withoutmen = {"women", "under-", "junior"}
        self.title2q = title2q

    def remove_quantifiers(self, title, title2q):
        tit = " ".join(
            [
                t
                for t in title.split()
                if not any(q in t.lower() for q in self.quantifiers)
            ]
        )
        if tit not in title2q:
            title2 = title.replace("women", "men")
            tit2 = " ".join(
                [
                    t
                    for t in title2.split()
                    if not any(q in t.lower() for q in self.quantifiers_withoutmen)
                ]
            )
            if tit2 not in title2q:
                return None
            else:
                return tit2
        else:
            return tit

    def has_quantifier(self, title):
        title = title.lower()
        return "women" in title or "under-" in title

    def apply(
        self,
        skeleton_batches: List[Batch],
        slice_membership: np.ndarray,
        batch: Batch,
        columns: List[str],
        *args,
        **kwargs
    ) -> Tuple[List[Batch], np.ndarray]:

        for i, (sentence, gold_title, gold_qid, cand_names, cand_probs) in enumerate(
            zip(*[batch[c] for c in columns])
        ):

            filtered_cand_names = [c for c in cand_names if not self.has_quantifier(c)]
            filtered_cand_probs = [
                p
                for j, p in enumerate(cand_probs)
                if not self.has_quantifier(cand_names[j])
            ]
            new_gold_title = self.remove_quantifiers(gold_title, self.title2q)
            new_gold_qid = self.title2q.get(new_gold_title, None)
            if new_gold_qid is None:
                print("NOT FOUND TITLE", new_gold_title, "FROM", gold_title)
                slice_membership[i] = 0
            else:
                # Fill in the changed sentence: all other columns are exactly the same
                skeleton_batches[0][columns[1]][i] = new_gold_title
                skeleton_batches[0][columns[2]][i] = new_gold_qid
                skeleton_batches[0][columns[3]][i] = filtered_cand_names
                skeleton_batches[0][columns[4]][i] = filtered_cand_probs

        return skeleton_batches, slice_membership
