import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import argh
import feather
import pandas as pd
import ujson
from fuzzywuzzy import fuzz
from guide_utils import (
    KG_KEY,
    TYPE_HY,
    TYPE_REL,
    TYPE_WD,
    prep_for_run,
    regiater_funcs,
    write_df_to_jsonl,
)
from metadata import WLMetadata
from tabulate import tabulate
from tqdm import tqdm

tqdm.pandas()

AIRPORT_KEYWORDS = {
    "international airport",
    "airport",
    "flights to",
    "flights from",
    "flight to",
    "flight from",
    "hub at",
    "bound for",
    "bound to",
    "connections to",
    "route to",
    "route from",
    "arriving at",
    "arrive at",
    "departing from",
    "depart from",
}
SPORT_KEYWORDS = {
    "match against",
    "win over",
    "debut for",
    "victory over",
    "goal",
    "score",
    "coach",
    "coached",
    "team",
    "teams",
    "loss too",
    "matches against",
    "defeat against",
    "played against",
    "final against",
    "match between",
    "goal against",
    "played for",
    "play",
    "game",
    "games",
    "win",
    "lost",
    "won",
}

airport_aliases = ujson.load(
    open(
        Path(
            "/dfs/scratch0/lorr1/projects/bootleg/notebooks/guidability/saved_aug_metadata/korealiases_title_1229_airport_aliases.json"
        )
    )
)
soccer_aliases = ujson.load(
    open(
        Path(
            "/dfs/scratch0/lorr1/projects/bootleg/notebooks/guidability/saved_aug_metadata/korealiases_title_1229_football_aliases.json"
        )
    )
)

aug_func = regiater_funcs()


class AugObj:
    def __init__(self, titles, qids, aliases, spans, golds, sources, sentences):
        assert type(titles) is list, f"titles must be type list"
        assert type(qids) is list, f"qids must be type list"
        assert type(aliases) is list, f"aliases must be type list"
        assert type(spans) is list, f"spans must be type list"
        assert type(golds) is list, f"golds must be type list"
        assert type(sources) is list, f"sources must be type list"
        assert type(sentences) is list, f"sentences must be type list"
        self.titles = titles
        self.qids = qids
        self.aliases = aliases
        self.spans = spans
        self.golds = golds
        self.sources = sources
        self.sentences = sentences

    @classmethod
    def create_empty(cls):
        return cls([], [], [], [], [], [], [])

    def append(self, titles, qids, aliases, spans, golds, sources, sentence):
        self.titles.append(titles)
        self.qids.append(qids)
        self.aliases.append(aliases)
        self.spans.append(spans)
        self.golds.append(golds)
        self.sources.append(sources)
        self.sentences.append(sentence)

    def extend(self, titles, qids, aliases, spans, golds, sources, sentence):
        self.titles.extend(titles)
        self.qids.extend(qids)
        self.aliases.extend(aliases)
        self.spans.extend(spans)
        self.golds.extend(golds)
        self.sources.extend(sources)
        self.sentences.extend(sentence)

    def __repr__(self):
        r = {
            "sentence": self.sentences,
            "titles": self.titles,
            "qids": self.qids,
            "aliases": self.aliases,
            "spans": self.spans,
            "gold": self.golds,
            "sources": self.sources,
        }
        return str(r)


def is_airport(gold_types):
    typs = {"airport"}
    bad_typs = {}
    for t in gold_types:
        if any(tp in t for tp in typs) and t not in bad_typs:
            return True
    return False


def is_location(gold_types):
    typs = {"city", "municipality", "country", "county", "town", "island"}
    for t in gold_types:
        if any(tp in t for tp in typs):
            return True
    return False


def is_soccer_team(title):
    t = title.lower()
    in_list = ["national", "football", "team"]
    not_in_list = [
        "competition",
        "season",
        "cup",
        "national team nomenclature",
        "teamsters",
    ]
    r = all(i in t for i in in_list) and all(i not in t for i in not_in_list)
    return r


@aug_func
def aug_airport(row, wl_metadata):
    num_swaps = 2
    filtered_swapable_aliases = []
    new_obj = AugObj.create_empty()
    if num_words_in_sentence(AIRPORT_KEYWORDS, row["sentence"]) <= 0:
        return new_obj
    for i, (orig_al, sp, gld_types, gld_kgs) in enumerate(
        zip(
            row["aliases"],
            row["spans"],
            row["wikidata_types_1229_gld"],
            row["kg_adj_1229_gld"],
        )
    ):
        # If not airport skip
        if not is_airport(gld_types):
            continue
        # If the alias to remove is part of a KG connection, do not swap as we will potentially create false co-occurrence information
        if len(gld_kgs) > 0:
            continue
        # Lazy eval
        if len(filtered_swapable_aliases) <= 0:
            filtered_swapable_aliases = [
                al for al in airport_aliases if al not in row["aliases"]
            ]
        # Swap airport alias
        aliases_to_swap = random.choices(filtered_swapable_aliases, k=num_swaps)
        qids_to_swap = [
            random.choice(airport_aliases[al])["to_add"] for al in aliases_to_swap
        ]
        new_sentences, new_spans, good_swaps = permute_aliases(
            row["sentence"],
            row["aliases"],
            row["spans"],
            [i] * num_swaps,
            aliases_to_swap,
        )
        # If bad swap, no new sentences will be returned
        if len(new_sentences) <= 0:
            continue
        new_aliases = [row["aliases"][:]] * len(good_swaps)
        new_qids = [row["qids"][:]] * len(good_swaps)
        new_golds = [row["gold"][:]] * len(good_swaps)
        new_sources = [row["sources"][:]] * len(good_swaps)
        for j, (n_s, n_sp, n_al, n_q, n_g, n_src) in enumerate(
            zip(new_sentences, new_spans, new_aliases, new_qids, new_golds, new_sources)
        ):
            # Adjust the other aliases, qids, and sources - add to overall list
            n_al[i] = aliases_to_swap[good_swaps[j]]
            n_q[i] = qids_to_swap[good_swaps[j]]
            n_src[i] = "airport_swap"
            new_titles = [wl_metadata.get_title(q) for q in n_q]
            new_obj.append(new_titles, n_q, n_al, n_sp, n_g, n_src, n_s)

    return new_obj


@aug_func
def aug_soccer(row, wl_metadata):
    num_swaps = 2
    filtered_swapable_aliases = []
    new_obj = AugObj.create_empty()
    if num_words_in_sentence(SPORT_KEYWORDS, row["sentence"]) <= 0:
        return new_obj
    for i, (orig_al, sp, gld_title, gld_kgs) in enumerate(
        zip(row["aliases"], row["spans"], row["titles"], row["kg_adj_1229_gld"])
    ):
        # If not soccer team skip
        if not is_soccer_team(gld_title):
            continue
        # If the alias to remove is part of a KG connection, do not swap as we will potentially create false co-occurrence information
        if len(gld_kgs) > 0:
            continue
        # Lazy eval
        if len(filtered_swapable_aliases) <= 0:
            filtered_swapable_aliases = [
                al for al in soccer_aliases if al not in row["aliases"]
            ]
        # Swap soccer alias
        aliases_to_swap = random.choices(filtered_swapable_aliases, k=num_swaps)
        qids_to_swap = [
            random.choice(soccer_aliases[al])["to_add"] for al in aliases_to_swap
        ]
        new_sentences, new_spans, good_swaps = permute_aliases(
            row["sentence"],
            row["aliases"],
            row["spans"],
            [i] * num_swaps,
            aliases_to_swap,
        )
        # If bad swap, no new sentences will be returned
        if len(new_sentences) <= 0:
            continue
        new_aliases = [row["aliases"][:]] * len(good_swaps)
        new_qids = [row["qids"][:]] * len(good_swaps)
        new_golds = [row["gold"][:]] * len(good_swaps)
        new_sources = [row["sources"][:]] * len(good_swaps)
        for j, (n_s, n_sp, n_al, n_q, n_g, n_src) in enumerate(
            zip(new_sentences, new_spans, new_aliases, new_qids, new_golds, new_sources)
        ):
            # Adjust the other aliases, qids, and sources - add to overall list
            n_al[i] = aliases_to_swap[good_swaps[j]]
            n_q[i] = qids_to_swap[good_swaps[j]]
            n_src[i] = "soccer_swap"
            new_titles = [wl_metadata.get_title(q) for q in n_q]
            new_obj.append(new_titles, n_q, n_al, n_sp, n_g, n_src, n_s)
    return new_obj


def num_words_in_sentence(words, sentence):
    sentence = " " + sentence.lower() + " "
    return sum([f" {w.lower()} " in sentence for w in words])


# If the new alias is longer or shorter, must shift all after alias spans
def adjust_spans_after_alias_swap(old_spans, new_al, span_to_remove):
    al_length_diff = len(new_al.split()) - (span_to_remove[1] - span_to_remove[0])
    sp_st = span_to_remove[0]
    new_spans = []
    for sp in old_spans:
        if sp[0] < sp_st:
            new_spans.append(sp)
        elif sp[0] == sp_st:
            new_spans.append([sp[0], sp[1] + al_length_diff])
        else:
            new_spans.append([sp[0] + al_length_diff, sp[1] + al_length_diff])
    return new_spans


def permute_aliases(
    sentence, aliases, spans, aliases_to_remove_idx, aliases_to_swap, cap_aliases=True
):
    """
    Performs len(aliases_to_remove_idx) alias swaps on the given sentence.

    Args:
        sentence: sentence
        aliases: aliases
        spans: spans
        aliases_to_remove_idx: list of alias indices to swap out - one per augmentation (ie., len 2 means you are augmenting twice)
        qids_to_add: list of qids to swap in - one per augmentation (ie., len 2 means you are augmenting twice)
        aliases_to_swap: list of aliases to swap in - one per augmentation (ie., len 2 means you are augmenting twice)
        cap_aliases: Whether to capitalize the alias being added

    Returns: augmented sentences, adjusted spans, bad swaps indices (for postprocessing filtering)

    """
    new_sentences = []
    new_spans = []
    good_swaps = []
    for i, (al, alias_to_remove_idx) in enumerate(
        zip(aliases_to_swap, aliases_to_remove_idx)
    ):

        span_to_remove = spans[alias_to_remove_idx]
        alias_to_remove = aliases[alias_to_remove_idx]
        sentence_to_change = sentence.split()
        # Replace new alias in sentence, capitalizing if necessary
        al_to_replace = al
        if cap_aliases:
            al_to_replace = al.title()
        new_sentence = (
            " ".join(sentence_to_change[: span_to_remove[0]])
            + " "
            + al_to_replace
            + " "
            + " ".join(sentence_to_change[span_to_remove[1] :])
        )
        all_new_spans = adjust_spans_after_alias_swap(spans, al, span_to_remove)
        # Sanity checkes
        split_sent = new_sentence.split()
        passes = True
        # print("*****\nREMOVED", alias_to_remove, "AND SWAPPED FOR", al,  "\nOLD SPANS", spans, "\nNEW SPANS", all_new_spans, "\nSENTENCe", new_sentence)
        sp = all_new_spans[alias_to_remove_idx]
        if " ".join(split_sent[sp[0] : sp[1]]).lower() in [
            "his",
            "her",
            "hers",
            "he",
            "she",
        ]:
            continue
        if fuzz.ratio(" ".join(split_sent[sp[0] : sp[1]]).lower(), al.lower()) < 0.8:
            print(f"BAD AT {alias_to_remove_idx} {sp}")
            print("NEW")
            print(al, new_spans, new_sentence)
            print("OLD")
            print(alias_to_remove, spans, sentence)
            passes = False
        if passes:
            new_sentences.append(new_sentence)
            new_spans.append(all_new_spans)
            good_swaps.append(i)

    return new_sentences, new_spans, good_swaps


def augment_data(df, lfs, wl_metadata_dir, max_swaps):
    """
    Applies relabelled weak label function to the input dataframe. Will return original labels if WL function is
    does not trigger

    Args:
        df: data frame
        lfs: list of LFs
        wl_metadata: weak label metadata

    Returns: titles, qids, aliases, spans, golds metrics, row_indices where LF was applied

    """
    wl_metadata = WLMetadata.load(wl_metadata_dir)
    max_sent_idx = int(df["sent_idx"].max())
    final_data = AugObj.create_empty()
    final_sent_indx = []
    metrics = defaultdict(int)
    old_indices = defaultdict(list)
    new_indices = defaultdict(lambda: defaultdict(list))
    new_index = max_sent_idx + 1
    for i, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
        final_data.append(
            row["titles"],
            row["qids"],
            row["aliases"],
            row["spans"],
            row["gold"],
            row["sources"],
            row["sentence"],
        )
        final_sent_indx.append(row["sent_idx"])
        for j, lf in enumerate(lfs):
            # If we have reached the max number per LF, don't augment anymore
            if 0 < max_swaps <= metrics[lf.__name__]:
                continue
            aug_obj = lf(row, wl_metadata)
            if aug_obj.titles is None or len(aug_obj.titles) <= 0:
                continue
            assert (
                type(aug_obj.titles) is list and type(aug_obj.titles[0]) is list
            ), f"We assume you return a list of lists of items from the augmentation LF"
            final_data.extend(
                aug_obj.titles,
                aug_obj.qids,
                aug_obj.aliases,
                aug_obj.spans,
                aug_obj.golds,
                aug_obj.sources,
                aug_obj.sentences,
            )
            # Add new unique sentence index for augmented data
            for k in range(len(aug_obj.titles)):
                final_sent_indx.append(new_index)
                new_indices[lf.__name__][i].append(len(final_sent_indx) - 1)
                new_index += 1
            metrics[lf.__name__] += len(aug_obj.titles)
            old_indices[lf.__name__].append(i)
    new_df = pd.DataFrame(
        data={
            "sentence": final_data.sentences,
            "sent_idx": final_sent_indx,
            "titles": final_data.titles,
            "qids": final_data.qids,
            "aliases": final_data.aliases,
            "spans": final_data.spans,
            "gold": final_data.golds,
            "sources": final_data.sources,
        }
    )
    assert (
        len(final_data.titles)
        == len(final_data.qids)
        == len(final_data.aliases)
        == len(final_data.spans)
        == len(final_data.golds)
        == len(final_data.sources)
        == len(final_data.sentences)
    )
    return new_df, metrics, old_indices, new_indices


@argh.arg("input_file", help="input train file", type=str)
@argh.arg("output_file", help="output train file", type=str)
@argh.arg("--max_swaps", help="maximum new sentences added per LF", type=int)
@argh.arg("--num_workers", help="parallelism", type=int)
@argh.arg("--overwrite", help="overwrite", action="store_true")
def main(input_file, output_file, max_swaps=-1, num_workers=40, overwrite=False):
    """
    Each augmentation function returns newly generated data from the augmentation. It does NOT return copies of the original data.
    This is different than the weak label assumptions.

    Args:
        input_file: input train file
        output_file: output file to save
        num_workers: number processes
        overwrite: overwrite saved metadata or not

    Returns:

    """
    input_path = Path(input_file)
    input_dir = input_path.parent
    last_dir = input_path.parent.name
    basename = os.path.splitext(input_path.name)[0]
    emb_dir = Path("/dfs/scratch0/lorr1/projects/bootleg-data/embs")
    cache_dir = Path("saved_data")
    cache_metadata_dir = cache_dir / f"{last_dir}_saved_metadata"
    cache_file = cache_dir / f"{last_dir}_{basename}.feather"

    # This creates the training data and ensure the metadata is loaded
    train_df = prep_for_run(
        cache_file,
        cache_dir,
        cache_metadata_dir,
        input_dir,
        emb_dir,
        input_file,
        overwrite,
    )

    lfs = list(aug_func.all.values())
    assert len(lfs) > 0, f"You must specify LFs via decorates to label data"
    print(f"Using LFs {[l.__name__ for l in lfs]}")
    new_df, metrics, old_indices, new_indices = augment_data(
        train_df, lfs, cache_metadata_dir, max_swaps
    )
    debug_output_dir = Path("debug_outputs")
    os.makedirs(debug_output_dir, exist_ok=True)

    for lf_name, old_indexes in old_indices.items():
        temp_old = train_df.iloc[old_indexes]
        temp_new = pd.DataFrame()
        for j in range(temp_old.shape[0]):
            new_indx = new_indices[lf_name][old_indexes[j]]
            temp_new = pd.concat([temp_new, new_df.iloc[new_indx]])
        save_file = debug_output_dir / f"{lf_name}_old_df.feather"
        feather.write_dataframe(temp_old, save_file)
        print(
            f"Saving augmented original rows for {lf_name} at {save_file}. Load with feather.load_dataframe()."
        )
        save_file = debug_output_dir / f"{lf_name}_new_df.feather"
        feather.write_dataframe(temp_new, save_file)
        print(
            f"Saving augmented new rows for {lf_name} at {save_file}. Load with feather.load_dataframe()."
        )

    print(f"Writing out file to {output_file}")
    print(f"Original Data Size {train_df.shape[0]}")
    for k in metrics:
        print(f"{k} Metrics: {metrics[k]} ({metrics[k]/train_df.shape[0]}%)")
    write_df_to_jsonl(new_df, output_file)


if __name__ == "__main__":
    argh.dispatch_command(main)
