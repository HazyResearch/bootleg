import os
import sys
from collections import defaultdict
from pathlib import Path

import argh
import pandas as pd
import ujson
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

UNIV_KEYWORDS = {
    "studied at",
    "studied at the",
    "studies at",
    "studies at the",
    "educated at",
    "educated at the",
    "graduated from",
    "graduated from the",
    "department at",
    "department at the",
    "degree from",
    "degree from the",
    "attended",
    "attended the",
    "professor at",
    "professor at the",
    "taught at",
    "taught at the",
    "univeristy of",
}

wl_func = regiater_funcs()


def is_university(gold_types):
    typs = {"university", "educational institution", "college", "school"}
    bad_typs = {"college athletic conference"}
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


@wl_func
def lf_univ_keyword(row, wl_metadata):
    titles, qids, als, sps, glds, srcs = (
        row["titles"],
        row["qids"],
        row["aliases"],
        row["spans"],
        row["gold"],
        row["sources"],
    )
    updated_cnt = 0
    for i, (orig_al, spans, gld_types, cand_qids, cand_names) in enumerate(
        zip(
            row["aliases"],
            row["spans"],
            row["wikidata_types_1229_gld"],
            row["cands_qids"],
            row["cands_names"],
        )
    ):
        # If already a university, skip
        if is_university(gld_types) or not is_location(gld_types):
            continue
        # Check if university keywords
        span_l, span_r = spans
        sent_left = row["sentence_split"][span_l - 3 : span_l]
        keyword_found = False
        for univ_k in UNIV_KEYWORDS:
            if univ_k == " ".join(sent_left[-len(univ_k.split()) :]):
                keyword_found = True
                break
        if not keyword_found:
            continue
        new_label = None
        # Find a university correct answer
        for j, cand_qid in enumerate(cand_qids):
            cand_types = wl_metadata.get_types(cand_qid, TYPE_WD)
            if is_university(cand_types):
                new_label = cand_names[j]
                break
        if not new_label:
            continue
        updated_cnt += 1
        titles[i] = new_label
        qids[i] = wl_metadata.get_qid(new_label)
        srcs[i] = "univ_key"
    return titles, qids, als, sps, glds, srcs, updated_cnt


def weaklabel_data(df, lfs, wl_metadata_dir):
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
    new_titles = []
    new_qids = []
    new_aliases = []
    new_spans = []
    new_golds = []
    new_sources = []
    new_sentences = []
    new_sent_idx = []
    metrics = defaultdict(int)
    row_indices = defaultdict(list)
    for i, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
        new_sentences.append(row["sentence"])
        new_sent_idx.append(row["sent_idx"])
        labelled = False
        for j, lf in enumerate(lfs):
            if labelled:
                break
            titles, qids, als, sps, glds, srcs, updated_cnt = lf(row, wl_metadata)
            if j == 0:
                new_titles.append(titles)
                new_qids.append(qids)
                new_aliases.append(als)
                new_spans.append(sps)
                new_golds.append(glds)
                new_sources.append(srcs)
            else:
                new_titles[-1] = titles
                new_qids[-1] = qids
                new_aliases[-1] = als
                new_spans[-1] = sps
                new_golds[-1] = glds
                new_sources[-1] = srcs
            if updated_cnt > 0:
                metrics[lf.__name__] += updated_cnt
                row_indices[lf.__name__].append(i)
                labelled = True
    assert (
        len(new_titles)
        == len(new_qids)
        == len(new_aliases)
        == len(new_spans)
        == len(new_golds)
        == len(new_sources)
        == len(new_sentences)
    )
    new_df = pd.DataFrame(
        data={
            "sentence": new_sentences,
            "sent_idx": new_sent_idx,
            "titles": new_titles,
            "qids": new_qids,
            "aliases": new_aliases,
            "spans": new_spans,
            "gold": new_golds,
            "sources": new_sources,
        }
    )
    return new_df, metrics, row_indices, row_indices


@argh.arg("input_file", help="input train file", type=str)
@argh.arg("output_file", help="output train file", type=str)
@argh.arg("--num_workers", help="parallelism", type=int)
@argh.arg("--overwrite", help="overwrite", action="store_true")
def main(input_file, output_file, num_workers=40, overwrite=False):
    """
    Each WL function returns either the original row (if LF didn't trigger) OR a row with new labels.

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

    lfs = list(wl_func.all.values())
    assert len(lfs) > 0, f"You must specify LFs via decorates to label data"
    print(f"Using LFs {[l.__name__ for l in lfs]}")
    new_df, metrics, old_indices, new_indices = weaklabel_data(
        train_df, lfs, cache_metadata_dir
    )
    print(f"Metrics: {ujson.dumps(metrics, indent=4)}")

    for lf_name, old_idx in old_indices.items():
        new_indx = new_indices[lf_name]
        temp_old = train_df.iloc[old_idx]
        temp_new = new_df.iloc[new_indx]
        print(lf_name)
        print("OLD")
        print(
            tabulate(
                temp_old[["sentence", "titles", "qids", "spans"]].head(
                    min(10, temp_old.shape[0])
                ),
                headers="keys",
                tablefmt="pretty",
            )
        )
        print("NEW")
        print(
            tabulate(
                temp_new[["sentence", "titles", "qids", "spans"]].head(
                    min(10, temp_new.shape[0])
                ),
                headers="keys",
                tablefmt="pretty",
            )
        )

    print(f"Writing out file to {output_file}")
    write_df_to_jsonl(new_df, output_file)


if __name__ == "__main__":
    argh.dispatch_command(main)
