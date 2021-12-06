"""
Sample eval data.

This will sample a jsonl train or eval data based on the slices in the data.
This is useful for subsampling a smaller eval dataset.py.

The output of this file is a files with a subset of sentences from the
input file samples such that for each slice in --args.slice, a minimum
of args.min_sample_size mentions are in the slice (if possible). Once
that is satisfied, we sample to get approximately --args.sample_perc of
mentions from each slice.
"""

import argparse
import multiprocessing
import os
import random
import shutil
from collections import defaultdict

import numpy as np
import ujson
from tqdm import tqdm

from bootleg.utils import utils

FINAL_COUNTS_PREFIX = "final_counts"
FINAL_SLICE_TO_SENT_PREFIX = "final_slice_to_sent"
FINAL_SENT_TO_SLICE_PREFIX = "final_sent_to_slices"


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="merged.jsonl")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/wiki_title_0122",
        help="Where files saved",
    )
    parser.add_argument(
        "--out_file_name",
        type=str,
        default="merged_sample.jsonl",
        help="Where files saved",
    )
    parser.add_argument(
        "--sample_perc", type=float, default=0.005, help="Perc of each slice to sample"
    )
    parser.add_argument(
        "--min_sample_size",
        type=int,
        default=5000,
        help="Min number of mentions per slice",
    )
    parser.add_argument(
        "--slice",
        default=[],
        action="append",
        required=True,
        help="Slices to consider when sampling",
    )

    args = parser.parse_args()
    return args


def get_slice_stats(num_processes, file):
    """Get true anchor slice counts."""
    pool = multiprocessing.Pool(processes=num_processes)
    final_counts = defaultdict(int)
    final_slice_to_sent = defaultdict(set)
    final_sent_to_slices = defaultdict(lambda: defaultdict(int))
    temp_out_dir = os.path.join(os.path.dirname(file), "_temp")
    os.mkdir(temp_out_dir)

    all_lines = [li for li in open(file, encoding="utf-8")]
    num_lines = len(all_lines)
    chunk_size = int(np.ceil(num_lines / num_processes))
    line_chunks = [
        all_lines[i : i + chunk_size] for i in range(0, num_lines, chunk_size)
    ]

    input_args = [
        [i, line_chunks[i], i * chunk_size, temp_out_dir]
        for i in range(len(line_chunks))
    ]

    for i in tqdm(
        pool.imap_unordered(get_slice_stats_hlp, input_args, chunksize=1),
        total=len(line_chunks),
        desc="Gathering slice counts",
    ):
        cnt_res = utils.load_json_file(
            os.path.join(temp_out_dir, f"{FINAL_COUNTS_PREFIX}_{i}.json")
        )
        sent_to_slices = utils.load_json_file(
            os.path.join(temp_out_dir, f"{FINAL_SENT_TO_SLICE_PREFIX}_{i}.json")
        )
        slice_to_sent = utils.load_json_file(
            os.path.join(temp_out_dir, f"{FINAL_SLICE_TO_SENT_PREFIX}_{i}.json")
        )
        for k in cnt_res:
            final_counts[k] += cnt_res[k]
        for k in slice_to_sent:
            final_slice_to_sent[k].update(set(slice_to_sent[k]))
        for k in sent_to_slices:
            final_sent_to_slices[k].update(sent_to_slices[k])
    shutil.rmtree(temp_out_dir)
    return dict(final_counts), dict(final_slice_to_sent), dict(final_sent_to_slices)


def get_slice_stats_hlp(args):
    """Get slice count helper."""
    i, lines, offset, temp_out_dir = args

    res = defaultdict(int)  # slice -> cnt
    slice_to_sent = defaultdict(set)  # slice -> sent_idx (for sampling)
    sent_to_slices = defaultdict(
        lambda: defaultdict(int)
    )  # sent_idx -> slice -> cnt (for sampling)
    for line in tqdm(lines, total=len(lines), desc=f"Processing lines for {i}"):
        line = ujson.loads(line)
        slices = line.get("slices", {})
        anchors = line["gold"]
        for slice_name in slices:
            for al_str in slices[slice_name]:
                if anchors[int(al_str)] is True and slices[slice_name][al_str] > 0.5:
                    res[slice_name] += 1
                    slice_to_sent[slice_name].add(int(line["sent_idx_unq"]))
        sent_to_slices[int(line["sent_idx_unq"])].update(res)

    utils.dump_json_file(
        os.path.join(temp_out_dir, f"{FINAL_COUNTS_PREFIX}_{i}.json"), res
    )
    utils.dump_json_file(
        os.path.join(temp_out_dir, f"{FINAL_SENT_TO_SLICE_PREFIX}_{i}.json"),
        sent_to_slices,
    )
    # Turn into list for dumping
    for slice_name in slice_to_sent:
        slice_to_sent[slice_name] = list(slice_to_sent[slice_name])
    utils.dump_json_file(
        os.path.join(temp_out_dir, f"{FINAL_SLICE_TO_SENT_PREFIX}_{i}.json"),
        slice_to_sent,
    )
    return i


def main():
    """Run."""
    args = parse_args()
    print(ujson.dumps(vars(args), indent=4))
    num_processes = int(0.8 * multiprocessing.cpu_count())

    in_file = os.path.join(args.data_dir, args.file)
    print(f"Getting slice counts from {in_file}")
    slice_counts, slice_to_sents, sent_to_slices = get_slice_stats(
        num_processes, in_file
    )

    print("****SLICE COUNTS*****")
    print(ujson.dumps(slice_counts, indent=4))

    desired_slices = args.slice
    final_sentences = set()
    new_counts = defaultdict(int)
    for sl_name in desired_slices:
        cur_count = new_counts[sl_name]
        sample_size = max(
            min(args.min_sample_size - cur_count, len(slice_to_sents[sl_name])),
            min(
                int(args.sample_perc * slice_counts[sl_name]) - cur_count,
                len(slice_to_sents[sl_name]),
            ),
            0,
        )
        if sample_size > 0:
            sents_to_add = random.sample(list(slice_to_sents[sl_name]), k=sample_size)
            final_sentences.update(sents_to_add)
            new_counts = defaultdict(int)
            for sent_id in final_sentences:
                for sl_name2 in sent_to_slices.get(sent_id, {}):
                    new_counts[sl_name2] += sent_to_slices.get(sent_id, {}).get(
                        sl_name2, 0
                    )

    out_file = os.path.join(args.data_dir, args.out_file_name)
    print(f"Outputting results to {out_file}")
    num_lines = sum([1 for _ in open(in_file, encoding="utf-8")])
    final_cnt = 0
    final_slice_cnts = defaultdict(int)
    with open(out_file, "w", encoding="utf-8") as out_f:
        for line in tqdm(
            [ujson.loads(li.strip()) for li in open(in_file, encoding="utf-8")],
            desc="Writing out file",
            total=num_lines,
        ):
            if int(line["sent_idx_unq"]) in final_sentences:
                out_f.write(ujson.dumps(line) + "\n")
                for sl_name in line.get("slices", {}):
                    for al_idx in line["slices"][sl_name]:
                        if (
                            line["slices"][sl_name][al_idx] > 0.5
                            and line["gold"][int(al_idx)] is True
                        ):
                            final_slice_cnts[sl_name] += 1
                final_cnt += 1
    print(f"Wrote out {final_cnt} lines to {out_file}")
    print("****FINAL SLICE COUNTS*****")
    print(ujson.dumps(final_slice_cnts, indent=4))


if __name__ == "__main__":
    main()
