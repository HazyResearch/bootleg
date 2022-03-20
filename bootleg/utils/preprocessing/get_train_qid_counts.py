"""
Compute QID counts.

Helper function that computes a dictionary of QID -> count in training data.

If a QID is not in this dictionary, it has a count of zero.
"""

import argparse
import multiprocessing
from collections import defaultdict

import ujson
from tqdm import tqdm

from bootleg.utils import utils


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/wiki_title_0114/train.jsonl",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/wiki_title_0114/train_qidcnt.json",
        help="Regularization of each qid",
    )

    args = parser.parse_args()
    return args


def get_counts(num_processes, file):
    """Get true anchor slice counts."""
    pool = multiprocessing.Pool(processes=num_processes)
    num_lines = sum(1 for _ in open(file))
    qid_cnts = defaultdict(int)
    for res in tqdm(
        pool.imap_unordered(get_counts_hlp, open(file), chunksize=1000),
        total=num_lines,
        desc="Gathering counts",
    ):
        for qid in res:
            qid_cnts[qid] += res[qid]
    pool.close()
    pool.join()
    return qid_cnts


def get_counts_hlp(line):
    """Get count helper."""
    res = defaultdict(int)  # qid -> cnt
    line = ujson.loads(line)
    for qid in line["qids"]:
        res[qid] += 1
    return res


def main():
    """Run."""
    args = parse_args()
    print(ujson.dumps(vars(args), indent=4))
    num_processes = int(0.8 * multiprocessing.cpu_count())
    print(f"Getting slice counts from {args.train_file}")
    qid_cnts = get_counts(num_processes, args.train_file)
    utils.dump_json_file(args.out_file, qid_cnts)


if __name__ == "__main__":
    main()
