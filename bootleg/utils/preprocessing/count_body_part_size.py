"""Util file to count the number of mentions in each "body part" (head, torso,
tail, toes) such that they have more than one candidate."""

import argparse
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path

import ujson
from tqdm import tqdm

FINAL_COUNTS_PREFIX = "final_counts"
FINAL_SLICE_TO_SENT_PREFIX = "final_slice_to_sent"
FINAL_SENT_TO_SLICE_PREFIX = "final_sent_to_slices"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/lfs/raiders10/0/lorr1/data/wiki_dump/wiki_title_1229/",
        help="Where files saved",
    )

    parser.add_argument("--processes", type=int, default=20)
    args = parser.parse_args()
    return args


def init_pool(qid_cnt, a2q_f):
    global qid_cnt_global
    qid_cnt_global = qid_cnt
    global a2q_global
    a2q_global = ujson.load(open(a2q_f))


def get_slice_counts(num_processes, qid_cnt, a2q_f, files):
    """Gets true anchor slice counts."""
    pool = multiprocessing.Pool(
        processes=num_processes, initializer=init_pool, initargs=[qid_cnt, a2q_f]
    )
    final_counts = Counter()
    for res in tqdm(
        pool.imap_unordered(get_slice_counts_hlp, files, chunksize=1),
        total=len(files),
        desc="Gathering slice counts",
    ):
        final_counts.update(res)

    return dict(final_counts)


def get_slice_counts_hlp(args):
    file = args
    cnts = Counter()
    with open(file) as in_f:
        for line in in_f:
            line = ujson.loads(line)
            for (a, al, q) in zip(line["gold"], line["aliases"], line["qids"]):
                cnt = qid_cnt_global.get(q, 0)
                if a:
                    if cnt <= 0:
                        cnts["toes"] += 1
                        if len(a2q_global[al]) > 1:
                            cnts["toes_NS"] += 1
                    if 0 <= cnt <= 10:
                        cnts["tail"] += 1
                        if len(a2q_global[al]) > 1:
                            cnts["tail_NS"] += 1
                    if 11 <= cnt <= 1000:
                        cnts["torso"] += 1
                        if len(a2q_global[al]) > 1:
                            cnts["torso_NS"] += 1
                    if 1001 <= cnt:
                        cnts["head"] += 1
                        if len(a2q_global[al]) > 1:
                            cnts["head_NS"] += 1
    return cnts


def get_counts(num_processes, files):
    """Gets true anchor slice counts."""
    pool = multiprocessing.Pool(processes=num_processes)
    qid_cnts = defaultdict(int)
    for res in tqdm(
        pool.imap_unordered(get_counts_hlp, files, chunksize=1),
        total=len(files),
        desc="Gathering counts",
    ):
        for qid in res:
            qid_cnts[qid] += res[qid]
    return qid_cnts


def get_counts_hlp(file):
    res = defaultdict(int)  # qid -> cnt
    with open(file) as in_f:
        for line in in_f:
            line = ujson.loads(line)
            for qid in line["qids"]:
                res[qid] += 1
    return res


def main():
    args = parse_args()
    print(ujson.dumps(vars(args), indent=4))
    num_processes = min(args.processes, int(0.8 * multiprocessing.cpu_count()))
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    train_files = list(train_dir.glob("*.jsonl"))
    print(f"Getting slice counts from {len(train_files)} train files")
    qid_cnts = get_counts(num_processes, train_files)

    a2q_f = data_dir / "entity_db" / "entity_mappings" / "alias2qids.json"

    print("Getting slice counts from test and dev")
    for prefix in ["test", "dev"]:
        subfolder = data_dir / prefix
        input_files = list(subfolder.glob("*.jsonl"))
        slice_counts = get_slice_counts(num_processes, qid_cnts, a2q_f, input_files)

        print(f"****FINAL SLICE COUNTS {prefix}*****")
        print(ujson.dumps(slice_counts, indent=4))


if __name__ == "__main__":
    main()
