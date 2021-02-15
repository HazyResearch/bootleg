"""Builds our hierarchical regularization where an entity is regularization
proportional to the power of its popularity in the training data.

The output of this is a csv file with columns QID and regulatization. This is then used in the config for an Entity Embedding.

ent_embeddings:
   - key: learned
     load_class: LearnedEntityEmb
     freeze: false
     cpu: true
     args:
       learned_embedding_size: 256
       regularize_mapping: <path to csv regularization files>
"""

import argparse
import multiprocessing
import os
import random
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import ujson
from tqdm import tqdm

from bootleg.symbols.entity_symbols import EntitySymbols

REG = lambda x: 0.95 * (np.power(x, -0.32))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument(
        "--entity_dir",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/tacred_title_0122/entity_db",
        help="Path to entities inside data_dir",
    )
    parser.add_argument(
        "--entity_map_dir",
        type=str,
        default="entity_mappings",
        help="Path to entities inside data_dir",
    )
    parser.add_argument(
        "--alias_cand_map",
        type=str,
        default="alias2qids.json",
        help="Path to alias candidate map",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/tacred_title_0122",
        help="Where files saved",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/tacred_title_0122/qid2reg_pow.csv",
        help="Regularization of each qid",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=50,
        help="Number of processes",
    )

    args = parser.parse_args()
    return args


def get_counts(num_processes, file):
    """Gets true anchor slice counts."""
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
    return qid_cnts


def get_counts_hlp(line):
    res = defaultdict(int)  # qid -> cnt
    line = ujson.loads(line)
    for qid in line["qids"]:
        res[qid] += 1
    return res


def build_reg_csv(qid_cnt, es):
    regs = []
    qids = []
    for qid in es.get_all_qids():
        r = 0.05
        cnt = qid_cnt.get(qid, 0)
        if cnt <= 1:
            r = 0.95
        elif cnt > 10000:
            r = 0.05
        else:
            r = REG(cnt)
        qids.append(qid)
        regs.append(r)

    df = pd.DataFrame(data={"qid": qids, "regularization": regs})
    return df


def main():
    args = parse_args()
    print(ujson.dumps(args, indent=4))
    num_processes = min(args.processes, int(0.8 * multiprocessing.cpu_count()))
    print("Loading entity symbols")
    entity_symbols = EntitySymbols(
        os.path.join(args.entity_dir, args.entity_map_dir),
        alias_cand_map_file=args.alias_cand_map,
    )

    in_file = os.path.join(args.data_dir, args.train_file)
    print(f"Getting slice counts from {in_file}")
    qid_cnts = get_counts(num_processes, in_file)
    with open(os.path.join(args.data_dir, "qid_cnts_train.json"), "w") as out_f:
        ujson.dump(qid_cnts, out_f)
    df = build_reg_csv(qid_cnts, entity_symbols)

    df.to_csv(args.out_file, index=False)
    print(f"Saved file to {args.out_file}")


if __name__ == "__main__":
    main()
