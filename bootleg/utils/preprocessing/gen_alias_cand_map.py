import argparse
from collections import defaultdict

import jsonlines
import ujson
from tqdm import tqdm

from bootleg.utils.utils import get_lnrm

# generates the alias candidate map from a provided a training file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alias2qids_file", type=str, required=True, help="Path to write alias2qids"
    )
    parser.add_argument(
        "--train_file", type=str, required=True, help="Path for train file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    alias2qids_dict = defaultdict(set)
    qid2freq = defaultdict(int)
    with jsonlines.open(args.train_file) as f:
        for line in f:
            # this includes weakly labelled aliases
            for qid, alias in zip(line["qids"], line["aliases"]):
                # aliases are lower-cased
                alias2qids_dict[get_lnrm(alias, strip=True, lower=True)].add(qid)
                qid2freq[qid] += 1

    alias2qids = {}
    for al in tqdm(alias2qids_dict):
        qid_cands = [[q, qid2freq[q]] for q in alias2qids_dict[al]]
        qid_cands = sorted(qid_cands, key=lambda x: x[1], reverse=True)
        alias2qids[al] = qid_cands

    with open(args.alias2qids_file, "w") as f:
        ujson.dump(alias2qids, f)


if __name__ == "__main__":
    main()
