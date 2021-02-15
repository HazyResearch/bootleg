import random
from pathlib import Path

import argh
import numpy as np
import ujson as json


def convert_data(file_in, file_out):
    """
    Load data from json files, preprocess and prepare batches.
    """
    with open(file_in) as infile:
        data = json.load(infile)
        print(len(data))

    with open(file_out, "w") as outfile:
        for d in data:
            tokens = d["token"]
            example = " ".join(tokens)
            entry = {"sentence": example}
            json.dump(entry, outfile)
            outfile.write("\n")


def normalize_glove(tokens):
    mapping = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    for i in range(len(tokens)):
        if tokens[i] in mapping:
            # print(tokens)
            tokens[i] = mapping[tokens[i]]
    return tokens


def extract_subj_obj(tokens, d):
    masked_tokens = ["-"] * len(tokens)
    ss, se = d["subj_start"], d["subj_end"]
    os, oe = d["obj_start"], d["obj_end"]
    masked_tokens[ss : se + 1] = tokens[ss : se + 1]
    masked_tokens[os : oe + 1] = tokens[os : oe + 1]
    return masked_tokens


def create_one_file(train, dev, test, all_out, subjobj=False):
    with open(train) as infile:
        data_train = json.load(infile)
        print("TRAIN LEN:", len(data_train))

    with open(dev) as infile:
        data_dev = json.load(infile)
        print("DEV LEN:", len(data_dev))

    with open(test) as infile:
        data_test = json.load(infile)
        print("TEST LEN:", len(data_test))

    with open(all_out, "w") as outfile:
        for d in data_train:
            tokens = d["token"]
            tokens = normalize_glove(tokens)
            if subjobj:
                tokens = extract_subj_obj(tokens, d)
            example = " ".join(tokens)
            entry = {"sentence": example, "id": d["id"]}
            json.dump(entry, outfile)
            outfile.write("\n")

        for d in data_dev:
            tokens = d["token"]
            tokens = normalize_glove(tokens)
            if subjobj:
                tokens = extract_subj_obj(tokens, d)
            example = " ".join(tokens)
            entry = {"sentence": example, "id": d["id"]}
            json.dump(entry, outfile)
            outfile.write("\n")

        for d in data_test:
            tokens = d["token"]
            tokens = normalize_glove(tokens)
            if subjobj:
                tokens = extract_subj_obj(tokens, d)
            example = " ".join(tokens)
            entry = {"sentence": example, "id": d["id"]}
            json.dump(entry, outfile)
            outfile.write("\n")
    print("Data is saved to", all_out)


@argh.arg("source_path", help="Path to tacred data")
def main(source_path="/dfs/scratch0/lorr1/projects/bootleg-data/downstream/tacred/"):
    inname_train = Path(source_path) / "train.json"
    inname_dev = Path(source_path) / "dev.json"
    inname_test = Path(source_path) / "test.json"
    outname_all = Path(source_path) / "all_tacred_bootinput.jsonl"
    create_one_file(inname_train, inname_dev, inname_test, outname_all, subjobj=False)


if __name__ == "__main__":
    argh.dispatch_command(main)
