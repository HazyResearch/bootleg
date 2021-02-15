import argparse
import csv
import json
import os
import pickle
import random
import sys
from collections import OrderedDict, defaultdict

import jsonlines
import numpy as np
import pandas as pd
import ujson
from tqdm import tqdm


def load_mentions(file):
    lines = []
    with jsonlines.open(file) as f:
        for line in f:
            new_line = {
                "id": line["id"],
                "sentence": line["sentence"],
                "aliases": line["aliases"],
                "spans": line["spans"],
                "gold": line["gold"],
                "cand_probs": line["cand_probs"],
                "qids": line["qids"],
                "sent_idx_unq": line["sent_idx_unq"],
                "probs": line["probs"],
                "ctx_emb_ids": line["ctx_emb_ids"],
                "entity_ids": line["entity_ids"],
            }
            lines.append(new_line)
    return pd.DataFrame(lines)


def generate_features(bootleg_labels_df, threshold):
    ctx_emb_id_dict = {}
    ctx_emb_id_dict_first = {}
    qid_dict = {}
    qid_dict_first = {}

    num_removed = 0
    for ind, row in bootleg_labels_df.iterrows():
        ctx_emb_ids = row["ctx_emb_ids"]
        qids = row["qids"]
        spans = row["spans"]

        # get sentence length
        example = row["sentence"]
        tokens = example.split(" ")
        length = len(tokens)

        # initialize result datastructures
        ctx_emb_id_result = [-1] * length
        qid_result = ["UNK"] * length

        ctx_emb_id_result_first = [-1] * length
        qid_result_first = ["UNK"] * length

        for i in range(len(spans)):
            span = spans[i]
            start, end = span[0], span[1]
            span_len = end - start

            prob = row["probs"][i]
            if prob < threshold:
                num_removed += 1
                continue

            # contextual
            ctx_emb_id = ctx_emb_ids[i]
            ctx_emb_id_lst = [ctx_emb_id] * span_len
            ctx_emb_id_result[start:end] = ctx_emb_id_lst
            ctx_emb_id_result_first[start] = ctx_emb_id

            # qids
            qid = qids[i]
            qid_lst = [qid] * span_len
            qid_result[start:end] = qid_lst
            qid_result_first[start] = qid

        idx = row["id"]
        if idx in ctx_emb_id_dict:
            raise ValueError("ERROR: duplicate indices!")

        ctx_emb_id_dict[idx] = ctx_emb_id_result
        qid_dict[idx] = qid_result

        ctx_emb_id_dict_first[idx] = ctx_emb_id_result_first
        qid_dict_first[idx] = qid_result_first
    print(
        f"Removed {num_removed} out of {bootleg_labels_df.shape[0]} with threshold {threshold}"
    )
    return ctx_emb_id_dict, qid_dict, ctx_emb_id_dict_first, qid_dict_first


def main(bootleg_directory, base_data, threshold=0.0):
    # load the features to add
    boot_labels_file = os.path.join(bootleg_directory, "bootleg_labels.jsonl")
    bootleg_labels_df = load_mentions(boot_labels_file)
    print(bootleg_labels_df.columns)
    print(bootleg_labels_df.shape)

    # load the base tacred data
    train_file = "{}/train.json".format(base_data)
    with open(train_file) as train:
        df_train = json.load(train)
        df_train = pd.DataFrame.from_dict(df_train, orient="columns")
        print("TRAIN SHAPE: ", df_train.shape)

    dev_file = "{}/dev.json".format(base_data)
    with open(dev_file) as dev:
        df_dev = json.load(dev)
        df_dev = pd.DataFrame.from_dict(df_dev, orient="columns")
        print("DEV SHAPE: ", df_dev.shape)

    test_file = "{}/test.json".format(base_data)
    with open(test_file) as test:
        df_test = json.load(test)
        df_test = pd.DataFrame.from_dict(df_test, orient="columns")
        print("TEST SHAPE", df_test.shape)

    (
        ctx_emb_id_dict,
        qid_dict,
        ctx_emb_id_dict_first,
        qid_dict_first,
    ) = generate_features(bootleg_labels_df, threshold)

    # add features to the data
    dfs = [df_train, df_dev, df_test]
    for df in dfs:
        df["entity_emb_id"] = np.nan
        df["entity_emb_id_first"] = np.nan
        df["ent_id"] = np.nan
        df["ent_id_first"] = np.nan

        dict_ctx_emb_id = {}
        dict_ctx_emb_id_first = {}
        dict_qid = {}
        dict_qid_first = {}

        for ind, row in df.iterrows():
            idx = row["id"]
            tokens = row["token"]
            length = len(tokens)

            # initialize result datastructures
            ctx_emb_id_default = [-1] * length
            qid_default = ["UNK"] * length

            # contextual
            if idx in ctx_emb_id_dict:
                dict_ctx_emb_id[idx] = ctx_emb_id_dict[idx]
            else:
                dict_ctx_emb_id[idx] = ctx_emb_id_default

            if idx in ctx_emb_id_dict_first:
                dict_ctx_emb_id_first[idx] = ctx_emb_id_dict_first[idx]
            else:
                dict_ctx_emb_id_first[idx] = ctx_emb_id_default

            # qids
            if idx in qid_dict:
                dict_qid[idx] = qid_dict[idx]
            else:
                dict_qid[idx] = qid_default

            if idx in qid_dict_first:
                dict_qid_first[idx] = qid_dict_first[idx]
            else:
                dict_qid_first[idx] = qid_default

        assert len(dict_ctx_emb_id.keys()) == df.shape[0]
        assert len(dict_ctx_emb_id_first.keys()) == df.shape[0]
        assert len(dict_qid.keys()) == df.shape[0]
        assert len(dict_qid_first.keys()) == df.shape[0]
        df["entity_emb_id"] = df["id"].map(dict_ctx_emb_id)
        df["entity_emb_id_first"] = df["id"].map(dict_ctx_emb_id_first)
        df["ent_id"] = df["id"].map(dict_qid)
        df["ent_id_first"] = df["id"].map(dict_qid_first)

    # Save tacred data with Bootleg features
    train_out = df_train.to_json(
        r"{}/train_ent.json".format(base_data), orient="records"
    )
    dev_out = df_dev.to_json(r"{}/dev_ent.json".format(base_data), orient="records")
    test_out = df_test.to_json(r"{}/test_ent.json".format(base_data), orient="records")
    print(
        "Saved datasets with Bootleg features to train_ent.json, dev_ent.json, test_ent.json in",
        base_data,
        "directory",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootleg_directory",
        type=str,
        help="location where ctx_embeddings.npy and bootleg_labels.jsonl saved",
    )
    parser.add_argument(
        "--tacred_directory",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/downstream/tacred",
        help="location where base tacred data is located",
    )
    args = parser.parse_args()

    main(args.bootleg_directory, args.tacred_directory)
