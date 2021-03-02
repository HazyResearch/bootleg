"""
This file
1. Reads in raw wikipedia sentences from /lfs/raiders7/0/lorr1/sentences
2. Reads in map of WPID-Title-QID from /lfs/raiders7/0/lorr1/title_to_all_ids.jsonl
3. Computes frequencies for alias-QID mentions over Wikipedia. Keeps only alias-QID mentions which occur at least args.min_frequency times
4. Merges alias-QID map with alias-QID map extracted from Wikidata
2. Saves alias-qid map as alias_to_qid_filter.json to args.data_dir

After this, run remove_bad_aliases.py

Example run command:
python3.6 -m contextual_embeddings.bootleg_data_prep.curate_aliases
"""

import argparse
import glob
import multiprocessing
import os
import shutil
import time

import numpy as np
import ujson
import ujson as json
from tqdm import tqdm

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils


def get_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--contextual_cand_data",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/korealiases_title_1229/cands_for_laurel",
        help="Where files saved",
    )
    parser.add_argument(
        "--entity_dump",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/korealiases_title_1229/entity_db/entity_mappings",
        help="Where files saved",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/dfs/scratch0/lorr1/projects/bootleg-data/data/korealiases_title_1229",
        help="Where files saved",
    )
    parser.add_argument(
        "--out_subdir",
        type=str,
        default="contextual_cands",
        help="Where files saved",
    )
    parser.add_argument("--train_in_candidates", action="store_true")
    parser.add_argument(
        "--keep_orig",
        action="store_true",
        help="This will keep the original Bootleg maps but add contextual candidates to max out at 30",
    )
    parser.add_argument("--max_candidates", type=int, default=int(30))
    parser.add_argument("--processes", type=int, default=int(50))
    return parser


def init_process(entity_dump_f):
    global ed_global
    ed_global = EntitySymbols(load_dir=entity_dump_f)


def merge_data(
    num_processes,
    train_in_candidates,
    keep_orig,
    max_candidates,
    file_pairs,
    entity_dump_f,
):
    # File pair is in file, cand map file, out file, is_train

    # Chunk file for parallel writing
    create_ex_indir = os.path.join(
        os.path.dirname(file_pairs[0]), "_bootleg_temp_indir"
    )
    utils.ensure_dir(create_ex_indir)
    create_ex_indir_cands = os.path.join(
        os.path.dirname(file_pairs[0]), "_bootleg_temp_indir2"
    )
    utils.ensure_dir(create_ex_indir_cands)
    create_ex_outdir = os.path.join(
        os.path.dirname(file_pairs[0]), "_bootleg_temp_outdir"
    )
    utils.ensure_dir(create_ex_outdir)
    print(f"Counting lines")
    total_input = sum(1 for _ in open(file_pairs[0]))
    total_input_cands = sum(1 for _ in open(file_pairs[1]))
    assert (
        total_input_cands == total_input
    ), f"{total_input} lines of orig data != {total_input_cands} of cand data"
    chunk_input_size = int(np.ceil(total_input / num_processes))
    total_input_from_chunks, input_files_dict = utils.chunk_file(
        file_pairs[0], create_ex_indir, chunk_input_size
    )
    total_input_cands_from_chunks, input_files_cands_dict = utils.chunk_file(
        file_pairs[1], create_ex_indir_cands, chunk_input_size
    )

    input_files = list(input_files_dict.keys())
    input_cand_files = list(input_files_cands_dict.keys())
    assert len(input_cand_files) == len(input_files)
    input_file_lines = [input_files_dict[k] for k in input_files]
    input_cand_file_lines = [input_files_cands_dict[k] for k in input_cand_files]
    for p_l, p_r in zip(input_file_lines, input_cand_file_lines):
        assert (
            p_l == p_r
        ), f"The matching chunk files don't have matching sizes {p_l} versus {p_r}"
    output_files = [
        in_file_name.replace(create_ex_indir, create_ex_outdir)
        for in_file_name in input_files
    ]
    assert (
        total_input == total_input_from_chunks
    ), f"Lengths of files {total_input} doesn't match {total_input_from_chunks}"
    assert (
        total_input_cands == total_input_cands_from_chunks
    ), f"Lengths of files {total_input_cands} doesn't match {total_input_cands_from_chunks}"
    # file_pairs is input file, cand map file, output file, is_train
    input_args = [
        [
            train_in_candidates,
            keep_orig,
            max_candidates,
            input_files[i],
            input_file_lines[i],
            input_cand_files[i],
            output_files[i],
            file_pairs[3],
        ]
        for i in range(len(input_files))
    ]

    pool = multiprocessing.Pool(
        processes=num_processes, initializer=init_process, initargs=[entity_dump_f]
    )

    new_alias2qids = {}
    total_seen = 0
    total_dropped = 0
    for res in pool.imap(merge_data_hlp, input_args, chunksize=1):
        temp_alias2qids, seen, dropped = res
        total_seen += seen
        total_dropped += dropped
        for k in temp_alias2qids:
            assert k not in new_alias2qids, f"{k}"
            new_alias2qids[k] = temp_alias2qids[k]
    print(
        f"Overall Recall for {file_pairs[0]}: {(total_seen - total_dropped) / total_seen} for seeing {total_seen}"
    )
    # Merge output files to final file
    print(f"Merging output files")
    with open(file_pairs[2], "wb") as outfile:
        for filename in glob.glob(os.path.join(create_ex_outdir, "*")):
            if filename == file_pairs[2]:
                # don't want to copy the output into the output
                continue
            with open(filename, "rb") as readfile:
                shutil.copyfileobj(readfile, outfile)
    # Remove temporary files/folders
    shutil.rmtree(create_ex_indir)
    shutil.rmtree(create_ex_indir_cands)
    shutil.rmtree(create_ex_outdir)
    return new_alias2qids


def merge_data_hlp(args):
    (
        train_in_candidates,
        keep_orig,
        max_candidates,
        input_file,
        total_input,
        input_cand_file,
        output_file,
        is_train,
    ) = args
    sent2cands = {}
    sent2probs = {}
    new_alias2qids = {}
    with open(input_cand_file, "r") as f_in:
        for line in tqdm(f_in, total=total_input, desc="Processing cand data"):
            line = ujson.loads(line)
            sent2cands[line["sent_idx_unq"]] = line["ents"]
            if "probs" in line:
                sent2probs[line["sent_idx_unq"]] = line["probs"]
    total_dropped = 0
    total_seen = 0
    total_len = 0
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        tag = os.path.splitext(os.path.basename(input_file))[0]
        for line in tqdm(f_in, total=total_input, desc="Processing data"):
            line = ujson.loads(line)
            sent_idx_unq = line["sent_idx_unq"]
            if sent_idx_unq not in sent2cands:
                assert (
                    len(line["aliases"]) == 0
                ), f"{sent_idx_unq} not in cand maps but there are aliases"
            cands = sent2cands[sent_idx_unq]
            probs = sent2probs.get(
                sent_idx_unq,
                [[500 - j for j in range(len(cand_set))] for cand_set in cands],
            )
            assert len(cands) == len(
                line["aliases"]
            ), f"The length of aliases does not match cands in {sent_idx_unq}"
            assert len(probs) == len(
                line["aliases"]
            ), f"The length of aliases does not match probs in {sent_idx_unq}"

            new_als, new_qids, new_spans, new_golds = [], [], [], []
            new_slices = {}
            j = 0
            for i in range(len(line["aliases"])):
                total_seen += 1
                new_al = f"al_{sent_idx_unq}_{i}_{tag}"
                orig_cand_pairs = ed_global.get_qid_count_cands(line["aliases"][i])
                assert len(orig_cand_pairs) <= max_candidates
                new_cand_pairs = [
                    [c, p]
                    for c, p in zip(cands[i], probs[i])
                    if ed_global.qid_exists(c)
                ]
                if keep_orig:
                    final_cand_pairs = orig_cand_pairs
                    final_cand_set = set(map(lambda x: x[0], final_cand_pairs))
                    for ctx_q, ctx_val in sorted(
                        new_cand_pairs, key=lambda x: x[1], reverse=False
                    ):
                        if len(final_cand_pairs) >= max_candidates:
                            break
                        if ctx_q not in final_cand_set:
                            final_cand_pairs.append([ctx_q, ctx_val])
                else:
                    final_cand_pairs = new_cand_pairs[:max_candidates]
                total_len += len(final_cand_pairs)
                # We are training in candidates and gold is not in list, discard
                if (
                    is_train
                    and train_in_candidates
                    and line["qids"][i] not in [p[0] for p in final_cand_pairs]
                ):
                    total_dropped += 1
                    continue
                new_alias2qids[new_al] = final_cand_pairs
                new_als.append(new_al)
                new_qids.append(line["qids"][i])
                new_spans.append(line["spans"][i])
                new_golds.append(line["gold"][i])
                for slice_name in line["slices"]:
                    if slice_name not in new_slices:
                        new_slices[slice_name] = {}
                    new_slices[slice_name][str(j)] = line["slices"][slice_name][str(i)]
                j += 1
            line["old_aliases"] = line["aliases"][:]
            line["aliases"] = new_als
            line["qids"] = new_qids
            line["spans"] = new_spans
            line["gold"] = new_golds
            line["slices"] = new_slices
            f_out.write(ujson.dumps(line) + "\n")
    print(
        f"Total Seen: {total_seen}, Total Dropped: {total_dropped}, Recall: {(total_seen - total_dropped) / total_seen}, Avg Cand Len: {total_len / (total_seen)} for {input_file}"
    )
    return new_alias2qids, total_seen, total_dropped


def main():
    gl_start = time.time()
    multiprocessing.set_start_method("spawn")
    args = get_arg_parser().parse_args()
    print(json.dumps(vars(args), indent=4))
    utils.ensure_dir(args.data_dir)

    out_dir = os.path.join(args.data_dir, args.out_subdir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # Reading in files
    in_files_train = glob.glob(os.path.join(args.data_dir, "*.jsonl"))
    in_files_cand = glob.glob(os.path.join(args.contextual_cand_data, "*.jsonl"))
    in_files = []
    for file in in_files_train:
        file_name = os.path.basename(file)
        tag = os.path.splitext(file_name)[0]
        is_train = "train" in tag
        if is_train:
            print(f"{file_name} is a training dataset...will be processed as such")
        pair = None
        for f in in_files_cand:
            if tag in f:
                pair = f
                break
        assert pair is not None, f"{file_name} name, {tag} tag"
        out_file = os.path.join(out_dir, file_name)
        in_files.append([file, pair, out_file, is_train])
    final_cand_map = {}
    max_cands = 0
    for pair in in_files:
        print(f"Reading in {pair[0]} with cand maps {pair[1]} and dumping to {pair[2]}")
        new_alias2qids = merge_data(
            args.processes,
            args.train_in_candidates,
            args.keep_orig,
            args.max_candidates,
            pair,
            args.entity_dump,
        )
        for al in new_alias2qids:
            assert al not in final_cand_map, f"{al} is already in final_cand_map"
            final_cand_map[al] = new_alias2qids[al]
            max_cands = max(max_cands, len(final_cand_map[al]))

    print(f"Buidling new entity symbols")
    entity_dump = EntitySymbols(load_dir=args.entity_dump)
    entity_dump_new = EntitySymbols(
        max_candidates=max_cands,
        max_alias_len=1,
        alias2qids=final_cand_map,
        qid2title=entity_dump.get_qid2title(),
    )
    out_dir = os.path.join(out_dir, "entity_db/entity_mappings")
    entity_dump_new.dump(out_dir)
    print(f"Finished in {time.time() - gl_start}s")


if __name__ == "__main__":
    main()
