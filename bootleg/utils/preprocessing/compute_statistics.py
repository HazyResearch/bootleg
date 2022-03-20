"""
Compute statistics over data.

Helper file for computing various statistics over our data such as mention
frequency, mention text frequency in the data (even if not labeled as an
anchor), ...

etc.
"""

import argparse
import logging
import multiprocessing
import os
import time
from collections import Counter

import marisa_trie
import nltk
import numpy as np
import ujson
import ujson as json
from tqdm import tqdm

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils
from bootleg.utils.utils import get_lnrm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Data dir for training data"
    )
    parser.add_argument(
        "--save_dir", type=str, default="data/", help="Data dir for saving stats"
    )
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument(
        "--entity_symbols_dir",
        type=str,
        default="entity_db/entity_mappings",
        help="Path to entities inside data_dir",
    )
    parser.add_argument("--lower", action="store_true", help="Lower aliases")
    parser.add_argument("--strip", action="store_true", help="Strip punc aliases")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers to parallelize", default=2
    )
    args = parser.parse_args()
    return args


def compute_histograms(save_dir, entity_symbols):
    """Compute histogram."""
    al_counts = Counter()
    for al in entity_symbols.get_all_aliases():
        num_entities = len(entity_symbols.get_qid_cands(al))
        al_counts.update([num_entities])
    utils.dump_json_file(
        filename=os.path.join(save_dir, "candidate_counts.json"), contents=al_counts
    )
    return


def get_num_lines(input_src):
    """Get number of lines."""
    # get number of lines
    num_lines = 0
    with open(input_src, "r", encoding="utf-8") as in_file:
        try:
            for line in in_file:
                num_lines += 1
        except Exception as e:
            logging.error("ERROR READING IN TRAINING DATA")
            logging.error(e)
            return []
    return num_lines


def chunk_text_data(input_src, chunk_files, chunk_size, num_lines):
    """Chunk text data."""
    logging.info(f"Reading in {input_src}")
    start = time.time()
    # write out chunks as text data
    chunk_id = 0
    num_lines_in_chunk = 0
    # keep track of what files are written
    out_file = open(chunk_files[chunk_id], "w")
    with open(input_src, "r", encoding="utf-8") as in_file:
        for i, line in enumerate(in_file):
            out_file.write(line)
            num_lines_in_chunk += 1
            # move on to new chunk when it hits chunk size
            if num_lines_in_chunk == chunk_size:
                chunk_id += 1
                # reset number of lines in chunk and open new file if not at end
                num_lines_in_chunk = 0
                out_file.close()
                if i < (num_lines - 1):
                    out_file = open(chunk_files[chunk_id], "w")
    out_file.close()
    logging.info(f"Wrote out data chunks in {round(time.time() - start, 2)}s")


def compute_occurrences_single(args, max_alias_len=6):
    """Compute statistics single process."""
    data_file, aliases_file, lower, strip = args
    num_lines = sum(1 for _ in open(data_file))
    all_aliases = ujson.load(open(aliases_file))
    all_aliases = marisa_trie.Trie(all_aliases)
    # entity histogram
    ent_occurrences = Counter()
    # alias histogram
    alias_occurrences = Counter()
    # alias text occurrances
    alias_text_occurrences = Counter()
    # number of aliases per sentence
    alias_pair_occurrences = Counter()
    # alias|entity histogram
    alias_entity_pair = Counter()
    with open(data_file, "r") as in_file:
        for line in tqdm(in_file, total=num_lines):
            line = json.loads(line.strip())
            for n in range(max_alias_len + 1, 0, -1):
                grams = nltk.ngrams(line["sentence"].split(), n)
                for gram_words in grams:
                    gram_attempt = get_lnrm(" ".join(gram_words), lower, strip)
                    if gram_attempt in all_aliases:
                        alias_text_occurrences[gram_attempt] += 1
            # Get aliases in wikipedia _before_ the swapping - these represent the true textual aliases
            aliases = line["unswap_aliases"]
            qids = line["qids"]
            for qid, alias in zip(qids, aliases):
                ent_occurrences[qid] += 1
                alias_occurrences[alias] += 1
                alias_entity_pair[alias + "|" + qid] += 1
            alias_pair_occurrences[len(aliases)] += 1
    results = {
        "ent_occurrences": ent_occurrences,
        "alias_occurrences": alias_occurrences,
        "alias_text_occurrences": alias_text_occurrences,
        "alias_pair_occurrences": alias_pair_occurrences,
        "alias_entity_pair": alias_entity_pair,
    }
    return results


def compute_occurrences(save_dir, data_file, entity_dump, lower, strip, num_workers=8):
    """Compute statistics."""
    all_aliases = entity_dump.get_all_aliases()
    chunk_file_path = os.path.join(save_dir, "tmp")
    all_aliases_f = os.path.join(chunk_file_path, "all_aliases.json")
    utils.ensure_dir(chunk_file_path)
    ujson.dump(all_aliases, open(all_aliases_f, "w"), ensure_ascii=False)
    # divide up data into chunks
    num_lines = get_num_lines(data_file)
    num_processes = min(num_workers, int(multiprocessing.cpu_count()))
    logging.info(f"Using {num_processes} workers...")
    chunk_size = int(np.ceil(num_lines / (num_processes)))
    utils.ensure_dir(chunk_file_path)
    chunk_infiles = [
        os.path.join(f"{chunk_file_path}", f"data_chunk_{chunk_id}_in.jsonl")
        for chunk_id in range(num_processes)
    ]
    chunk_text_data(data_file, chunk_infiles, chunk_size, num_lines)

    pool = multiprocessing.Pool(processes=num_processes)
    subprocess_args = [
        [chunk_infiles[i], all_aliases_f, lower, strip] for i in range(num_processes)
    ]
    results = pool.map(compute_occurrences_single, subprocess_args)
    pool.close()
    pool.join()
    logging.info("Finished collecting counts")
    logging.info("Merging counts....")
    # merge counters together
    ent_occurrences = Counter()
    # alias histogram
    alias_occurrences = Counter()
    # alias text occurrances
    alias_text_occurrences = Counter()
    # number of aliases per sentence
    alias_pair_occurrences = Counter()
    # alias|entity histogram
    alias_entity_pair = Counter()
    for result_set in tqdm(results, desc="Merging"):
        ent_occurrences += result_set["ent_occurrences"]
        alias_occurrences += result_set["alias_occurrences"]
        alias_text_occurrences += result_set["alias_text_occurrences"]
        alias_pair_occurrences += result_set["alias_pair_occurrences"]
        alias_entity_pair += result_set["alias_entity_pair"]
    # save counters
    utils.dump_json_file(
        filename=os.path.join(save_dir, "entity_count.json"), contents=ent_occurrences
    )
    utils.dump_json_file(
        filename=os.path.join(save_dir, "alias_counts.json"), contents=alias_occurrences
    )
    utils.dump_json_file(
        filename=os.path.join(save_dir, "alias_text_counts.json"),
        contents=alias_text_occurrences,
    )
    utils.dump_json_file(
        filename=os.path.join(save_dir, "alias_pair_occurrences.json"),
        contents=alias_pair_occurrences,
    )
    utils.dump_json_file(
        filename=os.path.join(save_dir, "alias_entity_counts.json"),
        contents=alias_entity_pair,
    )


def main():
    """Run."""
    args = parse_args()
    logging.info(json.dumps(vars(args), indent=4))
    entity_symbols = EntitySymbols.load_from_cache(
        load_dir=os.path.join(args.data_dir, args.entity_symbols_dir)
    )
    train_file = os.path.join(args.data_dir, args.train_file)
    save_dir = os.path.join(args.save_dir, "stats")
    logging.info(f"Will save data to {save_dir}")
    utils.ensure_dir(save_dir)
    # compute_histograms(save_dir, entity_symbols)
    compute_occurrences(
        save_dir,
        train_file,
        entity_symbols,
        args.lower,
        args.strip,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
