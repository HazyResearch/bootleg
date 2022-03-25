"""
Extract mentions.

This file takes in a jsonlines file with sentences
and extract aliases and spans using a pre-computed alias table.
"""
import argparse
import logging
import multiprocessing
import os
import time

import jsonlines
import numpy as np
from tqdm.auto import tqdm

from bootleg.symbols.constants import ANCHOR_KEY
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils.classes.nested_vocab_tries import VocabularyTrie
from bootleg.utils.mention_extractor_utils import (
    ngram_spacy_extract_aliases,
    spacy_extract_aliases,
)

logger = logging.getLogger(__name__)

MENTION_EXTRACTOR_OPTIONS = {
    "ngram_spacy": ngram_spacy_extract_aliases,
    "spacy": spacy_extract_aliases,
}


def parse_args():
    """Generate args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file", type=str, required=True, help="File to extract mentions from"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="File to write extracted mentions to",
    )
    parser.add_argument(
        "--entity_db_dir", type=str, required=True, help="Path to entity db"
    )
    parser.add_argument(
        "--extract_method",
        type=str,
        choices=list(MENTION_EXTRACTOR_OPTIONS.keys()),
        default="ngram_spacy",
    )
    parser.add_argument("--min_alias_len", type=int, default=1)
    parser.add_argument("--max_alias_len", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def create_out_line(sent_obj, final_aliases, final_spans, found_char_spans):
    """Create JSON output line.

    Args:
        sent_obj: input sentence JSON
        final_aliases: list of final aliases
        final_spans: list of final spans
        found_char_spans: list of final char spans

    Returns: JSON object
    """
    sent_obj["aliases"] = final_aliases
    sent_obj["spans"] = final_spans
    sent_obj["char_spans"] = found_char_spans
    # we don't know the true QID (or even if there is one) at this stage
    # we assign to the most popular candidate for now so models w/o NIL can also evaluate this data
    sent_obj["qids"] = ["Q-1"] * len(final_aliases)
    # global alias2qids
    # sent_obj["qids"] = [alias2qids[alias][0] for alias in final_aliases]
    sent_obj[ANCHOR_KEY] = [True] * len(final_aliases)
    return sent_obj


def chunk_text_data(input_src, chunk_files, chunk_size, num_lines):
    """Chunk text input file into chunk_size chunks.

    Args:
        input_src: input file
        chunk_files: list of chunk file names
        chunk_size: chunk size in number of lines
        num_lines: total number of lines
    """
    logger.debug(f"Reading in {input_src}")
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
    logger.debug(f"Wrote out data chunks in {round(time.time() - start, 2)}s")


def subprocess(args):
    """
    Extract mentions single process.

    Args:
        args: subprocess args
    """
    in_file = args["in_file"]
    out_file = args["out_file"]
    extact_method = args["extract_method"]
    min_alias_len = args["min_alias_len"]
    max_alias_len = args["max_alias_len"]
    verbose = args["verbose"]
    all_aliases = VocabularyTrie(load_dir=args["all_aliases_trie_f"])
    num_lines = sum(1 for _ in open(in_file))
    with jsonlines.open(in_file) as f_in, jsonlines.open(out_file, "w") as f_out:
        for line in tqdm(
            f_in, total=num_lines, disable=not verbose, desc="Processing data"
        ):
            found_aliases, found_spans, found_char_spans = MENTION_EXTRACTOR_OPTIONS[
                extact_method
            ](line["sentence"], all_aliases, min_alias_len, max_alias_len)
            f_out.write(
                create_out_line(line, found_aliases, found_spans, found_char_spans)
            )


def merge_files(chunk_outfiles, out_filepath):
    """Merge output files.

    Args:
        chunk_outfiles: list of chunk files
        out_filepath: final output file path
    """
    sent_idx_unq = 0
    with jsonlines.open(out_filepath, "w") as f_out:
        for file in chunk_outfiles:
            with jsonlines.open(file) as f_in:
                for line in f_in:
                    if "sent_idx_unq" not in line:
                        line["sent_idx_unq"] = sent_idx_unq
                    f_out.write(line)
                    sent_idx_unq += 1


def extract_mentions(
    in_filepath,
    out_filepath,
    entity_db_dir,
    extract_method="ngram_spacy",
    min_alias_len=1,
    max_alias_len=6,
    num_workers=8,
    num_chunks=None,
    verbose=False,
):
    """Extract mentions from file.

    Args:
        in_filepath: input file
        out_filepath: output file
        entity_db_dir: path to entity db
        extract_method: mention extraction method
        min_alias_len: minimum alias length (in words)
        max_alias_len: maximum alias length (in words)
        num_workers: number of multiprocessing workers
        num_chunks: number of subchunks to feed to workers
        verbose: verbose boolean
    """
    assert os.path.exists(in_filepath), f"{in_filepath} does not exist"
    entity_symbols: EntitySymbols = EntitySymbols.load_from_cache(entity_db_dir)
    all_aliases_trie: VocabularyTrie = entity_symbols.get_all_alias_vocabtrie()
    if num_chunks is None:
        num_chunks = num_workers
    start_time = time.time()
    # multiprocessing
    if num_workers > 1:
        prep_dir = os.path.join(os.path.dirname(out_filepath), "prep")
        os.makedirs(prep_dir, exist_ok=True)

        all_aliases_trie_f = os.path.join(prep_dir, "mention_extract_alias.marisa")
        all_aliases_trie.dump(all_aliases_trie_f)

        # chunk file for multiprocessing
        num_lines = sum([1 for _ in open(in_filepath)])
        num_processes = min(num_workers, int(multiprocessing.cpu_count()))
        num_chunks = min(num_lines, num_chunks)
        logger.debug(f"Using {num_processes} workers...")
        chunk_size = int(np.ceil(num_lines / num_chunks))
        chunk_file_path = os.path.join(prep_dir, "data_chunk")
        chunk_infiles = [
            f"{chunk_file_path}_{chunk_id}_in.jsonl" for chunk_id in range(num_chunks)
        ]
        chunk_text_data(in_filepath, chunk_infiles, chunk_size, num_lines)
        logger.debug("Calling subprocess...")
        # call subprocesses on chunks
        pool = multiprocessing.Pool(processes=num_processes)
        chunk_outfiles = [
            f"{chunk_file_path}_{chunk_id}_out.jsonl" for chunk_id in range(num_chunks)
        ]
        subprocess_args = [
            {
                "in_file": chunk_infiles[i],
                "out_file": chunk_outfiles[i],
                "extract_method": extract_method,
                "min_alias_len": min_alias_len,
                "max_alias_len": max_alias_len,
                "all_aliases_trie_f": all_aliases_trie_f,
                "verbose": verbose,
            }
            for i in range(num_chunks)
        ]
        pool.map(subprocess, subprocess_args)
        pool.close()
        pool.join()
        logger.debug("Merging files...")
        # write all chunks back in single file
        merge_files(chunk_outfiles, out_filepath)
        logger.debug("Removing temporary files...")
        # clean up and remove chunked files
        for file in chunk_infiles:
            try:
                os.remove(file)
            except PermissionError:
                pass
        for file in chunk_outfiles:
            try:
                os.remove(file)
            except PermissionError:
                pass
        try:
            os.remove(all_aliases_trie_f)
        except PermissionError:
            pass

    # single process
    else:
        logger.debug("Using 1 worker...")
        with jsonlines.open(in_filepath, "r") as in_file, jsonlines.open(
            out_filepath, "w"
        ) as out_file:
            sent_idx_unq = 0
            for line in in_file:
                (
                    found_aliases,
                    found_spans,
                    found_char_spans,
                ) = MENTION_EXTRACTOR_OPTIONS[extract_method](
                    line["sentence"], all_aliases_trie, min_alias_len, max_alias_len
                )
                new_line = create_out_line(
                    line, found_aliases, found_spans, found_char_spans
                )
                if "sent_idx_unq" not in new_line:
                    new_line["sent_idx_unq"] = sent_idx_unq
                    sent_idx_unq += 1
                out_file.write(new_line)

    logger.debug(
        f"Finished in {time.time() - start_time} seconds. Wrote out to {out_filepath}"
    )


def main():
    """Run."""
    args = parse_args()
    in_file = args.in_file
    out_file = args.out_file

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(args)
    extract_mentions(
        in_file,
        out_file,
        entity_db_dir=args.entity_db_dir,
        min_alias_len=args.min_alias_len,
        max_alias_len=args.max_alias_len,
        num_workers=args.num_workers,
        num_chunks=args.num_chunks,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
