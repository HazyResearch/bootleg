"""
Extract mentions.

This file takes in a jsonlines file with sentences
and extract aliases and spans using a pre-computed alias table.
"""
import argparse
import logging
import multiprocessing
import os
import string
import time
from collections import defaultdict

import jsonlines
import marisa_trie
import nltk
import numpy as np
import spacy
import ujson
from spacy.cli.download import download as spacy_download
from tqdm import tqdm

from bootleg.symbols.constants import ANCHOR_KEY
from bootleg.utils.utils import get_lnrm

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    logger.warning(
        "Spacy models en_core_web_sm not found.  Downloading and installing."
    )
    try:
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        nlp = None

# We want this to pass gracefully in the case Readthedocs is trying to build.
# This will fail later on if a user is actually trying to run Bootleg without mention extraction
if nlp is not None:
    ALL_STOPWORDS = nlp.Defaults.stop_words
else:
    ALL_STOPWORDS = {}
PUNC = string.punctuation
KEEP_POS = {"PROPN", "NOUN"}  # ADJ, VERB, ADV, SYM
PLURAL = {"s", "'s"}
table = str.maketrans(
    dict.fromkeys(PUNC)
)  # OR {key: None for key in string.punctuation}


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
        "--cand_map", type=str, required=True, help="Alias table to use"
    )
    parser.add_argument("--min_alias_len", type=int, default=1)
    parser.add_argument("--max_alias_len", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def create_out_line(sent_obj, final_aliases, final_spans):
    """Create JSON output line.

    Args:
        sent_obj: input sentence JSON
        final_aliases: list of final aliases
        final_spans: list of final spans

    Returns: JSON object
    """
    sent_obj["aliases"] = final_aliases
    sent_obj["spans"] = final_spans
    # we don't know the true QID (or even if there is one) at this stage
    # we assign to the most popular candidate for now so models w/o NIL can also evaluate this data
    sent_obj["qids"] = ["Q-1"] * len(final_aliases)
    # global alias2qids
    # sent_obj["qids"] = [alias2qids[alias][0] for alias in final_aliases]
    sent_obj[ANCHOR_KEY] = [True] * len(final_aliases)
    return sent_obj


def get_all_aliases(alias2qidcands, verbose):
    """Load all aliases.

    Args
        alias2qidcands: Dict of alias to list of QID candidates with score
        verbose: verbose boolean flag

    Returns: marisa trie of all aliases
    """
    # Load alias2qids
    global alias2qids
    alias2qids = {}
    logger.debug("Loading candidate mapping...")
    for al in tqdm(alias2qidcands, disable=not verbose, desc="Reading candidate map"):
        alias2qids[al] = [c[0] for c in alias2qidcands[al]]
    logger.debug(f"Loaded candidate mapping with {len(alias2qids)} aliases.")
    all_aliases = marisa_trie.Trie(alias2qids.keys())
    return all_aliases


def get_new_to_old_dict(split_sentence):
    """Return a mapped dictionary from new tokenized words with Spacy to old.

    (Spacy sometimes splits words with - and other punc).

    Args:
        split_sentence: list of words in sentence

    Returns: Dict of new word id -> old word id
    """
    old_w = 0
    new_w = 0
    new_to_old = defaultdict(list)
    while old_w < len(split_sentence):
        old_word = split_sentence[old_w]
        tokenized_word = nlp(old_word)
        new_w_ids = list(range(new_w, new_w + len(tokenized_word)))
        for i in new_w_ids:
            new_to_old[i] = old_w
        new_w = new_w + len(tokenized_word)
        old_w += 1
    new_to_old[new_w] = old_w
    new_to_old = dict(new_to_old)
    return new_to_old


def find_aliases_in_sentence_tag(
    sentence, all_aliases, min_alias_len=1, max_alias_len=6
):
    """Extract function.

    Args:
        sentence: text
        all_aliases: Trie of all aliases in our save
        min_alias_len: minimum length (in words) of an alias
        max_alias_len: maximum length (in words) of an alias

    Returns: list of aliases, list of span offsets
    """
    used_aliases = []
    # Remove multiple spaces and replace with single - tokenization eats multiple spaces but
    # ngrams doesn't which can cause parse issues
    sentence = " ".join(sentence.strip().split())

    doc = nlp(sentence)
    split_sent = sentence.split()
    new_to_old_span = get_new_to_old_dict(split_sent)
    # find largest aliases first
    for n in range(max_alias_len, min_alias_len - 1, -1):
        grams = nltk.ngrams(doc, n)
        j_st = -1
        j_end = n - 1
        for gram_words in grams:
            j_st += 1
            j_end += 1
            if j_st not in new_to_old_span or j_end not in new_to_old_span:
                print("BAD")
                print(sentence)
                return [], []
            j_st_adjusted = new_to_old_span[j_st]
            j_end_adjusted = new_to_old_span[j_end]
            # Check if nlp has split the word and we are looking at a subword mention - which we don't want
            is_subword = j_st_adjusted == j_end_adjusted
            if j_st > 0:
                is_subword = is_subword | (j_st_adjusted == new_to_old_span[j_st - 1])
            # j_end is exclusive and should be a new word from the previous j_end-1
            is_subword = is_subword | (j_end_adjusted == new_to_old_span[j_end - 1])
            if is_subword:
                continue
            # Assert we are a full word
            assert (
                j_st_adjusted != j_end_adjusted
            ), f"Something went wrong getting mentions for {sentence}"
            # If single word and not in a POS we care about, skip
            if len(gram_words) == 1 and gram_words[0].pos_ not in KEEP_POS:
                continue
            # If multiple word and not any word in a POS we care about, skip
            if len(gram_words) > 1 and not any(g.pos_ in KEEP_POS for g in gram_words):
                continue
            # print("@", gram_words, [g.pos_ for g in gram_words])
            # If we are part of a proper noun, make sure there isn't another part of the proper noun to the
            # left or right - this means we didn't have the entire name in our alias and we should skip
            if len(gram_words) == 1 and gram_words[0].pos_ == "PROPN":
                if j_st > 0 and doc[j_st - 1].pos_ == "PROPN":
                    continue
                # End spans are exclusive so no +1
                if j_end < len(doc) and doc[j_end].pos_ == "PROPN":
                    continue
            # print("3", j_st, gram_words, [g.pos_ for g in gram_words])
            # We don't want punctuation words to be used at the beginning/end unless it's capitalized
            # or first word of sentence
            if (
                gram_words[-1].text in PLURAL
                or gram_words[0].text in PLURAL
                or (
                    gram_words[0].text.lower() in ALL_STOPWORDS
                    and (not gram_words[0].text[0].isupper() or j_st == 0)
                )
            ):
                continue
            # If the word starts with punctuation and there is a space in between, also continue; keep
            # if punctuation is part of the word boundary
            # print("4", j_st, gram_words, [g.pos_ for g in gram_words])
            if (
                gram_words[0].text in PUNC
                and (
                    j_st + 1 >= len(doc)
                    or new_to_old_span[j_st] != new_to_old_span[j_st + 1]
                )
            ) or (
                gram_words[-1].text in PUNC
                and (
                    j_end - 2 < 0
                    or new_to_old_span[j_end - 1] != new_to_old_span[j_end - 2]
                )
            ):
                continue
            joined_gram = " ".join(split_sent[j_st_adjusted:j_end_adjusted])
            # If 's in alias, make sure we remove the space and try that alias, too
            joined_gram_merged_plural = joined_gram.replace(" 's", "'s")
            # If PUNC in alias, make sure we remove the space and try that alias, too
            joined_gram_merged_nopunc = joined_gram_merged_plural.translate(table)
            gram_attempt = get_lnrm(joined_gram, strip=True, lower=True)
            gram_attempt_merged_plural = get_lnrm(
                joined_gram_merged_plural, strip=True, lower=True
            )
            gram_attempt_merged_nopunc = get_lnrm(
                joined_gram_merged_nopunc, strip=True, lower=True
            )
            # Remove numbers
            if (
                gram_attempt.isnumeric()
                or joined_gram_merged_plural.isnumeric()
                or gram_attempt_merged_nopunc.isnumeric()
            ):
                continue
            final_gram = None
            # print("4", gram_attempt, [g.pos_ for g in gram_words])
            if gram_attempt in all_aliases:
                final_gram = gram_attempt
            elif gram_attempt_merged_plural in all_aliases:
                final_gram = gram_attempt_merged_plural
            elif gram_attempt_merged_nopunc in all_aliases:
                final_gram = gram_attempt_merged_nopunc
                # print("5", final_gram, [g.pos_ for g in gram_words])
            # print("FINAL GRAM", final_gram)
            if final_gram is not None:
                keep = True
                # We start from the largest n-grams and go down in size. This prevents us from adding an alias that
                # is a subset of another. For example: "Tell me about the mother on how I met you mother" will find
                # "the mother" as alias and "mother". We want to only take "the mother" and not "mother" as it's
                # likely more descriptive of the real entity.
                for u_al in used_aliases:
                    u_j_st = u_al[1]
                    u_j_end = u_al[2]
                    if j_st_adjusted < u_j_end and j_end_adjusted > u_j_st:
                        keep = False
                        break
                if not keep:
                    continue
                used_aliases.append(tuple([final_gram, j_st_adjusted, j_end_adjusted]))
    # sort based on span order
    aliases_for_sorting = sorted(used_aliases, key=lambda elem: [elem[1], elem[2]])
    used_aliases = [a[0] for a in aliases_for_sorting]
    spans = [[a[1], a[2]] for a in aliases_for_sorting]
    assert all([sp[1] <= len(doc) for sp in spans]), f"{spans} {sentence}"
    return used_aliases, spans


def get_num_lines(input_src):
    """Count number of lines in file.

    Args:
        input_src: input file

    Returns: number of lines
    """
    # get number of lines
    num_lines = 0
    with open(input_src, "r", encoding="utf-8") as in_file:
        try:
            for line in in_file:
                num_lines += 1
        except Exception as e:
            logger.error("ERROR READING IN TRAINING DATA")
            logger.error(e)
            return []
    return num_lines


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
    min_alias_len = args["min_alias_len"]
    max_alias_len = args["max_alias_len"]
    verbose = args["verbose"]
    all_aliases = marisa_trie.Trie().load(args["all_aliases_trie_f"])
    num_lines = sum(1 for _ in open(in_file))
    with jsonlines.open(in_file) as f_in, jsonlines.open(out_file, "w") as f_out:
        for line in tqdm(
            f_in, total=num_lines, disable=not verbose, desc="Processing data"
        ):
            found_aliases, found_spans = find_aliases_in_sentence_tag(
                line["sentence"], all_aliases, min_alias_len, max_alias_len
            )
            f_out.write(create_out_line(line, found_aliases, found_spans))


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
    cand_map_file,
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
        cand_map_file: alias to candidate file
        min_alias_len: minimum alias length (in words)
        max_alias_len: maximum alias length (in words)
        num_workers: number of multiprocessing workers
        num_chunks: number of subchunks to feed to workers
        verbose: verbose boolean
    """
    assert os.path.exists(in_filepath), f"{in_filepath} does not exist"
    candidate_map = ujson.load(open(cand_map_file))
    all_aliases_trie = get_all_aliases(candidate_map, verbose)
    if num_chunks is None:
        num_chunks = num_workers
    start_time = time.time()
    # multiprocessing
    if num_workers > 1:
        prep_dir = os.path.join(os.path.dirname(out_filepath), "prep")
        os.makedirs(prep_dir, exist_ok=True)

        all_aliases_trie_f = os.path.join(prep_dir, "mention_extract_alias.marisa")
        all_aliases_trie.save(all_aliases_trie_f)

        # chunk file for multiprocessing
        num_lines = get_num_lines(in_filepath)
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
            os.remove(file)
        for file in chunk_outfiles:
            os.remove(file)
        os.remove(all_aliases_trie_f)

    # single process
    else:
        logger.debug("Using 1 worker...")
        with jsonlines.open(in_filepath, "r") as in_file, jsonlines.open(
            out_filepath, "w"
        ) as out_file:
            sent_idx_unq = 0
            for line in in_file:
                found_aliases, found_spans = find_aliases_in_sentence_tag(
                    line["sentence"], all_aliases_trie, min_alias_len, max_alias_len
                )
                new_line = create_out_line(line, found_aliases, found_spans)
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
        cand_map_file=args.cand_map,
        min_alias_len=args.min_alias_len,
        max_alias_len=args.max_alias_len,
        num_workers=args.num_workers,
        num_chunks=args.num_chunks,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
