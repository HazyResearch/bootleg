# This file takes in a jsonlines file with sentences and extract aliases and spans using a pre-computed alias table.

import argparse
import collections
import logging
import string
import jsonlines
import ujson
import multiprocessing
import os
import time
import marisa_trie
from collections import defaultdict
from tqdm import tqdm
import string
import unicodedata
import numpy as np
import nltk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True, help="File to extract mentions from")
    parser.add_argument('--out_file', type=str, required=True, help="File to write extracted mentions to")
    parser.add_argument('--cand_map', type=str, required=True, help="Alias table to use")
    parser.add_argument('--max_alias_len', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()

def create_out_line(sent_obj, final_aliases, final_spans):
    sent_obj["aliases"] = final_aliases
    sent_obj["spans"] = final_spans
    # we don't know the true QID (or even if there is one) at this stage
    # we assign to the most popular candidate for now so models w/o NIL can also evaluate this data
    # sent_obj["qids"] = ["Q-1"]*len(final_aliases)
    global alias2qids
    sent_obj["qids"] = [alias2qids[alias][0] for alias in final_aliases]
    sent_obj["anchor"] = [True]*len(final_aliases)
    return sent_obj

def get_lnrm(s):
    """Convert a string to its lnrm form
    We form the lower-cased normalized version l(s) of a string s by canonicalizing
    its UTF-8 characters, eliminating diacritics, lower-casing the UTF-8 and
    throwing out all ASCII-range characters that are not alpha-numeric.
    from http://nlp.stanford.edu/pubs/subctackbp.pdf Section 2.3
    Args:
        input string
    Returns:
        the lnrm form of the string
    """
    lnrm = unicodedata.normalize('NFD', str(s))
    lnrm = lnrm.lower()
    lnrm = ''.join([x for x in lnrm if (not unicodedata.combining(x)
                                        and x.isalnum() or x == ' ')]).strip()
    # will remove if there are any duplicate white spaces e.g. "the  alias    is here"
    lnrm = " ".join(lnrm.split())
    return lnrm

def get_all_aliases(alias2qidcands, logger):
    # Load alias2qids
    global alias2qids
    alias2qids = {}
    logger.info("Loading candidate mapping...")
    for al in tqdm(alias2qidcands):
        alias2qids[al] = [c[0] for c in alias2qidcands[al]]
    logger.info(f"Loaded candidate mapping with {len(alias2qids)} aliases.")
    all_aliases = marisa_trie.Trie(alias2qids.keys())
    return all_aliases

# TODO: simplify -- remove extra filters
def find_aliases_in_sentence_tag(sentence, all_aliases, max_alias_len = 5):
    PUNC = string.punctuation
    table = str.maketrans(dict.fromkeys(PUNC))  # OR {key: None for key in string.punctuation}
    used_aliases = []
    sentence_split_raw = sentence.split()
    # find largest aliases first
    for n in range(max_alias_len+1, 0, -1):
        grams = nltk.ngrams(sentence_split_raw, n)
        j_st = -1
        j_end = n-1
        for gram_words in grams:
            j_st += 1
            j_end += 1
            # We don't want punctuation words to be used at the beginning/end
            if len(gram_words[0].translate(table).strip()) == 0 or len(gram_words[-1].translate(table).strip()) == 0:
                continue
            gram_attempt = get_lnrm(" ".join(gram_words))
            # TODO: remove possessives from alias table
            if len(gram_attempt) > 1:
                if gram_attempt[-1] == 's' and gram_attempt[-2] == ' ':
                    continue
            if gram_attempt in all_aliases:
                keep = True
                # We start from the largest n-grams and go down in size. This prevents us from adding an alias that is a subset of another.
                # For example: "Tell me about the mother on how I met you mother" will find "the mother" as alias and "mother". We want to
                # only take "the mother" and not "mother" as it's likely more descriptive of the real entity.
                for u_al in used_aliases:
                    u_j_st = u_al[1]
                    u_j_end = u_al[2]
                    if j_st < u_j_end and j_end > u_j_st:
                        keep = False
                        break
                if not keep:
                    continue
                used_aliases.append(tuple([gram_attempt, j_st, j_end]))
    # sort based on span order
    aliases_for_sorting = sorted(used_aliases, key=lambda elem: [elem[1], elem[2]])
    used_aliases = [a[0] for a in aliases_for_sorting]
    spans = ["{}:{}".format(a[1], a[2]) for a in aliases_for_sorting]
    return used_aliases, spans

def get_num_lines(input_src):
    # get number of lines
    num_lines = 0
    with open(input_src, "r", encoding='utf-8') as in_file:
        try:
            for line in in_file:
                num_lines += 1
        except Exception as e:
            logging.error("ERROR READING IN TRAINING DATA")
            logging.error(e)
            return []
    return num_lines

def chunk_text_data(input_src, chunk_files, chunk_size, num_lines, logger):
    logger.info(f"Reading in {input_src}")
    start = time.time()
    # write out chunks as text data
    chunk_id = 0
    num_lines_in_chunk = 0
    # keep track of what files are written
    out_file = open(chunk_files[chunk_id], 'w')
    with open(input_src, 'r', encoding='utf-8') as in_file:
        for i, line in enumerate(in_file):
            out_file.write(line)
            num_lines_in_chunk += 1
            # move on to new chunk when it hits chunk size
            if num_lines_in_chunk == chunk_size:
                chunk_id += 1
                # reset number of lines in chunk and open new file if not at end
                num_lines_in_chunk = 0
                out_file.close()
                if i < (num_lines-1):
                    out_file = open(chunk_files[chunk_id], 'w')
    out_file.close()
    logger.info(f'Wrote out data chunks in {round(time.time() - start, 2)}s')

def subprocess(args):
    in_file = args['in_file']
    out_file = args['out_file']
    max_alias_len = args['max_alias_len']
    with jsonlines.open(in_file) as f_in, jsonlines.open(out_file, 'w') as f_out:
        for line in f_in:
            found_aliases, found_spans = find_aliases_in_sentence_tag(line["sentence"], global_all_aliases, max_alias_len)
            f_out.write(create_out_line(line, found_aliases, found_spans))

def merge_files(chunk_outfiles, out_filepath):
    sent_idx_unq = 0
    with jsonlines.open(out_filepath, 'w') as f_out:
        for file in chunk_outfiles:
            with jsonlines.open(file) as f_in:
                for line in f_in:
                    if 'sent_idx_unq' not in line:
                        line['sent_idx_unq'] = sent_idx_unq
                    f_out.write(line)
                    sent_idx_unq += 1

def extract_mentions(in_filepath, out_filepath, cand_map_file, max_alias_len=6, num_workers=8,
    logger=logging.getLogger()):
    candidate_map = ujson.load(open(cand_map_file))
    all_aliases_trie = get_all_aliases(candidate_map, logger=logger)
    start_time = time.time()
    # multiprocessing
    if num_workers > 1:
        global global_all_aliases
        global_all_aliases = all_aliases_trie

        prep_dir = os.path.join(os.path.dirname(out_filepath), 'prep')
        os.makedirs(prep_dir, exist_ok=True)

        # chunk file for multiprocessing
        num_lines = get_num_lines(in_filepath)
        num_processes = min(num_workers, int(multiprocessing.cpu_count()))
        logger.info(f'Using {num_processes} workers...')
        chunk_size = int(np.ceil(num_lines/(num_processes)))
        num_chunks = int(np.ceil(num_lines / chunk_size))
        chunk_file_path = os.path.join(prep_dir, 'data_chunk')
        chunk_infiles = [f'{chunk_file_path}_{chunk_id}_in.jsonl' for chunk_id in range(num_chunks)]
        chunk_text_data(in_filepath, chunk_infiles, chunk_size, num_lines, logger=logger)
        logger.info("Calling subprocess...")
        # call subprocesses on chunks
        pool = multiprocessing.Pool(processes=num_processes)
        chunk_outfiles = [f'{chunk_file_path}_{chunk_id}_out.jsonl' for chunk_id in range(num_chunks)]
        subprocess_args = [{'in_file': chunk_infiles[i],
                            'out_file': chunk_outfiles[i],
                            'max_alias_len': max_alias_len}
                            for i in range(num_chunks)]
        pool.map(subprocess, subprocess_args)
        pool.close()
        pool.join()
        logger.info("Merging files...")
        # write all chunks back in single file
        merge_files(chunk_outfiles, out_filepath)
        logger.info("Removing temporary files...")
        # clean up and remove chunked files
        for file in chunk_infiles:
            os.remove(file)
        for file in chunk_outfiles:
            os.remove(file)

    # single process
    else:
        logger.info(f'Using 1 worker...')
        with jsonlines.open(in_filepath, 'r') as in_file, jsonlines.open(out_filepath, 'w') as out_file:
            for line in in_file:
                found_aliases, found_spans = find_aliases_in_sentence_tag(line["sentence"], all_aliases_trie, max_alias_len)
                out_file.write(create_out_line(line, found_aliases, found_spans))

    logger.info(f"Finished in {time.time() - start_time} seconds. Wrote out to {out_filepath}")

def main():
    args = parse_args()
    in_file = args.in_file
    out_file = args.out_file

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    extract_mentions(in_file, out_file, cand_map_file=args.cand_map, max_alias_len=args.max_alias_len, num_workers=args.num_workers, logger=logging.getLogger())

if __name__ == '__main__':
    main()
