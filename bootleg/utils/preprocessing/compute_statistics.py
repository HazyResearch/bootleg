import argparse, os
from collections import defaultdict, Counter
import ujson as json
import nltk
import unicodedata

from bootleg.symbols.constants import *
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.type_symbols import TypeSymbols
from bootleg.utils import utils

from itertools import combinations
from tqdm import tqdm
import marisa_trie
import multiprocessing

import logging
import numpy as np
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='Data dir for training data')
    parser.add_argument('--save_dir', type=str, default='data/', help='Data dir for saving stats')
    parser.add_argument('--train_file', type=str, default="train.jsonl")
    parser.add_argument('--entity_symbols_dir', type=str, default='entity_db/entity_mappings', help='Path to entities inside data_dir')
    parser.add_argument('--emb_dir', type=str, default='/dfs/scratch0/lorr1/bootleg/embs', help='Path to embeddings')
    parser.add_argument('--max_types', type=int, default=3, help='Max types to load')
    parser.add_argument('--no_types', action='store_true', help='Do not compute type statistics')
    parser.add_argument('--lower', action='store_true', help='Lower aliases')
    parser.add_argument('--strip', action='store_true', help='Strip punc aliases')
    parser.add_argument('--num_workers', type=int, help='Number of workers to parallelize')
    args = parser.parse_args()
    return args

def compute_histograms(save_dir, entity_symbols):
    al_counts = Counter()
    for al in entity_symbols.get_all_aliases():
        num_entities = len(entity_symbols.get_qid_cands(al))
        al_counts.update([num_entities])
    utils.dump_json_file(filename=os.path.join(save_dir, "candidate_counts.json"), contents=al_counts)
    return

def get_all_aliases(alias2qidcands):
    # Load alias2qids
    alias2qids = {}
    for al in tqdm(alias2qidcands):
        alias2qids[al] = [c[0] for c in alias2qidcands[al]]
    logging.info(f"Loaded entity dump with {len(alias2qids)} aliases.")
    all_aliases = marisa_trie.Trie(alias2qids.keys())
    return all_aliases

def get_lnrm(s, strip, lower):
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
    if not strip and not lower:
        return s
    lnrm = str(s)
    if lower:
        lnrm = lnrm.lower()
    if strip:
        lnrm = unicodedata.normalize('NFD', lnrm)
        lnrm = ''.join([x for x in lnrm if (not unicodedata.combining(x)
                                            and x.isalnum() or x == ' ')]).strip()
    # will remove if there are any duplicate white spaces e.g. "the  alias    is here"
    lnrm = " ".join(lnrm.split())
    return lnrm

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

def chunk_text_data(input_src, chunk_files, chunk_size, num_lines):
    logging.info(f"Reading in {input_src}")
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
    logging.info(f'Wrote out data chunks in {round(time.time() - start, 2)}s')

def compute_occurrences_single(args, max_alias_len=6):
    data_file, lower, strip = args
    global all_aliases
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
        for line in in_file:
            line = json.loads(line.strip())
            for n in range(max_alias_len+1, 0, -1):
                grams = nltk.ngrams(line['sentence'].split(), n)
                for gram_words in grams:
                    gram_attempt = get_lnrm(" ".join(gram_words), lower, strip)
                    if gram_attempt in all_aliases:
                        alias_text_occurrences[gram_attempt] += 1
            aliases = line['aliases']
            qids = line['qids']
            for qid, alias in zip(qids, aliases):
                ent_occurrences[qid] += 1
                alias_occurrences[alias] += 1
                alias_entity_pair[alias + "|" + qid] += 1
            alias_pair_occurrences[len(aliases)] += 1
    results = {'ent_occurrences': ent_occurrences,
        'alias_occurrences': alias_occurrences,
        'alias_text_occurrences': alias_text_occurrences,
        'alias_pair_occurrences': alias_pair_occurrences,
        'alias_entity_pair': alias_entity_pair}
    return results

def compute_occurrences(save_dir, data_file, entity_dump, lower, strip, num_workers=8):
    global all_aliases
    all_aliases = get_all_aliases(entity_dump._alias2qids)

    # divide up data into chunks
    num_lines = get_num_lines(data_file)
    num_processes = min(num_workers, int(multiprocessing.cpu_count()))
    logging.info(f'Using {num_processes} workers...')
    chunk_size = int(np.ceil(num_lines/(num_processes)))
    chunk_file_path = os.path.join(save_dir, 'tmp')
    utils.ensure_dir(chunk_file_path)
    chunk_infiles = [os.path.join(f'{chunk_file_path}', f'data_chunk_{chunk_id}_in.jsonl') for chunk_id in range(num_processes)]
    chunk_text_data(data_file, chunk_infiles, chunk_size, num_lines)

    pool = multiprocessing.Pool(processes=num_processes)
    subprocess_args = [[chunk_infiles[i], lower, strip] for i in range(num_processes)]
    results = pool.map(compute_occurrences_single, subprocess_args)
    pool.close()
    pool.join()
    logging.info('Finished collecting counts')
    logging.info('Merging counts....')
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
    for result_set in results:
        ent_occurrences += result_set['ent_occurrences']
        alias_occurrences += result_set['alias_occurrences']
        alias_text_occurrences += result_set['alias_text_occurrences']
        alias_pair_occurrences += result_set['alias_pair_occurrences']
        alias_entity_pair += result_set['alias_entity_pair']
    # save counters
    utils.dump_json_file(filename=os.path.join(save_dir, "entity_count.json"), contents=ent_occurrences)
    utils.dump_json_file(filename=os.path.join(save_dir, "alias_counts.json"), contents=alias_occurrences)
    utils.dump_json_file(filename=os.path.join(save_dir, "alias_text_counts.json"), contents=alias_text_occurrences)
    utils.dump_json_file(filename=os.path.join(save_dir, "alias_pair_occurrences.json"), contents=alias_pair_occurrences)
    utils.dump_json_file(filename=os.path.join(save_dir, "alias_entity_counts.json"), contents=alias_entity_pair)


def compute_type_occurrences(save_dir, prefix, entity_symbols, qid2typenames, data_file):
    # type histogram
    type_occurances = defaultdict(int)
    # type intersection histogram (frequency of type co-occurring together)
    type_pair_occurances = defaultdict(int)
    # computes number of aliases in a sentence with the number of shared types amongst those aliases; just outputs number of types for single alias
    num_al_num_match_type = defaultdict(int)
    type_pairs = defaultdict(int)
    with open(data_file, "r") as in_file:
        for line in in_file:
            line = json.loads(line.strip())
            aliases = line['aliases']
            qids = line['qids']
            all_ex_types = set()
            i = 0
            # for all pairs of qid, types, get the intersection of types
            # if the intersection > 1, write type pairs and increment
            for qid, alias in zip(qids, aliases):
                types = qid2typenames.get(qid, [])
                for ty in types:
                    type_occurances[ty] += 1
                if i == 0:
                    all_ex_types = set(types)
                else:
                    all_ex_types = all_ex_types.intersection(set(types))
                i += 1
            if len(aliases) > 1:
                num_al_num_match_type[f"{len(aliases)}|{len(set(all_ex_types))}"] += 1
                type_pair_occurances[tuple(sorted(list(all_ex_types)))] += 1

            alias_subsets = list(combinations(qids, 2))
            for qid1, qid2 in alias_subsets:
                types1 = qid2typenames.get(qid1, [])
                types2 = qid2typenames.get(qid2, [])
                overlap_types = set(types1).intersection(set(types2))
                if len(overlap_types) > 0:
                    type_pairs[tuple(overlap_types)] += 1
    logging.info(f"Saving type data...")
    utils.dump_json_file(filename=os.path.join(save_dir, f"{prefix}_type_occurances.json"), contents=type_occurances)
    utils.dump_json_file(filename=os.path.join(save_dir, f"{prefix}_type_pair_occurances.json"), contents=type_pair_occurances)
    utils.dump_json_file(filename=os.path.join(save_dir, f"{prefix}_num_al_num_match_type.json"), contents=num_al_num_match_type)
    utils.dump_json_file(filename=os.path.join(save_dir, f"{prefix}_num_type_pairs.json"), contents=type_pairs)

def main():
    args = parse_args()
    logging.info(json.dumps(args, indent=4))
    entity_symbols = EntitySymbols(load_dir = os.path.join(args.data_dir, args.entity_symbols_dir))
    train_file = os.path.join(args.data_dir, args.train_file)
    save_dir = os.path.join(args.save_dir, "stats")
    logging.info(f"Will save data to {save_dir}")
    utils.ensure_dir(save_dir)
    # compute_histograms(save_dir, entity_symbols)
    compute_occurrences(save_dir, train_file, entity_symbols, args.lower, args.strip, num_workers=args.num_workers)
    if not args.no_types:
        type_symbols = TypeSymbols(entity_symbols=entity_symbols, emb_dir=args.emb_dir, max_types=args.max_types,
                                   emb_file="hyena_type_emb.pkl", type_vocab_file="hyena_type_graph.vocab.pkl", type_file="hyena_types.txt")
        compute_type_occurrences(save_dir, "orig", entity_symbols, type_symbols.qid2typenames, train_file)

if __name__ == '__main__':
    main()