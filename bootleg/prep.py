import argparse
import ujson as json
import multiprocessing
import os
import time
import sys
import glob
import numpy as np
import logging
import jsonlines
import pickle
from collections import defaultdict

from bootleg.symbols.alias_entity_table import AliasEntityTable
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.constants import *
from bootleg.utils import utils, train_utils
import bootleg.utils.prep_utils as prep_utils
from bootleg.utils import sentence_utils
import bootleg.utils.data_utils as data_utils
from bootleg.utils.data_utils import generate_save_data_name, load_wordsymbols, get_data_prep_dir
from bootleg.utils.parser_utils import get_full_config
from bootleg.utils.utils import import_class

"""
This file preps all data, slices, and embeddings needed to train a disambiguation model.
We additionally batch prep embeddings per data sample if the 'batch_prep' flag is used in the embedding (prep is called fist)
(e.g. which candidates in this sentence share a relation with the other candidates in the sentence).
Memory mapped files are used for the main data and batch_prepped_embedding_data.
"""

#===========================================#
# Helpers functions for multiprocessing (_helper)
#===========================================#

def get_entities(args):
    # Build alias entity table in advance for quick loading later
    entity_symbols = EntitySymbols(load_dir=os.path.join(args.data_config.entity_dir, args.data_config.entity_map_dir),
        alias_cand_map_file=args.data_config.alias_cand_map)
    # Collect stats needed later
    num_cands_K = entity_symbols.max_candidates + (not args.data_config.train_in_candidates)
    num_aliases_with_pad = len(entity_symbols.get_all_aliases()) + 1
    return entity_symbols, num_cands_K, num_aliases_with_pad

def prep_all_embs(args):
    entity_symbols = EntitySymbols(load_dir=os.path.join(args.data_config.entity_dir, args.data_config.entity_map_dir),
        alias_cand_map_file=args.data_config.alias_cand_map)
    word_symbols = load_wordsymbols(args.data_config)
    for emb in args.data_config.ent_embeddings:
        mod, load_class = import_class("bootleg.embeddings", emb.load_class)
        try:
            getattr(mod, load_class).prep(main_args=args, emb_args=emb['args'],
                entity_symbols=entity_symbols, word_symbols=word_symbols, log_func=logging.debug)
        except AttributeError:
            logging.debug(f'No prep method found for {emb.load_class}')

def chunk_text_data(args, input_src, chunk_prep_dir):
    logging.debug(f"Reading in {input_src}")
    start = time.time()
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

    # compute number of chunks
    num_processes = min(args.run_config.dataset_threads, int(multiprocessing.cpu_count()))
    chunk_size = int(np.ceil(num_lines/(num_processes)))

    chunk_file_path = os.path.join(chunk_prep_dir, 'data_chunk')
    out_files = [f'{chunk_file_path}_{chunk_id}.jsonl' for chunk_id in range(num_processes)]

    # write out chunks as text data
    chunk_id = 0
    num_lines_in_chunk = 0
    # keep track of what files are written
    chunk_list = [out_files[chunk_id]]
    out_file = open(out_files[chunk_id], 'w')

    # since we have to read through all lines to chunk files,
    # we check for unique sentence indices here
    seen_sent_indices = set()
    with open(input_src, 'r', encoding='utf-8') as in_file:
        for i, line in enumerate(in_file):
            line_json = json.loads(line)
            assert 'sent_idx_unq' in line
            assert line_json['sent_idx_unq'] not in seen_sent_indices, f'Sentence indices must be unique. {line_json["sent_idx_unq"]} already seen.'
            seen_sent_indices.add(line_json['sent_idx_unq'])
            out_file.write(line)
            num_lines_in_chunk += 1
            # move on to new chunk when it hits chunk size
            if num_lines_in_chunk == chunk_size:
                chunk_id += 1
                # reset number of lines in chunk and open new file if not at end
                num_lines_in_chunk = 0
                out_file.close()
                if i < (num_lines-1):
                    chunk_list.append(out_files[chunk_id])
                    out_file = open(out_files[chunk_id], 'w')
    out_file.close()
    metadata = {'chunks': chunk_list, 'chunksize': chunk_size}
    utils.dump_json_file(os.path.join(chunk_prep_dir, 'metadata.json'), metadata)
    logging.debug(f'Wrote out data chunks in {round(time.time() - start, 2)}s')

def generate_data_subsent_helper(input_args):
    args = input_args['args']
    use_weak_label = input_args['use_weak_label']
    dataset_is_eval = input_args['dataset_is_eval']
    text_file = input_args['text_file']
    out_file = input_args['out_file']
    data_args = args.data_config
    word_symbols = load_wordsymbols(data_args)
    entity_symbols = EntitySymbols(load_dir=os.path.join(data_args.entity_dir, data_args.entity_map_dir),
        alias_cand_map_file=data_args.alias_cand_map)
    # keep track of number of samples including subsentences and with filtering
    # due to tokenization
    num_samples = 0
    dropped_lines = 0
    with jsonlines.open(text_file, 'r') as f, jsonlines.open(out_file, 'w') as writer:
        for line in f:
            assert 'sent_idx_unq' in line
            assert 'aliases' in line
            assert 'qids' in line
            assert 'spans' in line
            assert 'sentence' in line
            assert ANCHOR_KEY in line
            sent_idx = line['sent_idx_unq']
            # aliases are assumed to be lower-cased in candidate map
            aliases = [alias.lower() for alias in line['aliases']]
            qids = line['qids']
            spans = line['spans']
            phrase = line['sentence']
            assert len(spans) == len(aliases) == len(qids), 'lengths of alias-related values not equal'
            all_false_anchors = False
            # For datasets, we see all aliases, unless use_weak_label is turned off
            aliases_seen_by_model = [i for i in range(len(aliases))]
            anchor = [True for i in range(len(aliases))]
            if ANCHOR_KEY in line:
                anchor = line[ANCHOR_KEY]
                assert len(aliases) == len(anchor)
                assert all(isinstance(a, bool) for a in anchor)
                all_false_anchors = all([a is False for a in anchor])
            # checks that filtering was done correctly
            for alias in aliases:
                assert (alias in entity_symbols.get_all_aliases()), f"alias {alias} is not in entity_symbols"
            for span in spans:
                assert len(span) == 2,  f"Span is too large {span}"
                # assert int(span.split(":")[0]) <= len(phrase.split(" "))-1, f"Span is too large {span}"
            if not use_weak_label:
                aliases = [aliases[i] for i in range(len(anchor)) if anchor[i] is True]
                qids = [qids[i] for i in range(len(anchor)) if anchor[i] is True]
                spans = [spans[i] for i in range(len(anchor)) if anchor[i] is True]
                aliases_seen_by_model = [i for i in range(len(aliases))]
                anchor = [True for i in range(len(aliases))]
            if (dataset_is_eval and all_false_anchors) or len(aliases) == 0:
                dropped_lines += 1
                # logging.warning(f'Dropping sentence. There are no aliases to predict for {line}.')
                continue
            # Extract the subphrase in the sentence
            # idxs_arr is list of lists of indexes of qids, aliases, spans that are part of each subphrase
            # aliases_seen_by_model represents the aliases to be sent through the model
            # The output of aliases_to_predict_per_split represents which aliases are to be scored in which subphrase.
            # Ex:
            # If I have 15 aliases and M = 10, then I can split it into two chunks of 10 aliases each
            # idxs_arr = [0,1,2,3,4,5,6,7,8,9] and [5,6,7,8,9,10,11,12,13,14]
            # However, I do not want to score aliases 5-9 twice. Therefore, aliases_to_predict_per_split represents which ones to score
            # aliases_to_predict_per_split = [0,1,2,3,4,5,6,7] and [3,4,5,6,7,8,9]
            # These are indexes into the idx_arr (the first aliases to be scored in the second list is idx = 3, representing the 8th aliases in
            # the original sequence.
            idxs_arr, aliases_to_predict_per_split, spans_arr, phrase_tokens_arr = sentence_utils.split_sentence(max_aliases=data_args.max_aliases,
                                                                                                              phrase=phrase, spans=spans,
                                                                                                              aliases=aliases,
                                                                                                              aliases_seen_by_model=aliases_seen_by_model,
                                                                                                              seq_len=data_args.max_word_token_len,
                                                                                                              word_symbols=word_symbols)
            word_indices_arr = [word_symbols.convert_tokens_to_ids(pt) for pt in phrase_tokens_arr]
            aliases_arr = [[aliases[idx] for idx in idxs] for idxs in idxs_arr]
            anchor_arr = [[anchor[idx] for idx in idxs] for idxs in idxs_arr]
            qids_arr = [[qids[idx] for idx in idxs] for idxs in idxs_arr]
            # write out results to json lines file
            for subsent_idx in range(len(idxs_arr)):
                # This contains, pre eval dataset filtering, the mapping of aliases seen by model to the ones to be scores. We need this mapping
                # during eval to create on "prediction" per alias, True or False anchor.
                train_aliases_to_predict_arr = aliases_to_predict_per_split[subsent_idx][:]
                # aliases_to_predict_arr is an index into idxs_arr/anchor_arr/aliases_arr. It should only include true anchors if eval dataset
                # During training want to backpropagate on false anchors as well
                if dataset_is_eval:
                    aliases_to_predict_arr = [a2p_ex for a2p_ex in aliases_to_predict_per_split[subsent_idx] if anchor_arr[subsent_idx][a2p_ex] is True]
                else:
                    aliases_to_predict_arr = aliases_to_predict_per_split[subsent_idx]
                # If no aliases to predict or it's an eval dataset and all aliases to predict are false, then drop
                # Note that this may occur because of the windowing above. If aliases get split up, you have have True anchors in the sentence
                # but all aliases to predict anchors will be False.
                if (len(aliases_to_predict_arr) <= 0): #or (dataset_is_eval and sum([anchor_arr[subsent_idx][i] for i in aliases_to_predict_per_split[subsent_idx]]) == 0):
                    # logging.warning('Dropping example. There are no aliases to predict for the example.')
                    continue
                num_samples += 1
                writer.write({
                    'sent_idx': sent_idx,
                    'subsent_idx': subsent_idx,
                    'alias_list_pos': idxs_arr[subsent_idx],
                    'aliases_to_predict': aliases_to_predict_arr,
                    'train_aliases_to_predict_arr': train_aliases_to_predict_arr,
                    'spans': spans_arr[subsent_idx],
                    'word_indices': word_indices_arr[subsent_idx],
                    'aliases': aliases_arr[subsent_idx],
                    'qids': qids_arr[subsent_idx]})
    logging.debug(f"Finished processing data. Dropped {dropped_lines} from dataset. If use_weak_label is false, this is because lines had all false anchors. If use_weak_label"
          f" is true, then this is because this is a dev/test slice where there were no True anchors.")
    return num_samples

def process_data_subsent_helper(input_args):
    args = input_args['args']
    data_args = args.data_config
    in_file = input_args['in_file']
    mmap_file_name = input_args['out_file']
    storage_type = input_args['storage_type']
    data_len = input_args['len']
    # write into correct location in memory mapped file (each process gets it's assigned location)
    data_start_idx = input_args['start_idx']

    # we can create the memory mapped file in advance and fill in!
    # this is bc we should not be dropping examples at this point
    mmap_file = np.memmap(mmap_file_name, dtype=storage_type, mode='r+')
    entity_symbols = EntitySymbols(load_dir=os.path.join(data_args.entity_dir, data_args.entity_map_dir),
        alias_cand_map_file=data_args.alias_cand_map)
    with jsonlines.open(in_file, 'r') as f:
        for line_idx, line in enumerate(f):
            train_aliases_to_predict_arr = line['train_aliases_to_predict_arr']
            aliases_to_predict = line['aliases_to_predict']
            # drop example if we have nothing to predict (no valid aliases)
            spans = line['spans']
            word_indices = line['word_indices']
            aliases = line['aliases']
            qids = line['qids']
            alias_list_pos = line['alias_list_pos']
            assert len(aliases_to_predict) >= 0, f'There are no aliases to predict for an example. This should not happen at this point.'
            assert len(aliases) <= data_args.max_aliases, f'Each example should have no more that {data_args.max_aliases} max aliases. {json.dumps(line)} does.'
            example_aliases = np.ones(data_args.max_aliases) * PAD_ID
            example_aliases_locs_start = np.ones(data_args.max_aliases) * PAD_ID
            example_aliases_locs_end = np.ones(data_args.max_aliases) * PAD_ID
            example_alias_list_pos = np.ones(data_args.max_aliases) * PAD_ID
            # this stores the true entities we use to compute losses; some aliases don't meet filter criteria for testslices
            example_true_entities_for_loss = np.ones(data_args.max_aliases) * PAD_ID
            # this stores the true entities for all aliases seen by model, whether or not they are scored by model
            example_true_entities_for_train = np.ones(data_args.max_aliases) * PAD_ID
            # used to keep track of original alias index in the list
            for idx, (alias, span_idx, qid, alias_pos) in enumerate(zip(aliases, spans, qids, alias_list_pos)):
                span_start_idx,  span_end_idx = span_idx
                # generate indexes into alias table.
                alias_trie_idx = entity_symbols.get_alias_idx(alias)
                alias_qids = np.array(entity_symbols.get_qid_cands(alias))
                if not qid in alias_qids:
                    assert not data_args.train_in_candidates
                    # set class label to be "not in candidate set"
                    true_entity_idx = 0
                else:
                    # Here we are getting the correct class label for training.
                    # Our training is "which of the max_entities entity candidates is the right one (class labels 1 to max_entities) or is it none of these (class label 0)".
                    # + (not discard_noncandidate_entities) is to ensure label 0 is reserved for "not in candidate set" class
                    true_entity_idx = np.nonzero(alias_qids == qid)[0][0] + (not data_args.train_in_candidates)
                example_aliases[idx:idx+1] = alias_trie_idx
                example_aliases_locs_start[idx:idx+1] = span_start_idx
                # The span_idxs are [start, end). We want [start, end]. So subtract 1 from end idx.
                example_aliases_locs_end[idx:idx+1] = span_end_idx-1
                # If the final token is greater than the sentence length, truncate it. This happens due to windowing which enusre that start is within
                # the max token len. These edge aliases are scored in another subsent.
                if example_aliases_locs_end[idx:idx+1] >= data_args.max_word_token_len:
                    example_aliases_locs_end[idx:idx+1] = data_args.max_word_token_len-1
                example_alias_list_pos[idx:idx+1] = alias_pos
                # leave as -1 if it's not an alias we want to predict; we get these if we split a sentence and need to only predict subsets
                if idx in aliases_to_predict:
                    example_true_entities_for_loss[idx:idx+1] = true_entity_idx
                if idx in train_aliases_to_predict_arr:
                    example_true_entities_for_train[idx:idx+1] = true_entity_idx
            # drop example if we have nothing to predict (no valid aliases) -- make sure this doesn't cause problems when we start using unk aliases...
            if (all(example_aliases == PAD_ID)):
                logging.error(f"There were 0 aliases in this example {line}. This shouldn't happen.")
                sys.exit(0)
            data_array = [example_aliases_locs_start,
                            example_aliases_locs_end,
                            example_aliases,
                            word_indices,
                            example_true_entities_for_loss,
                            example_true_entities_for_train,
                            example_alias_list_pos,
                            int(line['sent_idx']),
                            int(line['subsent_idx'])]
            mmap_file[line_idx+data_start_idx] = np.array([tuple(data_array)], dtype=storage_type)
    mmap_file.flush()

def generate_slice_data_helper(input_args):
    args = input_args['args']
    data_args = args.data_config
    dataset_is_eval = input_args['dataset_is_eval']
    text_file = input_args['text_file']
    out_file = input_args['out_file']
    use_weak_label = input_args['use_weak_label']
    slice_names = input_args['slice_names']

    entity_symbols = EntitySymbols(load_dir=os.path.join(data_args.entity_dir, data_args.entity_map_dir),
        alias_cand_map_file=data_args.alias_cand_map)
    max_cands = entity_symbols.max_candidates + (not args.data_config.train_in_candidates)
    add_null_cand = not data_args.train_in_candidates
    dropped_lines = 0
    num_lines = 0
    # The memmap stores things differently when you have two integers and we want to keep a2p as an array
    # Therefore for force max the minimum max_a2p to be 2
    max_a2p = 2
    max_sent_idx = -1
    with jsonlines.open(text_file, 'r') as reader, jsonlines.open(out_file, 'w') as writer:
        for line in reader:
            assert 'sent_idx_unq' in line
            assert 'aliases' in line
            sent_idx = int(line['sent_idx_unq'])
            # keep track of max sent index for building sent idx row mapping
            max_sent_idx = max(sent_idx, max_sent_idx)
            aliases = line['aliases']
            num_a2p = len(aliases)
            max_cands_per_alias = prep_utils.get_max_cands_per_alias(add_null_cand, aliases, entity_symbols)
            slices = prep_utils.get_slice_values(slice_names, line)

            # dict from slice_names -> aliases_to_predict
            # only used for slicing models
            anchor = [True for i in range(len(aliases))]
            if ANCHOR_KEY in line:
                anchor = line[ANCHOR_KEY]
                assert len(aliases) == len(anchor)
                assert all(isinstance(a, bool) for a in anchor)
                if dataset_is_eval:
                    # Reindex aliases to predict to be where anchor == True because we only ever want to predict those (it will see all aliases in
                    # the forward pass but we will only score the True anchors)
                    for slice_name in slices:
                        # ignore slice names that we don't use
                        aliases_to_predict = slices[slice_name]
                        slices[slice_name] = {i:aliases_to_predict[i] for i in aliases_to_predict if anchor[int(i)] is True}
                    max_cands_per_alias = {i:max_cands_per_alias[i] for i in max_cands_per_alias if anchor[int(i)] is True}
            # Base slice
            slices[BASE_SLICE] = data_utils.get_base_slice(anchor, slices, slice_names, dataset_is_eval)
            if dataset_is_eval:
                slices[FINAL_LOSS] = {str(i):1.0 for i in range(len(aliases)) if anchor[i] is True}
            else:
                slices[FINAL_LOSS] = {str(i):1.0 for i in range(len(aliases))}

            # If not use_weak_label, only the anchor is True aliases will be given to the model
            # We must re-index alias to predict to be in terms of anchors == True
            # Ex: anchors = [F, T, T, F, F, T]
            #     If dataset_is_eval, let
            #     a2p = [2,5]     (a2p must only be for T anchors)
            #     AFTER NOT USE_WEAK_LABEL, DATA WILL BE ONLY THE TRUE ANCHORS
            #     a2p needs to be [1, 2] for the 3rd and 6th true become the 2nd and 3rd after not weak labelling
            #     If dataset_is_eval is False, let
            #     a2p = [0,2,4,5]     (a2p can be anything)
            if not use_weak_label:
                assert ANCHOR_KEY in line, 'Cannot toggle off data weak labelling without anchor info'
                # The number of aliases will be reduced to the number of true anchors
                num_a2p = sum(anchor)
                # We must correct all mappings with alias indexes in them (filterings, importances, ...)
                # This is because the indexing will change when we remove False anchors (see comment example above)
                slices = data_utils.correct_not_augmented_dict_values(anchor, slices)
                max_cands_per_alias = data_utils.correct_not_augmented_max_cands(anchor, max_cands_per_alias)
            # logging.debug(line, slices)
            # Remove slices that have no aliases to predict
            for slice_name in list(slices.keys()):
                if len(slices[slice_name]) <= 0:
                    del slices[slice_name]
            if len(slices) <= 1:
                if len(slices) == 1:
                    assert list(slices.keys())[0] == FINAL_LOSS, f'Slice includes {list(slices.keys())[0]}'
                dropped_lines += 1
                # log_func(f'Dropping sentence. There are no aliases to predict for {line}.')
                continue
            # For nicer code downstream, we make sure FINAL_LOSS is in here
            assert FINAL_LOSS in slices
            if train_utils.is_slicing_model(args) and not dataset_is_eval:
                assert len(slices) > 1, f'{slices} only has FINAL_LOSS and this is a train dataset with slicing'
            # We store aliases_to_predict as an index array into the aliases. The max is therefore based on the number of aliases
            max_a2p = max(max_a2p, num_a2p)
            # Write out updated and filtered slices with the sentence index for the next step in processing
            writer.write({'slices': slices, 'max_cands_per_alias': max_cands_per_alias, 'sent_idx': sent_idx})
            num_lines += 1

        logging.debug(f"Finished processing data. Dropped {dropped_lines} lines. If use_weak_label is false, this is because there were only false anchors.")
        logging.debug(f"Max aliases2predict for chunk is {max_a2p}")
        return {'max_a2p': max_a2p, 'max_cands': max_cands, 'num_lines': num_lines, 'max_sent_idx': max_sent_idx}


def process_slice_data_helper(input_args):
    sent_idx_file = input_args['sent_idx_file']
    slice_idx_file = input_args['slice_idx_file']
    in_text_file = input_args['in_file']
    args = input_args['args']
    storage_type = input_args['storage_type']
    slice_names = input_args['slice_names']
    data_len = input_args['len']
    # write into correct location in memory mapped file (each process gets it's assigned location)
    data_start_idx = input_args['start_idx']
    max_a2p = input_args['max_a2p']
    max_cands = input_args['max_cands']

    # load up memory mapped files
    sent_idx_arr = np.memmap(sent_idx_file, dtype=np.int, mode='r+')
    slice_idx_arr = np.memmap(slice_idx_file, dtype=storage_type, mode='r+')
    with jsonlines.open(in_text_file, 'r') as f:
        for line_idx, line in enumerate(f):
            sent_idx = line['sent_idx']
            slices = line['slices']
            max_cands_per_alias = line['max_cands_per_alias']
            # Keep track of mapping of sent_idx to original row_idx -- this memory mapped file isn't chunked
            # across processes in blocks but sentence indices are unique so no processes
            # should write the same row
            sent_idx_arr[sent_idx] = line_idx+data_start_idx
            row_data = np.recarray((1,), dtype=storage_type)
            for slice_name in slice_names:
                # We use the information in "slices" key to generate two pieces of info
                # 1. Binary info for if a mention is in the slice, this is equivalent to the old slice indices and used
                # in eval_wrapper and to determine which samples to use for the prediction heads
                # 2. Probabilistic info for the prob the mention is in the slice, this is used to train indicator heads
                if slice_name in slices:
                    # Set indices of aliases to predict relevant to slice to 1-hot vector
                    slice_indexes = np.array([0]*(max_a2p))
                    for idx in slices[slice_name]:
                        # We consider an example as "in the slice" if it's probability is greater than 0.5
                        slice_indexes[int(idx)] = slices[slice_name][idx] > 0.5
                    base_a2p = slice_indexes
                else:
                    # Set to zero for all aliases if no aliases in example occur in the slice
                    base_a2p = np.array([0]*max_a2p)
                # Add probabilistic labels for training indicators
                if slice_name in slices:
                    # padded values are -1 so they are masked in score function
                    slices_padded = np.array([-1.0]*(max_a2p))
                    for idx in slices[slice_name]:
                        # The indexes needed to be string for json
                        slices_padded[int(idx)] = slices[slice_name][idx]
                    base_a_prob = slices_padded
                else:
                    base_a_prob = np.array([-1]*max_a2p)

                # Write slice indices into record array
                row_data[slice_name]['sent_idx'] = [sent_idx]
                row_data[slice_name]['alias_to_predict'] = base_a2p
                row_data[slice_name]['prob_labels'] = base_a_prob
            # Write row of slice_dt objects into the memory mapped file
            slice_idx_arr[line_idx+data_start_idx] = row_data
    slice_idx_arr.flush()
    sent_idx_arr.flush()

def process_emb_helper(input_args):
    idx = input_args['chunk_idx']
    args = input_args['args']
    out_data = input_args['out_data']
    data_len = input_args['len']
    data_start_idx = input_args['start_idx']
    data_file = input_args['in_file']
    in_storage_type = input_args['in_storage_type']
    num_cands_K = input_args['num_cands_K']
    num_aliases_with_pad = input_args['num_aliases_with_pad']
    example_data = np.memmap(data_file, dtype=in_storage_type, mode='r')
    data_args = args.data_config
    out_files = {}
    # alias entity table and embeddings have already been prepped by now so
    # we can pass None for entity_symbols to avoid huge memory cost of
    # duplicating entity_symbols across processes
    alias2entity_table, _ = AliasEntityTable.prep(args=args, entity_symbols=None,
        num_aliases_with_pad=num_aliases_with_pad, num_cands_K=num_cands_K,
        log_func=logging.debug)
    logging.debug(f'alias table size {len(pickle.dumps(alias2entity_table, -1))}')
    batch_prepped = {}
    for emb in args.data_config.ent_embeddings:
        if 'batch_prep' in emb and emb['batch_prep']:
            mod, load_class = import_class("bootleg.embeddings", emb.load_class)
            batch_prepped[emb.key] = getattr(mod, load_class)(main_args=args,
            emb_args=emb['args'], model_device=None, entity_symbols=None, word_symbols=None, word_emb=None, key=emb.key)
            # load memory mapped file for the corresponding embedding to write
            out_files[emb.key] = np.memmap(out_data[emb.key]['file_name'], dtype=out_data[emb.key]['dtype'], shape=out_data[emb.key]['shape'], mode='r+')
    logging.debug('Loaded entity embeddings to batch_prep')
    # read in relevant portion of data chunk, perform batch prepping and write out
    # to relevant portion of data chunk for relevant memory mapped file
    count = 0
    start = time.time()
    for i in range(data_start_idx, data_start_idx + data_len):
        count += 1
        example = example_data[i]
        entity_indices = alias2entity_table[example['alias_idx']]
        for emb_name, emb in batch_prepped.items():
            # TODO: refactor conditionals out of loop
            out_files[emb_name][i] = emb.batch_prep(example['alias_idx'], entity_indices)
        if count % 10000 == 0:
            logging.debug(f'done with {count} out of {data_len} or {count/data_len} samples in {time.time() - start}')
    # flush files to be sure they are written
    for emb_name, emb in batch_prepped.items():
        out_files[emb_name].flush()

#===========================================#
# Parent functions of multiprocesses
#===========================================#

def generate_data_subsent(args, use_weak_label, dataset_is_eval, chunk_prep_dir, predata_prep_dir):
    logging.debug('Starting to extract subsentences')
    start = time.time()
    chunk_metadata = utils.load_json_file(os.path.join(chunk_prep_dir, 'metadata.json'))
    num_chunks = len(chunk_metadata['chunks'])
    num_processes = min(args.run_config.dataset_threads, int(multiprocessing.cpu_count()))
    logging.debug("Parallelizing with " + str(num_processes) + " threads.")
    sent_file_path = os.path.join(predata_prep_dir, 'sent_chunk')
    out_files = [f'{sent_file_path}_{i}.jsonl' for i in range(num_chunks)]
    all_process_args = [{'chunk_idx': i,
                         'args': args,
                         'use_weak_label': use_weak_label,
                         'dataset_is_eval': dataset_is_eval,
                         'text_file': chunk_metadata['chunks'][i],
                         'out_file': out_files[i]} for i in range(num_chunks)]
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(generate_data_subsent_helper, all_process_args)
    pool.close()
    pool.join()
    chunks = []
    idx = 0
    for i in range(num_chunks):
        data_len = results[i]
        chunks.append({'path': out_files[i],
                     'len': data_len,
                     'start_idx': idx})
        idx += data_len
    metadata = {'chunks': chunks, 'total_num_exs': idx}
    utils.dump_json_file(os.path.join(predata_prep_dir, 'metadata.json'), metadata)
    logging.debug(f'Extracted {idx} sub-sentences in {round(time.time() - start,2)}s')


def process_data_subsent(args, predata_prep_dir, data_prep_dir, dataset_name):
    logging.debug('Building data chunks...')
    start = time.time()
    chunk_metadata = utils.load_json_file(os.path.join(predata_prep_dir, 'metadata.json'))
    num_chunks = len(chunk_metadata['chunks'])

    is_bert = load_wordsymbols(args.data_config).is_bert

    num_processes = min(args.run_config.dataset_threads, int(multiprocessing.cpu_count()))
    logging.debug("Parallelizing with " + str(num_processes) + " threads.")

    num_aliases_M = args.data_config.max_aliases
    # i1 is 8-bit ints, i2 is 16-bit ints, i4 is 32-bit ints, i8 is 64-bit ints
    # embedding indexing in python requires 64-bit (long), we could potentially store more
    # as 32-bit though if needed
    # https://discuss.pytorch.org/t/problems-with-target-arrays-of-int-int32-types-in-loss-functions/140/2
    storage_type = [
         ('start_idx_in_sent', 'i2', num_aliases_M), # the sentence length should not be that long so we use 16-bit
         ('end_idx_in_sent', 'i2', num_aliases_M), # the sentence length should not be that long so we use 16-bit
         ('alias_idx', 'i8', num_aliases_M),
         ('word_indices', 'i8', (args.data_config.max_word_token_len + 2*int(is_bert))),
         ('true_entity_idx', 'i8', num_aliases_M),
         ('true_entity_idx_for_train', 'i8', num_aliases_M),
         ('alias_list_pos', 'i4', num_aliases_M), # there should not be that many aliases in a sentence
         ('sent_idx', 'i8'),
         ('subsent_idx', 'i8')]

    # dump storage type for loading in wiki_dataset
    storage_type_file = data_utils.get_storage_file(dataset_name)
    pickle.dump(storage_type, open(storage_type_file, 'wb'))

    mmap_file = np.memmap(dataset_name, dtype=storage_type, mode='w+', shape=(chunk_metadata['total_num_exs'],))

    all_process_args = [{'chunk_idx': i,
                        'args': args,
                        'in_file': chunk_metadata['chunks'][i]['path'],
                        'len': chunk_metadata['chunks'][i]['len'],
                        'start_idx':  chunk_metadata['chunks'][i]['start_idx'],
                        'out_file': dataset_name,
                        'storage_type': storage_type
                        } for i in range(num_chunks)]
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(process_data_subsent_helper, all_process_args)
    pool.close()
    pool.join()
    # store some metadata from previous step to help in batch prepping
    # emb data
    metadata = {'data_file': dataset_name, 'storage_type': storage_type, 'total_num_exs': chunk_metadata['total_num_exs'], 'chunks': chunk_metadata['chunks']}
    utils.dump_json_file(os.path.join(data_prep_dir, 'metadata.json'), metadata)
    logging.debug(f'Finished building chunks in {round(time.time() - start,2)}s')

def generate_slice_data(args, use_weak_label, dataset_is_eval, chunk_prep_dir, slice_prep_dir):
    logging.debug('Starting to extract slice data')
    start = time.time()

    chunk_metadata = utils.load_json_file(os.path.join(chunk_prep_dir, 'metadata.json'))
    num_chunks = len(chunk_metadata['chunks'])

    # Get slice names
    slice_names = train_utils.get_data_slices(args, dataset_is_eval)
    assert FINAL_LOSS in slice_names
    if train_utils.model_has_base_head_loss(args):
        assert BASE_SLICE in slice_names

    num_processes = min(args.run_config.dataset_threads, int(multiprocessing.cpu_count()))
    logging.debug("Parallelizing with " + str(num_processes) + " threads.")
    slice_file_path = os.path.join(slice_prep_dir, 'slice_chunk')
    out_files = [f'{slice_file_path}_{i}.jsonl' for i in range(num_chunks)]
    all_process_args = [{'chunk_idx': i,
                         'args': args,
                         'use_weak_label': use_weak_label,
                         'slice_names': slice_names,
                         'dataset_is_eval': dataset_is_eval,
                         'text_file': chunk_metadata['chunks'][i],
                         'out_file': out_files[i]} for i in range(num_chunks)]
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(generate_slice_data_helper, all_process_args)
    pool.close()
    pool.join()
    chunks = []
    idx = 0
    global_max_a2p = -1
    global_max_sent_idx = -1
    global_max_cands = -1
    for i in range(num_chunks):
        max_a2p = results[i]['max_a2p']
        max_cands = results[i]['max_cands']
        data_len = results[i]['num_lines']
        max_sent_idx = results[i]['max_sent_idx']
        chunks.append({'path': out_files[i],
                     'len': data_len,
                     'start_idx': idx})
        idx += data_len
        global_max_a2p = max(max_a2p, global_max_a2p)
        global_max_sent_idx = max(max_sent_idx, global_max_sent_idx)
        global_max_cands = max(max_cands, global_max_cands)
    logging.debug(f"Max aliases2predict GLOBAL is {global_max_a2p}")
    logging.debug(f"Max sentidx GLOBAL is {global_max_sent_idx}")
    metadata = {'chunks': chunks,
        'total_num_exs': idx,
        'max_a2p': global_max_a2p,
        'max_cands': global_max_cands,
        'max_sent_idx': global_max_sent_idx,
        'slice_names': slice_names}
    utils.dump_json_file(os.path.join(slice_prep_dir, 'metadata.json'), metadata)
    logging.debug(f'Extracted {idx} slice data in {round(time.time() - start,2)}s')

def process_slice_data(args, dataset_name, sent_idx_file, slice_prep_dir, slice_final_prep_dir):
    logging.debug('Building slice indices...')
    start = time.time()
    chunk_metadata = utils.load_json_file(os.path.join(slice_prep_dir, 'metadata.json'))
    num_chunks = len(chunk_metadata['chunks'])
    max_a2p = chunk_metadata['max_a2p']
    max_cands = chunk_metadata['max_cands']
    max_sent_idx = chunk_metadata['max_sent_idx']
    slice_names = chunk_metadata['slice_names']
    total_num_exs = chunk_metadata['total_num_exs']
    num_processes = min(args.run_config.dataset_threads, int(multiprocessing.cpu_count()))
    logging.debug("Parallelizing with " + str(num_processes) + " threads.")

    # Create shared memory mapped file for saving slice indices
    # For each row in the dataset, we have a slice_dt object for each slice in slice_names
    # Nested structure recarray dtype to store slice, sent_idx, aliases_to_predict, and importance_values
    slice_dt = np.dtype([('sent_idx', int), ('alias_to_predict', int, max_a2p), ('prob_labels', float, max_a2p)])
    storage_type = np.dtype([(slice_name, slice_dt) for slice_name in slice_names])
    np.memmap(dataset_name, dtype=storage_type, mode='w+', shape=(total_num_exs, 1))

    # create shared memory mapped file for row/sent_idx mapping to easily retrieve indices
    # given a sent_idx
    # +1 bc length needs to be one greater than index
    sent_idx_arr = np.memmap(sent_idx_file, dtype=np.int, mode='w+', shape=(max_sent_idx+1,))
    # initialize sentence indices as -1 as not all sentence indices will be set
    sent_idx_arr[:] = np.ones(max_sent_idx+1).astype(int) * -1

    # launch processes to write their share of the memory mapped file
    all_process_args = [{'args': args,
                        'in_file': chunk_metadata['chunks'][i]['path'],
                        'len': chunk_metadata['chunks'][i]['len'],
                        'start_idx':  chunk_metadata['chunks'][i]['start_idx'],
                        'sent_idx_file': sent_idx_file,
                        'slice_idx_file': dataset_name,
                        'storage_type': storage_type,
                        'slice_names': slice_names,
                        'max_a2p': max_a2p,
                        'max_cands': max_cands
                        } for i in range(num_chunks)]
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(process_slice_data_helper, all_process_args)
    pool.close()
    pool.join()
    # save metadata
    logging.debug(f'Finished building slice indices in {round(time.time() - start,2)}s')
    return storage_type

def process_emb(args, data_prep_dir, data_feats_prep_dir, num_candidates_K, num_aliases_with_pad):
    # want to reuse objects that have been prepped at this point
    orig_preprocessed_val = args.data_config.overwrite_preprocessed_data
    args.data_config.overwrite_preprocessed_data = False
    logging.debug('Preprocessing embeddings for the data...')
    start = time.time()
    # open memory mapped file for data -- needed for candidates
    data_metadata = utils.load_json_file(os.path.join(data_prep_dir, 'metadata.json'))
    num_chunks = len(data_metadata['chunks'])
    data_file = data_metadata['data_file']
    batch_prepped_data_config_file = data_utils.get_batch_prep_config(data_file)
    data_storage_type = [tuple(type) for type in data_metadata['storage_type']]
    # create memory mapped file for each batch_prepped embedding -- we keep these separate for modularity and so we can easily iterate on these components
    batch_prepped_emb_data = defaultdict(dict)
    num_aliases_M = args.data_config.max_aliases

    for emb in args.data_config.ent_embeddings:
        # ASSUMES THAT PREPROCESSED INFO IS MxKxdim and can be represented with an int16 number
        if 'batch_prep' in emb and emb['batch_prep']:
            # values are summed over all candidates in sentence so not just binary
            # TODO: more specific embedding tag
            assert "dim" in emb, f"You need the dim key for {emb} in the embedding config: \"dim:\" dim for MxKxdim embedding for batch_prep to occur."
            assert "dtype" in emb, f"You need the dtype key for {emb} in the embedding config: \"dtype:\" (int16, float,...) for batch_prep to occur."
            shape = (data_metadata['total_num_exs'],num_aliases_M*num_candidates_K*emb.dim)
            dtype = emb.dtype
            logging.debug(f"Setting dtype of {emb.key} to {dtype}")
            batch_prep_file_name = f'{os.path.splitext(data_file)[0]}_{emb.key}_{dtype}.pt'
            mmap_file = np.memmap(batch_prep_file_name, dtype=dtype, mode='w+', shape=shape)
            batch_prepped_emb_data[emb.key]['dtype'] = dtype
            batch_prepped_emb_data[emb.key]['shape'] = shape
            batch_prepped_emb_data[emb.key]['file_name'] = batch_prep_file_name

    # make sure embeddings to batch_prep have been prepped
    prep_all_embs(args)

    # call subprocesses
    num_processes = min(args.run_config.dataset_threads, int(multiprocessing.cpu_count()))
    logging.debug("Parallelizing with " + str(num_processes) + " threads.")

    all_process_args = [{'chunk_idx': i,
                        'args': args,
                        'in_file': data_file,
                        'in_storage_type': data_storage_type, # storage type of original memory mapped file
                        'len': data_metadata['chunks'][i]['len'],
                        'start_idx': data_metadata['chunks'][i]['start_idx'],
                        'out_data': batch_prepped_emb_data,
                        'num_cands_K': num_candidates_K,
                        'num_aliases_with_pad': num_aliases_with_pad
                        } for i in range(num_chunks)]
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(process_emb_helper, all_process_args)
    pool.close()
    pool.join()

    # save metadata about storage types to be read in by wiki_dataset
    if os.path.exists(batch_prepped_data_config_file):
        batch_prepped_data_config = utils.load_json_file(batch_prepped_data_config_file)
    else:
        batch_prepped_data_config = {}
    batch_prepped_data_config.update(batch_prepped_emb_data)
    utils.dump_json_file(filename=batch_prepped_data_config_file, contents=batch_prepped_data_config)

    metadata = {'batch_prepped_files': batch_prepped_emb_data, 'total_num_exs': data_metadata['total_num_exs'], 'data_config': batch_prepped_data_config_file}
    utils.dump_json_file(os.path.join(data_feats_prep_dir, 'metadata.json'), metadata)
    logging.debug(f'Finished batch prepping embs in {round(time.time() - start,2)}s')
    # revert back args
    args.data_config.overwrite_preprocessed_data = orig_preprocessed_val

def get_idx_mapping(data_prep_dir, dataset_name):
    logging.debug('Getting sentence to data idx mapping')
    start = time.time()
    data_metadata = json.load(open(os.path.join(data_prep_dir, 'metadata.json'), 'r'))
    storage_type = [tuple(type) for type in data_metadata['storage_type']]
    data = np.memmap(dataset_name, dtype=storage_type, mode='r')
    sent_idx_to_idx = {}
    for id, row in enumerate(data):
        # can be multiple subsents per sent_idx
        if row['sent_idx'] in sent_idx_to_idx:
            sent_idx_to_idx[row['sent_idx']].append(id)
        else:
            sent_idx_to_idx[row['sent_idx']] = [id]
    sent_idx_file = os.path.splitext(dataset_name)[0] + "_sent_idx.json"
    utils.dump_json_file(sent_idx_file, sent_idx_to_idx)
    logging.debug(f'Finished mapping in {round(time.time() - start,2)}s')

#===========================================#
# Main prep functions
#===========================================#

def prep_data(args, use_weak_label, dataset_is_eval, input_src='', chunk_data=True, ext_subsent=True, build_data=True, batch_prep_embeddings=True,
    prep_dir='', dataset_name='', keep_all=False):
    data_tag = os.path.splitext(os.path.basename(dataset_name))[0]
    chunk_prep_dir = f'{prep_dir}/{data_tag}/chunks'
    predata_prep_dir = f'{prep_dir}/{data_tag}/predata'
    data_prep_dir = f'{prep_dir}/{data_tag}/data'
    data_feats_prep_dir = f'{prep_dir}/{data_tag}/data_feats'
    utils.ensure_dir(chunk_prep_dir)
    utils.ensure_dir(predata_prep_dir)
    utils.ensure_dir(data_prep_dir)
    utils.ensure_dir(data_feats_prep_dir)

    # Default to running all if no specific commands set
    run_all = (((not build_data) and (not chunk_data) and (not ext_subsent) and (not batch_prep_embeddings))
        or (build_data and chunk_data and ext_subsent and batch_prep_embeddings))

    entity_symbols, num_cands_K, num_aliases_with_pad = get_entities(args)
    AliasEntityTable.prep(args, entity_symbols, num_cands_K=num_cands_K, num_aliases_with_pad=num_aliases_with_pad,
    log_func=logging.debug)

    # Chunk text into multiple files
    if chunk_data or run_all:
        chunk_text_data(args, input_src, chunk_prep_dir)

    # Extract subsentences
    if ext_subsent or run_all:
        generate_data_subsent(args, use_weak_label, dataset_is_eval, chunk_prep_dir, predata_prep_dir)

    # Build data arrays
    if build_data or run_all:
        # dataset name needed for optional dumping of filtered data
        process_data_subsent(args, predata_prep_dir, data_prep_dir, dataset_name)

        # Get sentence idx mapping to be able to subsample slices
        get_idx_mapping(data_prep_dir, dataset_name)

    # Preprocess embeddings
    if batch_prep_embeddings or run_all:
        process_emb(args, data_prep_dir, data_feats_prep_dir, num_candidates_K=num_cands_K, num_aliases_with_pad=num_aliases_with_pad)

    # TODO: way to check for existence to decide to skip a step?
    # TODO: clean up and remove unnecessary files -- leave metadata?
    # by default we remove the data chunks
    if not keep_all:
        logging.debug('Cleaning up and removing chunk files...')
        text_chunk_files = glob.glob(f'{chunk_prep_dir}/*jsonl')
        for file in text_chunk_files:
            os.remove(file)

        predata_chunk_files = glob.glob(f'{predata_prep_dir}/*jsonl')
        for file in predata_chunk_files:
            os.remove(file)

        data_chunk_files = glob.glob(f'{data_prep_dir}/*bin')
        for file in data_chunk_files:
            os.remove(file)

def prep_slice(args, file, use_weak_label, dataset_is_eval, dataset_name,
    sent_idx_file, storage_config, keep_all=False):
    logging.debug(f'Prepping slice {file} ...')
    prep_dir = get_data_prep_dir(args)
    data_tag = os.path.splitext(os.path.basename(dataset_name))[0]
    chunk_prep_dir = f'{prep_dir}/{data_tag}/chunks'
    slice_prep_dir = f'{prep_dir}/{data_tag}/slice_chunks'
    slice_final_prep_dir = f'{prep_dir}/{data_tag}/slice_final'
    utils.ensure_dir(chunk_prep_dir)
    utils.ensure_dir(slice_prep_dir)
    utils.ensure_dir(slice_final_prep_dir)

    # TODO: check if chunked data already exists first, reuse one from main dataset?
    # chunk text data
    chunk_text_data(args=args, input_src=os.path.join(args.data_config.data_dir, file),
        chunk_prep_dir=chunk_prep_dir)

    # update slice data
    generate_slice_data(args=args, use_weak_label=use_weak_label, dataset_is_eval=dataset_is_eval,
        chunk_prep_dir=chunk_prep_dir, slice_prep_dir=slice_prep_dir)

    # save slice data into memory mapped file
    storage_type = process_slice_data(args=args, dataset_name=dataset_name,
        sent_idx_file=sent_idx_file, slice_prep_dir=slice_prep_dir, slice_final_prep_dir=slice_final_prep_dir)
    # save storage type file
    np.save(storage_config, storage_type, allow_pickle=True)

    logging.debug("Done prepping slice")

    if not keep_all:
        logging.debug('Cleaning up and removing chunk files...')
        text_chunk_files = glob.glob(f'{chunk_prep_dir}/*jsonl')
        for file in text_chunk_files:
            os.remove(file)

        slice_chunk_files = glob.glob(f'{slice_prep_dir}/*jsonl')
        for file in slice_chunk_files:
            os.remove(file)

    return storage_type

def main():
    start = time.time()
    multiprocessing.set_start_method("forkserver", force=True)
    config_parser = argparse.ArgumentParser('Where is config script?')
    config_parser.add_argument('--config_script', type=str, default='run_config.json',
                        help='This config should mimc the config.py config json with parameters you want to override.')
    args, unknown = config_parser.parse_known_args()
    args = get_full_config(args.config_script, unknown)
    train_utils.setup_train_heads_and_eval_slices(args)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    numeric_level = getattr(logging, args.run_config.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s %(message)s')
    log = logging.getLogger()

    prep_dir = get_data_prep_dir(args)
    utils.ensure_dir(prep_dir)
    log_name = os.path.join(prep_dir, "log")
    if not os.path.exists(log_name): os.system("touch " + log_name)

    fh = logging.FileHandler(log_name, mode='w')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Dump command line arguments
    logging.debug("Machine: " + os.uname()[1])
    logging.debug("CMD: python " + " ".join(sys.argv))

    # Prep embs first since they may be used for batch prepping when preparing datasets
    if args.prep_config.prep_embs:
        logging.info('Prepping embs...')
        start = time.time()
        prep_all_embs(args)
        logging.info(f'Done prepping embs in {round(time.time() - start, 2)}s!')

    if args.prep_config.prep_train:
        logging.info('Prepping train...')
        dataset = generate_save_data_name(
            data_args=args.data_config,
            use_weak_label=args.data_config.train_dataset.use_weak_label,
            split_name=os.path.splitext(args.data_config.train_dataset.file)[0])
        prep_data(args, use_weak_label=args.data_config.train_dataset.use_weak_label, dataset_is_eval=False,
            input_src=os.path.join(args.data_config.data_dir, args.data_config.train_dataset.file),
            chunk_data=args.prep_config.chunk_data,
            ext_subsent=args.prep_config.ext_subsent,
            build_data=args.prep_config.build_data,
            batch_prep_embeddings=args.prep_config.batch_prep_embeddings,
            prep_dir=prep_dir,
            dataset_name=os.path.join(prep_dir, dataset), keep_all=args.prep_config.keep_all)
        logging.debug(f'Prepped {args.data_config.train_dataset.file}.')
        logging.info(f'Done with train prep in {round(time.time() - start, 2)}s!')

    if args.prep_config.prep_dev:
        logging.info('Prepping dev...')
        start = time.time()
        dataset = generate_save_data_name(
            data_args=args.data_config,
            use_weak_label=args.data_config.dev_dataset.use_weak_label,
            split_name=os.path.splitext(args.data_config.dev_dataset.file)[0])
        prep_data(args, use_weak_label=args.data_config.dev_dataset.use_weak_label, dataset_is_eval=True,
              input_src=os.path.join(args.data_config.data_dir, args.data_config.dev_dataset.file),
              chunk_data=args.prep_config.chunk_data,
              ext_subsent=args.prep_config.ext_subsent,
              build_data=args.prep_config.build_data,
              batch_prep_embeddings=args.prep_config.batch_prep_embeddings,
              prep_dir=prep_dir,
              dataset_name=os.path.join(prep_dir, dataset), keep_all=args.prep_config.keep_all)
        logging.debug(f'Prepped {args.data_config.dev_dataset.file}')
        logging.info(f'Done with dev prep in {round(time.time() - start, 2)}s!')

    if args.prep_config.prep_test:
        logging.info('Prepping test...')
        start = time.time()
        dataset = generate_save_data_name(
            data_args=args.data_config,
            use_weak_label=args.data_config.test_dataset.use_weak_label,
            split_name=os.path.splitext(args.data_config.test_dataset.file)[0])
        prep_data(args, use_weak_label=args.data_config.test_dataset.use_weak_label, dataset_is_eval=True,
              input_src=os.path.join(args.data_config.data_dir, args.data_config.test_dataset.file),
              chunk_data=args.prep_config.chunk_data,
              ext_subsent=args.prep_config.ext_subsent,
              build_data=args.prep_config.build_data,
              batch_prep_embeddings=args.prep_config.batch_prep_embeddings,
              prep_dir=prep_dir,
              dataset_name=os.path.join(prep_dir, dataset), keep_all=args.prep_config.keep_all)
        logging.debug(f'Prepped {args.data_config.test_dataset.file}')
        logging.info(f'Done with test prep in {round(time.time() - start, 2)}s!')

    if args.prep_config.prep_train_slices:
        logging.info('Prepping train slices...')
        start = time.time()
        dataset_name = data_utils.generate_slice_name(args, args.data_config, use_weak_label=args.data_config.train_dataset.use_weak_label,
            split_name="slice_" + os.path.splitext(args.data_config.train_dataset.file)[0],
            dataset_is_eval=False)
        full_dataset_name = os.path.join(prep_dir, dataset_name)
        config_dataset_name = os.path.join(prep_dir, data_utils.get_slice_storage_file(dataset_name))
        sent_idx_to_idx_file = os.path.join(prep_dir, data_utils.get_sent_idx_file(dataset_name))
        prep_slice(args, args.data_config.train_dataset.file, args.data_config.train_dataset.use_weak_label,
            dataset_is_eval=False, dataset_name=full_dataset_name,
            sent_idx_file=sent_idx_to_idx_file, storage_config=config_dataset_name,
            keep_all=args.prep_config.keep_all)
        logging.debug(f'Prepped slices from {args.data_config.train_dataset.file} to {full_dataset_name}.')
        logging.info(f'Done with train slice prep in {round(time.time() - start, 2)}s!')

    if args.prep_config.prep_dev_eval_slices:
        logging.info('Prepping dev slices...')
        start = time.time()
        dataset_name = data_utils.generate_slice_name(args, args.data_config, use_weak_label=args.data_config.dev_dataset.use_weak_label,
            split_name="slice_" + os.path.splitext(args.data_config.dev_dataset.file)[0],
             dataset_is_eval=True)
        full_dataset_name = os.path.join(prep_dir, dataset_name)
        config_dataset_name = os.path.join(prep_dir, data_utils.get_slice_storage_file(dataset_name))
        sent_idx_to_idx_file = os.path.join(prep_dir, data_utils.get_sent_idx_file(dataset_name))
        prep_slice(args, args.data_config.dev_dataset.file, args.data_config.dev_dataset.use_weak_label,
            dataset_is_eval=True, dataset_name=full_dataset_name,
            sent_idx_file=sent_idx_to_idx_file, storage_config=config_dataset_name,
            keep_all=args.prep_config.keep_all)
        logging.debug(f'Prepped slices from {args.data_config.dev_dataset} to {full_dataset_name}.')
        logging.info(f'Done with dev slice prep in {round(time.time() - start, 2)}s!')

    if args.prep_config.prep_test_eval_slices:
        logging.info('Prepping test slices...')
        start = time.time()
        dataset_name = data_utils.generate_slice_name(args, args.data_config, use_weak_label=args.data_config.test_dataset.use_weak_label,
            split_name="slice_" + os.path.splitext(args.data_config.test_dataset.file)[0],
             dataset_is_eval=True)
        full_dataset_name = os.path.join(prep_dir, dataset_name)
        config_dataset_name = os.path.join(prep_dir, data_utils.get_slice_storage_file(dataset_name))
        sent_idx_to_idx_file = os.path.join(prep_dir, data_utils.get_sent_idx_file(dataset_name))
        prep_slice(args, args.data_config.test_dataset.file, args.data_config.test_dataset.use_weak_label,
            dataset_is_eval=True, dataset_name=full_dataset_name,
            sent_idx_file=sent_idx_to_idx_file, storage_config=config_dataset_name,
            keep_all=args.prep_config.keep_all)
        logging.debug(f'Prepped slices from {args.data_config.test_dataset} to {full_dataset_name}.')
        logging.info(f'Done with test slice prep in {round(time.time() - start, 2)}s!')

if __name__ == '__main__':
    main()