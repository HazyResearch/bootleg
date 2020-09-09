import unittest

import marisa_trie
import numpy as np
import os

from bootleg.symbols.constants import BASE_SLICE, FINAL_LOSS
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.dataloaders.wiki_slices import WikiSlices
from bootleg.dataloaders.wiki_dataset import WikiDataset
from bootleg.utils import data_utils, parser_utils, train_utils

ARR_KEYS_TO_COMPARE = [
    'start_idx_in_sent','end_idx_in_sent','alias_idx','word_indices','alias_list_pos',
    f'slice:{FINAL_LOSS}_pred',f'slice:{FINAL_LOSS}_ind',
    f'slice:{BASE_SLICE}_pred',f'slice:{BASE_SLICE}_ind',
    f'slice:slice1_pred',f'slice:slice1_ind',
    f'slice:slice2_pred',f'slice:slice2_ind',
]

# max_aliases-len(aliases) aliases will be -1 padded
# Other aliases get 1s for their candidates and -1 otherwise (as these are padded candidates)
def get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands):
    max_cands = 6
    arr = np.ones((max_aliases, max_cands+int(not train_in_cands)))*-1
    for al_idx, alias in enumerate(aliases):
        cand_arr = np.ones(max_cands+int(not train_in_cands))*-1
        for i in range(len(alias2wpids[alias])):
            cand_arr[i+int(not train_in_cands)] = 1
        if not train_in_cands:
            cand_arr[0] = 1
        arr[al_idx] = cand_arr
    return arr

def get_data_train_in_candidates():
    """ Return artificial datasets when train_in_candidates = True """

    sent_idx_to_idx = {}

    # Manually set alias map
    alias2wpids = {
        'alias1': [["Q1", 10], ["Q4", 6]],
        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
        'alias3': [["Q1", 30]],
        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
    }
    alias_trie = marisa_trie.Trie(alias2wpids.keys())
    max_aliases = 4
    # Manually set sentence data. Remember each line in the data file corresponds to an independent sample, with:
    # {sentence_index}|{alias to predict}~*~...|{alias}~*~...|{true QID}~*~...|{spans}~*~|{sentence}
    truedata = [None, None, None, None]
    # Sentence 1:
    # 0|0~*~1|alias1~*~multi word alias2|Q1~*~Q4|0:1~*~2:5|alias1 or multi word alias2
    aliases = ['alias1', 'multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[0] = [0]
    truedata[0] = {
        'sent_idx': 0,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2, -1, -1]),
        'end_idx_in_sent': np.array([0, 4, -1, -1]),
        'alias_idx': np.array([alias_trie['alias1'], alias_trie['multi word alias2'], -1, -1]),
        'word_indices': np.array([1, 6, 5, 0, 2]),
        'alias_list_pos': np.array([0, 1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 2, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 2, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, 0, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, 0, -1, -1]),
    }

    # Sentence 2:
    # 1|0~*~1|alias3~*~alias4|Q1~*~Q4|0:1~*~2:3|alias3 cat alias4
    aliases = ['alias3', 'alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[1] = [1]
    truedata[1] = {
        'sent_idx': 1,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2, -1,-1]),
        'end_idx_in_sent': np.array([0, 2, -1,-1]),
        'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias4'], -1, -1]),
        'word_indices': np.array([3, 7, 4, -1, -1]),
        'alias_list_pos': np.array([0, 1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 0, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 0, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, 0, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, 0, -1, -1])
    }

    # Sentence 3:
    # 2|0|multi word alias2|Q4|1:4|cat multi word alias2
    aliases = ['multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[2] = [2]
    truedata[2] = {
        'sent_idx': 2,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([1,-1, -1, -1]),
        'end_idx_in_sent': np.array([3,-1, -1, -1]),
        'alias_idx': np.array([alias_trie['multi word alias2'], -1, -1, -1]),
        'word_indices': np.array([7, 5, 0, 2, -1]),
        'alias_list_pos': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([2, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([2, -1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, -1, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, -1, -1, -1]),
    }

    # Sentence 4:
    #3|0|alias4|Q3|2:3|alias3 cat alias4
    aliases = ['alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[3] = [3]
    truedata[3] = {
        'sent_idx': 3,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([2, -1, -1, -1]),
        'end_idx_in_sent': np.array([2, -1, -1, -1]),
        'alias_idx': np.array([alias_trie['alias4'], -1, -1, -1]),
        'word_indices': np.array([3, 7, 4, -1, -1]),
        'alias_list_pos': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, -1, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, -1, -1, -1]),
    }

    return alias2wpids, truedata, sent_idx_to_idx


def get_data_missing_gold():
    """ Return artificial dataset where the gold ID for one alias is not contained in the alias list"""

    sent_idx_to_idx = {}

    # Manually set alias map
    alias2wpids = {
        'alias1': [["Q1", 10], ["Q4", 6]],
        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
        'alias3': [["Q1", 30]],
        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
    }
    alias_trie = marisa_trie.Trie(alias2wpids.keys())
    max_aliases = 4
    # Manually set sentence data. Remember each line in the data file corresponds to an independent sample, with:
    # {sentence_index}|{alias to predict}~*~...|{alias}~*~...|{true QID}~*~...|{spans}~*~|{sentence}
    truedata = [None, None, None, None]
    # Sentence 1:
    # 0|0~*~1|alias1~*~multi word alias2|Q1~*~Q4|0:1~*~2:5|alias1 or multi word alias2
    aliases = ['alias1', 'multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=False)
    sent_idx_to_idx[0] = [0]
    truedata[0] = {
        'sent_idx': 0,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2, -1, -1]),
        'end_idx_in_sent': np.array([0, 4, -1, -1]),
        'alias_idx': np.array([alias_trie['alias1'], alias_trie['multi word alias2'], -1, -1]),
        'word_indices': np.array([1, 6, 5, 0, 2]),
        'alias_list_pos': np.array([0, 1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, 3, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, 3, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, 0, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, 0, -1, -1]),
    }

    # Sentence 2:
    # 1|0~*~1|alias3~*~alias4|Q1~*~Q4|0:1~*~2:3|alias3 cat alias4
    aliases = ['alias3', 'alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=False)
    sent_idx_to_idx[1] = [1]
    truedata[1] = {
        'sent_idx': 1,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2, -1,-1]),
        'end_idx_in_sent': np.array([0, 2, -1,-1]),
        'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias4'], -1, -1]),
        'word_indices': np.array([3, 7, 4, -1, -1]),
        'alias_list_pos': np.array([0, 1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, 1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, 1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, 0, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, 0, -1, -1])
    }

    # Sentence 3:
    # 2|0|multi word alias2|Q4|1:4|cat multi word alias2
    aliases = ['multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=False)
    sent_idx_to_idx[2] = [2]
    truedata[2] = {
        'sent_idx': 2,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([1,-1, -1, -1]),
        'end_idx_in_sent': np.array([3,-1, -1, -1]),
        'alias_idx': np.array([alias_trie['multi word alias2'], -1, -1, -1]),
        'word_indices': np.array([7, 5, 0, 2, -1]),
        'alias_list_pos': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([3, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([3, -1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, -1, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, -1, -1, -1])
    }

    # Sentence 4: Note that Q2 is not a candidate for alias3!
    #3|0~*~1|alias3~*~alias4|Q2~*~Q3|0:1~*~2:3|alias3 cat alias4
    aliases = ['alias3', 'alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=False)
    sent_idx_to_idx[3] = [3]
    truedata[3] = {
        'sent_idx': 3,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2, -1, -1]),
        'end_idx_in_sent': np.array([0, 2, -1, -1]),
        'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias4'],  -1, -1]),
        'word_indices': np.array([3, 7, 4, -1, -1]),
        'alias_list_pos': np.array([0, 1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 2, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 2, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, 0, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, 0, -1, -1])
    }
    return alias2wpids, truedata, sent_idx_to_idx

def get_data_train_long_sentence():
    """ Return artificial dataset where a sentence exceeds the sentence limit """

    sent_idx_to_idx = {}

    # Manually set alias map
    alias2wpids = {
        'alias1': [["Q1", 10], ["Q4", 6]],
        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
        'alias3': [["Q1", 30]],
        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
    }
    alias_trie = marisa_trie.Trie(alias2wpids.keys())
    max_aliases = 4
    # Manually set sentence data. Remember each line in the data file corresponds to an independent sample, with:
    # {sentence_index}|{alias to predict}~*~...|{alias}~*~...|{true QID}~*~...|{spans}~*~|{sentence}|
    truedata = [None, None, None, None, None]
    # Sentence 1:
    # 0|0~*~1|alias1~*~multi word alias2|Q1~*~Q4|0:1~*~2:5|alias1 or multi word alias2|"slices":{"base": [0,1], "slice1": [0], "slice2": [0]}
    # "filters": "slice1": {"0": [0], "1": [1,2]}, "slice2": {"0": [0], "1": [1,2]}}}
    aliases = ['alias1', 'multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    no_filt_arr_s1 = np.array(no_filt_arr, copy=True)
    # Make candidate 1 not be in filter (alias1 has two candidates)
    no_filt_arr_s1[0, 1] = 0
    # Make candidate 0 not be in filter (multi word alias2 has three candidates)
    no_filt_arr_s1[1, 0] = 0
    no_filt_arr_s2 = np.array(no_filt_arr, copy=True)
    # Make candidate 1 not be in filter (alias1 has two candidates)
    no_filt_arr_s2[0, 1] = 0
    # Make candidate 0 not be in filter (multi word alias2 has three candidates)
    no_filt_arr_s2[1, 0] = 0

    sent_idx_to_idx[0] = [0]
    truedata[0] = {
        'sent_idx': 0,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2, -1, -1]),
        'end_idx_in_sent': np.array([0, 4, -1, -1]),
        'alias_idx': np.array([alias_trie['alias1'], alias_trie['multi word alias2'], -1, -1]),
        'word_indices': np.array([1, 6, 5, 0, 2]),
        'alias_list_pos': np.array([0, 1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 2, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 2, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1, -1, -1]),
        f'slice:slice1_pred': np.array([0, -1, -1, -1]),
        f'slice:slice1_ind': np.array([1, 0, -1, -1]),
        f'slice:slice2_pred': np.array([0, -1, -1, -1]),
        f'slice:slice2_ind': np.array([1, 0, -1, -1])
    }

    # Sentence 2:
    # 1|0~*~1|alias3~*~alias4|Q1~*~Q4|0:1~*~2:3|alias3 cat cat cat cat cat cat alias4|"slices":{"base": [0,1], "slice1": [0], "slice2": [1]}}
    aliases = ['alias3']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    truedata[1] = {
        'sent_idx': 1,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, -1, -1,-1]),
        'end_idx_in_sent': np.array([0, -1, -1,-1]),
        'alias_idx': np.array([alias_trie['alias3'], -1, -1, -1]),
        'word_indices': np.array([3, 7, 7, 7, 7]),
        'alias_list_pos': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, -1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1, -1, -1]),
        f'slice:slice1_pred': np.array([0, -1, -1, -1]),
        f'slice:slice1_ind': np.array([1, -1, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, -1, -1, -1])
    }

    aliases = ['alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[1] = [1, 2]
    truedata[2] = {
        'sent_idx': 1,
        'subsent_idx': 1,
        'start_idx_in_sent': np.array([4, -1, -1,-1]),
        'end_idx_in_sent': np.array([4, -1, -1,-1]),
        'alias_idx': np.array([alias_trie['alias4'], -1, -1, -1]),
        'word_indices': np.array([7, 7, 7, 7, 4]),
        'alias_list_pos': np.array([1, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, -1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, -1, -1, -1]),
        f'slice:slice2_pred': np.array([0, -1, -1, -1]),
        f'slice:slice2_ind': np.array([1, -1, -1, -1])
    }
        # sentence is divided into two subsentences
        # sentence is divided into two subsentence

    # Sentence 3:
    # 2|0|multi word alias2|Q4|1:4|cat multi word alias2|"slices":{"base": [0], "slice1": [], "slice2": []}
    aliases = ['multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[2] = [3]
    truedata[3] = {
        'sent_idx': 2,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([1,-1, -1, -1]),
        'end_idx_in_sent': np.array([3,-1, -1, -1]),
        'alias_idx': np.array([alias_trie['multi word alias2'], -1, -1, -1]),
        'word_indices': np.array([7, 5, 0, 2, -1]),
        'alias_list_pos': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([2, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([2, -1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1, -1, -1]),
        f'slice:slice1_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([0, -1, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, -1, -1, -1]),
    }

    # Sentence 4: Because this sentence is long, we'll focus on the first alias and recenter.
    #3|0|alias4|Q3|2:3|alias3 cat alias4 cat cat cat cat|"slices":{"base": [0], "slice1": [0], "slice2": []}
    aliases = ['alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[3] = [4]
    truedata[4] = {
        'sent_idx': 3,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([1, -1, -1, -1]),
        'end_idx_in_sent': np.array([1, -1, -1, -1]),
        'alias_idx': np.array([alias_trie['alias4'], -1, -1, -1]),
        'word_indices': np.array([7, 4, 7, 7, 7]),
        'alias_list_pos': np.array([0, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, -1, -1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, -1, -1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1, -1, -1]),
        f'slice:slice1_pred': np.array([1, -1, -1, -1]),
        f'slice:slice1_ind': np.array([1, -1, -1, -1]),
        f'slice:slice2_pred': np.array([-1, -1, -1, -1]),
        f'slice:slice2_ind': np.array([0, -1, -1, -1]),
    }

    return alias2wpids, truedata, sent_idx_to_idx

def get_data_train_many_alias():
    """ Return artificial dataset where the number of aliases exceeds the limit """

    sent_idx_to_idx = {}

    # Manually set alias map
    alias2wpids = {
        'alias1': [["Q1", 10], ["Q4", 6]],
        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
        'alias3': [["Q1", 30]],
        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
    }
    alias_trie = marisa_trie.Trie(alias2wpids.keys())
    max_aliases = 2
    # Manually set sentence data. Remember each line in the data file corresponds to an independent sample, with:
    # {sentence_index}|{alias to predict}~*~...|{alias}~*~...|{true QID}~*~...|{spans}~*~|{sentence}
    truedata = [None, None, None, None, None]

    # Sentence 1:
    # 0|0~*~1|alias1~*~multi word alias2|Q1~*~Q4|0:1~*~2:5|alias1 or multi word alias2|"slices":{"base": [0,1], "slice1": [1]}
    aliases = ['alias1', 'multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[0] = [0]
    truedata[0] = {
        'sent_idx': 0,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2]),
        'end_idx_in_sent': np.array([0, 4]),
        'alias_idx': np.array([alias_trie['alias1'], alias_trie['multi word alias2']]),
        'word_indices': np.array([1, 6, 5, 0, 2]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 2]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 2]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1]),
        f'slice:slice1_pred': np.array([-1, 2]),
        f'slice:slice1_ind': np.array([0, 0.9]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, 0])
    }

    # Sentence 2:
    # 1|0~*~1~*~2|alias3~*~alias4~*~alias3|Q1~*~Q4~*~Q1|0:1~*~2:3~*~3:4|alias3 cat alias4 alias3|"slices":{"base": [0,1,2], "slice1": [1,2]}
    aliases = ['alias3', 'alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    no_filt_arr_s1 = np.array(no_filt_arr, copy=True)
    # Make candidate 1 and 2 not be in filter (alias4 has three candidates)
    no_filt_arr_s1[1, 1] = 0
    no_filt_arr_s1[1, 2] = 0
    truedata[1] = {
        'sent_idx': 1,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2]),
        'end_idx_in_sent': np.array([0, 2]),
        'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias4']]),
        'word_indices': np.array([3, 7, 4, 3, -1]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 0]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 0]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1]),
        f'slice:slice1_pred': np.array([-1, 0]),
        f'slice:slice1_ind': np.array([0, 1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, 0])
    }

    aliases = ['alias3']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[1] = [1, 2]
    truedata[2] = {
        'sent_idx': 1,
        'subsent_idx': 1,
        'start_idx_in_sent': np.array([3, -1]),
        'end_idx_in_sent': np.array([3, -1]),
        'alias_idx': np.array([alias_trie['alias3'], -1]),
        'word_indices': np.array([3, 7, 4, 3, -1]),
        'alias_list_pos': np.array([2, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([0, -1]),
        f'slice:slice1_ind': np.array([1, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
        # across sentences
    }

    # Sentence 3:
    # 2|0|multi word alias2|Q4|1:4|cat multi word alias2|"slices":{"base": [0], "slice1": []}
    aliases = ['multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[2] = [3]
    truedata[3] = {
        'sent_idx': 2,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([1,-1]),
        'end_idx_in_sent': np.array([3,-1]),
        'alias_idx': np.array([alias_trie['multi word alias2'], -1]),
        'word_indices': np.array([7, 5, 0, 2, -1]),
        'alias_list_pos': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([2, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([2, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([0, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
    }

    # Sentence 4:
    #3|0|alias4|Q3|2:3|alias3 cat alias4|"slices":{"base": [0], "slice1": []}
    aliases = ['alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[3] = [4]
    truedata[4] = {
        'sent_idx': 3,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([2, -1]),
        'end_idx_in_sent': np.array([2, -1]),
        'alias_idx': np.array([alias_trie['alias4'], -1]),
        'word_indices': np.array([3, 7, 4, -1, -1]),
        'alias_list_pos': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([0, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
    }

    return alias2wpids, truedata, sent_idx_to_idx

def get_data_train_anchors_noaug():
    """ Return artificial dataset where the False anchors are removed"""

    sent_idx_to_idx = {}

    # Manually set alias map
    alias2wpids = {
        'alias1': [["Q1", 10], ["Q4", 6]],
        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
        'alias3': [["Q1", 30]],
        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
    }
    alias_trie = marisa_trie.Trie(alias2wpids.keys())
    max_aliases = 2
    # Manually set sentence data. Remember each line in the data file corresponds to an independent sample, with:
    # {sentence_index}|{alias to predict}~*~...|{alias}~*~...|{true QID}~*~...|{spans}~*~|{sentence}
    truedata = [None, None, None, None, None]

    # Sentence 1:
    # 0|0~*~1|alias1~*~multi word alias2|Q1~*~Q4|0:1~*~2:5|alias1 or multi word alias2|"slices":{"base": [0,1], "slice1": [1]}
    aliases = ['alias1', 'multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[0] = [0]
    truedata[0] = {
        'sent_idx': 0,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2]),
        'end_idx_in_sent': np.array([0, 4]),
        'alias_idx': np.array([alias_trie['alias1'], alias_trie['multi word alias2']]),
        'word_indices': np.array([1, 6, 5, 0, 2]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 2]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 2]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1]),
        f'slice:slice1_pred': np.array([-1, 2]),
        f'slice:slice1_ind': np.array([0, 0.9]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, 0])
    }

    # Sentence 2: we drop the second alias and then to sentence parsing after its removal
    # 1|0~*~1~*~2|alias3~*~alias4~*~alias3|true~*~false~*~true|Q1~*~Q4~*~Q1|0:1~*~2:3~*~3:4|alias3 cat alias4 alias3|"slices":{"base": [0,1,2], "slice1": [1,2]}, "importance": {"base": {"0": 1.0, "1": 0.0, "2": 0.0}, "slice1": {"0": 0.0, "1": 1.0, "2": 1.0}}, "filters": {"slice1": {"0": [0], "1": [0], "2": [0]}}
    aliases = ['alias3', 'alias3']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[1] = [1]
    truedata[1] = {
        'sent_idx': 1,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 3]),
        'end_idx_in_sent': np.array([0, 3]),
        'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias3']]),
        'word_indices': np.array([3, 7, 4, 3, -1]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 0]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 0]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1]),
        f'slice:slice1_pred': np.array([-1, 0]),
        f'slice:slice1_ind': np.array([0, 1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, 0])
    }

    # Sentence 3: droped because anchor is false

    # Sentence 4:
    #3|0|alias4|Q3|2:3|alias3 cat alias4|"slices":{"base": [0], "slice1": []}
    aliases = ['alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[3] = [2]
    truedata[2] = {
        'sent_idx': 3,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([2, -1]),
        'end_idx_in_sent': np.array([2, -1]),
        'alias_idx': np.array([alias_trie['alias4'], -1]),
        'word_indices': np.array([3, 7, 4, -1, -1]),
        'alias_list_pos': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([0, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
    }

    return alias2wpids, truedata, sent_idx_to_idx

def get_data_train_anchors_eval_yesaug():
    sent_idx_to_idx = {}

    # Manually set alias map
    alias2wpids = {
        'alias1': [["Q1", 10], ["Q4", 6]],
        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
        'alias3': [["Q1", 30]],
        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
    }
    alias_trie = marisa_trie.Trie(alias2wpids.keys())
    max_aliases = 2
    # Manually set sentence data. Remember each line in the data file corresponds to an independent sample, with:
    # {sentence_index}|{alias to predict}~*~...|{alias}~*~...|{true QID}~*~...|{spans}~*~|{sentence}
    truedata = [None, None, None, None]

    # Sentence 1:
    # 0|0~*~1|alias1~*~multi word alias2|Q1~*~Q4|0:1~*~2:5|true,true|alias1 or multi word alias2|"slices":{"base": [0,1], "slice1": [1]}
    aliases = ['alias1', 'multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[0] = [0]
    truedata[0] = {
        'sent_idx': 0,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2]),
        'end_idx_in_sent': np.array([0, 4]),
        'alias_idx': np.array([alias_trie['alias1'], alias_trie['multi word alias2']]),
        'word_indices': np.array([1, 6, 5, 0, 2]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 2]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 2]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1]),
        f'slice:slice1_pred': np.array([-1, 2]),
        f'slice:slice1_ind': np.array([0, 0.9]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, 0])
    }

    # dataset_is_eval is set in test so we want to NOT predict on indicators or prediction heads for
    # false anchors

    # Sentence 2:
    # 1|0~*~1~*~2|alias3~*~alias4~*~alias3|Q1~*~Q4~*~Q1|0:1~*~2:3~*~3:4|true,false,true|alias3 cat alias4 alias3|"slices":{"base": [0,1,2], "slice1": [1,2]}
    # Alias4 gets seen by the model, but not predicted on because it's a False anchor and this is eval (we know it gets seen because alias_idx != -1)
    aliases = ['alias3', 'alias4']
    # Limit to the first alias because anchor is false for the second
    no_filt_arr = get_no_filter(max_aliases, aliases[:1], alias2wpids, train_in_cands=True)
    truedata[1] = {
        'sent_idx': 1,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2]),
        'end_idx_in_sent': np.array([0, 2]),
        'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias4']]),
        'word_indices': np.array([3, 7, 4, 3, -1]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([0, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
    }

    aliases = ['alias3']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[1] = [1, 2]
    truedata[2] = {
        'sent_idx': 1,
        'subsent_idx': 1,
        'start_idx_in_sent': np.array([3, -1]),
        'end_idx_in_sent': np.array([3, -1]),
        'alias_idx': np.array([alias_trie['alias3'], -1]),
        'word_indices': np.array([3, 7, 4, 3, -1]),
        'alias_list_pos': np.array([2, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([0, -1]),
        f'slice:slice1_ind': np.array([1, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
        # across sentences
        # across sentence
    }

    # Sentence 3:
    # dropped because only false anchor

    # Sentence 4:
    #3|0|alias4|Q3|2:3|true|alias3 cat alias4|"slices":{"base": [0], "slice1": []}

    aliases = ['alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[3] = [3]
    truedata[3] = {
        'sent_idx': 3,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([2, -1]),
        'end_idx_in_sent': np.array([2, -1]),
        'alias_idx': np.array([alias_trie['alias4'], -1]),
        'word_indices': np.array([3, 7, 4, -1, -1]),
        'alias_list_pos': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([0, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
    }

    return alias2wpids, truedata, sent_idx_to_idx

def get_data_train_anchors_eval_yesaug_long():
    """ Return artificial dataset where the number of aliases exceeds the limit and such that one of the splits has all false aliases.
    In a eval setting, these subsentences need to be dropped, too."""

    # Since this data is to test eval, we want predictions on false aliases to be masked out (set to -1)

    sent_idx_to_idx = {}

    # Manually set alias map
    alias2wpids = {
        'alias1': [["Q1", 10], ["Q4", 6]],
        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
        'alias3': [["Q1", 30]],
        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
    }
    alias_trie = marisa_trie.Trie(alias2wpids.keys())
    max_aliases = 2
    # Manually set sentence data. Remember each line in the data file corresponds to an independent sample, with:
    # {sentence_index}|{alias to predict}~*~...|{alias}~*~...|{true QID}~*~...|{spans}~*~|{sentence}
    truedata = [None, None, None, None, None]

    # Sentence 1:
    # 0|0~*~1|alias1~*~multi word alias2|Q1~*~Q4|0:1~*~2:5|true,true|alias1 or multi word alias2|"slices":{"base": [0,1], "slice1": [1]}
    aliases = ['alias1', 'multi word alias2']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[0] = [0]
    truedata[0] = {
        'sent_idx': 0,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([0, 2]),
        'end_idx_in_sent': np.array([0, 4]),
        'alias_idx': np.array([alias_trie['alias1'], alias_trie['multi word alias2']]),
        'word_indices': np.array([1, 6, 5, 0, 2, -1, -1, -1, -1, -1]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 2]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 2]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1]),
        f'slice:slice1_pred': np.array([-1, 2]),
        f'slice:slice1_ind': np.array([0, 1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, 0])
    }

    # Sentence 2:
    # 1|0~*~1~*~2|alias3~*~alias4~*~alias3|Q1~*~Q4~*~Q1~*~Q4|0:1~*~2:3~*~3:4|false,false,true,true|alias3 cat alias4 alias3 alias4|"slices":{"base": [0,1,2,4], "slice1": [1,2]}
    # First part of sentence gets dropped because of two false anchors
    # truedata[1] = {
    #     'sent_idx': 1,
    #     'subsent_idx': 0,
    #     'start_idx_in_sent': np.array([0, 2]),
    #     'end_idx_in_sent': np.array([0, 2]),
    #     'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias4']]),
    #     'word_indices': np.array([3, 7, 4, 3, -1, -1, -1, -1, -1, -1]),
    #     'alias_list_pos': np.array([0, 1]),
    #     f'slice:{FINAL_LOSS}_pred' np.array([0, -1]),
    #     f'slice:{FINAL_LOSS}_ind' np.array([0, -1]),
    #     f'slice:{BASE_SLICE}_ind': np.array([1, 0]),
    #     f'slice:slice1_pred': np.array([-1, -1]),
    #     f'slice:slice1_ind': np.array([0, 0]),
    #     f'slice:slice2_pred': np.array([-1, -1]),
    #     f'slice:slice2_ind': np.array([0, 0])
    # }

    aliases = ['alias3', 'alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[1] = [1]
    truedata[1] = {
        'sent_idx': 1,
        'subsent_idx': 1,
        'start_idx_in_sent': np.array([3, 4]),
        'end_idx_in_sent': np.array([3, 4]),
        'alias_idx': np.array([alias_trie['alias3'], alias_trie['alias4']]),
        'word_indices': np.array([3, 7, 4, 3, 4, -1, -1, -1, -1, -1]),
        'alias_list_pos': np.array([2, 3]),
        f'slice:{FINAL_LOSS}_pred': np.array([0, 0]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([0, 0]),
        f'slice:{BASE_SLICE}_ind': np.array([1, 1]),
        f'slice:slice1_pred': np.array([0, -1]),
        f'slice:slice1_ind': np.array([1, 0]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, 0])
        # across sentences
        # across sentence
    }

    # Sentence 3:
    # {"aliases":["multi word alias2", "alias4", "alias4", "alias4"],"anchor":[false,true,false,true],"qids":["Q4","Q4","Q4","Q4"],"sent_idx_unq":"2",
    # "sentence":"cat multi word alias2 alias4 alias4 cat cat cat alias4","slices":{"base": [0,1,2,3], "slice1": []}}
    # First two aliases in this subsentence
    aliases = ['multi word alias2', 'alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    # First alias is False and gets -1 labels
    no_filt_arr[0,:] = -1
    sent_idx_to_idx[2] = [2, 3]
    truedata[2] = {
        'sent_idx': 2,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([1, 4]),
        'end_idx_in_sent': np.array([3, 4]),
        'alias_idx': np.array([alias_trie['multi word alias2'], alias_trie['alias4']]),
        'word_indices': np.array([7,  5,  0,  2,  4,  4,  7,  7,  7,  4]),
        'alias_list_pos': np.array([0, 1]),
        f'slice:{FINAL_LOSS}_pred': np.array([-1, 0]),
        f'slice:{FINAL_LOSS}_ind': np.array([-1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([-1, 0]),
        f'slice:{BASE_SLICE}_ind': np.array([-1, 1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([-1, 0]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([-1, 0])
    }

    aliases = ['alias4', 'alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases[1:], alias2wpids, train_in_cands=True)
    truedata[3] = {
        'sent_idx': 2,
        'subsent_idx': 1,
        'start_idx_in_sent': np.array([5, 6]),
        'end_idx_in_sent': np.array([5, 6]),
        'alias_idx': np.array([alias_trie['alias4'], alias_trie['alias4']]),
        'word_indices': np.array([7,  5,  0,  2,  4,  4,  7,  7,  7,  4]),
        'alias_list_pos': np.array([2, 3]),
        f'slice:{FINAL_LOSS}_pred': np.array([-1, 0]),
        f'slice:{FINAL_LOSS}_ind': np.array([-1, 1]),
        f'slice:{BASE_SLICE}_pred': np.array([-1, 0]),
        f'slice:{BASE_SLICE}_ind': np.array([-1, 1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([-1, 0]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([-1, 0])
    }

    aliases = ['alias4']
    no_filt_arr = get_no_filter(max_aliases, aliases, alias2wpids, train_in_cands=True)
    sent_idx_to_idx[3] = [4]
    truedata[4] = {
        'sent_idx': 3,
        'subsent_idx': 0,
        'start_idx_in_sent': np.array([2, -1]),
        'end_idx_in_sent': np.array([2, -1]),
        'alias_idx': np.array([alias_trie['alias4'], -1]),
        'word_indices': np.array([3, 7, 4, -1, -1, -1, -1, -1, -1, -1]),
        'alias_list_pos': np.array([0, -1]),
        f'slice:{FINAL_LOSS}_pred': np.array([1, -1]),
        f'slice:{FINAL_LOSS}_ind': np.array([1, -1]),
        f'slice:{BASE_SLICE}_pred': np.array([1, -1]),
        f'slice:{BASE_SLICE}_ind': np.array([1, -1]),
        f'slice:slice1_pred': np.array([-1, -1]),
        f'slice:slice1_ind': np.array([0, -1]),
        f'slice:slice2_pred': np.array([-1, -1]),
        f'slice:slice2_ind': np.array([0, -1])
    }

    return alias2wpids, truedata, sent_idx_to_idx

def get_slice_data_test_anchors_yesaug():
    ''' Get artificial test dataset (where you can onlp predict anchors with True) for slice indexes where use_weak_label is True'''
    # The max number of aliases to predict in the data will be 2 (for the number of true anchors)
    slice_dt = np.dtype([('sent_idx', int), ('alias_to_predict', int, 3), ('prob_labels', float, 3)])
    storage_type = np.dtype([(slice_name, slice_dt, 1) for slice_name in [FINAL_LOSS, BASE_SLICE, "slice1", "slice2"]])

    # Slices enforce the size of alias_to_predict to be at least 2 so that we can always assume matrix for memmap

    # {"aliases":["alias1","multi word alias2"],"anchor":[true,true],"qids":["Q1","Q4"],"sent_idx":"0","sent_idx_unq":"0",
    #       "sentence":"alias1 or multi word alias2","spans":["0:1","2:5"],
    #       "importance": {"base": {"0":0.2, "1":1.0}, "slice1": {"0": 0.8}},
    #       "slices": {"slice1": {"0":1.0, "1":0.9}, "slice2": {"0": 0.0, "1":0.0}}}
    # {"aliases":["alias3","alias4","alias3"],"anchor":[true,false,true],"qids":["Q1","Q4","Q1"],"sent_idx":"1","sent_idx_unq":"1",
    #       "sentence":"alias3 cat alias4 alias3","spans":["0:1","2:3","3:4"],
    #       "importance": {"base": {"0":1.0, "1":0.1, "2":1.0}, "slice2": {"1": 0.9}},
    #       "slices": {"slice1": {"0":0.0, "1":0.0, "2":0.0}, "slice2": {"0": 0.0, "1":1.0, "2":1.0}}}
    # {"aliases":["multi word alias2"],"anchor":[false],"qids":["Q4"],"sent_idx":"2","sent_idx_unq":"2",
    #       "sentence":"cat multi word alias2","spans":["1:4"],
    #       "importance": {"base": {"0":1.0}},
    #       "slices": {"slice1": {"0": 0.0}, "slice2": {"0": 0.0}}}
    # {"aliases":["alias4"],"anchor":[true],"qids":["Q3"],"sent_idx":"3","sent_idx_unq":"3",
    #       "sentence":"alias3 cat alias4","spans":["2:3"],
    #       "importance": {"base": {"0":0.0}, "slice1": {"0": 1.0}},
    #       "slices": {"slice1": {"0": 1.0}, "slice2": {"0": 0.0}}}

    # Each subarray is sent_idx followed but aliases to predict, each array is a row in the data

    # Note that for the prob labels, only slices recieve them. The -1 is for unk aliases
    ex1 = [np.rec.array([(0), (1, 1, 0),  (1.0, 1.0, -1.0)], dtype=slice_dt), np.rec.array([(0), (1, 1, 0), (1.0, 1.0, -1.0)], dtype=slice_dt),
            np.rec.array([(0), (1, 1, 0), (1.0, 0.9, -1.0)], dtype=slice_dt), np.rec.array([(0), (0, 0, 0), (0.0, 0.0, -1.0)], dtype=slice_dt)]
    ex2 = [np.rec.array([(1), (1, 0, 1), (1.0, -1.0, 1.0)], dtype=slice_dt), np.rec.array([(1), (1, 0, 1), (1.0, -1.0, 1.0)], dtype=slice_dt),
           np.rec.array([(1), (0, 0, 0), (0.0, -1.0, 0.0)], dtype=slice_dt), np.rec.array([(1), (0, 0, 1), (0.0, -1.0, 1.0)], dtype=slice_dt)]
    ex3 = [np.rec.array([(3), (1, 0, 0), (1.0, -1.0, -1.0)], dtype=slice_dt), np.rec.array([(3), (1, 0, 0), (1.0, -1.0, -1.0)], dtype=slice_dt),
           np.rec.array([(3), (1, 0, 0), (1.0, -1.0, -1.0)], dtype=slice_dt), np.rec.array([(3), (0, 0, 0), (0.0, -1.0, -1.0)], dtype=slice_dt)]
    # There is weird behavior giving a rec array a list of exs with this storage_type
    mat1 = np.rec.array(ex1, dtype=storage_type).reshape(1,1)
    mat2 = np.rec.array(ex2, dtype=storage_type).reshape(1,1)
    mat3 = np.rec.array(ex3, dtype=storage_type).reshape(1,1)
    res = np.vstack((mat1,mat2,mat3))
    return res


def get_slice_data_test_anchors_noaug():
    ''' Get artificial test dataset (where you can onlp predict anchors with True) for slice indexes where use_weak_label is False'''
    # The max number of aliases to predict in the data will be 2 (for the number of true anchors)
    slice_dt = np.dtype([('sent_idx', int), ('alias_to_predict', int, 2), ('prob_labels', float, 2)])
    storage_type = np.dtype([(slice_name, slice_dt, 1) for slice_name in [FINAL_LOSS, BASE_SLICE, "slice1", "slice2"]])

    # Slices enforce the size of alias_to_predict to be at least 2 so that we can always assume matrix for memmap

    # {"aliases":["alias1","multi word alias2"],"anchor":[true,true],"qids":["Q1","Q4"],"sent_idx":"0","sent_idx_unq":"0",
    #       "sentence":"alias1 or multi word alias2","spans":["0:1","2:5"],
    #       "importance": {"base": {"0":0.2, "1":1.0}, "slice1": {"0": 0.8}},
    #       "slices": {"slice1": {"0":1.0, "1":0.9}, "slice2": {"0": 0.0, "1":0.0}}}
    # {"aliases":["alias3","alias4","alias3"],"anchor":[true,false,true],"qids":["Q1","Q4","Q1"],"sent_idx":"1","sent_idx_unq":"1",
    #       "sentence":"alias3 cat alias4 alias3","spans":["0:1","2:3","3:4"],
    #       "importance": {"base": {"0":1.0, "1":0.1, "2":1.0}, "slice2": {"1": 0.9}},
    #       "slices": {"slice1": {"0":0.0, "1":0.0, "2":0.0}, "slice2": {"0": 0.0, "1":1.0, "2":1.0}}}
    # {"aliases":["multi word alias2"],"anchor":[false],"qids":["Q4"],"sent_idx":"2","sent_idx_unq":"2",
    #       "sentence":"cat multi word alias2","spans":["1:4"],
    #       "importance": {"base": {"0":1.0}},
    #       "slices": {"slice1": {"0": 0.0}, "slice2": {"0": 0.0}}}
    # {"aliases":["alias4"],"anchor":[true],"qids":["Q3"],"sent_idx":"3","sent_idx_unq":"3",
    #       "sentence":"alias3 cat alias4","spans":["2:3"],
    #       "importance": {"base": {"0":0.0}, "slice1": {"0": 1.0}},
    #       "slices": {"slice1": {"0": 1.0}, "slice2": {"0": 0.0}}}

    # Each subarray is sent_idx followed but aliases to predict, each array is a row in the data
    # There is one less alias here as all false aliases were removed and max length went down to two
    ex1 =  [np.rec.array([(0), (1, 1), (1.0, 1.0)], dtype=slice_dt), np.rec.array([(0), (1, 1), (1.0, 1.0)], dtype=slice_dt),
            np.rec.array([(0), (1, 1), (1.0, 0.9)], dtype=slice_dt), np.rec.array([(0), (0, 0), (0.0, 0.0)], dtype=slice_dt)]
    ex2 =  [np.rec.array([(1), (1, 1), (1.0, 1.0)], dtype=slice_dt), np.rec.array([(1), (1, 1), (1.0, 1.0)], dtype=slice_dt),
           np.rec.array([(1), (0, 0),  (0.0, 0.0)], dtype=slice_dt), np.rec.array([(1), (0, 1), (0.0, 1.0)], dtype=slice_dt)]
    ex3 =  [np.rec.array([(3), (1, 0), (1.0, -1.0)], dtype=slice_dt), np.rec.array([(3), (1, 0),(1.0, -1.0)], dtype=slice_dt),
           np.rec.array([(3), (1, 0),  (1.0, -1.0)], dtype=slice_dt), np.rec.array([(3), (0, 0),(0.0, -1.0)], dtype=slice_dt)]
    # There is weird behavior giving a rec array a list of exs with this storage_type
    mat1 = np.rec.array(ex1, dtype=storage_type).reshape(1,1)
    mat2 = np.rec.array(ex2, dtype=storage_type).reshape(1,1)
    mat3 = np.rec.array(ex3, dtype=storage_type).reshape(1,1)
    res = np.vstack((mat1,mat2,mat3))
    return res


class DataLoaderTest(unittest.TestCase):

    def setUp(self) -> None:
        # tests that the sampling is done correctly on indices
        # load data from directory
        self.args = parser_utils.get_full_config("test/run_args/test_data.json")
        train_utils.setup_train_heads_and_eval_slices(self.args)

    def test_load_data_candidate_missing_complains(self):
        # If train_in_candidates == True, we should throw an exception if the gold ID is *not* contained in
        # the alias list for an example
        # load data from directory
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)
        with self.assertRaises(Exception) as context:
            data = WikiDataset(
                args=self.args,
                use_weak_label=False,
                input_src=os.path.join(self.args.data_config.data_dir, "train_small_keep_cands.jsonl"),
                dataset_name=os.path.join(self.args.data_config.data_dir,
                                          data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_small_keep_cands")),
                is_writer=True,
                distributed=self.args.run_config.distributed,
                word_symbols=word_symbols,
                entity_symbols=entity_symbols
            )
        self.assertTrue(type(context.exception) == AssertionError)

    def test_load_data_candidate_missing_works(self):
        # If train_in_candidates == False, we should be able to load data even if the gold ID is *not* contained
        # in the alias list for an example

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_missing_gold()

        self.args.data_config.train_in_candidates = False
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_small_keep_cands.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="slice_train_small_keep_cands")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_small_keep_cands.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_small_keep_cands")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )
        self.assertEqual(len(data), 4)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(4):
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'])
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_load_data(self):
        # If train_in_candidates == True, we should be able to load data where all gold IDs are contained the candidate

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_in_candidates()

        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="slice_train_small")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_small")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )

        self.assertEqual(len(data), 4)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(4):
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'])
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_load_data_eval(self):
        # With Normal mode and eval, we should see the slice heads in data

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_in_candidates()

        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="slice_train_small")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=True
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_small")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=True
        )

        self.assertEqual(len(data), 4)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(4):
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'])
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_load_data_normal_train(self):
        # With Normal mode and NOT eval, we should not see a base head

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_in_candidates()
        # Slice method changes what slices we see so we need to regen the args
        args = parser_utils.get_full_config("test/run_args/test_data.json")
        args.train_config.slice_method = "Normal"
        train_utils.setup_train_heads_and_eval_slices(args)
        word_symbols = data_utils.load_wordsymbols(args.data_config)
        entity_symbols = EntitySymbols(os.path.join(args.data_config.entity_dir, args.data_config.entity_map_dir),
            alias_cand_map_file=args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=args,
            use_weak_label=False,
            input_src=os.path.join(args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=args.data_config, use_weak_label=False, split_name="slice_train_small")),
            is_writer=True,
            distributed=args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=args,
            use_weak_label=False,
            input_src=os.path.join(args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=args.data_config, use_weak_label=False, split_name="train_small")),
            is_writer=True,
            distributed=args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )

        self.assertEqual(len(data), 4)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        keys_to_ignore = [f'slice:{BASE_SLICE}_pred', f'slice:{BASE_SLICE}_ind']
        for i in range(4):
            for key in keys_to_ignore:
                assert key not in data[i]
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'])
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                if key not in keys_to_ignore:
                    np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_load_data_normal_train_no_filt(self):
        # With Normal mode and NOT eval, we should not see a base head

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_in_candidates()
        # Slice method changes what slices we see so we need to regen the args
        args = parser_utils.get_full_config("test/run_args/test_data.json")
        args.train_config.slice_method = "Normal"
        train_utils.setup_train_heads_and_eval_slices(args)
        word_symbols = data_utils.load_wordsymbols(args.data_config)
        entity_symbols = EntitySymbols(os.path.join(args.data_config.entity_dir, args.data_config.entity_map_dir),
            alias_cand_map_file=args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=args,
            use_weak_label=False,
            input_src=os.path.join(args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=args.data_config, use_weak_label=False, split_name="slice_train_small")),
            is_writer=True,
            distributed=args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=args,
            use_weak_label=False,
            input_src=os.path.join(args.data_config.data_dir, "train_small.jsonl"),
            dataset_name=os.path.join(args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=args.data_config, use_weak_label=False, split_name="train_small")),
            is_writer=True,
            distributed=args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )

        self.assertEqual(len(data), 4)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        keys_to_ignore = [f'slice:{BASE_SLICE}_pred', f'slice:{BASE_SLICE}_ind']
        for i in range(4):
            for key in keys_to_ignore:
                assert key not in data[i]
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'])
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                if key not in keys_to_ignore:
                    np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_long_sentence(self):
        # Test loading data when the sentence is very very long

        # If train_in_candidates == True, we should be able to load data where all gold IDs are contained the candidate

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_long_sentence()
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_long.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="slice_train_long")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_long.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_long")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )

        self.assertEqual(len(data), 5)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(5):
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'])
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_max_aliases(self):
        # Test loading data when the number of aliases in the sentence exceeds our limit

        # If train_in_candidates == True, we should be able to load data where all gold IDs are contained the candidate

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_many_alias()

        # Set limit on max_aliases
        self.args.data_config.max_aliases = 2
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_many_aliases.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="slice_train_many_aliases")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_many_aliases.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_many_aliases")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )

        self.assertEqual(len(data), 5)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(5):
            print(i)
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'], f"index: {i}")
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")


    def test_anchors(self):
        # This will first test that when use_weak_label is True, the loading keeps False anchors.
        # We use the same data as many_aliases except we flip some aliases to have False anchors.
        # Therefore for the first test, the loaded data should be the same as the many_aliases test
        # For the second test, we set use_weak_label to False and make sure aliases are dropped.

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_many_alias()

        # Set limit on max_aliases
        self.args.data_config.max_aliases = 2
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)

        slices = WikiSlices(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="slice_train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )
        # Test with use_weak_label of True
        self.assertEqual(len(data), 5)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(5):
            print(i)
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'], f"index: {i}")
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

        # Testing with use_weak_label is False
        _, truedata, correct_sent_idx_to_idx = get_data_train_anchors_noaug()
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="slice_train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=False
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=False
        )

        self.assertEqual(len(data), 3)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(3):
            print(i)
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'], f"index: {i}")
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_anchors_with_eval(self):
        # This will first test that when use_weak_label is True, the loading keeps False anchors.
        # We use the same data as many_aliases except we flip some aliases to have False anchors.
        # Therefore for the first test, the loaded data should be the same as the many_aliases test
        # For the second test, we set use_weak_label to False and make sure aliases are dropped.

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_anchors_eval_yesaug()

        # Set limit on max_aliases
        self.args.data_config.max_aliases = 2
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)

        slices = WikiSlices(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="slice_train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=True
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=True
        )
        # Test with use_weak_label of True
        self.assertEqual(len(data), 4)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(4):
            print(i)
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'], f"index: {i}")
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

        # Testing with use_weak_label is False
        # The noaug is the same with or without dataset_is_eval
        _, truedata, correct_sent_idx_to_idx = get_data_train_anchors_noaug()
        slices = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="slice_train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=True
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=True
        )

        self.assertEqual(len(data), 3)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(3):
            print(i)
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'], f"index: {i}")
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_anchors_with_eval_long(self):
        # This will first test that when use_weak_label is True, the loading keeps False anchors.
        # We use the same data as many_aliases except we flip some aliases to have False anchors.
        # Therefore for the first test, the loaded data should be the same as the many_aliases test
        # For the second test, we set use_weak_label to False and make sure aliases are dropped.

        alias2wpids, truedata, correct_sent_idx_to_idx = get_data_train_anchors_eval_yesaug_long()

        # Set limit on max_aliases
        self.args.data_config.max_aliases = 2
        self.args.data_config.max_word_token_len = 10
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)

        slices = WikiSlices(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor_long.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="slice_train_anchor_long")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=True
        )
        data = WikiDataset(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor_long.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="train_anchor_long")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            slice_dataset=slices,
            dataset_is_eval=True
        )
        # Test with use_weak_label of True
        self.assertEqual(len(data), 5)
        self.assertEqual(data.sent_idx_to_idx, correct_sent_idx_to_idx)
        for i in range(3):
            print(i)
            self.assertEqual(data[i]['sent_idx'], truedata[i]['sent_idx'], f"index: {i}")
            self.assertEqual(data[i]['subsent_idx'], truedata[i]['subsent_idx'])
            for key in ARR_KEYS_TO_COMPARE:
                np.testing.assert_array_equal(data[i][key], truedata[i][key], err_msg=f"At index {i} key {key}")

    def test_eval_anchors_slice(self):
        # Same test as above except dataset_is_eval is set to True. This means aliases to predict must be altered to be only of True anchors.

        truedata = get_slice_data_test_anchors_yesaug()

        # Set limit on max_aliases
        self.args.data_config.max_aliases = 2
        # Test with use_weak_label of True
        data = WikiSlices(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor_slice.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="train_anchor_slice")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=True
        )

        self.assertEqual(len(data), 3)
        for i in range(len(data)):
            np.testing.assert_array_equal(truedata[i], data.data[i])

        # Testing with use_weak_label is False
        truedata = get_slice_data_test_anchors_noaug()
        data = WikiSlices(
            args=self.args,
            use_weak_label=False,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor_slice.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=False, split_name="train_anchor_slice")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=True
        )

        self.assertEqual(len(data), 3)
        for i in range(len(data)):
            np.testing.assert_array_equal(truedata[i], data.data[i])

    def test_slices_sampling(self):
        # set seed as there is randomness here
        self.args.train_config.seed = 0
        # Set limit on max_aliases; 3 ensures no sentence splitting
        self.args.data_config.max_aliases = 3
        word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        entity_symbols = EntitySymbols(os.path.join(self.args.data_config.entity_dir, self.args.data_config.entity_map_dir),
            alias_cand_map_file=self.args.data_config.alias_cand_map)

        # Test with use_weak_label of True
        data = WikiDataset(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor_slice.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="test_anchor")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            word_symbols=word_symbols,
            entity_symbols=entity_symbols,
            dataset_is_eval=False
        )

        # Test with use_weak_label of True
        data_slices = WikiSlices(
            args=self.args,
            use_weak_label=True,
            input_src=os.path.join(self.args.data_config.data_dir, "train_anchor_slice.jsonl"),
            dataset_name=os.path.join(self.args.data_config.data_dir,
                                      data_utils.generate_save_data_name(data_args=self.args.data_config, use_weak_label=True, split_name="train_anchor_slice")),
            is_writer=True,
            distributed=self.args.run_config.distributed,
            dataset_is_eval=True
        )

        indices = data_utils.get_eval_slice_subset_indices(self.args, data_slices, data)
        # we want one example per slice
        # base_slice gets mapped to data idx 0, 1, 3
        # slice1 gets mapped to data idx 0, 3
        # slice2 gets mapped to data idx 1
        # We need a covering set of indices for sampling
        # This means we will always have 1, and will either have 0 and 3, just 0, or just 3
        assert len(indices) == 3 or len(indices) == 2
        assert 1 in indices
        assert 0 in indices or 3 in indices
        # assert there are only 0, 1, or 3 in indices
        assert len(set(indices).difference(set([0,1,3]))) == 0



if __name__ == "__main__":
    unittest.main()
