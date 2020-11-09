import json
import random
import numpy as np

def convert_data(file_in, file_out):
    """
    Load data from json files, preprocess and prepare batches.
    """
    with open(file_in) as infile:
        data = json.load(infile)
        print(len(data))

    with open(file_out, 'w') as outfile:
        for d in data:
            tokens = d['token']
            example = ' '.join(tokens)
            entry = {"sentence": example}
            json.dump(entry, outfile)
            outfile.write('\n')

def normalize_glove(tokens):
    mapping = {'-LRB-': '(',
                '-RRB-': ')',
                '-LSB-': '[',
                '-RSB-': ']',
                '-LCB-': '{',
                '-RCB-': '}'}
    for i in range(len(tokens)):
        if tokens[i] in mapping:
            #print(tokens)
            tokens[i] = mapping[tokens[i]]
    return tokens


def extract_subj_obj(tokens, d):
    masked_tokens = ['-']*len(tokens)
    ss, se = d['subj_start'], d['subj_end']
    os, oe = d['obj_start'], d['obj_end']
    masked_tokens[ss:se+1] = tokens[ss:se+1]
    masked_tokens[os:oe+1] = tokens[os:oe+1]
    return masked_tokens

def create_one_file(train, dev, test, all_out, subjobj=False):
    with open(train) as infile:
        data_train = json.load(infile)
        print(len(data_train))

    with open(dev) as infile:
        data_dev = json.load(infile)
        print(len(data_dev))

    with open(test) as infile:
        data_test = json.load(infile)
        print(len(data_test))

    with open(all_out, 'w') as outfile:
        for d in data_train:
            tokens = d['token']
            tokens = normalize_glove(tokens)
            if subjobj:
                tokens = extract_subj_obj(tokens, d)
            example = ' '.join(tokens)
            entry = {"sentence": example, "id": d['id']}
            json.dump(entry, outfile)
            outfile.write('\n')
        
        for d in data_dev:
            tokens = d['token']
            tokens = normalize_glove(tokens)
            if subjobj:
                tokens = extract_subj_obj(tokens, d)
            example = ' '.join(tokens)
            entry = {"sentence": example, "id": d['id']}
            json.dump(entry, outfile)
            outfile.write('\n')

        for d in data_test:
            tokens = d['token']
            tokens = normalize_glove(tokens)
            if subjobj:
                tokens = extract_subj_obj(tokens, d)
            example = ' '.join(tokens)
            entry = {"sentence": example, "id": d['id']}
            json.dump(entry, outfile)
            outfile.write('\n')

source_path = '../dataset/tacred/'
inname_train = "{}train.json".format(source_path)
inname_dev = "{}dev.json".format(source_path)
inname_test = "{}test.json".format(source_path)
outname_all = '{}all_tacred_bootinput.jsonl'.format(source_path)
create_one_file(inname_train, inname_dev, inname_test, outname_all, subjobj = False)

