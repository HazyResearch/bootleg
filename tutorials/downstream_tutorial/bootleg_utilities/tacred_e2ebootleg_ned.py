# Given a file (.jsonl) produced from the bootleg mention extractor, produce the {bootleg_labels.jsonl, bootleg_embs.npy, and 
# static_entity_embs.npy} files. 

import sys
sys.path.append('../../')
import numpy as np 
import pandas as pd
import ujson
from utils import load_mentions
import logging
from importlib import reload
import argparse
reload(logging)
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
import os
import torch 
use_cpu = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../dataset/tacred')
parser.add_argument('--bootleg_model_dir', type=str, default='../../../models/bootleg_wiki', help='directory of checkpointed bootleg models')
parser.add_argument('--bootleg_model_name', type=str, default='bootleg_model', help='checkpointed model to use for inference')
parser.add_argument('--bootleg_config_name', type=str, default='bootleg_config.json', help='name of model config file')
parser.add_argument('--cand_map_name', type=str, default='alias2qids.json', help='ill in name of candidatae map')
parser.add_argument('--root_dir', type=str, default='../../../', help='fill in path to model resources')
parser.add_argument('--infile_name', type=str, default='all_tacred_bootinput.jsonl', help='jsonl format downstream task data')
parser.add_argument('--outfile_name', type=str, default='all_tacred_bootoutput.jsonl', help='output file containing infile contents with candidates extracted')
args = parser.parse_args()

# MENTION EXTRACTION
root_dir = args.root_dir
infile = f'{args.data_dir}/{args.infile_name}'
outfile = f'{args.data_dir}/{args.outfile_name}'
cand_map = f'{root_dir}/data/wiki_entity_data/entity_mappings/alias2qids_wiki.json'

from bootleg.extract_mentions import extract_mentions
extract_mentions(in_filepath=infile, out_filepath=outfile, cand_map_file=cand_map, logger=logger)  


# BOOTLEG INFERENCE
from bootleg import run
from bootleg.utils.parser_utils import get_full_config
config_path = f'{root_dir}/models/bootleg_wiki/bootleg_config.json'
config_args = get_full_config(config_path)

# set the model checkpoint path 
config_args.run_config.init_checkpoint = f'{root_dir}/models/bootleg_wiki/bootleg_model.pt'

# set the path for the entity db and candidate map
config_args.data_config.entity_dir = f'{root_dir}/data/wiki_entity_data'
config_args.data_config.alias_cand_map = 'alias2qids_wiki.json'

# set the data path and file we want to run inference over
config_args.data_config.data_dir = args.data_dir
config_args.data_config.test_dataset.file = args.outfile_name

# set the embedding paths 
config_args.data_config.emb_dir =  f'{root_dir}/data/emb_data'
config_args.data_config.word_embedding.cache_dir =  f'{root_dir}/pretrained_bert_models'

# set the save directory 
config_args.run_config.save_dir = f'{args.data_dir}/results'

# set whether to run inference on the CPU
config_args.run_config.cpu = use_cpu
 
# run inference (take note of where the bootleg_label_file and bootleg_emb_file save to!)
bootleg_label_file, bootleg_emb_file = run.model_eval(args=config_args, mode="dump_embs", logger=logger, is_writer=True)

