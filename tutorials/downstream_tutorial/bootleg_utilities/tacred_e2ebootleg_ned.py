# Given a file (.jsonl) produced from the bootleg_emmental mention extractor, produce the {bootleg_labels.jsonl, bootleg_embs.npy, and
# static_entity_embs.npy} files.

import sys

from bootleg.run import run_model

sys.path.append("../../")
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../dataset/tacred")
parser.add_argument(
    "--bootleg_model_dir",
    type=str,
    default="../../../models/bootleg_wiki",
    help="directory of checkpointed bootleg_emmental models",
)
parser.add_argument(
    "--bootleg_model_name",
    type=str,
    default="bootleg_model",
    help="checkpointed model to use for inference",
)
parser.add_argument(
    "--bootleg_config_name",
    type=str,
    default="bootleg_config.json",
    help="name of model config file",
)
parser.add_argument(
    "--cand_map_name",
    type=str,
    default="alias2qids.json",
    help="ill in name of candidatae map",
)
parser.add_argument(
    "--root_dir", type=str, default="../../../", help="fill in path to model resources"
)
parser.add_argument(
    "--infile_name",
    type=str,
    default="all_tacred_bootinput.jsonl",
    help="jsonl format downstream task data",
)
parser.add_argument(
    "--outfile_name",
    type=str,
    default="all_tacred_bootoutput.jsonl",
    help="output file containing infile contents with candidates extracted",
)
args = parser.parse_args()

# MENTION EXTRACTION
root_dir = Path(args.root_dir)
infile = Path(args.data_dir) / args.infile_name
outfile = Path(args.data_dir) / args.outfile_name
cand_map = Path(root_dir) / "data/wiki_entity_data/entity_mappings/alias2qids_wiki.json"

from bootleg.end2end.extract_mentions import extract_mentions

extract_mentions(in_filepath=infile, out_filepath=outfile, cand_map_file=cand_map)

# BOOTLEG INFERENCE

from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args
from bootleg.utils.utils import load_yaml_file

config_in_path = root_dir / "models/bootleg_wiki/bootleg_wiki_config.yaml"

config_args = load_yaml_file(config_in_path)

# decrease number of data threads as this is a small file
config_args["run_config"]["dataset_threads"] = 2
config_args["run_config"]["log_level"] = "info"
# set the model checkpoint path
config_args["emmental"]["model_path"] = str(
    root_dir / "models/bootleg_wiki/bootleg_wiki.pth"
)
config_args["emmental"]["log_path"] = str(Path(args.data_dir) / "bootleg_results")
# set the path for the entity db and candidate map
config_args["data_config"]["entity_dir"] = str(root_dir / "data/wiki_entity_data")
config_args["data_config"]["alias_cand_map"] = str(cand_map)

config_args["data_config"]["data_dir"] = args.data_dir
config_args["data_config"]["test_dataset"]["file"] = args.outfile_name

# set the embedding paths
config_args["data_config"]["emb_dir"] = str(root_dir / "data/emb_data")
config_args["data_config"]["word_embedding"]["cache_dir"] = str(
    root_dir / "data/emb_data/pretrained_bert_models"
)


config_args = parse_boot_and_emm_args(
    config_args
)  # or you can pass in the config_out_path

bootleg_label_file, bootleg_emb_file = run_model(mode="dump_embs", config=config_args)
