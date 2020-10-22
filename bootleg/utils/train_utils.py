
import os
from time import strftime
import torch

from bootleg.symbols.constants import *
from bootleg.utils import utils

def get_gpu_mem():
    return f"{torch.cuda.memory_allocated()/1024**3}, {torch.cuda.memory_cached()/1024**3}"

def is_a_timestamp(x):
    try:
        tuple([int(x.split("_")[-2]), int(x.split("_")[-1])])
    except:
        return False
    return True

def setup_run_folders(args, mode):
    if args.run_config.timestamp == "":
        # create a timestamp for directory for saving results
        start_date = strftime("%Y%m%d")
        start_time = strftime("%H%M%S")
        args.run_config.timestamp = "{:s}_{:s}".format(start_date, start_time)
        utils.ensure_dir(get_save_folder(args.run_config))
    return

def get_save_folder(run_args):
    save_folder = os.path.join(run_args.save_dir, run_args.timestamp)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder

def get_eval_folder(args, file):
    return os.path.join(get_save_folder(args.run_config), os.path.basename(file).split('.jsonl')[0], "eval",
        os.path.basename(args.run_config.init_checkpoint).replace(".pt", ""))

def is_slicing_model(args):
    return args.train_config.slice_method != NORMAL_SLICE_METHOD

def model_has_base_head_loss(args):
    return args.train_config.slice_method in ["SBL"]

def get_head_key_to_idx(args):
    head_key_to_idx = {}
    idx = 0
    # We may still have train_heads for a Normal model as they serve as filter heads
    # if is_slicing_model(args):
    for head_name in args.train_config.train_heads:
        head_key_to_idx[get_slice_head_pred_name(head_name)] = idx
        idx += 1
    for stage_idx in range(args.model_config.num_model_stages-1):
        head_key_to_idx[get_stage_head_name(stage_idx)] = idx
        idx += 1
    head_key_to_idx[FINAL_LOSS] = idx
    idx += 1
    return head_key_to_idx

def get_stage_head_name(stage_idx):
    return f'{FINAL_LOSS}_stage_{stage_idx}'

def get_prestage():
    return f'{FINAL_LOSS}_prestage'

def get_type_head_name():
    return f"{FINAL_LOSS}_type"

def get_slice_head_pred_name(slice_name):
    return f"slice:{slice_name}_pred"

def get_slice_head_ind_name(slice_name):
    return f"slice:{slice_name}_ind"

def get_slice_head_eval_name(slice_name):
    return f"{slice_name}_head"

def get_inv_head_name_eval(head_name):
    return head_name.replace("_head", "")

def get_inv_slice_head_pred_name(slice_name):
    name = slice_name.split("slice:")[1].split("_pred")[0]
    return name

def get_inv_slice_head_ind_name(slice_name):
    name = slice_name.split("slice:")[1].split("_ind")[0]
    return name

def is_name_slice_head_pred(slice_name):
    return (slice_name.startswith("slice:") and (slice_name.endswith("_pred")))

# We have eval slices, train heads, and train slices. The important parts is that when the data is generated, we need a slice for each output
# by the model (which includes FINAL_LOSS) for train (in order to score) and eval (in order to evaluate head performance). This derives the following taxonomy.
# If SBL (has base head loss):
#     train_heads must include BASE_SLICE and FINAL_LOSS
#     eval_slices must include BASE_SLICE and FINAL_LOSS
# If Normal:
#     train_heads must include FINAL_LOSS and can include those from the args (these train heads will be used to evaluate filter/ranking capabilities)
#     eval_slices must include FINAL_LOSS and can include BASE_SLICE
def get_data_slices(args, dataset_is_eval):
    if dataset_is_eval:
        slice_names = args.run_config.eval_slices[:]
        if BASE_SLICE not in slice_names:
            slice_names.insert(0, BASE_SLICE)
    else:
        slice_names = args.train_config.train_heads[:]
        if model_has_base_head_loss(args):
            if BASE_SLICE not in slice_names:
                slice_names.insert(0, BASE_SLICE)
    # FINAL LOSS is in ALL MODELS for ALL SLICES
    if FINAL_LOSS not in slice_names:
        slice_names.insert(0, FINAL_LOSS)
    return slice_names


# We have eval slices, train slices, and train heads. The train heads are used in the slice heads module.
# The train slices are used in data generation to make sure we have slice data for each output of the module.
# Mainly, this incorporates the fact that FINAL_LOSS is included in the model output and isn't a train head.
# For eval slices, we need to include every slice we want to measure, including FINAL_LOSS and BASE_SLICE and train heads.
# This derives the following taxonomy.
# If SBL (has base head loss):
#     train_heads must include BASE_SLICE and not include FINAL_LOSS
#     eval_slices must include BASE_SLICE and FINAL_LOSS
# If Normal:
#     train_heads can include those from the args (these train heads will be used to evaluate filter/ranking capabilities)
#     eval_slices must include FINAL_LOSS and can include BASE_SLICE
def setup_train_heads_and_eval_slices(args):
    if BASE_SLICE in args.train_config.train_heads:
        print(f"WARNING: removing {BASE_SLICE} from train_heads")
        args.train_config.train_heads.remove(BASE_SLICE)
    if FINAL_LOSS in args.train_config.train_heads:
        print(f"WARNING: removing {FINAL_LOSS} from train_heads")
        args.train_config.train_heads.remove(FINAL_LOSS)
    assert BASE_SLICE not in args.train_config.train_heads, f"Do not add {BASE_SLICE} to train_heads, we do that for you"
    for train_head in args.train_config.train_heads:
        assert FINAL_LOSS not in train_head, f"{FINAL_LOSS} cannot be part of a train_head name. You have {train_head}. This is due to our assumption that any head out of the backbone has final_loss key (in scorer and eval_utils)"
    if model_has_base_head_loss(args):
        if BASE_SLICE not in args.train_config.train_heads:
            args.train_config.train_heads.insert(0, BASE_SLICE)
    if BASE_SLICE not in args.run_config.eval_slices:
        args.run_config.eval_slices.insert(0, BASE_SLICE)
    if FINAL_LOSS not in args.run_config.eval_slices:
        args.run_config.eval_slices.insert(0, FINAL_LOSS)
    # All train heads must be eval slices
    for slice_name in args.train_config.train_heads:
        if slice_name not in args.run_config.eval_slices:
            args.run_config.eval_slices.append(slice_name)
    return

def get_file_suffix(args):
    to_add = ""
    if args.run_config.init_checkpoint != "":
        to_add += f'_{os.path.basename(args.run_config.init_checkpoint).split(".pt")[0]}'
    if args.run_config.model_suffix != "":
        to_add += f'_{args.run_config.model_suffix}'
    return to_add

# Used in entity embedding to turn the alias prior counts to entity counts to determine the tails
def generate_qid_counts(alias_qid_counts):
    qid_count_dict = {}
    for al in alias_qid_counts:
        for q in alias_qid_counts[al]:
            if q not in qid_count_dict:
                qid_count_dict[q] = 0
            qid_count_dict[q] += alias_qid_counts[al][q]
    return qid_count_dict