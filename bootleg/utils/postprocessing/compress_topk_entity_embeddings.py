"""Takes the learned embeddings of the model, and compresses them to only
contain the top K% determined by popularity. If no model is give, we only
output the qid2topk_eid json which can be used for training.

This can be used for out TopKEntityEmb::

    ent_embeddings:
       - key: learned
         load_class: TopKEntityEmb
         freeze: false
         cpu: false
         args:
           learned_embedding_size: 256
           perc_emb_drop: 0.95 # This MUST match the percent given to this method
           qid2topk_eid: <path to output json file>

If using this to update an already trained model, you **must** have the learned embedding key be ``learned`` for this method to work. If not, you can
 simply change the value of ``learned`` to whatever the right key is in the ``ENTITY_EMB_KEYS``, ``ENTITY_EID_KEYS``, ``ENTITY_REG_KEYS`` values set
 in this file below.

 After this runs, you need to change the LearnedEntityEmb config by

#. Changing the ``load_class`` to be ``TopKEntityEmb``
#. Add ``perc_emb_drop`` to the custom ``args``
#. Remove the ``regularize_mapping`` param in ``args`` if there is one

**Note** the key value must stay the same.
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import ujson as json
from tqdm import tqdm

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils

ENTITY_EMB_KEYS = ["module_pool", "learned", "learned_entity_embedding.weight"]
ENTITY_EID_KEYS = ["module_pool", "learned", "eid2topkeid"]
ENTITY_REG_KEYS = ["module_pool", "learned", "eid2reg"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qid2count", type=str, required=True, help="Path for qid2count file"
    )
    parser.add_argument(
        "--perc_emb_drop",
        type=float,
        required=True,
        help="Percentage of embeddings to remove by popularity",
    )
    parser.add_argument(
        "--alias_cand_map_file",
        type=str,
        default="alias2qids.json",
        help="Name of file to read the alias2qids in the entity_dir",
    )
    parser.add_argument(
        "--save_qid2topk_file",
        type=str,
        required=True,
        help="Where to save topk qid2eid mapping in entity_db/entity_mappings",
    )
    parser.add_argument(
        "--save_qid2topk_reg_file",
        type=str,
        required=True,
        help="Where to save topk qid2reg mapping",
    )
    parser.add_argument(
        "--entity_dir", type=str, required=True, help="Directory to read entity_db"
    )
    parser.add_argument(
        "--entity_map_dir",
        type=str,
        default="entity_mappings",
        help="Directory to read entity_mappings inside entity_dir",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Which model to load. If empty, we just generate the entity mappings",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        help="Where to save the model. If empty, we just generate the entity mappings",
    )
    args = parser.parse_args()
    return args


def get_nested_item(d, list_of_keys):
    """Returns the item from a nested dictionary. Each key in list_of_keys is
    accessed in order.

    Args:
        d: dictionary
        list_of_keys: list of keys

    Returns: item in d[list_of_keys[0]][list_of_keys[1]]...
    """
    dct = d
    for i, k in enumerate(list_of_keys):
        assert (
            k in dct
        ), f"Key {k} is not in dictionary after seeing {i+1} keys from {list_of_keys}"
        dct = dct[k]
    return dct


def set_nested_item(d, list_of_keys, value):
    """Sets the item from a nested dictionary. Each key in list_of_keys is
    accessed in order and last item is the one set. If value is None, item is
    removed.

    Args:
        d: dictionary
        list_of_keys: list of keys
        value: new value

    Returns: d such that d[list_of_keys[0]][list_of_keys[1]] == value
    """
    dct = d
    for i, k in enumerate(list_of_keys[:-1]):
        assert (
            k in dct
        ), f"Key {k} is not in dictionary after seeing {i+1} keys from {list_of_keys}"
        dct = dct[k]
    final_k = list_of_keys[-1]
    if value is None:
        del dct[final_k]
    else:
        dct[final_k] = value
    return d


def load_statedict(model_path):
    """Loads model state dict.

    Args:
        model_path: model path

    Returns: state dict
    """
    print(f"Loading model from {model_path}.")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("Loaded model.")
    # Remove distributed naming if model trained in distributed mode
    model_state_dict = OrderedDict()
    for k, v in state_dict["model"].items():
        if k.startswith("module."):
            name = k[len("module.") :]
            model_state_dict[name] = v
        else:
            model_state_dict[k] = v
    return state_dict, model_state_dict


def filter_qids(perc_emb_drop, entity_db, qid2count):
    """Creates a new QID -> EID mapping to have the top (1-perc_emb_drop)
    percent most popular entities. Also creates a regularization mapping if
    needed that adds 95% regularization on the single toe embedding (this toe
    embedding replaces the perc_emb_drop entities). This does not have to be
    used.

    Args:
        perc_emb_drop: percent to drop
        entity_db: entity dump
        qid2count: qid to count in training data dictionary

    Returns: qid2topk_eid, qid2topk_reg, old2new_eid, new_toes_eid, num_topk_entities
    """
    old2new_eid = {}
    print(f"Removing the least popular {perc_emb_drop} embeddings")
    qid2count_with_tails = {}
    for qid in entity_db.get_all_qids():
        qid2count_with_tails[qid] = qid2count.get(qid, 0)

    # Sort smallest to largest
    qid2count_list = sorted(
        list(qid2count_with_tails.items()), key=lambda x: x[1], reverse=False
    )
    assert (
        len(qid2count_list) == len(entity_db.get_all_qids()) == entity_db.num_entities
    )
    to_drop = int(perc_emb_drop * len(qid2count_list))
    print(f"Dropping {to_drop} qids out of {len(qid2count_list)}")
    # Find a tail embedding row
    qid_vec = [0] * entity_db.num_entities_with_pad_and_nocand
    # Set the padded eid to be -1 so we don't count it as a "tail" qid
    qid_vec[-1] = -1
    for qid in entity_db.get_all_qids():
        if qid in qid2count:
            eid = entity_db.get_eid(qid)
            weight = qid2count[qid]
            qid_vec[eid] = weight
    # Do not take the pad or nocand values so filter [1:-1] then add 1 to account for [1:-1] indexing being off by 1
    old_toes_eid = np.argmin(np.array(qid_vec[1:-1])) + 1
    # We start indexing eids by 1 (0 is reserved). So the +1 is correct for an index for an eid.
    num_topk_entities = len(qid2count_list) - to_drop + 1
    new_toes_eid = num_topk_entities
    new_toes_reg = 0.95
    non_toes_reg = 0.05
    new_eid_idx = 1
    # Build qid2neweid mapping
    qid2topk_eid = {}
    qid2topk_reg = {}
    for i, qid_pair in enumerate(qid2count_list):
        qid = qid_pair[0]
        eid = entity_db.get_eid(qid)
        if i < to_drop:
            qid2topk_eid[qid] = new_toes_eid
            qid2topk_reg[qid] = new_toes_reg
            old2new_eid[old_toes_eid] = new_toes_eid
        else:
            qid2topk_eid[qid] = new_eid_idx
            qid2topk_reg[qid] = non_toes_reg
            old2new_eid[eid] = new_eid_idx
            new_eid_idx += 1
    assert -1 not in old2new_eid
    assert 0 not in old2new_eid
    old2new_eid[0] = 0
    old2new_eid[-1] = -1
    assert (
        new_eid_idx == new_toes_eid
    ), f"{new_eid_idx} is not {new_toes_eid} {to_drop} {old_toes_eid} {perc_emb_drop} {len(qid2count_list)} {num_topk_entities}"
    assert len(qid2topk_eid) == entity_db.num_entities
    return qid2topk_eid, qid2topk_reg, old2new_eid, new_toes_eid, num_topk_entities


def filter_embs(
    new_num_topk_entities,
    entity_db_old,
    old2new_eid,
    qid2topk_eid,
    toes_eid,
    state_dict,
):
    """Using the outputs from filter_eids, this filters the embedding matrix
    for the model to only have the top (1-perc_emb_drop) entities.

    Args:
        new_num_topk_entities: number topK entites
        entity_db_old: old entity database
        old2new_eid: dict of old -> new EID
        qid2topk_eid: dict of QID -> topK EID
        toes_eid: the toe eid
        state_dict: model state dict

    Returns: updated model state dict with updated entity embedding mapping
    """
    entity_weights = get_nested_item(state_dict, ENTITY_EMB_KEYS)
    assert (
        entity_weights.shape[0] == entity_db_old.num_entities_with_pad_and_nocand
    ), f"{entity_db_old.num_entities_with_pad_and_nocand} does not match entity weights shape of {entity_weights.shape[0]}"
    # +2 is for pad and unk
    entity_weights_new = torch.zeros(new_num_topk_entities + 2, entity_weights.shape[1])

    modified_eids = set()
    for eid_old, eid_new in tqdm(old2new_eid.items(), desc="Modifying weights"):
        assert (
            eid_new not in modified_eids
        ), f"You have modified {eid_new} more than once"
        modified_eids.add(eid_new)
        entity_weights_new[eid_new, :] = entity_weights[eid_old, :]

    # Check that mapping was fine
    total_cnt = 0
    cnt_same = 0
    for qid in tqdm(entity_db_old.get_all_qids(), desc="Verifying embeddings"):
        total_cnt += 1
        old_eid = entity_db_old.get_eid(qid)
        new_eid = qid2topk_eid[qid]
        # If the embedding should be the same
        if new_eid != toes_eid:
            cnt_same += 1
            assert torch.equal(
                entity_weights_new[new_eid], entity_weights[old_eid]
            ), f"{old_eid} {new_eid} {qid} {entity_weights_new[new_eid]} {entity_weights[old_eid]}"
    print(f"Verified {cnt_same/total_cnt} percent of embeddings are the same")
    state_dict = set_nested_item(state_dict, ENTITY_EMB_KEYS, entity_weights_new)
    return state_dict


def main():
    args = parse_args()
    print(json.dumps(args, indent=4))
    assert 0 < args.perc_emb_drop < 1, f"perc_emb_drop must be between 0 and 1"
    print(
        f"Loading entity symbols from {os.path.join(args.entity_dir, args.entity_map_dir)}"
    )
    entity_db = EntitySymbols(
        os.path.join(args.entity_dir, args.entity_map_dir), args.alias_cand_map_file
    )
    print(f"Loading qid2count from {args.qid2count}")
    qid2count = utils.load_json_file(args.qid2count)
    print(f"Filtering qids")
    (
        qid2topk_eid,
        qid2topk_reg,
        old2new_eid,
        toes_eid,
        new_num_topk_entities,
    ) = filter_qids(args.perc_emb_drop, entity_db, qid2count)
    if len(args.model_path) > 0:
        assert (
            len(args.save_checkpoint) > 0
        ), f"If you give a model path, you must give a save checkpoint"
        print(f"Filtering embeddings")
        state_dict, model_state_dict = load_statedict(args.model_path)
        try:
            get_nested_item(model_state_dict, ENTITY_EMB_KEYS)
        except:
            print(f"ERROR: All of {ENTITY_EMB_KEYS} are not in model_state_dict")
            raise
        model_state_dict = filter_embs(
            new_num_topk_entities,
            entity_db,
            old2new_eid,
            qid2topk_eid,
            toes_eid,
            model_state_dict,
        )
        # Generate the new old2new_eid weight vector to save in model_state_dict
        oldeid2topkeid = torch.arange(0, entity_db.num_entities_with_pad_and_nocand)
        # The +2 is to account for pads and unks. The -1 is as there are issues with -1 in the indexing for entity embeddings. So we must manually make it the last entry
        oldeid2topkeid[-1] = new_num_topk_entities + 2 - 1
        for qid, new_eid in tqdm(qid2topk_eid.items(), desc="Setting new ids"):
            old_eid = entity_db.get_eid(qid)
            oldeid2topkeid[old_eid] = new_eid
        assert oldeid2topkeid[0] == 0, f"The 0 eid shouldn't be changed"
        assert (
            oldeid2topkeid[-1] == new_num_topk_entities + 2 - 1
        ), "The -1 eid should still map to the last row"
        model_state_dict = set_nested_item(
            model_state_dict, ENTITY_EID_KEYS, oldeid2topkeid
        )
        # Remove the eid2reg value as that was with the old entity id mapping
        try:
            model_state_dict = set_nested_item(model_state_dict, ENTITY_REG_KEYS, None)
        except:
            print(
                f"Could not remove regularization. If your model was trained with regularization mapping on the learned entity embedding, this should not happen."
            )
        print(model_state_dict["module_pool"]["learned"].keys())
        state_dict["model"] = model_state_dict
        print(f"Saving model at {args.save_checkpoint}")
        torch.save(state_dict, args.save_checkpoint)
    print(
        f"Saving topk to eid at {os.path.join(args.entity_dir, args.entity_map_dir, args.save_qid2topk_file)}"
    )
    utils.dump_json_file(
        os.path.join(args.entity_dir, args.entity_map_dir, args.save_qid2topk_file),
        qid2topk_eid,
    )
    print(f"Saving topk to reg at {args.save_qid2topk_reg_file}")
    with open(args.save_qid2topk_reg_file, "w") as out_f:
        out_f.write(f"qid,regularization\n")
        for k, v in qid2topk_reg.items():
            out_f.write(f"{k},{v}\n")


if __name__ == "__main__":
    main()
