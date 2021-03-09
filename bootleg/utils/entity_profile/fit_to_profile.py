"""Preconditions of the different profiles:

- We do not support new types/relations
- All entity_db data will need to be reprepped anyways - that is independent of model
- We _only_ need to change the entity embeddings
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import ujson
import ujson as json
from tqdm import tqdm

from bootleg.symbols.entity_profile import EntityProfile
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils

ENTITY_EMB_KEYS = ["module_pool", "learned", "learned_entity_embedding.weight"]
ENTITY_REG_KEYS = ["module_pool", "learned", "eid2reg"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init_vec_file", type=str, default=None, help="Path for new entity vec file"
    )
    parser.add_argument(
        "--train_entity_profile",
        type=str,
        required=True,
        help="Path to entity profile used for training",
    )
    parser.add_argument(
        "--new_entity_profile",
        type=str,
        required=True,
        help="Path to new entity profile we want to fit to",
    )
    parser.add_argument(
        "--oldqid2newqid",
        type=str,
        default=None,
        help="Path to mapping of old qid to new qid in the case a reidentify is called",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Which model to load. If empty, we just generate the entity mappings",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        required=True,
        help="Where to save the model.",
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
        ), f"Key {k} is not in dictionary after seeing {i + 1} keys from {list_of_keys}"
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
        ), f"Key {k} is not in dictionary after seeing {i + 1} keys from {list_of_keys}"
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


def match_entities(old_ep, new_ep, oldqid2newqid, newqid2oldqid):
    """
    Return the left difference, intersection, and right difference of the sets of entities and mapping of old to new
    entity IDs for the same entities
    Args:
        old_ep: old entity profile
        new_ep: new entity profile
        oldqid2newqid: Dict of old entity QID to new entity QID (if reidentified)
        newqid2oldqid: Dict of new entity QID to old entity QID (if reidentified)

    Returns: removed ents, same ents, new ents, old entity ID -> new entity ID
    """
    old_ents = set([oldqid2newqid.get(q, q) for q in old_ep.get_all_qids()])
    modified_ents = set(new_ep.get_all_qids())
    # np_ stands for new profile
    np_removed_ents = old_ents - modified_ents
    np_new_ents = modified_ents - old_ents
    np_same_ents = old_ents & modified_ents
    oldeid2neweid = {
        old_ep.get_eid(newqid2oldqid.get(q, q)): new_ep.get_eid(q) for q in np_same_ents
    }
    return np_removed_ents, np_same_ents, np_new_ents, oldeid2neweid


def refit_weights(
    np_same_ents,
    neweid2oldeid,
    train_entity_profile,
    new_entity_profile,
    vector_for_new_ent,
    state_dict,
):
    entity_weights = get_nested_item(state_dict, ENTITY_EMB_KEYS)
    try:
        entity_reg = get_nested_item(state_dict, ENTITY_REG_KEYS)
    except AssertionError:
        # Is not eid2reg in model
        entity_reg = None
    assert (
        entity_weights.shape[0]
        == train_entity_profile.get_num_entities_with_pad_and_nocand()
    ), f"{train_entity_profile.get_num_entities_with_pad_and_nocand()} does not match entity weights shape of {entity_weights.shape[0]}"
    # Last row is pad row of all zeros
    assert torch.equal(
        entity_weights[-1], torch.zeros(entity_weights.shape[1])
    ), "Last row of train data wasn't zero"

    new_entity_weight = np.zeros(
        (
            new_entity_profile.get_num_entities_with_pad_and_nocand(),
            entity_weights.shape[1],
        ),
    )

    # Create index map from new eid to old eid for the entities that are shared
    shared_neweid2old = {
        new_entity_profile.get_eid(qid): neweid2oldeid[new_entity_profile.get_eid(qid)]
        for qid in np_same_ents
    }
    # shared_newindex: the new index set of the shared entities
    # shared_oldindex: the old index set of the shared entities
    # The 0 represents the NC entity embedding. We always want to copy this over so we add it to both (i.e., 0 row maps to 0 row)
    shared_newindex = [0]
    shared_oldindex = [0]
    newent_index = []
    # We start at 1 because we already handled 0 above.
    # We end at -1 as we want the last row to always be zero
    for i in range(1, new_entity_weight.shape[0] - 1):
        # If the entity id is shared with the old set
        if i in shared_neweid2old:
            shared_newindex.append(i)
            shared_oldindex.append(shared_neweid2old[i])
        # If the entity id is a new entity
        else:
            newent_index.append(i)
    shared_newindex = np.array(shared_newindex)
    shared_oldindex = np.array(shared_oldindex)
    newent_index = np.array(newent_index)
    print(
        "OG",
        new_entity_weight.shape[0],
        "SHARED",
        len(shared_newindex),
        "NEW",
        len(newent_index),
    )
    # Copy over the old weights
    new_entity_weight[shared_newindex, :] = entity_weights[shared_oldindex, :]
    # Assign new entities
    new_entity_weight[newent_index, :] = vector_for_new_ent
    new_entity_weight = torch.from_numpy(new_entity_weight).float()
    # Last row is pad row of all zeros
    assert torch.equal(
        new_entity_weight[-1], torch.zeros(new_entity_weight.shape[1])
    ), "Last row of new data wasn't zero"
    state_dict = set_nested_item(state_dict, ENTITY_EMB_KEYS, new_entity_weight)
    # Create regularization file if need to
    if entity_reg is not None:
        new_entity_reg = np.zeros(
            (new_entity_profile.get_num_entities_with_pad_and_nocand())
        )
        # Copy over the old weights
        new_entity_reg[shared_newindex] = entity_reg[shared_oldindex]
        # Assign default value in the middle for these new entities. If finetuned, it shouldn't hurt performance.
        new_entity_reg[newent_index] = 0.5
        new_entity_reg = torch.from_numpy(new_entity_reg).float()
        state_dict = set_nested_item(state_dict, ENTITY_REG_KEYS, new_entity_reg)
    return state_dict


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    print(f"Loading train entity profile from {args.train_entity_profile}")
    train_entity_profile = EntityProfile.load_from_cache(
        load_dir=args.train_entity_profile
    )
    print(f"Loading new entity profile from {args.new_entity_profile}")
    new_entity_profile = EntityProfile.load_from_cache(load_dir=args.new_entity_profile)

    oldqid2newqid = dict()
    newqid2oldqid = dict()
    if args.oldqid2newqid is not None and len(args.oldqid2newqid) > 0:
        with open(args.oldqid2newqid) as in_f:
            oldqid2newqid = ujson.load(in_f)
            newqid2oldqid = {v: k for k, v in oldqid2newqid.items()}
            assert len(oldqid2newqid) == len(
                newqid2oldqid
            ), f"The dicts of oldqid2newqid and its inverse do not have the same length"

    np_removed_ents, np_same_ents, np_new_ents, oldeid2neweid = match_entities(
        train_entity_profile, new_entity_profile, oldqid2newqid, newqid2oldqid
    )
    neweid2oldeid = {v: k for k, v in oldeid2neweid.items()}
    assert len(oldeid2neweid) == len(
        neweid2oldeid
    ), f"The lengths of oldeid2neweid and neweid2oldeid don't match"
    state_dict, model_state_dict = load_statedict(args.model_path)

    try:
        weight_shape = get_nested_item(model_state_dict, ENTITY_EMB_KEYS).shape[1]
    except:
        raise ValueError(f"ERROR: All of {ENTITY_EMB_KEYS} are not in model_state_dict")

    if args.init_vec_file is not None and len(args.init_vec_file) > 0:
        vector_for_new_ent = np.load(args.init_vec_file)
    else:
        print(f"Setting init vector to be all zeros")
        vector_for_new_ent = np.zeros(weight_shape)
    new_model_state_dict = refit_weights(
        np_same_ents,
        neweid2oldeid,
        train_entity_profile,
        new_entity_profile,
        vector_for_new_ent,
        model_state_dict,
    )
    print(new_model_state_dict["module_pool"]["learned"].keys())
    state_dict["model"] = new_model_state_dict
    print(f"Saving model at {args.save_model_path}")
    torch.save(state_dict, args.save_model_path)


if __name__ == "__main__":
    main()
