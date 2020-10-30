""" Takes the learned embeddings of the model, and compresses them to only contain the top K% determined by popularity."""

import argparse
import os
import ujson as json
import torch
import numpy as np
from collections import OrderedDict

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import utils

ENTITY_EMB_KEY = "emb_layer.entity_embs.learned.learned_entity_embedding.weight"
ENTITY_EID_KEY = "emb_layer.entity_embs.learned.eid2topkeid"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qid2count', type=str, required=True, help='Path for qid2count file')
    parser.add_argument('--perc_emb_drop', type=float, required=True, help='Percentage of embeddings to remove by popularity')
    parser.add_argument('--alias_cand_map_file', type=str, default='alias2qids.json', help='Name of file to read the alias2qids in the entity_dir')
    parser.add_argument('--save_qid2topk_file', type=str, required=True, help='What to save topk qid2eid mapping in entity_db/entity_mappings')
    parser.add_argument('--entity_dir', type=str, required=True, help='Directory to read entity_db')
    parser.add_argument('--entity_map_dir', type=str, default='entity_mappings', help='Directory to read entity_mappings inside entity_dir')
    parser.add_argument('--init_checkpoint', type=str, required=True, help='Which model to load')
    parser.add_argument('--save_checkpoint', type=str, required=True, help='Where to save the model')
    args = parser.parse_args()
    return args

def load_statedict(init_checkpoint):
    print(f'Loading model from {init_checkpoint}.')
    state_dict = torch.load(init_checkpoint, map_location=lambda storage, loc: storage)
    print('Loaded model.')
    # Remove distributed naming if model trained in distributed mode
    model_state_dict = OrderedDict()
    for k, v in state_dict['model'].items():
        if k.startswith('module.'):
            name = k[len('module.'):]
            model_state_dict[name] = v
        else:
            model_state_dict[k] = v
    return state_dict, model_state_dict

def filter_qids(perc_emb_drop, entity_db, qid2count):
    old2new_eid = {}
    print(f"Removing the least popular {perc_emb_drop} embeddings")
    qid2count_with_tails = {}
    for qid in entity_db.get_all_qids():
        qid2count_with_tails[qid] = qid2count.get(qid, 0)

    # Sort smallest to largest
    qid2count_list = sorted(list(qid2count_with_tails.items()), key=lambda x: x[1], reverse=False)
    assert len(qid2count_list) == len(entity_db.get_all_qids()) == entity_db.num_entities
    to_drop = int(perc_emb_drop*len(qid2count_list))
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
    old_toes_eid = np.argmin(np.array(qid_vec[1:-1]))+1
    # We start indexing eids by 1 (0 is reserved). So the +1 is correct for an index for an eid.
    num_topk_entities = len(qid2count_list) - to_drop + 1
    new_toes_eid = num_topk_entities
    new_eid_idx = 1
    # Build qid2neweid mapping
    qid2topk_eid = {}
    for i, qid_pair in enumerate(qid2count_list):
        qid = qid_pair[0]
        eid = entity_db.get_eid(qid)
        if i < to_drop:
            qid2topk_eid[qid] = new_toes_eid
            old2new_eid[old_toes_eid] = new_toes_eid
        else:
            qid2topk_eid[qid] = new_eid_idx
            old2new_eid[eid] = new_eid_idx
            new_eid_idx += 1
    assert -1 not in old2new_eid
    assert 0 not in old2new_eid
    old2new_eid[0] = 0
    old2new_eid[-1] = -1
    assert new_eid_idx == new_toes_eid, f"{new_eid_idx} is not {new_toes_eid} {to_drop} {old_toes_eid} {perc_emb_drop} {len(qid2count_list)} {num_topk_entities}"
    assert len(qid2topk_eid) == entity_db.num_entities
    return qid2topk_eid, old2new_eid, new_toes_eid, num_topk_entities

def filter_embs(new_num_topk_entities, entity_db_old, old2new_eid, qid2topk_eid, toes_eid, state_dict):
    entity_weights = state_dict[ENTITY_EMB_KEY]
    assert entity_weights.shape[0] == entity_db_old.num_entities_with_pad_and_nocand, f"{entity_db_old.num_entities_with_pad_and_nocand} does not match entity weights shape of {entity_weights.shape[0]}"
    # +2 is for pad and unk
    entity_weights_new = torch.zeros(new_num_topk_entities+2, entity_weights.shape[1])

    modified_eids = set()
    for eid_old, eid_new in old2new_eid.items():
        assert eid_new not in modified_eids, f"You have modified {eid_new} more than once"
        modified_eids.add(eid_new)
        entity_weights_new[eid_new, :] = entity_weights[eid_old, :]

    # Check that mapping was fine
    total_cnt = 0
    cnt_same = 0
    for qid in entity_db_old.get_all_qids():
        total_cnt += 1
        old_eid = entity_db_old.get_eid(qid)
        new_eid = qid2topk_eid[qid]
        # If the embedding should be the same
        if new_eid != toes_eid:
            cnt_same += 1
            assert torch.equal(entity_weights_new[new_eid], entity_weights[old_eid]), f"{old_eid} {new_eid} {qid} {entity_weights_new[new_eid]} {entity_weights[old_eid]}"
    print(f"Verified {cnt_same/total_cnt} percent of embeddings are the same")
    state_dict[ENTITY_EMB_KEY] = entity_weights_new
    return state_dict

def main():
    args = parse_args()
    print(json.dumps(args, indent=4))
    assert 0 < args.perc_emb_drop < 1, f"perc_emb_drop must be between 0 and 1"
    state_dict, model_state_dict = load_statedict(args.init_checkpoint)
    assert ENTITY_EMB_KEY in model_state_dict
    print(f"Loading entity symbols from {os.path.join(args.entity_dir, args.entity_map_dir)}")
    entity_db = EntitySymbols(os.path.join(args.entity_dir, args.entity_map_dir), args.alias_cand_map_file)
    print(f"Loading qid2count from {args.qid2count}")
    qid2count = utils.load_json_file(args.qid2count)
    print(f"Filtering qids")
    qid2topk_eid, old2new_eid, toes_eid, new_num_topk_entities = filter_qids(args.perc_emb_drop, entity_db, qid2count)
    print(f"Filtering embeddings")
    model_state_dict = filter_embs(new_num_topk_entities, entity_db, old2new_eid, qid2topk_eid, toes_eid, model_state_dict)
    # Generate the new old2new_eid weight vector to save in model_state_dict
    oldeid2topkeid = torch.arange(0, entity_db.num_entities_with_pad_and_nocand)
    # The +2 is to account for pads and unks. The -1 is as there are issues with -1 in the indexing for entity embeddings. So we must manually make it the last entry
    oldeid2topkeid[-1] = new_num_topk_entities+2-1
    for qid, new_eid in qid2topk_eid.items():
        old_eid = entity_db.get_eid(qid)
        oldeid2topkeid[old_eid] = new_eid
    assert oldeid2topkeid[0] == 0, f"The 0 eid shouldn't be changed"
    assert oldeid2topkeid[-1] == new_num_topk_entities+2-1, "The -1 eid should still map to the last row"
    model_state_dict[ENTITY_EID_KEY] = oldeid2topkeid
    state_dict["model"] = model_state_dict
    print(f"Saving model at {args.save_checkpoint}")
    torch.save(state_dict, args.save_checkpoint)
    print(f"Saving entity_db at {os.path.join(args.entity_dir, args.entity_map_dir, args.save_qid2topk_file)}")
    utils.dump_json_file(os.path.join(args.entity_dir, args.entity_map_dir, args.save_qid2topk_file), qid2topk_eid)

if __name__ == '__main__':
    main()