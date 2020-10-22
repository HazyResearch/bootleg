"""
Used Wikidata extractor to get a list of triples for all qids in our dump such that the qid was the head of some relation. The tail could be any qid,
in our dump or not. (See wikidata_extractor get_all_wikipedia_triples.py)
"""
import json
from collections import defaultdict

pid_names = json.load(open("/dfs/scratch0/lorr1/contextual-embeddings/wikidata_mappings/pid_names.json", "r"))


triples = open("kg_triples_0905.txt", "r")
head_trips_dict = defaultdict(set)
pred_count = defaultdict(int)
for line in triples:
    head, pred, tail = line.strip().split()
    head_trips_dict[head].add(pred)
    pred_count[pred] += 1



rel_ids = defaultdict(int)
rel_ids_internal = defaultdict(int)
num = 0
for pid in pred_count:
    rel_ids[pid_names.get(pid,pid)] = num
    rel_ids_internal[pid] = num
    num += 1

# Current list of types...wanted to make sure the keys matched. This is optional
all_type_qids = {}
with open("wikidata_types_0905.json", "r") as in_f:
    all_type_qids = json.load(in_f)



out_file = open("kg_relation_types_09052.json", "w")
max_len = 0
final_qid2type = {}
for qid in all_type_qids:
    types = []
    if qid in head_trips_dict:
        type_ids = head_trips_dict[qid]
        max_len = max(max_len, len(type_ids))
        # Sort by least to most common; will help filter to more useful relation types
        type_ids = sorted(type_ids, key=lambda x: [pred_count[x], x])
        type_ids = [rel_ids_internal[i] for i in type_ids]
        # Type 0 is the "instance of" type that is not discriminative at all
        types = [i for i in type_ids if i != 0]
    final_qid2type[qid] = types


json.dump(final_qid2type, out_file)

out_file.close()
with open("relation_to_typeid_09052.json", "w") as out_f:
    json.dump(rel_ids, out_f)

print("MAX LEN", max_len)
