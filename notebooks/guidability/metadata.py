import os

import marisa_trie
import ujson
from bootleg_data_prep.utils.classes.record_trie_collection import RecordTrieCollection
from tqdm import tqdm

ALIAS2QID = "alias2qid"
MAX_RELATIONS = 150
MAX_TYPS = 50


class WLMetadata:
    def __init__(
        self,
        entity_dump=None,
        dict_type_syms=None,
        dict_kg_syms=None,
        tri_collection=None,
        q2title=None,
        type_vocabs=None,
    ):
        if dict_kg_syms is None:
            dict_kg_syms = []
        if dict_type_syms is None:
            dict_type_syms = []
        if entity_dump is not None:
            # This maps our keys that we use in the helper functions below to the right tri in tri collection.
            # The values are specific strings as outlines in the record trie collection class
            fmt_types = {ALIAS2QID: "qid_cand_with_score"}
            max_values = {ALIAS2QID: entity_dump.max_candidates}
            input_dicts = {ALIAS2QID: entity_dump.get_alias2qids()}

            # Add kg relations
            kg_keys = []
            for kg_tag, kg_syms in dict_kg_syms.items():
                kg_keys.append(kg_tag)
                fmt_types[kg_tag] = "kg_relations"
                max_values[kg_tag] = MAX_RELATIONS
                truncated_kg_dict = {}
                for k, v in tqdm(kg_syms.qid2connections.items(), desc="Filtering KG"):
                    if len(v) > 0:
                        truncated_kg_dict[k] = list(v)[:MAX_RELATIONS]
                input_dicts[kg_tag] = truncated_kg_dict

            # Add types
            type_vocabs = {}
            for type_tag, typ_syms in dict_type_syms.items():
                assert (
                    type_tag not in kg_keys
                ), f"You must provide unique keys across types and kg input symbol dicts. {type_tag} is already used."
                type_vocabs[type_tag] = typ_syms.typeid2typename
                fmt_types[type_tag] = "type_ids"
                max_values[type_tag] = MAX_TYPS
                truncated_type_dict = {}
                for k, v in tqdm(typ_syms.qid2typeid.items(), desc="Filtering Type"):
                    if len(v) > 0:
                        truncated_type_dict[k] = list(v)[:MAX_TYPS]
                input_dicts[type_tag] = truncated_type_dict

            print(f"Max Values {max_values}")
            self.tri_collection = RecordTrieCollection(
                load_dir=None,
                input_dicts=input_dicts,
                vocabulary=entity_dump.get_qid2eid(),
                fmt_types=fmt_types,
                max_values=max_values,
            )
            self.q2title = entity_dump.get_qid2title()
            self.type_vocabs = type_vocabs
        else:
            assert tri_collection is not None, f"tri_collection is None"
            assert q2title is not None, f"q2title is None"
            assert type_vocabs is not None, f"type_vocabs is None"
            self.tri_collection = tri_collection
            self.q2title = q2title
            self.type_vocabs = type_vocabs
        self.title2q = {v: k for k, v in self.q2title.items()}

    @classmethod
    def get_tri_dir(cls, dump_dir):
        return os.path.join(dump_dir, "TRI")

    @classmethod
    def get_qid2title_file(cls, dump_dir):
        return os.path.join(dump_dir, "QID2TITLE.json")

    @classmethod
    def get_type_vocabs_file(cls, dump_dir):
        return os.path.join(dump_dir, "TYPE_VOCABS.json")

    def save(self, dump_dir):
        self.tri_collection.save(save_dir=self.get_tri_dir(dump_dir))
        with open(self.get_qid2title_file(dump_dir), "w") as out_f:
            ujson.dump(self.q2title, out_f)
        with open(self.get_type_vocabs_file(dump_dir), "w") as out_f:
            ujson.dump(self.type_vocabs, out_f)

    @classmethod
    def load(cls, dump_dir):
        tri_collection = RecordTrieCollection(load_dir=cls.get_tri_dir(dump_dir))
        with open(cls.get_qid2title_file(dump_dir)) as in_f:
            qid2title = ujson.load(in_f)
        with open(cls.get_type_vocabs_file(dump_dir)) as in_f:
            type_vocabs = ujson.load(in_f)
        return cls(
            entity_dump=None,
            dict_type_syms=None,
            dict_kg_syms=None,
            tri_collection=tri_collection,
            q2title=qid2title,
            type_vocabs=type_vocabs,
        )

    def contains_alias(self, alias):
        return self.tri_collection.is_key_in_trie(ALIAS2QID, alias)

    def get_num_cands(self, alias):
        assert self.contains_alias(alias), f"{alias} not in mapping"
        return len(self.tri_collection.get_value(ALIAS2QID, alias))

    def get_qid_cands(self, alias):
        assert self.contains_alias(alias), f"{alias} not in mapping"
        return self.tri_collection.get_value(ALIAS2QID, alias, getter=lambda x: x[0])

    def get_qid_count_cands(self, alias):
        assert self.contains_alias(alias), f"{alias} not in mapping"
        return self.tri_collection.get_value(ALIAS2QID, alias)

    def get_title(self, qid, default=None):
        return self.q2title.get(qid, default)

    def get_qid(self, title, default=None):
        return self.title2q.get(title, default)

    def qid_in_qid2typeid(self, qid, type_key):
        return self.tri_collection.is_tri_in_collection(
            type_key
        ) and self.tri_collection.is_key_in_trie(type_key, qid)

    def get_type_ids(self, qid, type_key):
        if not self.tri_collection.is_key_in_trie(type_key, qid):
            return []
        return self.tri_collection.get_value(type_key, qid)

    def get_types(self, qid, type_key):
        type_ids = self.get_type_ids(qid, type_key)
        return [self.type_vocabs[type_key][str(i)] for i in type_ids]

    def qid_in_rel_mapping(self, qid, rel_key):
        return self.tri_collection.is_tri_in_collection(
            rel_key
        ) and self.tri_collection.is_key_in_trie(rel_key, qid)

    def get_related_qids(self, qid, rel_key):
        if not self.tri_collection.is_key_in_trie(rel_key, qid):
            return []
        return self.tri_collection.get_value(rel_key, qid)

    def get_all_aliases(self):
        return self.tri_collection.get_keys(ALIAS2QID)

    def __getstate__(self):
        print(
            f"WARNING!!! DO NOT USE PICKLE OR PYTHON DUMPING. PLEASE USE THE load() and save() methods"
        )
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        print(
            f"WARNING!!! DO NOT USE PICKLE OR PYTHON DUMPING. PLEASE USE THE load() and save() methods"
        )
        self.__dict__.update(state)
