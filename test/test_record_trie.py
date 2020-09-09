import unittest
import os, sys
import numpy as np

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils, parser_utils
from bootleg.utils.classes.record_trie_collection import RecordTrieCollection
from bootleg.utils.sentence_utils import split_sentence


class TestRecordTrieCollection(unittest.TestCase):

    def test_construction_single(self):
        entity_dump_dir = "test/data/preprocessing/base/entity_db/entity_mappings"
        entity_symbols = EntitySymbols(load_dir=entity_dump_dir,
            alias_cand_map_file="alias2qids.json")
        fmt_types = {"trie1": "qid_cand_with_score"}
        max_values = {"trie1": 3}
        input_dicts = {"trie1": entity_symbols.get_alias2qids()}
        record_trie = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_symbols.get_qid2eid(),
                                 fmt_types=fmt_types, max_values=max_values)
        truealias2qids = {
                        'alias1': [["Q1", 10], ["Q4", 6]],
                        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
                        'alias3': [["Q1", 30]],
                        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
                        }

        for al in truealias2qids:
            self.assertEqual(truealias2qids[al], record_trie.get_value("trie1", al))

        for al in truealias2qids:
            self.assertEqual([x[0] for x in truealias2qids[al]], record_trie.get_value("trie1", al, getter=lambda x: x[0]))

        for al in truealias2qids:
            self.assertEqual([x[1] for x in truealias2qids[al]], record_trie.get_value("trie1", al, getter=lambda x: x[1]))

        self.assertEqual(set(truealias2qids.keys()), set(record_trie.get_keys("trie1")))

    def test_construction_double(self):
        truealias2qids = {
                        'alias1': [["Q1", 10], ["Q4", 6]],
                        'multi word alias2': [["Q2", 5], ["Q1", 3], ["Q4", 2]],
                        'alias3': [["Q1", 30]],
                        'alias4': [["Q4", 20], ["Q3", 15], ["Q2", 1]]
                        }
        truealias2typeids = {
                        'Q1': [1, 2, 3],
                        'Q2': [8],
                        'Q3': [4],
                        'Q4': [2,4]
                        }
        truerelations = {
                        'Q1': set(["Q2", "Q3"]),
                        'Q2': set(["Q1"]),
                        'Q3': set(["Q2"]),
                        'Q4': set(["Q1", "Q2", "Q3"])
                        }
        entity_dump_dir = "test/data/preprocessing/base/entity_db/entity_mappings"
        entity_symbols = EntitySymbols(load_dir=entity_dump_dir,
            alias_cand_map_file="alias2qids.json")
        fmt_types = {"trie1": "qid_cand_with_score", "trie2": "type_ids", "trie3": "kg_relations"}
        max_values = {"trie1": 3, "trie2": 3, "trie3": 3}
        input_dicts = {"trie1": entity_symbols.get_alias2qids(), "trie2": truealias2typeids, "trie3": truerelations}
        record_trie = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_symbols.get_qid2eid(),
                                 fmt_types=fmt_types, max_values=max_values)

        for al in truealias2qids:
            self.assertEqual(truealias2qids[al], record_trie.get_value("trie1", al))

        for qid in truealias2typeids:
            self.assertEqual(truealias2typeids[qid], record_trie.get_value("trie2", qid))

        for qid in truerelations:
            self.assertEqual(truerelations[qid], record_trie.get_value("trie3", qid))

    def test_load_and_save(self):
        entity_dump_dir = "test/data/preprocessing/base/entity_db/entity_mappings"
        entity_symbols = EntitySymbols(load_dir=entity_dump_dir,
            alias_cand_map_file="alias2qids.json")
        fmt_types = {"trie1": "qid_cand_with_score"}
        max_values = {"trie1": 3}
        input_dicts = {"trie1": entity_symbols.get_alias2qids()}
        record_trie = RecordTrieCollection(load_dir=None, input_dicts=input_dicts, vocabulary=entity_symbols.get_qid2eid(),
                                 fmt_types=fmt_types, max_values=max_values)

        record_trie.dump(save_dir=os.path.join(entity_dump_dir, "record_trie"))
        record_trie_loaded = RecordTrieCollection(load_dir=os.path.join(entity_dump_dir, "record_trie"))

        self.assertEqual(record_trie._fmt_types, record_trie_loaded._fmt_types)
        self.assertEqual(record_trie._max_values, record_trie_loaded._max_values)
        self.assertEqual(record_trie._stoi, record_trie_loaded._stoi)
        np.testing.assert_array_equal(record_trie._itos, record_trie_loaded._itos)
        self.assertEqual(record_trie._record_tris, record_trie_loaded._record_tris)

if __name__ == '__main__':
    unittest.main()
