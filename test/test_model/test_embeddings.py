import os
import shutil
import unittest

import marisa_trie
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer

import emmental
from bootleg.embeddings import (
    EntityEmb,
    KGAdjEmb,
    KGIndices,
    LearnedEntityEmb,
    StaticEmb,
    TitleEmb,
    TopKEntityEmb,
    TypeEmb,
)
from bootleg.layers.embedding_payload import EmbeddingPayload, EmbeddingPayloadBase
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.tasks.task_getters import get_embedding_tasks
from bootleg.utils import model_utils, utils
from bootleg.utils.classes.dotted_dict import DottedDict
from bootleg.utils.embedding_utils import get_max_candidates
from bootleg.utils.parser import parser_utils


class Noop(nn.Module):
    def forward(self, input, *args):
        return input


class NoopCat(nn.Module):
    def forward(self, input, *args):
        ret = torch.cat(input, 3)
        return ret


class NoopTakeFirst(nn.Module):
    def forward(self, input, *args):
        return input[0]


# Embedding class that does not declare a normalize attribute. We check that the normalize
# attribute is instantiated and need
# to test that this feature works.
class EntityEmbNoNorm(EntityEmb):
    def __init__(
        self,
        main_args,
        emb_args,
        model_device,
        entity_symbols,
        word_symbols,
        word_emb,
        key,
    ):
        super(EntityEmbNoNorm, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
        )
        self._dim = main_args.model_config.hidden_size

    # For some reason, the check on parameters only happens during the forward call if an embedding package is returned
    # So we need some mock forward call that does this.
    def forward(
        self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb
    ):
        emb = self._package(
            tensor=torch.randn(5, 5),
            start_span_idx=None,
            end_span_idx=None,
            alias_indices=None,
            mask=None,
        )
        return emb


class EntitySymbolsSubclass(EntitySymbols):
    def __init__(self):
        self.max_candidates = 3
        # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
        self.max_alias_len = 1
        self._qid2title = {"Q1": "a b c d e", "Q2": "f", "Q3": "dd a b", "Q4": "x y z"}
        self._qid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        self._alias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2


# same as the above but with an extra entity and alias to test adding synthetic profiles
class EntitySymbolsSubclassExtra(EntitySymbols):
    def __init__(self):
        self._qid2title = {
            "Q1": "a b c d e",
            "Q2": "f",
            "Q3": "dd a b",
            "Q4": "x y z",
            "Q5": "d",
        }
        self._qid2eid = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}
        self._alias2qids = {
            "alias1": [["Q1", 10.0], ["Q4", 6]],
            "multi word alias2": [["Q2", 5.0], ["Q1", 3], ["Q4", 2]],
            "alias2": [["Q5", 1.0]],
            "alias3": [["Q1", 30.0]],
            "alias4": [["Q4", 20], ["Q3", 15.0], ["Q2", 1]],
        }
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2


def load_tokenizer():
    return BertTokenizer.from_pretrained(
        "bert-base-cased",
        do_lower_case=False,
        cache_dir="test/data/emb_data/pretrained_bert_models",
    )


class TestLoadEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_embeddings.json"
        )
        emmental.init(log_dir="test/temp_log", config=self.args)
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)
        self.args.data_config.ent_embeddings = [
            DottedDict(
                {
                    "key": "learned1",
                    "load_class": "LearnedEntityEmb",
                    "dropout1d": 0.5,
                    "args": {"learned_embedding_size": 5, "tail_init": False},
                }
            ),
            DottedDict(
                {
                    "key": "learned2",
                    "dropout2d": 0.5,
                    "load_class": "LearnedEntityEmb",
                    "args": {"learned_embedding_size": 5, "tail_init": False},
                }
            ),
            DottedDict(
                {
                    "key": "learned3",
                    "load_class": "LearnedEntityEmb",
                    "freeze": True,
                    "args": {"learned_embedding_size": 5, "tail_init": False},
                }
            ),
            DottedDict(
                {
                    "key": "learned4",
                    "load_class": "LearnedEntityEmb",
                    "normalize": False,
                    "args": {"learned_embedding_size": 5, "tail_init": False},
                }
            ),
            DottedDict(
                {
                    "key": "learned5",
                    "load_class": "LearnedEntityEmb",
                    "cpu": True,
                    "args": {"learned_embedding_size": 5, "tail_init": False},
                }
            ),
        ]
        self.tokenizer = load_tokenizer()
        self.entity_symbols = EntitySymbolsSubclass()

    def tearDown(self):
        prep_dir = os.path.join(
            self.args.data_config.data_dir, self.args.data_config.data_prep_dir
        )
        if os.path.exists(prep_dir):
            shutil.rmtree(prep_dir, ignore_errors=True)
        dir = os.path.join("test/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_embedding_task_gen(self):
        (
            task_flows,
            module_pool,
            module_device_dict,
            extra_bert_embedding_layers,
            to_add_to_payload,
            total_sizes,
        ) = get_embedding_tasks(self.args, self.entity_symbols)

        gold_module_device_dict = {"learned5": -1}

        gold_total_sizes = {
            "learned1": 5,
            "learned2": 5,
            "learned3": 5,
            "learned4": 5,
            "learned5": 5,
        }

        gold_extra_bert_embedding_layers = []

        gold_to_add_to_payload = [
            ("embedding_learned1", 0),
            ("embedding_learned2", 0),
            ("embedding_learned3", 0),
            ("embedding_learned4", 0),
            ("embedding_learned5", 0),
        ]
        gold_task_flows = [
            {
                "name": f"embedding_learned1",
                "module": "learned1",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned2",
                "module": "learned2",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned3",
                "module": "learned3",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned4",
                "module": "learned4",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned5",
                "module": "learned5",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
        ]

        # Asserts that the order is the same
        self.assertEqual(to_add_to_payload, gold_to_add_to_payload)
        self.assertEqual(extra_bert_embedding_layers, gold_extra_bert_embedding_layers)
        self.assertDictEqual(module_device_dict, gold_module_device_dict)
        self.assertDictEqual(total_sizes, gold_total_sizes)
        assert len(task_flows) == len(gold_task_flows)
        for li, r in zip(task_flows, gold_task_flows):
            self.assertDictEqual(li, r)
        # Check defaults
        assert module_pool["learned1"].normalize is True
        assert module_pool["learned2"].dropout1d_perc == 0.0
        assert module_pool["learned1"].dropout2d_perc == 0.0
        # Check set values
        assert module_pool["learned1"].dropout1d_perc == 0.5
        assert module_pool["learned2"].dropout2d_perc == 0.5
        assert module_pool["learned4"].normalize is False

        for n, param in module_pool["learned3"].named_parameters():
            assert param.requires_grad is False

    def test_adding_title_nobert(self):
        self.args.data_config.ent_embeddings.append(
            DottedDict(
                {
                    "key": "title1",
                    "load_class": "TitleEmb",
                    "normalize": False,
                    "args": {"proj": 5},
                }
            )
        )
        (
            task_flows,
            module_pool,
            module_device_dict,
            extra_bert_embedding_layers,
            to_add_to_payload,
            total_sizes,
        ) = get_embedding_tasks(self.args, self.entity_symbols)

        gold_module_device_dict = {"learned5": -1}

        gold_extra_bert_embedding_layers = []

        gold_to_add_to_payload = [
            ("embedding_learned1", 0),
            ("embedding_learned2", 0),
            ("embedding_learned3", 0),
            ("embedding_learned4", 0),
            ("embedding_learned5", 0),
            ("embedding_title1", 0),
        ]

        gold_total_sizes = {
            "learned1": 5,
            "learned2": 5,
            "learned3": 5,
            "learned4": 5,
            "learned5": 5,
            "title1": 5,
        }

        gold_task_flows = [
            {
                "name": f"embedding_learned1",
                "module": "learned1",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned2",
                "module": "learned2",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned3",
                "module": "learned3",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned4",
                "module": "learned4",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned5",
                "module": "learned5",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_title1",
                "module": "title1",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
        ]
        # Asserts that the order is the same
        self.assertEqual(to_add_to_payload, gold_to_add_to_payload)
        self.assertEqual(extra_bert_embedding_layers, gold_extra_bert_embedding_layers)
        self.assertDictEqual(module_device_dict, gold_module_device_dict)
        self.assertDictEqual(total_sizes, gold_total_sizes)
        assert len(task_flows) == len(gold_task_flows)
        for li, r in zip(task_flows, gold_task_flows):
            self.assertDictEqual(li, r)

    def test_adding_title(self):
        self.args.data_config.ent_embeddings.append(
            DottedDict(
                {
                    "key": "title1",
                    "load_class": "TitleEmb",
                    "send_through_bert": True,
                    "normalize": False,
                    "args": {"proj": 5},
                }
            )
        )
        (
            task_flows,
            module_pool,
            module_device_dict,
            extra_bert_embedding_layers,
            to_add_to_payload,
            total_sizes,
        ) = get_embedding_tasks(self.args, self.entity_symbols)

        gold_module_device_dict = {"learned5": -1}

        gold_extra_bert_embedding_layers = [TitleEmbMock()]

        gold_to_add_to_payload = [
            ("embedding_learned1", 0),
            ("embedding_learned2", 0),
            ("embedding_learned3", 0),
            ("embedding_learned4", 0),
            ("embedding_learned5", 0),
            ("bert", 2),
        ]

        gold_total_sizes = {
            "learned1": 5,
            "learned2": 5,
            "learned3": 5,
            "learned4": 5,
            "learned5": 5,
            "title1": 5,
        }

        gold_task_flows = [
            {
                "name": f"embedding_learned1",
                "module": "learned1",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned2",
                "module": "learned2",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned3",
                "module": "learned3",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned4",
                "module": "learned4",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
            {
                "name": f"embedding_learned5",
                "module": "learned5",
                "inputs": [
                    ("_input_", "entity_cand_eid"),
                    (
                        "_input_",
                        "batch_on_the_fly_kg_adj",
                    ),  # special kg adjacency embedding prepped in dataloader
                ],
            },
        ]
        # Asserts that the order is the same
        self.assertEqual(to_add_to_payload, gold_to_add_to_payload)
        assert len(extra_bert_embedding_layers) == len(gold_extra_bert_embedding_layers)
        for i_l, i_r in zip(
            extra_bert_embedding_layers, gold_extra_bert_embedding_layers
        ):
            # These are classes so we can't do == but we can check other properties are correct
            assert type(i_l) is TitleEmb
            self.assertEqual(i_l.key, i_r.key)
            self.assertEqual(i_l.cpu, i_r.cpu)
            self.assertEqual(i_l.normalize, i_r.normalize)
            self.assertEqual(i_l.dropout1d_perc, i_r.dropout1d_perc)
            self.assertEqual(i_l.dropout2d_perc, i_r.dropout2d_perc)
        self.assertDictEqual(module_device_dict, gold_module_device_dict)
        self.assertDictEqual(total_sizes, gold_total_sizes)
        assert len(task_flows) == len(gold_task_flows)
        for li, r in zip(task_flows, gold_task_flows):
            self.assertDictEqual(li, r)


class TestLearnedEmbedding(unittest.TestCase):
    def setUp(self):
        emmental.init(log_dir="test/temp_log")
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)
        self.entity_symbols = EntitySymbolsSubclass()
        self.hidden_size = 30
        self.learned_embedding_size = 50
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_embeddings.json"
        )
        self.regularization_csv = os.path.join(
            self.args.data_config.data_dir, "test_reg.csv"
        )
        self.static_emb = os.path.join(self.args.data_config.data_dir, "static_emb.pt")
        self.qid2topkeid = os.path.join(
            self.args.data_config.data_dir, "test_eid2topk.json"
        )
        self.args.model_config.hidden_size = self.hidden_size
        self.args.data_config.ent_embeddings[0]["args"] = DottedDict(
            {"learned_embedding_size": self.learned_embedding_size}
        )

    def tearDown(self) -> None:
        if os.path.exists(self.regularization_csv):
            os.remove(self.regularization_csv)
        if os.path.exists(self.qid2topkeid):
            os.remove(self.qid2topkeid)
        if os.path.exists(self.static_emb):
            os.remove(self.static_emb)
        dir = os.path.join(
            self.args.data_config.data_dir, self.args.data_config.data_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        dir = os.path.join(
            self.args.data_config.entity_dir, self.args.data_config.entity_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        dir = os.path.join("test/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    # confirm dimension of output is correct
    def test_forward_dimension(self):
        learned_emb = LearnedEntityEmb(
            main_args=self.args,
            emb_args=self.args.data_config.ent_embeddings[0]["args"],
            entity_symbols=self.entity_symbols,
            key="learned",
            cpu=True,
            normalize=False,
            dropout1d_perc=0.0,
            dropout2d_perc=0.0,
        )
        entity_ids = torch.tensor([[[0, 1], [2, 2]]])
        actual_out = learned_emb(entity_ids, batch_on_the_fly_data={})
        assert [
            entity_ids.shape[0],
            entity_ids.shape[1],
            entity_ids.shape[2],
            self.learned_embedding_size,
        ] == list(actual_out.size())

    def test_initialization(self):
        self.learned_entity_embedding = nn.Embedding(
            self.entity_symbols.num_entities_with_pad_and_nocand,
            4,
            padding_idx=-1,
            sparse=True,
        )
        gold_emb = self.learned_entity_embedding.weight.data[:]

        gold_emb[1:] = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        init_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
        init_vec_out = model_utils.init_embeddings_to_vec(
            self.learned_entity_embedding, pad_idx=-1, vec=init_vec
        )
        assert torch.equal(init_vec, init_vec_out)
        assert torch.equal(gold_emb, self.learned_entity_embedding.weight.data)

    def test_load_regularization(self):
        regularization_data = [
            ["qid", "regularization"],
            ["Q1", "0.4"],
            ["Q3", "0.8"],
            ["Q4", "0.2"],
        ]
        with open(self.regularization_csv, "w") as out_f:
            for li in regularization_data:
                out_f.write(f"{','.join(li)}\n")
        self.args.data_config.ent_embeddings[0]["args"][
            "regularize_mapping"
        ] = self.regularization_csv
        eid2reg = LearnedEntityEmb.load_regularization_mapping(
            self.args.data_config, self.entity_symbols, self.regularization_csv
        )
        eid2reg_gold = torch.tensor([0.0, 0.4, 0.0, 0.8, 0.2, 0.0])
        assert torch.equal(eid2reg_gold, eid2reg)

    def test_topk_embedding(self):
        topkqid2eid = {"Q1": 1, "Q2": 3, "Q3": 3, "Q4": 2}
        utils.dump_json_file(self.qid2topkeid, topkqid2eid)
        self.args.data_config.ent_embeddings[0]["args"]["perc_emb_drop"] = 0.5
        self.args.data_config.ent_embeddings[0]["args"][
            "qid2topk_eid"
        ] = self.qid2topkeid
        learned_emb = TopKEntityEmb(
            main_args=self.args,
            emb_args=self.args.data_config.ent_embeddings[0]["args"],
            entity_symbols=self.entity_symbols,
            key="learned",
            cpu=True,
            normalize=False,
            dropout1d_perc=0.0,
            dropout2d_perc=0.0,
        )
        num_new_eids_padunk = 5
        eid2topkeid_gold = torch.tensor([0, 1, 3, 3, 2, num_new_eids_padunk - 1])
        assert torch.equal(eid2topkeid_gold, learned_emb.eid2topkeid)
        assert list(learned_emb.learned_entity_embedding.weight.shape) == [
            num_new_eids_padunk,
            self.learned_embedding_size,
        ]

    def test_static_embedding(self):
        emb = torch.randn(self.entity_symbols.num_entities_with_pad_and_nocand, 150)
        torch.save((self.entity_symbols.get_qid2eid(), emb), self.static_emb)

        self.args.data_config.ent_embeddings[0]["args"]["emb_file"] = self.static_emb
        del self.args.data_config.ent_embeddings[0]["args"]["learned_embedding_size"]

        learned_emb = StaticEmb(
            main_args=self.args,
            emb_args=self.args.data_config.ent_embeddings[0]["args"],
            entity_symbols=self.entity_symbols,
            key="learned",
            cpu=True,
            normalize=False,
            dropout1d_perc=0.0,
            dropout2d_perc=0.0,
        )
        # The first and last rows will be zero as no entities have those eids
        emb[0] = torch.zeros(emb.shape[1])
        emb[-1] = torch.zeros(emb.shape[1])
        np.testing.assert_array_equal(emb.numpy(), learned_emb.entity2static)


class TestTypeEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        emmental.init(log_dir="test/temp_log")
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_embeddings.json"
        )
        self.entity_symbols = EntitySymbolsSubclass()
        self.entity_symbols_extra = EntitySymbolsSubclassExtra()
        self.type_file = os.path.join(
            self.args.data_config.emb_dir, "temp_type_file.json"
        )
        self.type_vocab_file = os.path.join(
            self.args.data_config.emb_dir, "temp_type_vocab.json"
        )
        self.regularization_csv = os.path.join(
            self.args.data_config.data_dir, "test_reg.csv"
        )

    def tearDown(self) -> None:
        if os.path.exists(self.type_file):
            os.remove(self.type_file)
        if os.path.exists(self.type_vocab_file):
            os.remove(self.type_vocab_file)
        if os.path.exists(self.regularization_csv):
            os.remove(self.regularization_csv)
        dir = os.path.join(
            self.args.data_config.data_dir, self.args.data_config.data_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        dir = os.path.join(
            self.args.data_config.entity_dir, self.args.data_config.entity_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        dir = os.path.join("test/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_build_type_table(self):
        type_data = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [], "Q4": [7, 8, 9]}
        type_vocab = {
            "T1": 1,
            "T2": 2,
            "T3": 3,
            "T4": 4,
            "T5": 5,
            "T6": 6,
            "T7": 7,
            "T8": 8,
            "T9": 9,
        }
        utils.dump_json_file(self.type_file, type_data)
        utils.dump_json_file(self.type_vocab_file, type_vocab)

        true_type_table = torch.tensor(
            [[0, 0, 0], [1, 2, 3], [4, 5, 6], [0, 0, 0], [7, 8, 9], [0, 0, 0]]
        ).long()
        true_type2row = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        pred_type_table, type2row, max_labels = TypeEmb.build_type_table(
            self.type_file,
            self.type_vocab_file,
            max_types=3,
            entity_symbols=self.entity_symbols,
        )
        assert torch.equal(pred_type_table, true_type_table)
        self.assertDictEqual(true_type2row, type2row)
        # there are 9 real types so we expect (including unk and pad) there to be type indices up to 10
        assert max_labels == 10

    def test_build_type_table_pad_types(self):
        type_data = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [], "Q4": [7, 8, 9]}
        type_vocab = {
            "T1": 1,
            "T2": 2,
            "T3": 3,
            "T4": 4,
            "T5": 5,
            "T6": 6,
            "T7": 7,
            "T8": 8,
            "T9": 9,
        }
        utils.dump_json_file(self.type_file, type_data)
        utils.dump_json_file(self.type_vocab_file, type_vocab)

        true_type_table = torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 2, 3, 10],
                [4, 5, 6, 10],
                [0, 0, 0, 0],
                [7, 8, 9, 10],
                [0, 0, 0, 0],
            ]
        ).long()
        true_type2row = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        pred_type_table, type2row, max_labels = TypeEmb.build_type_table(
            self.type_file,
            self.type_vocab_file,
            max_types=4,
            entity_symbols=self.entity_symbols,
        )
        assert torch.equal(pred_type_table, true_type_table)
        self.assertDictEqual(true_type2row, type2row)
        # there are 9 real types so we expect (including unk and pad) there to be type indices up to 10
        assert max_labels == 10

    def test_build_type_table_too_many_types(self):
        type_data = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [], "Q4": [7, 8, 9]}
        type_vocab = {
            "T1": 1,
            "T2": 2,
            "T3": 3,
            "T4": 4,
            "T5": 5,
            "T6": 6,
            "T7": 7,
            "T8": 8,
            "T9": 9,
        }
        utils.dump_json_file(self.type_file, type_data)
        utils.dump_json_file(self.type_vocab_file, type_vocab)

        true_type_table = torch.tensor([[0], [1], [4], [0], [7], [0]]).long()
        true_type2row = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        pred_type_table, type2row, max_labels = TypeEmb.build_type_table(
            self.type_file,
            self.type_vocab_file,
            max_types=1,
            entity_symbols=self.entity_symbols,
        )
        assert torch.equal(pred_type_table, true_type_table)
        self.assertDictEqual(true_type2row, type2row)
        # there are 9 real types so we expect (including unk and pad) there to be type indices up to 10
        assert max_labels == 10

    def test_average_type(self):
        # "mock" the type embedding class
        class TypeEmbSubclass(TypeEmb):
            def __init__(self):
                self.num_types_with_pad_and_unk = 4

        type_emb = TypeEmbSubclass()
        typeids = torch.tensor([[[[1, 2], [1, 0], [3, 3], [0, 0]]]])
        embeds = torch.tensor([[1, 2, 3], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).float()
        embed_cands = embeds[typeids.long().unsqueeze(3)].squeeze(3)
        pred_avg_types = type_emb._selective_avg_types(typeids, embed_cands)
        exp_avg_types = torch.tensor(
            [[[[2.5, 2.5, 2.5], [2, 2, 2], [0, 0, 0], [1, 2, 3]]]]
        )
        np.testing.assert_array_equal(pred_avg_types, exp_avg_types)

    def test_load_regularization(self):
        regularization_data = [
            ["typeid", "regularization"],
            [0, 0.4],
            [1, 0.8],
            [5, 0.2],
            [6, 0.1],
            [7, 0.6],
            [8, 0.7],
        ]
        with open(self.regularization_csv, "w") as out_f:
            for li in regularization_data:
                out_f.write(f"{','.join(map(str, li))}\n")

        num_types_with_pad_and_unk = 11
        type2row_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9}
        typeid2reg = TypeEmb.load_regularization_mapping(
            self.args.data_config,
            num_types_with_pad_and_unk,
            type2row_dict,
            self.regularization_csv,
        )
        typeid2reg_gold = torch.tensor(
            [0.0, 0.4, 0.8, 0.0, 0.0, 0.0, 0.2, 0.1, 0.6, 0.7, 0.0]
        )
        assert torch.equal(typeid2reg_gold.float(), typeid2reg.float())


class TestKGEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        emmental.init(log_dir="test/temp_log")
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_embeddings.json"
        )
        self.entity_symbols = EntitySymbolsSubclass()
        self.kg_adj = os.path.join(self.args.data_config.emb_dir, "kg_adj.txt")
        self.kg_adj_json = os.path.join(self.args.data_config.emb_dir, "kg_adj.json")

    def tearDown(self) -> None:
        if os.path.exists(self.kg_adj):
            os.remove(self.kg_adj)
        if os.path.exists(self.kg_adj_json):
            os.remove(self.kg_adj_json)
        dir = os.path.join("test/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_load_kg_adj(self):
        kg_data = [["Q1", "Q2"], ["Q3", "Q2"]]
        with open(self.kg_adj, "w") as out_f:
            for li in kg_data:
                out_f.write(f"{' '.join(li)}\n")

        adj_out = KGAdjEmb.build_kg_adj(
            kg_adj_file=self.kg_adj,
            entity_symbols=self.entity_symbols,
            threshold=0,
            log_weight=False,
        )
        adj_out_gold = nx.adjacency_matrix(
            nx.Graph(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                )
            )
        )
        np.testing.assert_array_equal(adj_out.toarray(), adj_out_gold.toarray())

    def test_load_kg_adj_indices_txt(self):
        kg_data = [["Q1", "Q2"], ["Q3", "Q2"]]
        with open(self.kg_adj, "w") as out_f:
            for li in kg_data:
                out_f.write(f"{' '.join(li)}\n")

        adj_out = KGIndices.build_kg_adj(
            kg_adj_file=self.kg_adj,
            entity_symbols=self.entity_symbols,
            threshold=0,
            log_weight=False,
        )
        adj_out_gold = nx.adjacency_matrix(
            nx.Graph(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                )
            )
        )
        np.testing.assert_array_equal(adj_out.toarray(), adj_out_gold.toarray())

    def test_load_kg_adj_indices_json(self):
        kg_data = {"Q1": {"Q2": 100}, "Q3": {"Q2": 11}}
        utils.dump_json_file(self.kg_adj_json, kg_data)

        adj_out = KGIndices.build_kg_adj(
            kg_adj_file=self.kg_adj_json,
            entity_symbols=self.entity_symbols,
            threshold=10,
            log_weight=True,
        )
        adj_out_gold = nx.adjacency_matrix(
            nx.Graph(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, np.log(100), 0, 0, 0],
                        [0, np.log(100), 0, np.log(11), 0, 0],
                        [0, 0, np.log(11), 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                )
            )
        )
        np.testing.assert_allclose(adj_out.toarray(), adj_out_gold.toarray())

        # We filter out weights 10 or lower
        kg_data = {"Q1": {"Q2": 100}, "Q3": {"Q2": 10}}
        utils.dump_json_file(self.kg_adj_json, kg_data)

        adj_out = KGIndices.build_kg_adj(
            kg_adj_file=self.kg_adj_json,
            entity_symbols=self.entity_symbols,
            threshold=10,
            log_weight=True,
        )
        adj_out_gold = nx.adjacency_matrix(
            nx.Graph(
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, np.log(100), 0, 0, 0],
                        [0, np.log(100), 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                )
            )
        )
        np.testing.assert_allclose(adj_out.toarray(), adj_out_gold.toarray())


class TitleEmbMock(TitleEmb):
    def __init__(self):
        super(EntityEmb, self).__init__()
        self.M = 1
        self.K = 3
        self._dim = 5
        self.key = "title1"
        self.cpu = False
        self.normalize = False
        self.dropout1d_perc = 0.0
        self.dropout2d_perc = 0.0
        self.title_proj = Noop()
        self.merge_func = self.average_titles
        self.normalize_and_dropout_emb = Noop()


class TestTitleEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        emmental.init(log_dir="test/temp_log")
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_embeddings.json"
        )
        self.entity_symbols = EntitySymbolsSubclass()
        self.tokenizer = load_tokenizer()

    def tearDown(self) -> None:
        prep_dir = os.path.join(
            self.args.data_config.data_dir, self.args.data_config.data_prep_dir
        )
        if os.path.exists(prep_dir):
            shutil.rmtree(prep_dir)
        dir = os.path.join("test/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_title_table(self):
        (
            entity2titleid,
            entity2titlemask,
            entity2tokentypeid,
        ) = TitleEmb.build_title_table(
            tokenizer=self.tokenizer, entity_symbols=self.entity_symbols
        )
        # Check that we correctly convert tokens to IDs.
        # UNK word is id 0; alphabet starts at 1
        correct_ids = torch.zeros(6, 7)
        correct_mask = torch.zeros(6, 7)
        correct_tokentype = torch.zeros(6, 7)
        # QID1
        correct_ids[1] = torch.tensor([101, 170, 171, 172, 173, 174, 102])
        correct_mask[1] = torch.ones(7)
        correct_tokentype[1] = torch.ones(7)
        # QID2
        correct_ids[2, :3] = torch.tensor([101, 175, 102])
        correct_mask[2, :3] = torch.ones(3)
        correct_tokentype[2, :3] = torch.ones(3)
        # QID3
        correct_ids[3, :6] = torch.tensor([101, 173, 1181, 170, 171, 102])
        correct_mask[3, :6] = torch.ones(6)
        correct_tokentype[3, :6] = torch.ones(6)
        # QID4
        correct_ids[4, :5] = torch.tensor([101, 193, 194, 195, 102])
        correct_mask[4, :5] = torch.ones(5)
        correct_tokentype[4, :5] = torch.ones(5)
        assert torch.equal(correct_ids.long(), entity2titleid)
        assert torch.equal(correct_mask.long(), entity2titlemask)
        assert torch.equal(correct_tokentype.long(), entity2tokentypeid)

    def test_avg_basic(self):
        # size is batch*M*K, max_title_token_ids, dim (batch = 1, M = 1, K = 3, max_title_token_ids = 3)
        title_emb = torch.tensor(
            [
                [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ]
        )
        # This is what the BERT encoder downstream mask would be (1 for what to NOT pay attention to)
        subset_mask = torch.tensor([[0, 0, 0], [0, 1, 1], [1, 1, 1]]).bool()
        # Instantiating this so we can use the inherited average_titles method
        title_mock = TitleEmbMock()
        actual_embs = title_mock.average_titles(subset_mask, title_emb)
        expected_embs = torch.tensor(
            [[2.0, 2.0, 2.0, 2.0, 2.0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
        )
        assert torch.equal(actual_embs, expected_embs)

    def test_title_subset_idx(self):
        title_mock = TitleEmbMock()
        (
            entity2titleid,
            entity2titlemask,
            entity2tokentypeid,
        ) = TitleEmb.build_title_table(
            tokenizer=self.tokenizer, entity_symbols=self.entity_symbols
        )
        # batch x M x K
        full_idx = torch.tensor(
            [[[1, 5, 5], [2, 1, 3], [5, 5, 5]], [[4, 3, 5], [5, 5, 5], [5, 5, 5]]]
        )
        title_token_ids = entity2titleid[full_idx]
        title_mask = entity2titlemask[full_idx]
        title_token_type = entity2tokentypeid[full_idx]
        (
            subset_title_token_ids,
            subset_mask,
            subset_token_types,
            subset_idx,
        ) = title_mock.get_subset_title_embs(
            title_token_ids, title_mask, title_token_type
        )
        gold_subset_idx = torch.tensor([0, 3, 4, 5, 9, 10])
        # batch*M*K x title tokens
        gold_subset_title_token_ids = torch.tensor(
            [
                [101, 170, 171, 172, 173, 174, 102],
                [101, 175, 102, 0, 0, 0, 0],
                [101, 170, 171, 172, 173, 174, 102],
                [101, 173, 1181, 170, 171, 102, 0],
                [101, 193, 194, 195, 102, 0, 0],
                [101, 173, 1181, 170, 171, 102, 0],
            ]
        )
        # This is the mask used for BERT where 1 means "pay attention to"
        gold_subset_mask = (gold_subset_title_token_ids > 0).long()
        gold_subset_token_types = gold_subset_mask

        assert torch.equal(subset_idx, gold_subset_idx)
        assert torch.equal(subset_title_token_ids, gold_subset_title_token_ids)
        assert torch.equal(subset_mask, gold_subset_mask)
        assert torch.equal(subset_token_types, gold_subset_token_types)

    def test_title_reconstruct_subset_idx(self):
        title_mock = TitleEmbMock()
        subset_idx = torch.tensor([0, 1])
        title_emb = torch.tensor(
            [
                [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ]
        )
        # This is the Pytorch attention mask where 0 means "pay attention to"
        subset_mask = torch.tensor([[0, 0, 0], [0, 1, 1]]).bool()
        final_emb = title_mock.postprocess_embedding(
            title_emb, subset_mask, subset_idx, 1
        )
        gold_final_emb = torch.zeros(1, 1, 3, 5)
        gold_final_emb[0, 0, 0, :] = torch.tensor([2, 2, 2, 2, 2])
        gold_final_emb[0, 0, 1, :] = torch.tensor([1, 1, 1, 1, 1])

        assert torch.equal(final_emb, gold_final_emb)


class EmbeddingPayloadMock(EmbeddingPayload):
    def __init__(self, args, entity_symbols):
        super(EmbeddingPayloadBase, self).__init__()
        self.K = get_max_candidates(entity_symbols, args.data_config)
        self.M = args.data_config.max_aliases


class TestPayload(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_embeddings.json"
        )
        self.entity_symbols = EntitySymbolsSubclass()

    def test_payload(self):
        self.args.data_config.ent_embeddings.extend(
            [DottedDict({"key": "kg_emb"}), DottedDict({"key": "kg_rel"})]
        )
        emb_payload = EmbeddingPayloadMock(self.args, self.entity_symbols)
        emb_payload.linear_layers = nn.ModuleDict()
        emb_payload.linear_layers["project_embedding"] = NoopCat()
        emb_payload.position_enc = nn.ModuleDict()
        emb_payload.position_enc["alias"] = Noop()
        emb_payload.position_enc["alias_last_token"] = Noop()
        emb_payload.position_enc["alias_position_cat"] = NoopTakeFirst()

        batch_size = 3
        # Max aliases is set to 5 in the config so we need one position per alias per batch
        start_idx_pair = torch.zeros(batch_size, emb_payload.M)
        end_idx_pair = torch.zeros(batch_size, emb_payload.M)

        entity_embedding = {
            "kg_emb": torch.randn(batch_size, emb_payload.M, emb_payload.K, 5),
            "kg_rel": torch.randn(batch_size, emb_payload.M, emb_payload.K, 6),
        }
        alias_list = emb_payload(
            start_idx_pair,
            end_idx_pair,
            entity_embedding["kg_emb"],
            entity_embedding["kg_rel"],
        )
        # The embeddings have been concatenated together
        assert torch.isclose(
            alias_list,
            torch.cat([entity_embedding["kg_emb"], entity_embedding["kg_rel"]], dim=3),
        ).all()


if __name__ == "__main__":
    unittest.main()
