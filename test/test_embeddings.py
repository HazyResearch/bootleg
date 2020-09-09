import unittest
from argparse import Namespace
import pytest
import marisa_trie
import numpy as np
import torchtext
import torch
import torch.nn as nn

from bootleg.embeddings import AvgTitleEmb, LearnedEntityEmb, TypeEmb, EntityEmb
from bootleg.embeddings.word_embeddings import StandardWordEmb
from bootleg.layers.emb_combiner import EmbCombinerProj
from bootleg.layers.embedding_layers import EmbeddingLayer
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils, parser_utils, model_utils, train_utils
from bootleg.utils.classes.dotted_dict import DottedDict

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

class WordEmbeddingMock(StandardWordEmb):
    def __init__(self, args, main_args, word_symbols):
        super(WordEmbeddingMock, self).__init__(args, main_args, word_symbols)
        self.position_words = Noop()
        self.layer_norm = Noop()
        self.dropout = Noop()

class EntitySymbolsSubclass(EntitySymbols):
    def __init__(self):
        self.max_candidates = 3
        # Used if we need to do any string searching for aliases. This keep track of the largest n-gram needed.
        self.max_alias_len = 1
        self._qid2title = {"Q1": "a b c d e", "Q2": "f", "Q3": "dd a b", "Q4": "x y z"}
        self._qid2eid = {"Q1" : 1, "Q2": 2, "Q3": 3, "Q4": 4}
        self._alias2qids = {"alias1": [["Q1", 10], ["Q2", 3]], "alias2": [["Q3", 100]], "alias3": [["Q1", 15], ["Q4", 5]]}
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2

# same as the above but with an extra entity and alias to test adding synthetic profiles
class EntitySymbolsSubclassExtra(EntitySymbols):
    def __init__(self):
        self._qid2title = {"Q1": "a b c d e", "Q2": "f", "Q3": "dd a b", "Q4": "x y z", "Q5": "d"}
        self._qid2eid = {"Q1" : 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}
        self._alias2qids = {"alias1": [["Q1", 10], ["Q2", 3]], "alias2": [["Q3", 100]], "alias3": [["Q1", 15], ["Q4", 5]], "alias4": [["Q5", 1.0]]}
        self._alias_trie = marisa_trie.Trie(self._alias2qids.keys())
        self.num_entities = len(self._qid2eid)
        self.num_entities_with_pad_and_nocand = self.num_entities + 2

def get_pos_in_sent():
    return [None, None]

class TestLoadEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parser_utils.get_full_config(
            "test/run_args/test_embeddings.json")
        self.args.data_config.ent_embeddings = [
            DottedDict(
            {
                "key": "learned1",
                "load_class": "LearnedEntityEmb",
                "args": {
                    "learned_embedding_size": 5,
                    "tail_init": False
                }
            }),
            DottedDict(
            {
                "key": "learned2",
                "load_class": "LearnedEntityEmb",
                "args": {
                    "learned_embedding_size": 5,
                    "tail_init": False
                }
            }),
            DottedDict(
            {
                "key": "learned3",
                "load_class": "LearnedEntityEmb",
                "args": {
                    "learned_embedding_size": 5,
                    "tail_init": False
                }
            }),
            DottedDict(
            {
                "key": "learned4",
                "load_class": "LearnedEntityEmb",
                "args": {
                    "learned_embedding_size": 5,
                    "tail_init": False
                }
            }),
            DottedDict(
            {
                "key": "learned5",
                "load_class": "LearnedEntityEmb",
                "args": {
                    "learned_embedding_size": 5,
                    "tail_init": False
                }
            }),
        ]
        self.word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        self.entity_symbols = EntitySymbolsSubclass()

    def test_embedding_load(self):
        emb_layer = EmbeddingLayer(self.args, "cpu", self.entity_symbols, self.word_symbols)
        # Asserts that the order is the same
        assert [ent.key for ent in emb_layer.entity_embs.values()] == ["learned1", "learned2", "learned3", "learned4", "learned5"]

class TestWordEmbeddings(unittest.TestCase):

    def setUp(self) -> None:
        #TODO: replace with custom vocab file and not GloVE
        self.args = parser_utils.get_full_config(
            "test/run_args/test_embeddings.json")
        self.word_symbols = data_utils.load_wordsymbols(self.args.data_config)

    def test_word_symbol_convert_tokens_to_ids(self):
        # Check that we correctly convert tokens to IDs.
        word_embeddings = torchtext.vocab.Vectors(
            self.args.data_config.word_embedding.custom_vocab_embedding_file,
            cache=self.args.data_config.emb_dir)
        good_tokens = "a b c z".split(" ")
        tokens = "a b xx c dd z".split(" ")
        correct_ids = [word_embeddings.stoi[t]+1 for t in good_tokens]
        correct_ids.insert(2, 0)
        correct_ids.insert(4, 0)
        processed_tokens = self.word_symbols.convert_tokens_to_ids(tokens)
        self.assertListEqual(correct_ids, processed_tokens)

    def test_word_embedding_setup(self):
        word_embeddings = torchtext.vocab.Vectors(
            self.args.data_config.word_embedding.custom_vocab_embedding_file,
            cache=self.args.data_config.emb_dir)
        good_tokens = "a b c z".split(" ")
        tokens = "a b xx c dd z".split(" ")
        correct_vecs = [word_embeddings.vectors[word_embeddings.stoi[t]] for t in good_tokens]
        correct_vecs.insert(2, torch.zeros(word_embeddings.vectors.shape[1]))
        correct_vecs.insert(4, torch.zeros(word_embeddings.vectors.shape[1]))
        #batch size is 1 so must unsqueeze
        correct_vecs = torch.stack(correct_vecs).unsqueeze(0)
        self.word_emb = WordEmbeddingMock(
            self.args.data_config.word_embedding, self.args, self.word_symbols)
        model_vecs = self.word_emb(torch.tensor(
            self.word_symbols.convert_tokens_to_ids(tokens)).unsqueeze(0)).tensor
        assert torch.equal(correct_vecs, model_vecs)

class TestTitleEmbeddings(unittest.TestCase):

    def setUp(self) -> None:
        self.args = parser_utils.get_full_config(
            "test/run_args/test_embeddings.json")
        self.word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        self.entity_symbols = EntitySymbolsSubclass()
        self.word_emb = WordEmbeddingMock(
            self.args.data_config.word_embedding, self.args, self.word_symbols)
        self.title_emb = AvgTitleEmb(self.args,
            self.args.data_config.ent_embeddings[1],
            "cpu", self.entity_symbols, self.word_symbols,
            word_emb=self.word_emb, key="avg_title")

    def average_titles(self, entity_package):
        batch, M, K = entity_package.tensor.shape
        word_title_ids = self.title_emb.entity2titleid_table[entity_package.tensor.long()]
        subset_idx, word_title_ids_subset, mask_subset, embs_subset = self.title_emb.get_subset_title_embs(word_title_ids)
        # sent_emb used for soft attention
        embs_subset = self.title_emb.average_titles(entity_package, subset_idx, word_title_ids_subset, mask_subset, embs_subset, sent_emb=None)
        batch_title_emb = torch.zeros(batch, M, K, self.title_emb.orig_dim).to(self.title_emb.model_device)
        batch_title_emb[subset_idx.reshape(batch, M, K)] = embs_subset
        return batch_title_emb

    def test_title_table(self):
        # Check that we correctly convert tokens to IDs.
        # UNK word is id 0; alphabet starts at 1
        correct_ids = torch.ones(6, 5)*-1
        # QID1
        correct_ids[1] = torch.tensor([1, 2, 3, 4, 5])
        # QID2
        correct_ids[2] = torch.tensor([6, -1, -1, -1, -1])
        # QID3
        correct_ids[3] = torch.tensor([0, 1, 2, -1, -1])
        # QID4
        correct_ids[4] = torch.tensor([24, 25, 26, -1, -1])
        assert torch.equal(correct_ids.long(), self.title_emb.entity2titleid_table)

    def test_avg_basic(self):
        entity_ids = torch.tensor([[[1,1]]])
        entity_package = DottedDict(tensor=entity_ids, pos_in_sent=get_pos_in_sent(), alias_indices=None, mask=None, normalize=False, key="key", dim=0)
        actual_embs = self.average_titles(entity_package)
        expected_embs = torch.tensor([[[[0.2, 0.2, 0.2, 0.2, 0.2] + [0]*21, [0.2, 0.2, 0.2, 0.2, 0.2] + [0]*21]]])
        assert torch.equal(actual_embs, expected_embs)

    def test_avg_padded(self):
        entity_ids = torch.tensor([[[2]]])
        entity_package = DottedDict(tensor=entity_ids, pos_in_sent=get_pos_in_sent(), alias_indices=None, mask=None, normalize=False, key="key", dim=0)
        actual_embs = self.average_titles(entity_package)
        expected_embs = torch.tensor([[[[0, 0, 0, 0, 0, 1.0] + [0]*20]]])
        assert torch.equal(actual_embs, expected_embs)

    def test_avg_unk(self):
        entity_ids = torch.tensor([[[3]]])
        entity_package = DottedDict(tensor=entity_ids, pos_in_sent=get_pos_in_sent(), alias_indices=None, mask=None, normalize=False, key="key", dim=0)
        actual_embs = self.average_titles(entity_package)
        expected_embs = torch.tensor([[[[0.5, 0.5] + [0]*24]]])
        assert torch.equal(actual_embs, expected_embs)

class TestLearnedEmbedding(unittest.TestCase):
    def setUp(self):
        self.entity_symbols = EntitySymbolsSubclass()
        self.hidden_size = 30
        self.learned_embedding_size = 50
        self.args = parser_utils.get_full_config(
            "test/run_args/test_embeddings.json")
        self.args.model_config.hidden_size = self.hidden_size
        emb_args = DottedDict({'learned_embedding_size': self.learned_embedding_size})
        self.learned_emb = LearnedEntityEmb(main_args=self.args, emb_args=emb_args,
            model_device='cpu', entity_symbols=self.entity_symbols,
            word_symbols=None, word_emb=None, key="learned")

    # confirm dimension of output is correct
    def test_forward_dimension(self):
        entity_ids = torch.tensor([[[0,1], [2,2]]])
        entity_package = DottedDict(tensor=entity_ids, pos_in_sent=get_pos_in_sent(), alias_indices=None, mask=None, normalize=False, key="key", dim=0)
        actual_out = self.learned_emb(entity_package, batch_prepped_data={}, batch_on_the_fly_data={}, sent_emb=None)
        assert ([entity_ids.shape[0], entity_ids.shape[1], entity_ids.shape[2],
            self.learned_embedding_size] == list(actual_out.tensor.size()))

    def test_initialization(self):
        self.learned_entity_embedding = nn.Embedding(
            self.entity_symbols.num_entities_with_pad_and_nocand, 4, padding_idx=-1, sparse=True)
        gold_emb = self.learned_entity_embedding.weight.data[:]

        gold_emb[1:] = torch.tensor([
            [1.0,2.0,3.0,4.0],
            [1.0,2.0,3.0,4.0],
            [1.0,2.0,3.0,4.0],
            [1.0,2.0,3.0,4.0],
            [0.,0.,0.,0.]
        ])

        init_vec = torch.tensor([1.0,2.0,3.0,4.0])
        init_vec_out = model_utils.init_tail_embeddings(self.learned_entity_embedding, {}, self.entity_symbols, pad_idx=-1, vec=init_vec)
        assert torch.equal(init_vec, init_vec_out)
        assert torch.equal(gold_emb, self.learned_entity_embedding.weight.data)



# "mock" the type embedding class
class TypeEmbSubclass(TypeEmb):
    def __init__(self):
        self.num_types_with_pad_and_unk=4

class TestTypeEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.entity_symbols = EntitySymbolsSubclass()
        self.entity_symbols_extra = EntitySymbolsSubclassExtra()
        self.type_labels = 'test/data/entity_embeddings/type_file.json'
        self.type_labels_new = 'test/data/entity_embeddings/type_file_new.json'
        self.type_labels_new_two = 'test/data/entity_embeddings/type_file_new_two.json'

    def test_build_type_table(self):
        true_type_table = torch.tensor([
            [0,0,0],
            [1,2,3],
            [4,5,6],
            [0,0,0],
            [7,8,9],
            [0,0,0]
        ]).long()
        true_type2row = np.array([1,2,3,4,5,6,7,8,9])
        pred_type_table, type2row, max_labels = TypeEmb.build_type_table(self.type_labels, max_types=3,
            entity_symbols=self.entity_symbols)
        assert torch.equal(pred_type_table, true_type_table)
        np.testing.assert_array_equal(true_type2row, type2row)
        # there are 9 real types so we expect (including unk and pad) there to be type indices up to 10
        assert max_labels == 10

    def test_build_type_table_pad_types(self):
        true_type_table = torch.tensor([
            [0,0,0,0],
            [1,2,3,10],
            [4,5,6,10],
            [0,0,0,0],
            [7,8,9,10],
            [0,0,0,0]
        ]).long()
        true_type2row = np.array([1,2,3,4,5,6,7,8,9])
        pred_type_table, type2row, max_labels = TypeEmb.build_type_table(self.type_labels, max_types=4,
            entity_symbols=self.entity_symbols)
        print(true_type_table, pred_type_table)
        assert torch.equal(pred_type_table, true_type_table)
        np.testing.assert_array_equal(true_type2row, type2row)
        # there are 9 real types so we expect (including unk and pad) there to be type indices up to 10
        assert max_labels == 10

    def test_build_type_table_too_many_types(self):
        true_type_table = torch.tensor([
            [0],
            [1],
            [4],
            [0],
            [7],
            [0]
        ]).long()
        true_type2row = np.array([1,2,3,4,5,6,7,8,9])
        pred_type_table, type2row, max_labels = TypeEmb.build_type_table(self.type_labels, max_types=1,
            entity_symbols=self.entity_symbols)
        print(true_type_table, pred_type_table)
        assert torch.equal(pred_type_table, true_type_table)
        np.testing.assert_array_equal(true_type2row, type2row)
        # there are 9 real types so we expect (including unk and pad) there to be type indices up to 10
        assert max_labels == 10

    def test_average_type(self):
        type_emb = TypeEmbSubclass()
        typeids = torch.tensor([[[[1,2], [1,0], [3,3], [0,0]]]])
        embeds = torch.tensor([[1, 2, 3], [2, 2, 2], [3, 3, 3], [0, 0, 0]]).float()
        embed_cands = embeds[typeids.long().unsqueeze(3)].squeeze(3)
        pred_avg_types = type_emb._selective_avg_types(typeids, embed_cands)
        exp_avg_types = torch.tensor([[[[2.5, 2.5, 2.5], [2, 2, 2], [0, 0, 0], [1, 2, 3]]]])
        np.testing.assert_array_equal(pred_avg_types, exp_avg_types)


# Embedding class that does not declare a normalize attribute. We check that the normalize attribute is instantiated and need
# to test that this feature works.
class EntityEmbNoNorm(EntityEmb):
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(EntityEmbNoNorm, self).__init__(main_args=main_args, emb_args=emb_args, model_device=model_device,
            entity_symbols=entity_symbols, word_symbols=word_symbols, word_emb=word_emb, key=key)
        self._dim = main_args.model_config.hidden_size

    # For some reason, the check on parameters only happens during the forwad call if an embedding package is returned
    # So we need some mock forward call that does this.
    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        emb = self._package(tensor = torch.randn(5,5), pos_in_sent=get_pos_in_sent(), alias_indices=None, mask = None)
        return emb

class TestKGEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parser_utils.get_full_config(
            "test/run_args/test_embeddings.json")
        self.word_symbols = data_utils.load_wordsymbols(self.args.data_config)
        self.entity_symbols = EntitySymbolsSubclass()

    def test_kg_norm(self):
        emb_sizes = {"kg_emb": 5, "kg_rel": 6}
        sent_emb_size = 10
        self.args.data_config.ent_embeddings.extend([DottedDict({"key": "kg_emb"}), DottedDict({"key": "kg_rel"})])
        emb_combiner = EmbCombinerProj(self.args, emb_sizes,
            sent_emb_size, self.word_symbols, self.entity_symbols)
        emb_combiner.linear_layers = nn.ModuleDict()
        emb_combiner.linear_layers['project_embedding'] = NoopCat()
        emb_combiner.position_enc = nn.ModuleDict()
        emb_combiner.position_enc['alias'] = Noop()
        emb_combiner.position_enc['alias_last_token'] = Noop()
        emb_combiner.position_enc['alias_position_cat'] = NoopTakeFirst()
        batch_size = 3
        num_words = 5
        sent_embedding = DottedDict(
            tensor=torch.randn(batch_size, num_words, sent_emb_size),
            downstream_mask=None,
            mask=None,
            key="sent_emb",
            dim=10
        )
        # Max aliases is set to 5 in the config so we need one position per alias per batch
        alias_idx_pair_sent = [
            torch.tensor([[0]*emb_combiner.M, [0]*emb_combiner.M]),
            torch.tensor([[0]*emb_combiner.M, [0]*emb_combiner.M])
        ]

        entity_embedding = [
            DottedDict(
                tensor=torch.randn(batch_size, emb_combiner.M, emb_combiner.K, 5),
                pos_in_sent=get_pos_in_sent(),
                alias_indices=None,
                mask=None,
                normalize=True,
                key="kg_emb",
                dim=5
            ),
            DottedDict(
                tensor=torch.randn(batch_size, emb_combiner.M, emb_combiner.K, 6),
                pos_in_sent=get_pos_in_sent(),
                alias_indices=None,
                mask=None,
                normalize=False,
                key="kg_rel",
                dim=6
            )
        ]
        norm_1 = entity_embedding[0].tensor.norm(p=2, dim=3)
        norm_2 = entity_embedding[1].tensor.norm(p=2, dim=3)
        entity_mask = None
        _, res_package = emb_combiner(sent_embedding, alias_idx_pair_sent, entity_embedding, entity_mask)
        alias_list = res_package.tensor
        # The embeddings have been normalized and concatenated together
        assert torch.isclose(alias_list[:,:,:,:5].norm(p=2, dim=3), torch.ones_like(norm_1)).all()
        assert torch.isclose(alias_list[:,:,:,5:].norm(p=2, dim=3), norm_2).all()

    def test_no_norm(self):
        learned_emb = EntityEmbNoNorm(main_args=self.args, emb_args={},
            model_device='cpu', entity_symbols=self.entity_symbols,
            word_symbols=None, word_emb=None, key="test")
        # The exception on paramters happens in a forward call
        with self.assertRaises(Exception) as context:
            learned_emb(entity_package=None, batch_prepped_data={}, batch_on_the_fly_data={}, sent_emb=None)
        self.assertTrue(type(context.exception) == AttributeError)


if __name__ == '__main__':
    unittest.main()
